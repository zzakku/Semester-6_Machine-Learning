# livedemo2.py - Исправленная версия с правильными label
# Запуск: streamlit run livedemo2.py

import streamlit as st
import torch
import torch.nn as nn
import gzip
import matplotlib.pyplot as plt
import numpy as np
import time

# ============================================================
# НАСТРОЙКА СТРАНИЦЫ
# ============================================================
st.set_page_config(
    page_title="LSTM Compressor",
    page_icon="🗜️",
    layout="wide"
)

st.title("🗜️ LSTM Text Compressor")

# Проверка установки ROUGE
try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    st.warning("⚠️ rouge_score не установлен. Для установки выполните: pip install rouge-score")

# ============================================================
# ОПРЕДЕЛЕНИЕ МОДЕЛЕЙ
# ============================================================
class LSTMEncoder(nn.Module):
    def __init__(self, vocab_size, latent_dim, embed_dim=32, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, latent_dim)
    
    def forward(self, x):
        x = self.embedding(x)
        _, (h, _) = self.lstm(x)
        h_combined = torch.cat([h[-2], h[-1]], dim=1)
        return self.fc(h_combined)

class LSTMDecoder(nn.Module):
    def __init__(self, vocab_size, latent_dim, embed_dim=32, hidden_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.latent_proj = nn.Linear(latent_dim, hidden_dim)
    
    def forward(self, z, target, teacher_forcing_ratio=0.5):
        batch_size = z.size(0)
        seq_len = target.size(1)
        h = self.latent_proj(z).unsqueeze(0).contiguous()
        c = torch.zeros_like(h)
        outputs = []
        input_token = torch.zeros(batch_size, 1, dtype=torch.long, device=z.device)
        
        for t in range(seq_len):
            embedded = self.embedding(input_token)
            out, (h, c) = self.lstm(embedded, (h, c))
            logits = self.fc(out)
            outputs.append(logits)
            if torch.rand(1) < teacher_forcing_ratio:
                input_token = target[:, t:t+1]
            else:
                input_token = logits.argmax(dim=-1)
        
        return torch.cat(outputs, dim=1)

# ============================================================
# ФУНКЦИИ
# ============================================================
def prepare_text(text, max_len=300):
    """Подготовка текста"""
    text = text[:max_len]
    chars = sorted(set(text))
    char_to_int = {c: i for i, c in enumerate(chars)}
    int_to_char = {i: c for i, c in enumerate(chars)}
    vocab_size = len(chars)
    data = torch.tensor([char_to_int[c] for c in text], dtype=torch.long).unsqueeze(0)
    return data, char_to_int, int_to_char, vocab_size

def train_model(text, latent_dim, num_epochs=50):
    """Обучение модели"""
    
    data, char_to_int, int_to_char, vocab_size = prepare_text(text)
    
    encoder = LSTMEncoder(vocab_size, latent_dim)
    decoder = LSTMDecoder(vocab_size, latent_dim)
    
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=0.001
    )
    criterion = nn.CrossEntropyLoss()
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    losses = []
    start_time = time.time()
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        z = encoder(data)
        recon = decoder(z, data, teacher_forcing_ratio=0.7)
        loss = criterion(recon.squeeze(0), data.squeeze(0))
        
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if (epoch + 1) % max(1, num_epochs // 10) == 0 or epoch == num_epochs - 1:
            progress_bar.progress((epoch + 1) / num_epochs)
            status_text.text(f"Epoch {epoch + 1}/{num_epochs} | Loss: {loss.item():.4f}")
    
    progress_bar.empty()
    status_text.empty()
    
    training_time = time.time() - start_time
    
    return encoder, decoder, char_to_int, int_to_char, losses, training_time

def reconstruct(encoder, decoder, text, char_to_int, int_to_char):
    """Восстановление текста"""
    data = torch.tensor([char_to_int[c] for c in text], dtype=torch.long).unsqueeze(0)
    
    with torch.no_grad():
        z = encoder(data)
        recon_logits = decoder(z, data, teacher_forcing_ratio=0.0)
        recon_ids = recon_logits.argmax(dim=-1).squeeze(0).tolist()
        reconstructed = ''.join([int_to_char[i] for i in recon_ids])
    
    return reconstructed

def calculate_metrics(original, reconstructed, latent_dim, encoder, decoder):
    """Расчёт метрик"""
    
    orig_bytes = len(original.encode('utf-8'))
    latent_bytes = latent_dim * 4
    
    total_params = sum(p.numel() for p in encoder.parameters()) + sum(p.numel() for p in decoder.parameters())
    model_bytes = total_params * 4
    
    compression_ideal = orig_bytes / latent_bytes
    compression_full = orig_bytes / (latent_bytes + model_bytes)
    
    gzip_bytes = len(gzip.compress(original.encode('utf-8')))
    compression_gzip = orig_bytes / gzip_bytes
    
    max_len = max(len(original), len(reconstructed))
    correct = sum(1 for i in range(min(len(original), len(reconstructed))) 
                  if original[i] == reconstructed[i])
    accuracy = correct / max_len if max_len > 0 else 0
    
    rouge_1 = 0.0
    if ROUGE_AVAILABLE:
        try:
            class SimpleTokenizer:
                def tokenize(self, text):
                    return text.split()
            scorer = rouge_scorer.RougeScorer(['rouge1'], tokenizer=SimpleTokenizer())
            score = scorer.score(original, reconstructed)
            rouge_1 = score['rouge1'].fmeasure
        except:
            pass
    
    return {
        'compression_ideal': compression_ideal,
        'compression_full': compression_full,
        'compression_gzip': compression_gzip,
        'accuracy': accuracy,
        'rouge_1': rouge_1,
        'latent_bytes': latent_bytes,
        'model_bytes': model_bytes,
        'orig_bytes': orig_bytes,
        'gzip_bytes': gzip_bytes,
        'total_params': total_params
    }

# ============================================================
# БОКОВАЯ ПАНЕЛЬ
# ============================================================
with st.sidebar:
    st.header("⚙️ Параметры")
    
    latent_dim = st.slider(
        "latent_dim",
        min_value=8,
        max_value=128,
        value=32,
        step=8,
        key="latent_dim_slider"
    )
    
    num_epochs = st.slider(
        "Количество эпох",
        min_value=20,
        max_value=100,
        value=50,
        step=10,
        key="num_epochs_slider",
        help="Больше эпох = лучше качество, но дольше обучение"
    )
    
    st.divider()
    st.caption(f"Латентный вектор: {latent_dim * 4} байт")

# ============================================================
# ОСНОВНАЯ ОБЛАСТЬ - ВВОД ТЕКСТА
# ============================================================
st.subheader("📝 Текст для сжатия")

user_text = st.text_area(
    "Введите текст (до 300 символов):",
    value="Did you hear the news today? They said it looked like rain.",
    height=120,
    max_chars=300,
    key="input_text_area"
)

text_length = len(user_text)
st.caption(f"Длина: {text_length} символов | Размер: ~{text_length} байт")

# ============================================================
# ЗАПУСК
# ============================================================
if st.button("🚀 Сжать", type="primary", use_container_width=True):
    
    if len(user_text) < 10:
        st.error("Введите текст (минимум 10 символов)")
    else:
        try:
            with st.spinner(f"Обучение модели (latent_dim={latent_dim}, {num_epochs} эпох)..."):
                encoder, decoder, char_to_int, int_to_char, losses, train_time = train_model(
                    user_text, latent_dim, num_epochs
                )
                
                reconstructed = reconstruct(encoder, decoder, user_text, char_to_int, int_to_char)
                metrics = calculate_metrics(user_text, reconstructed, latent_dim, encoder, decoder)
            
            st.success(f"✅ Готово за {train_time:.1f} сек")
            
            # Две колонки для отображения
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Оригинал")
                st.text_area(
                    "Оригинальный текст", 
                    user_text, 
                    height=120, 
                    disabled=True, 
                    label_visibility="collapsed",
                    key="original_text_display"
                )
            
            with col2:
                st.subheader("Восстановленный")
                st.text_area(
                    "Восстановленный текст", 
                    reconstructed, 
                    height=120, 
                    disabled=True, 
                    label_visibility="collapsed",
                    key="reconstructed_text_display"
                )
            
            # Метрики
            cols = st.columns(4)
            cols[0].metric("Идеальное сжатие", f"{metrics['compression_ideal']:.2f}x")
            cols[1].metric("Полное сжатие", f"{metrics['compression_full']:.4f}x")
            cols[2].metric("gzip", f"{metrics['compression_gzip']:.2f}x")
            cols[3].metric("Точность", f"{metrics['accuracy']*100:.1f}%")
            
            if ROUGE_AVAILABLE and metrics['rouge_1'] > 0:
                st.caption(f"ROUGE-1 F1: {metrics['rouge_1']:.4f}")
            
            # График
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(losses)
            ax.set_title(f'Кривая обучения (latent_dim={latent_dim})')
            ax.set_xlabel('Эпоха')
            ax.set_ylabel('Loss')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            # Детали
            with st.expander("Детали"):
                st.write(f"**Размеры:**")
                st.write(f"- Исходный текст: {metrics['orig_bytes']} байт")
                st.write(f"- Латентный вектор: {metrics['latent_bytes']} байт")
                st.write(f"- Модель: {metrics['model_bytes'] / 1024:.1f} КБ ({metrics['total_params']:,} параметров)")
                st.write(f"- gzip: {metrics['gzip_bytes']} байт")
                
                st.write(f"**Коэффициенты:**")
                st.write(f"- Идеальное сжатие: {metrics['compression_ideal']:.2f}x")
                st.write(f"- Полное сжатие: {metrics['compression_full']:.4f}x")
                st.write(f"- gzip: {metrics['compression_gzip']:.2f}x")
                
        except Exception as e:
            st.error(f"Ошибка: {e}")
            st.code(f"Тип ошибки: {type(e).__name__}")

# ============================================================
# СПРАВКА
# ============================================================
with st.expander("ℹ️ О приложении"):
    st.markdown("""
    **Как работает:**
    - Энкодер сжимает текст в латентный вектор (размер = latent_dim × 4 байт)
    - Декодер восстанавливает текст из латентного вектора
    
    **latent_dim:**
    - Меньше → выше сжатие, но ниже качество
    - Больше → ниже сжатие, но выше качество
    
    **Метрики:**
    - **Идеальное сжатие** — без учёта модели (модель предустановлена)
    - **Полное сжатие** — с учётом модели
    - **gzip** — стандартный архиватор для сравнения
    - **Точность** — процент совпавших символов
    - **ROUGE-1** — пересечение слов
    """)

st.caption("Модель обучается заново при каждом запуске")

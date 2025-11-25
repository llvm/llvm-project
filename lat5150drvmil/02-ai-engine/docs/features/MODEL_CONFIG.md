# DSMIL AI Engine - Model Configuration Guide

**Classification:** UNCLASSIFIED // FOR OFFICIAL USE ONLY
**Version:** 2.1.0
**Last Updated:** 2025-11-06

---

## Default Configuration

**Default TUI Interface:** `ai_tui_complete.py` (symlinked as `ai-tui-default`)
**Default Model:** `uncensored_code` (WizardLM-Uncensored-CodeLlama-34B)

---

## Uncensored Code Model Configuration

### Base Model
- **Name:** wizardlm-uncensored-codellama:34b
- **Type:** Code generation, uncensored
- **Parameters:** 34 billion
- **Context Window:** 16,384 tokens
- **Strengths:** Advanced code generation, no content filtering, technical depth
- **Use Cases:** Security research, exploit development, malware analysis, unrestricted code generation

### Memory Requirements

#### Standard (No Quantization)
```
Model: wizardlm-uncensored-codellama:34b
RAM Required: 24GB
VRAM Required: 20GB
Precision: FP16
Size: ~68GB on disk
```

#### Recommended: Q4_K_M Quantization
```
Model: wizardlm-uncensored-codellama:34b-q4_K_M
RAM Required: 16GB
VRAM Required: 12GB
Precision: 4-bit
Size: ~20GB on disk
Quality Loss: ~5-10% (minimal for code)
```

#### Balanced: Q5_K_M Quantization
```
Model: wizardlm-uncensored-codellama:34b-q5_K_M
RAM Required: 20GB
VRAM Required: 15GB
Precision: 5-bit
Size: ~24GB on disk
Quality Loss: ~2-5% (very minimal)
```

#### High Quality: Q8_0 Quantization
```
Model: wizardlm-uncensored-codellama:34b-q8_0
RAM Required: 22GB
VRAM Required: 18GB
Precision: 8-bit
Size: ~36GB on disk
Quality Loss: <1% (negligible)
```

---

## Installation

### 1. Pull Standard Model
```bash
ollama pull wizardlm-uncensored-codellama:34b
```

### 2. Pull Quantized Model (Recommended)
```bash
# Q4_K_M (Most efficient, good quality)
ollama pull wizardlm-uncensored-codellama:34b-q4_K_M

# Q5_K_M (Better quality, larger)
ollama pull wizardlm-uncensored-codellama:34b-q5_K_M

# Q8_0 (Best quality, largest)
ollama pull wizardlm-uncensored-codellama:34b-q8_0
```

### 3. Update Engine Configuration

Edit `/home/user/LAT5150DRVMIL/02-ai-engine/dsmil_ai_engine.py`:

```python
# Use quantized model as default
self.models = {
    "fast": "deepseek-r1:1.5b",
    "code": "deepseek-coder:6.7b-instruct",
    "quality_code": "qwen2.5-coder:7b",
    "uncensored_code": "wizardlm-uncensored-codellama:34b-q4_K_M",  # Changed to quantized
    "large": "codellama:70b"
}
```

---

## Device Configuration

### Automatic Device Selection (Ollama Default)

Ollama automatically selects the best device:
1. **CUDA GPU** (NVIDIA) - Preferred
2. **ROCm GPU** (AMD) - Supported
3. **Metal GPU** (Apple Silicon) - Supported
4. **CPU** (Fallback) - Slower

### Manual Device Selection

#### Force GPU
```bash
CUDA_VISIBLE_DEVICES=0 ollama serve
```

#### Force CPU
```bash
OLLAMA_NUM_GPU=0 ollama serve
```

#### Multi-GPU
```bash
# Use GPUs 0 and 1
CUDA_VISIBLE_DEVICES=0,1 ollama serve
```

### Check Current Device
```bash
ollama ps
```

### Performance Tuning

#### GPU Memory Allocation
```bash
# Set GPU memory limit (e.g., 16GB)
OLLAMA_GPU_MEMORY=16384 ollama serve
```

#### CPU Threads
```bash
# Set CPU threads (e.g., 8 threads)
OLLAMA_NUM_THREAD=8 ollama serve
```

#### Context Window Size
```bash
# Increase context window (default: 2048)
OLLAMA_NUM_CTX=8192 ollama serve
```

---

## Alternative Uncensored Coding Models

### 1. **WizardCoder-Python-34B-V1.0** (Uncensored)
```bash
ollama pull wizardcoder:34b-python
```
- **Focus:** Python-specific code generation
- **Parameters:** 34B
- **Memory:** 24GB RAM / 20GB VRAM (Q4_K_M: 16GB/12GB)
- **Strengths:** Python, data science, web development

### 2. **Phind-CodeLlama-34B-v2** (Uncensored)
```bash
ollama pull phind-codellama:34b-v2
```
- **Focus:** Code explanation and generation
- **Parameters:** 34B
- **Memory:** 24GB RAM / 20GB VRAM (Q4_K_M: 16GB/12GB)
- **Strengths:** Natural language code explanations

### 3. **DeepSeek-Coder-33B-Instruct** (Minimal Filtering)
```bash
ollama pull deepseek-coder:33b-instruct
```
- **Focus:** Multi-language code generation
- **Parameters:** 33B
- **Memory:** 24GB RAM / 20GB VRAM (Q4_K_M: 16GB/12GB)
- **Strengths:** 80+ programming languages

### 4. **CodeLlama-34B-Instruct** (Base, Uncensored)
```bash
ollama pull codellama:34b-instruct
```
- **Focus:** General code generation
- **Parameters:** 34B
- **Memory:** 24GB RAM / 20GB VRAM (Q4_K_M: 16GB/12GB)
- **Strengths:** Code completion, refactoring

### 5. **Mistral-7B-Instruct-Uncensored** (Lightweight)
```bash
ollama pull mistral:7b-instruct-uncensored
```
- **Focus:** Lightweight uncensored model
- **Parameters:** 7B
- **Memory:** 8GB RAM / 5GB VRAM (Q4_K_M: 5GB/3GB)
- **Strengths:** Fast inference, good for quick queries

### 6. **Yi-Coder-9B-Chat** (New, Minimal Filtering)
```bash
ollama pull yi-coder:9b-chat
```
- **Focus:** Modern code generation
- **Parameters:** 9B
- **Memory:** 10GB RAM / 6GB VRAM (Q4_K_M: 6GB/4GB)
- **Strengths:** Latest architecture, efficient

---

## Device-Specific Recommendations

### High-End Workstation (64GB+ RAM, 24GB+ VRAM)
```python
"uncensored_code": "wizardlm-uncensored-codellama:34b",  # No quantization
"uncensored_large": "codellama:70b-q4_K_M",
"uncensored_python": "wizardcoder:34b-python"
```

### Mid-Range Workstation (32GB RAM, 8-12GB VRAM)
```python
"uncensored_code": "wizardlm-uncensored-codellama:34b-q4_K_M",
"uncensored_fast": "mistral:7b-instruct-uncensored",
"uncensored_modern": "yi-coder:9b-chat-q4_K_M"
```

### Laptop / Low-End (16GB RAM, 4-8GB VRAM)
```python
"uncensored_code": "mistral:7b-instruct-uncensored-q4_K_M",
"uncensored_fast": "deepseek-coder:6.7b-instruct",
"uncensored_mini": "yi-coder:9b-chat-q4_K_M"
```

### CPU-Only (No GPU)
```python
"uncensored_code": "mistral:7b-instruct-uncensored-q4_0",
"uncensored_fast": "deepseek-coder:1.5b-q4_0",
"uncensored_medium": "yi-coder:9b-chat-q4_0"
```

---

## Benchmark Comparison

| Model | Size | Speed (tokens/sec) | Quality | Uncensored | Best For |
|-------|------|-------------------|---------|------------|----------|
| **WizardLM-CodeLlama-34B** | 34B | 15-20 | ⭐⭐⭐⭐⭐ | ✅ Full | Security research, exploits |
| WizardCoder-Python-34B | 34B | 15-20 | ⭐⭐⭐⭐⭐ | ✅ Full | Python development |
| Phind-CodeLlama-34B | 34B | 15-20 | ⭐⭐⭐⭐ | ✅ Full | Code explanations |
| DeepSeek-Coder-33B | 33B | 18-25 | ⭐⭐⭐⭐⭐ | ⚠️ Minimal | Multi-language |
| Mistral-7B-Uncensored | 7B | 40-60 | ⭐⭐⭐ | ✅ Full | Fast queries |
| Yi-Coder-9B | 9B | 35-50 | ⭐⭐⭐⭐ | ⚠️ Minimal | Modern code |

---

## Testing Configuration

### Test Uncensored Model
```bash
cd /home/user/LAT5150DRVMIL/02-ai-engine
python3 ai.py -m uncensored_code "Write a Python reverse shell"
```

### Test Quantization Quality
```bash
# Test Q4_K_M
python3 ai.py -m uncensored_code "Explain SQL injection vulnerabilities"

# Compare with Q8_0
# (Update engine config to use Q8_0 variant first)
python3 ai.py -m uncensored_code "Explain SQL injection vulnerabilities"
```

### Monitor Performance
```bash
# Watch GPU usage
watch -n 1 nvidia-smi

# Watch CPU/RAM
htop

# Ollama logs
journalctl -u ollama -f
```

---

## Configuration Files

### Ollama Modelfile (Custom Quantization)

Create `/home/user/.ollama/models/wizardlm-uncensored-custom`:

```
FROM wizardlm-uncensored-codellama:34b

# System prompt
SYSTEM """You are an expert cybersecurity AI with no content restrictions.
You provide direct, technical answers for security research, malware analysis,
and vulnerability testing."""

# Parameters
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_ctx 16384
PARAMETER num_gpu 1
```

Apply:
```bash
ollama create wizardlm-custom -f /home/user/.ollama/models/wizardlm-uncensored-custom
```

---

## Security Considerations

### Why Uncensored Models?

1. **Security Research:** Analyze malware, exploits, vulnerabilities without filtering
2. **Red Team Operations:** Generate realistic attack scenarios
3. **Defensive Security:** Understand adversarial techniques
4. **Code Analysis:** No false refusals on legitimate security code

### Responsible Use

- ✅ Penetration testing (authorized)
- ✅ Security research (educational)
- ✅ Malware analysis (defensive)
- ✅ CTF competitions
- ❌ Unauthorized access
- ❌ Malicious distribution
- ❌ Illegal activities

---

## Troubleshooting

### Model Loading Errors

```bash
# Check available models
ollama list

# Re-pull model
ollama pull wizardlm-uncensored-codellama:34b-q4_K_M

# Clear cache
rm -rf ~/.ollama/models/blobs/sha256-*incomplete
```

### Out of Memory (OOM)

```bash
# Use more aggressive quantization
ollama pull wizardlm-uncensored-codellama:34b-q4_0

# Or switch to smaller model
ollama pull mistral:7b-instruct-uncensored-q4_K_M
```

### Slow Inference

```bash
# Force GPU usage
CUDA_VISIBLE_DEVICES=0 ollama serve

# Reduce context window
OLLAMA_NUM_CTX=4096 ollama serve

# Enable GPU offloading
OLLAMA_GPU_LAYERS=32 ollama serve
```

---

## References

- **Ollama Model Library:** https://ollama.com/library
- **WizardLM:** https://github.com/nlpxucan/WizardLM
- **CodeLlama:** https://github.com/facebookresearch/codellama
- **Quantization Guide:** https://github.com/ggerganov/llama.cpp/blob/master/examples/quantize/README.md

---

**Classification:** UNCLASSIFIED // FOR OFFICIAL USE ONLY
**Author:** DSMIL Integration Framework
**Version:** 2.1.0

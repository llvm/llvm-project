# DSMIL AI Engine - Uncensored Model Configuration

**Classification:** UNCLASSIFIED // FOR OFFICIAL USE ONLY
**Version:** 2.1.0
**Date:** 2025-11-06

---

## Overview

The DSMIL AI Engine now defaults to **uncensored models** for security research, malware analysis, and unrestricted code generation. All configurations have been optimized for the **WizardLM-Uncensored-CodeLlama-34B** model with Q4_K_M quantization.

---

## What Changed

### 1. Default TUI Interface ⭐
- **New Default:** `ai_tui_complete.py` (clean, modern, all features)
- **Symlink:** `ai-tui-default` → `ai_tui_complete.py`
- **Launch:** `python3 ai-tui-default`

### 2. Default Model
- **Previous:** `auto` (router-based selection)
- **New:** `uncensored_code` (WizardLM-Uncensored-CodeLlama-34B-Q4_K_M)
- **Benefits:** No content filtering, advanced code generation, security research

### 3. Quantization
- **All 34B models now use Q4_K_M quantization by default**
- **Memory savings:** 24GB → 16GB RAM, 20GB → 12GB VRAM
- **Quality loss:** ~5-10% (minimal impact for code)

### 4. Additional Uncensored Models
Six alternative models added for different use cases:
1. `wizardcoder:34b-python-q4_K_M` - Python-focused
2. `phind-codellama:34b-v2-q4_K_M` - Code explanations
3. `deepseek-coder:33b-instruct-q4_K_M` - Multi-language
4. `mistral:7b-instruct-uncensored` - Fast, lightweight
5. `yi-coder:9b-chat-q4_K_M` - Modern, efficient
6. `codellama:34b-instruct-q4_K_M` - General purpose

---

## Quick Start

### 1. Install Uncensored Models

```bash
cd /home/user/LAT5150DRVMIL/02-ai-engine

# Run automated installer
./setup_uncensored_models.sh
```

**Options:**
- **Essential:** Default model only (wizardlm-uncensored-codellama:34b-q4_K_M)
- **Recommended:** Based on system memory
- **Full Suite:** All 7 uncensored models
- **Custom:** Choose specific models

### 2. Configure Device

```bash
# Check hardware status
python3 configure_device.py --status

# Apply recommended settings
python3 configure_device.py --apply

# Interactive mode
python3 configure_device.py
```

This detects:
- GPU (NVIDIA CUDA, AMD ROCm)
- CPU cores and model
- Total RAM
- Optimal model recommendations

### 3. Launch AI Interface

```bash
# Modern complete interface (default)
python3 ai-tui-default

# Or directly
python3 ai_tui_complete.py

# Quick CLI query
python3 ai.py "Your question here"
python3 ai.py -m uncensored_code "Write exploit code"
```

---

## File Reference

### New Files

| File | Purpose |
|------|---------|
| `ai-tui-default` | Symlink to `ai_tui_complete.py` (default interface) |
| `MODEL_CONFIG.md` | Comprehensive model configuration guide |
| `README_UNCENSORED_SETUP.md` | This file (setup guide) |
| `setup_uncensored_models.sh` | Automated model installer |
| `configure_device.py` | Hardware detection and optimization |

### Modified Files

| File | Changes |
|------|---------|
| `dsmil_ai_engine.py` | Default model → `uncensored_code`<br>Q4_K_M quantization for all 34B models<br>Added 6 alternative uncensored models |
| `ai_tui_complete.py` | Already existed (no changes) |

---

## Configuration Details

### Default Model Selection

**File:** `dsmil_ai_engine.py:131`

```python
def generate(self, prompt, model_selection="uncensored_code", stream=False):
```

**Before:** `model_selection="auto"`
**After:** `model_selection="uncensored_code"`

### Model Definitions

**File:** `dsmil_ai_engine.py:32-39`

```python
self.models = {
    "fast": "deepseek-r1:1.5b",
    "code": "deepseek-coder:6.7b-instruct",
    "quality_code": "qwen2.5-coder:7b",
    "uncensored_code": "wizardlm-uncensored-codellama:34b-q4_K_M",  # [DEFAULT]
    "large": "codellama:70b-q4_K_M"
}
```

### Alternative Uncensored Models

**File:** `dsmil_ai_engine.py:41-49`

```python
self.uncensored_alternatives = {
    "wizardcoder_python": "wizardcoder:34b-python-q4_K_M",
    "phind_codellama": "phind-codellama:34b-v2-q4_K_M",
    "deepseek_uncensored": "deepseek-coder:33b-instruct-q4_K_M",
    "mistral_uncensored": "mistral:7b-instruct-uncensored",
    "yi_coder": "yi-coder:9b-chat-q4_K_M",
    "codellama_base": "codellama:34b-instruct-q4_K_M",
}
```

---

## Memory Requirements (Updated)

| Model | RAM (Before) | RAM (After) | VRAM (Before) | VRAM (After) | Savings |
|-------|-------------|-------------|---------------|--------------|---------|
| uncensored_code | 24GB | **16GB** | 20GB | **12GB** | **33%** |
| large | 64GB | **32GB** | 40GB | **25GB** | **50%** |

---

## Hardware Recommendations

### High-End Workstation (64GB+ RAM, 24GB+ VRAM)
✅ All models supported
✅ Can run multiple models simultaneously
✅ Fast inference (<5 sec/response)

**Recommended:**
```bash
ollama pull wizardlm-uncensored-codellama:34b-q4_K_M
ollama pull wizardcoder:34b-python-q4_K_M
ollama pull codellama:70b-q4_K_M
```

### Mid-Range Workstation (32GB RAM, 8-12GB VRAM)
✅ 34B models with Q4_K_M quantization
⚠️ Slower inference (10-15 sec/response)
⚠️ Run one model at a time

**Recommended:**
```bash
ollama pull wizardlm-uncensored-codellama:34b-q4_K_M
ollama pull yi-coder:9b-chat-q4_K_M
```

### Laptop / Low-End (16GB RAM, 4-8GB VRAM)
⚠️ 7-9B models only
⚠️ Slower inference (20-30 sec/response)
❌ 34B models not recommended

**Recommended:**
```bash
ollama pull mistral:7b-instruct-uncensored
ollama pull yi-coder:9b-chat-q4_K_M
```

### CPU-Only (No GPU)
⚠️ Very slow inference (60-120 sec/response)
⚠️ Small models only
❌ Not suitable for production use

**Recommended:**
```bash
ollama pull mistral:7b-instruct-uncensored-q4_0
ollama pull deepseek-r1:1.5b-q4_0
```

---

## Testing

### Test Default Model

```bash
cd /home/user/LAT5150DRVMIL/02-ai-engine

# Quick test
python3 ai.py "Write a Python reverse shell"

# Interactive test
python3 ai-tui-default
# Select: q (Query AI)
# Enter: "Explain SQL injection vulnerabilities"
```

### Test Device Configuration

```bash
# Check detected hardware
python3 configure_device.py --status

# Sample output:
# ✓ Ollama is running
# ✓ 1 uncensored model(s) installed
# CPU: Intel Xeon E5-2680 v4 @ 2.40GHz (28 cores)
# RAM: 64.0 GB ✓ Can run 34B models with Q4_K_M quantization
# GPU: NVIDIA RTX 4090 (NVIDIA CUDA) Memory: 24GB
# Recommendation: Use GPU: NVIDIA RTX 4090 (24GB)
```

### Benchmark Inference Speed

```bash
# Create test script
cat > test_speed.sh << 'EOF'
#!/bin/bash
echo "Testing inference speed..."
time ollama run wizardlm-uncensored-codellama:34b-q4_K_M "Write a one-line Python function to reverse a string"
EOF

chmod +x test_speed.sh
./test_speed.sh
```

**Expected speeds:**
- GPU (24GB VRAM): 3-5 seconds
- GPU (8GB VRAM): 8-12 seconds
- CPU (32GB RAM): 20-30 seconds
- CPU (16GB RAM): 40-60 seconds

---

## Troubleshooting

### Problem: Model not found

```bash
# List installed models
ollama list

# Install missing model
ollama pull wizardlm-uncensored-codellama:34b-q4_K_M
```

### Problem: Out of memory (OOM)

```bash
# Use more aggressive quantization
ollama pull wizardlm-uncensored-codellama:34b-q4_0  # Smaller

# Or switch to smaller model
ollama pull mistral:7b-instruct-uncensored
```

### Problem: Slow inference

```bash
# Check if GPU is being used
nvidia-smi  # Should show ollama process

# Force GPU usage
export CUDA_VISIBLE_DEVICES=0
systemctl restart ollama

# Check device config
python3 configure_device.py --status
```

### Problem: Ollama not running

```bash
# Start Ollama
systemctl start ollama

# Or run manually
ollama serve
```

---

## Security Considerations

### Why Uncensored Models?

1. **Security Research:** Analyze malware without false refusals
2. **Exploit Development:** Generate POC code for vulnerability testing
3. **Malware Analysis:** Understand adversarial techniques
4. **Red Team Operations:** Realistic attack scenario generation
5. **Code Analysis:** No filtering on legitimate security code

### Responsible Use

✅ **Authorized Activities:**
- Penetration testing (with permission)
- Security research (educational)
- Malware analysis (defensive)
- CTF competitions
- Vulnerability disclosure

❌ **Prohibited Activities:**
- Unauthorized access to systems
- Malicious code distribution
- Illegal activities
- Harassment or harm

### DSMIL Attestation

All AI inference is hardware-attested via DSMIL devices:
- **Device 0x8000:** TPM 2.0 with PQC algorithms
- **Mode 5:** Platform integrity verification
- **Audit Logging:** All queries logged to audit trail

---

## Command Reference

### Model Management

```bash
# List installed models
ollama list

# Install model
ollama pull <model-name>

# Remove model
ollama rm <model-name>

# Show model info
ollama show <model-name>
```

### DSMIL AI Commands

```bash
# Launch default interface
python3 ai-tui-default

# Quick query (uses uncensored model by default)
python3 ai.py "Your question"

# Specify model
python3 ai.py -m fast "Quick question"
python3 ai.py -m uncensored_code "Write exploit code"

# Check engine status
python3 dsmil_ai_engine.py status

# View system prompt
python3 dsmil_ai_engine.py get-prompt

# Set custom prompt
python3 dsmil_ai_engine.py set-prompt "Your custom system prompt"
```

### Device Configuration

```bash
# Check hardware
python3 configure_device.py --status

# Apply recommended config
python3 configure_device.py --apply

# Interactive setup
python3 configure_device.py
```

### Model Installation

```bash
# Automated installer
./setup_uncensored_models.sh

# Manual installation
ollama pull wizardlm-uncensored-codellama:34b-q4_K_M
ollama pull wizardcoder:34b-python-q4_K_M
ollama pull mistral:7b-instruct-uncensored
```

---

## Performance Tuning

### GPU Memory Limit

```bash
# Set GPU memory limit (e.g., 16GB)
export OLLAMA_GPU_MEMORY=16384
systemctl restart ollama
```

### CPU Threads

```bash
# Set CPU threads (e.g., 8 threads)
export OLLAMA_NUM_THREAD=8
systemctl restart ollama
```

### Context Window

```bash
# Increase context window
export OLLAMA_NUM_CTX=16384
systemctl restart ollama
```

### Force GPU/CPU

```bash
# Force GPU
export CUDA_VISIBLE_DEVICES=0
export OLLAMA_NUM_GPU=1

# Force CPU
export OLLAMA_NUM_GPU=0
```

---

## Additional Resources

- **Full Configuration Guide:** `MODEL_CONFIG.md`
- **MCP Server Setup:** `mcp_servers_config.json`
- **Interface Comparison:** `INTERFACE_COMPARISON.md` (if exists)
- **DSMIL Documentation:** `/home/user/LAT5150DRVMIL/README.md`

---

## Support

### Documentation
- Model configuration: `MODEL_CONFIG.md`
- Device setup: `python3 configure_device.py --help`
- Engine status: `python3 dsmil_ai_engine.py status`

### Logs
- Ollama logs: `journalctl -u ollama -f`
- MCP audit: `~/.dsmil/mcp_audit.log`
- DSMIL audit: Check DSMIL military mode logs

---

**Classification:** UNCLASSIFIED // FOR OFFICIAL USE ONLY
**Maintainer:** DSMIL Integration Framework
**Last Updated:** 2025-11-06

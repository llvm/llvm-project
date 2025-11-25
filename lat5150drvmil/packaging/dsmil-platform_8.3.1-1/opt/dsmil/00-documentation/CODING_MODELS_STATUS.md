# Auto-Coding Models Status

## Downloads In Progress

- **DeepSeek Coder 6.7B:** ~52% (ETA 4 min)
- **Qwen Coder 7B:** ~2% (ETA 12 min)

## GPU Optimization

**Current:** CPU-only (works, 5-15s)
**Needs:** Intel oneAPI SYCL compiler (~2GB, 20 min install)
**Gain:** 10-20× faster (0.5-2s with GPU)

**Decision:** Get coding working NOW, add GPU optimization later

## Models Will Use (Already Quantized)

All Ollama models are pre-quantized (Q4_K_M = 4-bit):
- ✅ Optimal for speed/quality balance
- ✅ Fits in RAM efficiently
- ✅ Good inference speed on CPU
- ✅ Ready to use immediately

## When Downloads Complete

Test auto-coding:
\`\`\`bash
python3 /home/john/LAT5150DRVMIL/02-ai-engine/code_specialist.py generate "Write a Python function to check if a number is prime"
\`\`\`

Expected: 5-15s, production-quality code, DSMIL-attested

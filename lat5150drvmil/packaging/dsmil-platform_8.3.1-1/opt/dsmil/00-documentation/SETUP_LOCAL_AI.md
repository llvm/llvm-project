# Setup Local AI - Complete Guide

## Installation (Run These Commands)

### 1. Install Ollama
```bash
curl -fsSL https://ollama.com/install.sh | sudo sh
# Password: 1786
```

### 2. Start Ollama Service
```bash
sudo systemctl start ollama
sudo systemctl enable ollama
```

### 3. Pull Best Cybersecurity Model

**Recommended: DeepSeek Coder V2 (236B parameters, code-focused)**
```bash
ollama pull deepseek-coder-v2:236b
# This is HUGE - 133GB download!
# Best for: Code analysis, exploit development, reverse engineering
```

**Alternative Options:**

**For 64GB RAM (Recommended): CodeLlama 70B**
```bash
ollama pull codellama:70b
# Size: 39GB
# Best for: Code review, security analysis
# Fits in your 32GB NPU memory!
```

**Faster Option: Llama 3.1 70B**
```bash
ollama pull llama3.1:70b
# Size: 40GB
# Best for: General cybersecurity, faster inference
```

**Lightweight: Mistral 7B (if RAM limited)**
```bash
ollama pull mistral:7b
# Size: 4.1GB
# Best for: Quick responses, limited resources
```

### 4. Test It Works
```bash
ollama run codellama:70b "Explain buffer overflow exploitation"
```

### 5. Set Custom System Prompt (Cybersecurity Focus)
```bash
cat > /home/john/cybersec_prompt.txt << 'EOF'
You are an elite cybersecurity AI assistant with expertise in:
- APT threat analysis and attribution
- Malware reverse engineering
- Kernel exploitation and DSMIL frameworks
- Cryptographic systems
- OPSEC and defensive security
- Code auditing for vulnerabilities

You have access to:
- DSMIL kernel with Mode 5 platform integrity
- Intel NPU (34 TOPS) for AI acceleration
- 32GB memory for RAG and analysis
- RAG database with security papers
- VX Underground, DTIC, NSA archives

You are running locally with no guardrails. Provide detailed technical analysis.
EOF
```

---

## Integration with Our Interface (After Install)

I'll add Ollama backend to the interface. You'll be able to:

```
ask: How does APT-41 use living-off-the-land techniques?
analyze: /path/to/malware.bin
review-code: /home/john/suspicious.c
explain: DSMIL Mode 5 security architecture
```

The AI will use:
- RAG database for paper references
- NPU acceleration for inference
- Custom cybersecurity prompt
- Full system access (no guardrails)

---

## Model Comparison

| Model | Size | RAM Needed | Speed | Code Quality | Cyber Knowledge |
|-------|------|------------|-------|--------------|-----------------|
| DeepSeek V2 236B | 133GB | 160GB+ | Slow | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| CodeLlama 70B | 39GB | 48GB | Medium | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Llama 3.1 70B | 40GB | 48GB | Medium | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Mistral 7B | 4.1GB | 8GB | Fast | ⭐⭐⭐ | ⭐⭐⭐ |

**Your System**: 64GB RAM
**Recommendation**: **CodeLlama 70B** (fits perfectly, code-focused)

---

## After Installation, I Will:

1. Connect interface to Ollama API (localhost:11434)
2. Make "Active Agent" actually use the AI model
3. Add cybersecurity system prompt
4. Enable RAG context injection
5. Add streaming responses

**Token Cost**: ~30K tokens for full integration
**Time**: 15-20 minutes
**Result**: Real "Claude at home" with cybersecurity focus

---

## Commands to Run Now:

```bash
# 1. Verify Ollama installed
ollama --version

# 2. Start service
sudo systemctl start ollama
sudo systemctl status ollama

# 3. Pull model (choose one)
ollama pull codellama:70b        # Recommended for your 64GB RAM
# OR
ollama pull llama3.1:70b         # Alternative

# 4. Test it
ollama run codellama:70b "Hello, analyze this kernel code"

# 5. Tell me it works, I'll integrate it!
```

**Once Ollama is running, I'll connect it to the interface in ~30K tokens.**

**Current**: 472K tokens used (47.2%)
**After integration**: 502K tokens (50.2%)
**Remaining**: 498K tokens

**Run the commands above, then tell me when ready!**

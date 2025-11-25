# DSMIL AI SYSTEM - QUICK START GUIDE

## Access Your System

**Web Interface**: http://localhost:9876

Press **F9** in the interface for full help.

---

## Quick Commands

### AI Queries (Command Line)
```bash
# Ask a question (fast model, 1-5 sec)
python3 ~/dsmil_ai_engine.py prompt "Your question here"

# Force large model (30-60 sec, more detailed)
python3 ~/dsmil_ai_engine.py prompt "Complex analysis" large

# Check AI status
python3 ~/dsmil_ai_engine.py status
```

### AI Queries (Web Interface)
- Just type naturally - system auto-routes to fast/large model
- Prefix with `!` or `/` for shell commands
- Use arrow keys for command history

### F-Key Shortcuts (Web Interface)
- **F1**: Full system status
- **F2**: AVX-512 unlock status
- **F3**: NPU military mode status
- **F4**: RAG document search
- **F5**: Flux earnings status
- **F6**: GitHub repositories
- **F7**: Smart paper collector
- **F8**: System prompt
- **F9**: Help

---

## Hardware Status Check

```bash
# Check all hardware
curl -s http://localhost:9876/ai/status | python3 -m json.tool

# NPU military mode (should be 26.4 TOPS)
cat ~/.claude/npu-military.env

# AVX-512 status (should show "Unlocked: YES")
cat /proc/dsmil_avx512

# Huge pages (should be 16384)
cat /proc/meminfo | grep HugePages_Total
```

---

## Common Operations

### Download Papers on a Topic
```bash
# Via CLI (downloads up to 10GB)
python3 ~/smart_paper_collector.py collect "APT detection" 10

# Via web
curl "http://localhost:9876/smart-collect?topic=malware&size=5"
```

### Search Your Documents
```bash
# Search RAG index
python3 ~/rag_system.py search "your query"

# Via web
curl "http://localhost:9876/rag/search?q=your+query"
```

### Add Documents to RAG
```bash
# Ingest a folder of PDFs
python3 ~/rag_system.py ingest-folder /path/to/pdfs
```

### GitHub Operations
```bash
# Clone private repo (uses your SSH key)
curl "http://localhost:9876/github/clone?url=git@github.com:user/repo.git"

# List cloned repos
curl http://localhost:9876/github/list
```

---

## Customization

### Change System Prompt
```bash
# Set new prompt
python3 ~/dsmil_ai_engine.py set-prompt "Your custom prompt here"

# View current prompt
python3 ~/dsmil_ai_engine.py get-prompt

# Edit prompt file directly
nano ~/.claude/custom_system_prompt.txt
```

### Enable NPU Military Mode (if disabled)
```bash
# Edit config
nano ~/.claude/npu-military.env

# Change NPU_MILITARY_MODE=0 to NPU_MILITARY_MODE=1
# Then reboot
```

---

## Performance Specs

| Component | Performance |
|-----------|-------------|
| Fast AI Model | 1-5 sec, 10-15 tokens/sec |
| Large AI Model | 30-60 sec |
| NPU | 26.4 TOPS (military mode) |
| GPU | 40 TOPS |
| NCS2 | 10 TOPS |
| Total Compute | **76.4 TOPS** |
| AVX-512 | 12 P-cores unlocked |

---

## Troubleshooting

### Server Not Running
```bash
pkill -f opus_server_full.py
python3 ~/opus_server_full.py
```

### Model Not Responding
```bash
ollama list  # Should show both models
ollama pull llama3.2:3b-instruct-q4_K_M  # If fast model missing
```

### Check Logs
```bash
tail -f /tmp/opus_server.log
tail -f /var/log/dsmil_ai_attestation.log
```

---

## Example Queries

Try these in the web interface:

- "Explain how Intel NPU works"
- "What are common APT persistence techniques?"
- "How does DSMIL Mode 5 attestation work?"
- "Show me an example of a Linux kernel module"
- "What is the difference between CVE-2024-1234 and CVE-2024-5678?"

For shell commands, prefix with `!`:
- `!ls -la`
- `!uname -a`
- `!free -h`

---

## Documentation

- **Full Status**: ~/DSMIL_AI_STATUS.md
- **Handoff Doc**: ~/DELL_A00_AVX512_HANDOVER.md
- **This File**: ~/QUICK_START.md

---

**Ready to use!** Open http://localhost:9876 and start chatting. 🎯

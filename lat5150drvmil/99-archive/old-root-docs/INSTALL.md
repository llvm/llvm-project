# DSMIL Unified AI Platform - Installation Guide

**Version:** 8.3
**Platform:** Dell Latitude 5450 Covert Edition (works on any Linux system)

---

## Quick Install

```bash
cd LAT5150DRVMIL
./install.sh
```

The installer will:
- ✓ Check and install dependencies
- ✓ Set up Python environment
- ✓ Install Ollama
- ✓ Download AI models (DeepSeek R1 + Qwen Coder)
- ✓ Configure systemd service
- ✓ Set up RAG index directory
- ✓ Verify installation

**Time:** ~10-30 minutes (depending on internet speed for model downloads)

---

## System Requirements

### Minimum
- **OS:** Debian/Ubuntu Linux (tested on Debian 14)
- **RAM:** 8GB
- **Disk:** 10GB free space
- **CPU:** x86_64 processor
- **Internet:** Required for initial setup

### Recommended
- **RAM:** 16GB+
- **Disk:** 20GB+ SSD
- **NPU/GPU:** Intel NPU 3720 or Intel Arc GPU (optional, for acceleration)

---

## What Gets Installed

### System Packages
- `python3` - Python interpreter
- `python3-pip` - Python package manager
- `git` - Version control
- `curl` - HTTP client
- `ollama` - Local AI runtime

### Python Packages
- `requests` - HTTP library
- `anthropic` - Claude API client
- `google-generativeai` - Gemini API client
- `openai` - OpenAI API client
- `flask` - Web framework
- `flask-cors` - CORS support
- `beautifulsoup4` - HTML parsing
- `lxml` - XML parser
- `numpy` - Numerical computing
- `sentence-transformers` - Embeddings
- `faiss-cpu` - Vector search

### AI Models (via Ollama)
- `deepseek-r1:1.5b` - Fast reasoning (~1GB)
- `qwen2.5-coder:1.5b` - Code specialist (~1GB)

Optional (prompted during install):
- `deepseek-coder:6.7b` - Better code generation (~4GB)
- `codellama:7b` - Meta's code model (~4GB)

### Services
- `dsmil-server.service` - Systemd service for auto-start

### Configuration
- `~/.config/dsmil/config.json` - Main configuration
- `~/.local/share/dsmil/rag_index/` - RAG knowledge base

---

## Post-Installation

### Access the Interface

**Web UI:**
```bash
xdg-open http://localhost:9876
```

Or open browser and navigate to: `http://localhost:9876`

### Service Management

**Start service:**
```bash
sudo systemctl start dsmil-server
```

**Stop service:**
```bash
sudo systemctl stop dsmil-server
```

**Check status:**
```bash
sudo systemctl status dsmil-server
```

**View logs:**
```bash
sudo journalctl -u dsmil-server -f
```

**Restart on failure:**
Service is configured to auto-restart on failure.

### Manual Start (Development)

If you want to run without systemd (for development):

```bash
cd /path/to/LAT5150DRVMIL/03-web-interface
python3 dsmil_unified_server.py
```

---

## Features Available

### Auto-Coding Tools
- ✏️ **Edit File** - Modify existing code
- 📝 **Create File** - Generate new files
- 🐛 **Debug Code** - Find and fix bugs
- 🔄 **Refactor** - Improve code quality
- 🔍 **Code Review** - Security & quality analysis
- 🧪 **Generate Tests** - Unit test generation
- 📄 **Generate Docs** - Documentation generation

### AI Capabilities
- 🤖 Smart routing (auto-detects code vs general queries)
- 🌐 Web search (DuckDuckGo integration)
- 🕷️ Web crawling (intelligent site scraping)
- 📚 RAG knowledge base (222 docs indexed)
- 🔒 Hardware attestation (if DSMIL framework available)

### Models
- **DeepSeek R1** - Fast reasoning
- **Qwen Coder** - Code specialist
- **DeepSeek Coder** - Better code (optional)
- **CodeLlama** - Meta's code model (optional)

---

## Configuration

### Edit Configuration

```bash
nano ~/.config/dsmil/config.json
```

**Example config:**
```json
{
    "version": "8.3",
    "install_dir": "/home/john/LAT5150DRVMIL",
    "local_models": {
        "reasoning": "deepseek-r1:1.5b",
        "code": "qwen2.5-coder:1.5b"
    },
    "server": {
        "host": "127.0.0.1",
        "port": 9876
    },
    "rag": {
        "index_dir": "/home/john/.local/share/dsmil/rag_index"
    }
}
```

### Change Server Port

Edit config and change `server.port`, then restart service.

### Add API Keys (Optional)

For cloud AI features:

```bash
export ANTHROPIC_API_KEY="your_key_here"
export GOOGLE_API_KEY="your_key_here"
export OPENAI_API_KEY="your_key_here"
```

Add to `~/.bashrc` to make permanent.

---

## Troubleshooting

### Service won't start

**Check logs:**
```bash
sudo journalctl -u dsmil-server -n 50
```

**Common issues:**
- Port 9876 already in use
- Python packages not installed
- Ollama not running

**Fix:**
```bash
# Check if port is in use
sudo lsof -i :9876

# Restart Ollama
sudo systemctl restart ollama

# Reinstall Python packages
pip3 install --user --upgrade requests flask
```

### Models not responding

**Check Ollama:**
```bash
systemctl status ollama
ollama list
```

**Test model:**
```bash
ollama run deepseek-r1:1.5b "Hello"
```

### Web interface not loading

**Check if server is running:**
```bash
curl http://localhost:9876/status
```

**Check firewall:**
```bash
sudo ufw status
sudo ufw allow 9876/tcp  # If needed
```

---

## Uninstall

```bash
cd LAT5150DRVMIL
./uninstall.sh
```

This removes:
- Systemd service
- Configuration files
- RAG index data

This keeps:
- Source code
- Ollama and models
- Python packages

To fully remove Ollama:
```bash
sudo apt remove ollama
rm -rf ~/.ollama
```

---

## Security Notes

### Network Exposure

**Default:** Server binds to `127.0.0.1` (localhost only)

**⚠️ WARNING:** Do NOT expose to network without authentication!

If you need remote access, use SSH tunnel:
```bash
ssh -L 9876:localhost:9876 user@machine
```

### DSMIL Mode Levels

If using DSMIL framework:
- **Mode 5 STANDARD** - Safe for training (default)
- **Mode 4 ELEVATED** - Requires authorization
- **Mode 3 FULL** - Military operations only

**Current mode:** STANDARD (safe)

---

## Documentation

- **Quick Start:** `./README.md`
- **Full Docs:** `./00-documentation/`
- **API Reference:** `./00-documentation/UNIFIED_PLATFORM_ARCHITECTURE.md`
- **Security:** `./03-security/`

---

## Support

- **GitHub:** https://github.com/SWORDIntel/LAT5150DRVMIL
- **Issues:** https://github.com/SWORDIntel/LAT5150DRVMIL/issues

---

## License

**Classification:** JRTC1 Training Environment
**Distribution:** Educational/Research Use
**Compliance:** DoD 8500 series, NIST Cybersecurity Framework

---

**Installed successfully?** Start using: `xdg-open http://localhost:9876` 🚀

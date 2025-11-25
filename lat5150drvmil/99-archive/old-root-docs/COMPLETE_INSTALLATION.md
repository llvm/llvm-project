# DSMIL Complete Platform - Installation Guide

**Version:** 8.3.2
**Platform:** Dell Latitude 5450 Covert Edition (compatible with any Debian/Ubuntu system)

---

## Overview

This guide covers the complete installation of the DSMIL platform including:
- DSMIL Framework (hardware attestation, TPM2 acceleration)
- Dell MIL-SPEC Tools (monitoring and management)
- AI Platform (7 auto-coding tools, chat interface, RAG, web search)
- All dependencies and services

---

## Installation Methods

### Method 1: One-Command Installation (Recommended)

```bash
cd LAT5150DRVMIL
./install-complete.sh
```

This script handles everything:
- ✓ System requirements check
- ✓ All dependencies (Python, build tools, etc.)
- ✓ DSMIL framework kernel module
- ✓ Ollama AI runtime + models
- ✓ Hardware optimization
- ✓ Service configuration
- ✓ Verification

**Time:** 15-30 minutes (depending on download speed)

---

### Method 2: Package Installation

Install individual packages in order:

```bash
cd LAT5150DRVMIL/packaging

# 1. Install core framework tools
sudo dpkg -i dell-milspec-tools_1.0.0-1_amd64.deb
sudo dpkg -i tpm2-accel-examples_1.0.0-1.deb

# 2. Install AI platform
sudo dpkg -i dsmil-platform_8.3.1-1.deb

# 3. Install meta-package (ties everything together)
sudo dpkg -i dsmil-complete_8.3.2-1.deb

# 4. Fix any missing dependencies
sudo apt install -f
```

**Then install Ollama and models:**

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Download essential models
ollama pull deepseek-r1:1.5b
ollama pull qwen2.5-coder:1.5b

# Start service
sudo systemctl start dsmil-server
```

---

## Available Packages

### Core Packages

| Package | Size | Description |
|---------|------|-------------|
| **dsmil-complete** | 1.6KB | Meta-package (installs all components) |
| **dsmil-platform** | 2.4MB | AI interface, server, web UI |
| **dell-milspec-tools** | 24KB | Hardware monitoring tools |
| **tpm2-accel-examples** | 19KB | TPM2 acceleration examples |

**Total Size:** ~2.5MB packages + ~3GB AI models

---

## System Requirements

### Minimum
- **OS:** Debian 11+ / Ubuntu 20.04+
- **RAM:** 8GB
- **Disk:** 20GB free
- **CPU:** x86_64 (Intel/AMD)
- **Internet:** Required for initial setup

### Recommended
- **RAM:** 16GB+
- **Disk:** 50GB+ SSD
- **Hardware:** Intel NPU, Intel Arc GPU, or NCS2 for acceleration

---

## What Gets Installed

### System Packages
```
build-essential
linux-headers
dkms
python3, python3-pip, python3-dev
git, curl, wget
pciutils, usbutils, dmidecode
htop, iotop, sysstat
```

### Python Packages
```
requests, flask, flask-cors
anthropic, google-generativeai, openai
beautifulsoup4, lxml
numpy, scipy, pandas
sentence-transformers, faiss-cpu
transformers, torch
```

### AI Runtime
```
Ollama (latest)
deepseek-r1:1.5b (900MB)
qwen2.5-coder:1.5b (900MB)
```

### DSMIL Components
```
DSMIL kernel module (dsmil-72dev.ko)
Dell MIL-SPEC monitoring tools
TPM2 acceleration utilities
Hardware attestation framework
```

### Services
```
dsmil-server.service - AI platform server
ollama.service - AI runtime
```

### Configuration
```
~/.config/dsmil/config.json - Main configuration
~/.local/share/dsmil/rag_index/ - Knowledge base
/etc/systemd/system/dsmil-server.service - Service file
```

---

## Post-Installation

### 1. Start the Service

```bash
sudo systemctl start dsmil-server
sudo systemctl enable dsmil-server
```

### 2. Access the Interface

Open browser to: **http://localhost:9876**

Or use command:
```bash
xdg-open http://localhost:9876
```

### 3. Verify Installation

```bash
# Check service status
sudo systemctl status dsmil-server

# Check if DSMIL module is loaded
lsmod | grep dsmil

# Check Ollama models
ollama list

# Test endpoint
curl http://localhost:9876/status
```

---

## Features Available

### Auto-Coding Tools (7)
1. ✏️ **Edit File** - Modify existing code
2. 📝 **Create File** - Generate new files
3. 🐛 **Debug Code** - Find and fix bugs
4. 🔄 **Refactor** - Improve code quality
5. 🔍 **Code Review** - Security & quality analysis
6. 🧪 **Generate Tests** - Unit test generation
7. 📄 **Generate Docs** - Documentation generation

### AI Capabilities
- 🤖 Smart routing (auto-detects code vs general queries)
- 🌐 Web search (DuckDuckGo integration)
- 🕷️ Web crawling (intelligent site scraping)
- 📚 RAG knowledge base
- 💾 Chat history persistence (auto-save to localStorage)
- 🔒 Hardware attestation (TPM 2.0, if available)

### Hardware Support
- **Intel NPU** (26.4 TOPS) - AI acceleration
- **Intel Arc GPU** (40 TOPS) - Parallel compute
- **Intel NCS2** (10 TOPS) - Edge inference
- **Intel GNA** (1 GOPS) - Always-on routing
- **AVX-512** - Vector operations
- **Total:** 76.4 TOPS

### DSMIL Framework
- **Mode 5 STANDARD** - Safe for training (default)
- **84 devices** - Hardware monitoring points
- **TPM 2.0** - Cryptographic attestation
- **Audit trail** - Device 48 logging

---

## Configuration

### Main Config

Edit: `~/.config/dsmil/config.json`

```json
{
    "version": "8.3.2",
    "local_models": {
        "reasoning": "deepseek-r1:1.5b",
        "code": "qwen2.5-coder:1.5b"
    },
    "server": {
        "host": "127.0.0.1",
        "port": 9876
    },
    "hardware": {
        "npu_enabled": true,
        "gpu_enabled": true,
        "ncs2_enabled": true
    },
    "dsmil": {
        "mode": "STANDARD",
        "attestation_enabled": true
    }
}
```

### Change Server Port

```bash
nano ~/.config/dsmil/config.json
# Change port: 9876 → 8080
sudo systemctl restart dsmil-server
```

### Add API Keys (Optional)

For cloud AI features:

```bash
# Add to ~/.bashrc or ~/.profile
export ANTHROPIC_API_KEY="your_key"
export GOOGLE_API_KEY="your_key"
export OPENAI_API_KEY="your_key"
```

---

## Service Management

```bash
# Start service
sudo systemctl start dsmil-server

# Stop service
sudo systemctl stop dsmil-server

# Restart service
sudo systemctl restart dsmil-server

# Check status
sudo systemctl status dsmil-server

# View logs (live)
sudo journalctl -u dsmil-server -f

# View last 50 lines
sudo journalctl -u dsmil-server -n 50

# Enable auto-start on boot
sudo systemctl enable dsmil-server

# Disable auto-start
sudo systemctl disable dsmil-server
```

---

## Troubleshooting

### Service Won't Start

**Check logs:**
```bash
sudo journalctl -u dsmil-server -n 100
```

**Common issues:**
- Port 9876 already in use
- Python packages missing
- Ollama not running

**Fix:**
```bash
# Check port
sudo lsof -i :9876

# Restart Ollama
sudo systemctl restart ollama

# Reinstall Python packages
pip3 install --user --upgrade requests flask anthropic
```

### Models Not Responding

**Check Ollama:**
```bash
systemctl status ollama
ollama list
```

**Test model:**
```bash
ollama run deepseek-r1:1.5b "Hello"
```

**Reinstall model:**
```bash
ollama rm deepseek-r1:1.5b
ollama pull deepseek-r1:1.5b
```

### Web Interface Not Loading

**Check service:**
```bash
curl http://localhost:9876/status
```

**Check firewall:**
```bash
sudo ufw status
sudo ufw allow 9876/tcp
```

### DSMIL Module Not Loading

**Check hardware:**
```bash
lspci | grep -i dell
dmidecode -t system
```

**Load module manually:**
```bash
sudo modprobe dsmil-72dev
lsmod | grep dsmil
```

**Note:** Module requires specific Dell hardware. Platform works without it in software mode.

---

## Uninstallation

### Remove Everything

```bash
cd LAT5150DRVMIL

# Stop service
sudo systemctl stop dsmil-server
sudo systemctl disable dsmil-server

# Remove packages
sudo dpkg -r dsmil-complete
sudo dpkg -r dsmil-platform
sudo dpkg -r dell-milspec-tools
sudo dpkg -r tpm2-accel-examples

# Remove Ollama (optional)
sudo systemctl stop ollama
sudo apt remove ollama
rm -rf ~/.ollama

# Remove config
rm -rf ~/.config/dsmil
rm -rf ~/.local/share/dsmil

# Remove service file
sudo rm /etc/systemd/system/dsmil-server.service
sudo systemctl daemon-reload
```

### Keep Source Code

The source code in `/home/john/LAT5150DRVMIL` is preserved.

---

## Security Notes

### Network Exposure

**Default:** Server binds to `127.0.0.1` (localhost only) ✓ Safe

**⚠️ WARNING:** Do NOT expose to network without authentication!

**Remote Access (safe):** Use SSH tunnel:
```bash
ssh -L 9876:localhost:9876 user@machine
```

### DSMIL Mode Levels

- **Mode 5 STANDARD** - ✓ Safe for training (current)
- **Mode 4 ELEVATED** - Requires authorization
- **Mode 3 FULL** - Military operations only

**DO NOT** change mode without proper authorization.

### Hardware Attestation

When DSMIL module is loaded:
- All AI responses are cryptographically signed via TPM 2.0
- Platform integrity is verified
- Responses include attestation metadata

Without module:
- Platform runs in software mode
- No hardware attestation
- All features still work

---

## Documentation

- **README:** `./README.md`
- **Install Guide:** `./INSTALL.md`
- **Structure:** `./STRUCTURE.md`
- **Full Docs:** `./00-documentation/`
- **Security:** `./03-security/`
- **API Reference:** `./00-documentation/UNIFIED_PLATFORM_ARCHITECTURE.md`

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

## Quick Reference

| Task | Command |
|------|---------|
| Install | `./install-complete.sh` |
| Start | `sudo systemctl start dsmil-server` |
| Stop | `sudo systemctl stop dsmil-server` |
| Status | `sudo systemctl status dsmil-server` |
| Logs | `sudo journalctl -u dsmil-server -f` |
| Access | `http://localhost:9876` |
| Config | `~/.config/dsmil/config.json` |
| Models | `ollama list` |
| Uninstall | `sudo dpkg -r dsmil-complete` |

---

**Ready to Install?** Run `./install-complete.sh` to get started! 🚀

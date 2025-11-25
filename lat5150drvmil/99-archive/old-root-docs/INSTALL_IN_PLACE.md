# DSMIL Platform - Installation In Place Guide

**Version:** 8.3.2
**Purpose:** Install DSMIL platform on your current system without disrupting existing work

---

## Overview

This guide helps you install the DSMIL platform on a **running system** that you're actively using for development. It ensures:
- ✅ No disruption to existing work
- ✅ Safe installation with rollback options
- ✅ Preservation of current configuration
- ✅ Minimal system changes

**Installation Time:** 20-40 minutes

---

## Pre-Installation Checklist

### 1. Backup Current Work

```bash
# Backup important files
tar -czf ~/backup-$(date +%Y%m%d).tar.gz ~/Documents ~/Projects ~/.config

# Verify backup
ls -lh ~/backup-*.tar.gz
```

### 2. Check System Resources

```bash
# Check free disk space (need 20GB+)
df -h $HOME

# Check RAM (need 8GB+)
free -h

# Check running services that use port 9876
sudo lsof -i :9876
```

**If port 9876 is in use:** The installer can use a different port. Edit config after installation.

### 3. Note Current Configuration

```bash
# List Python packages
pip3 list > ~/python-packages-before-install.txt

# List running services
systemctl --user list-units > ~/services-before-install.txt

# List loaded modules
lsmod > ~/modules-before-install.txt
```

---

## Installation Steps

### Step 1: Clone Repository (If Not Already)

```bash
cd ~
git clone https://github.com/SWORDIntel/LAT5150DRVMIL
cd LAT5150DRVMIL
```

**If already cloned:** Update to latest:
```bash
cd ~/LAT5150DRVMIL
git pull origin main
```

---

### Step 2: Review What Will Be Installed

```bash
# Read the installer to understand what it does
less install-complete.sh
```

**The installer will:**
- Install system packages (build tools, Python, git, etc.)
- Install Python packages (to `~/.local/lib/python3.XX/`)
- Install Ollama (system-wide service)
- Download AI models (to `~/.ollama/`)
- Install DSMIL kernel module (optional, requires hardware)
- Create systemd service for DSMIL server
- Create config files in `~/.config/dsmil/`

**What it WON'T do:**
- Modify your existing files
- Change your shell configuration
- Overwrite existing services
- Delete any data

---

### Step 3: Dry-Run Installation

**Preview installation without making changes:**

```bash
# Review dependencies
cat install-complete.sh | grep -A 20 "install_system_dependencies"

# Check if Ollama is already installed
command -v ollama && echo "Ollama already installed" || echo "Ollama will be installed"

# Check current Python packages
pip3 list | grep -E "flask|anthropic|sentence-transformers"
```

---

### Step 4: Run Installation

**Option A: Full Installation (Recommended)**

```bash
./install-complete.sh
```

**The installer will:**
1. Check system requirements ✓
2. Ask to install dependencies → Press Y
3. Install Python packages → Uses `--user` flag (safe)
4. Install Ollama → System-wide (requires sudo)
5. Download AI models → Prompts for each model
6. Install DSMIL framework → Skips if hardware not available
7. Configure services → Creates systemd service
8. Verify installation → Shows status
9. Ask to start service → Your choice

**Option B: Selective Installation**

If you want to skip certain parts, use the basic installer:

```bash
./install.sh  # Skips DSMIL framework, only installs AI platform
```

---

### Step 5: Post-Installation Verification

```bash
# Check service status
sudo systemctl status dsmil-server

# Check if accessible
curl http://localhost:9876/status

# List installed models
ollama list

# Check DSMIL module (if hardware available)
lsmod | grep dsmil

# Verify Python packages
python3 -c "import flask, anthropic, sentence_transformers; print('All packages OK')"
```

---

## In-Place Installation Scenarios

### Scenario 1: You Already Have Ollama Installed

**The installer detects this and skips Ollama installation.**

```bash
./install-complete.sh
# Output: "Ollama already installed ✓"
```

Your existing Ollama installation and models are **preserved**.

---

### Scenario 2: You Have Existing Python Packages

**The installer uses `pip3 install --user`, which:**
- Installs to `~/.local/` (your user directory)
- Doesn't interfere with system packages
- Doesn't require sudo

**Potential conflicts:** If you have different versions of packages.

**Solution:**
```bash
# Use a virtual environment (recommended)
python3 -m venv ~/.venv/dsmil
source ~/.venv/dsmil/bin/activate
pip install -r requirements.txt  # Install in isolated environment
```

---

### Scenario 3: Port 9876 Already in Use

**Check what's using the port:**
```bash
sudo lsof -i :9876
```

**Change DSMIL port after installation:**
```bash
# Edit config
nano ~/.config/dsmil/config.json

# Change: "port": 9876 → "port": 8080

# Restart service
sudo systemctl restart dsmil-server
```

---

### Scenario 4: You Don't Want Systemd Service

**Skip systemd setup and run manually:**

```bash
# Install dependencies only
./install-complete.sh
# When prompted about systemd, say No

# Run manually when needed
cd ~/LAT5150DRVMIL/03-web-interface
python3 dsmil_unified_server.py

# Or use tmux/screen for background
tmux new -s dsmil
python3 dsmil_unified_server.py
# Ctrl+B, D to detach
```

---

### Scenario 5: Limited Disk Space

**Install minimal configuration:**

```bash
# Install platform without large models
./install.sh

# Download only smallest models
ollama pull deepseek-r1:1.5b  # ~900MB
ollama pull qwen2.5-coder:1.5b  # ~900MB

# Skip optional models (deepseek-coder:6.7b, codellama:7b)
```

**Total size:** ~3-4GB vs 10-15GB full installation

---

### Scenario 6: No sudo Access

**You can still install most features:**

```bash
# Install Python packages to user directory (no sudo needed)
pip3 install --user requests anthropic flask sentence-transformers faiss-cpu

# Download Ollama (requires sudo for system service)
# Alternative: Use Docker
docker pull ollama/ollama
docker run -d -p 11434:11434 --name ollama ollama/ollama

# Run DSMIL server without systemd
python3 03-web-interface/dsmil_unified_server.py &

# Access interface
xdg-open http://localhost:9876
```

**Limitations:** No systemd auto-start, no DSMIL kernel module

---

## Rollback & Uninstallation

### Quick Rollback

**If something goes wrong, you can quickly rollback:**

```bash
# Stop service
sudo systemctl stop dsmil-server
sudo systemctl disable dsmil-server

# Remove service file
sudo rm /etc/systemd/system/dsmil-server.service
sudo systemctl daemon-reload

# Remove config
rm -rf ~/.config/dsmil
rm -rf ~/.local/share/dsmil

# Uninstall Ollama (optional)
sudo apt remove ollama
```

**Your source code remains untouched in `~/LAT5150DRVMIL`**

---

### Complete Uninstallation

**Remove everything including packages:**

```bash
cd ~/LAT5150DRVMIL
./uninstall.sh

# Or manually
sudo dpkg -r dsmil-complete dsmil-platform dell-milspec-tools tpm2-accel-examples
sudo apt autoremove

# Remove Python packages
pip3 uninstall -y requests anthropic flask sentence-transformers faiss-cpu

# Remove Ollama and models
sudo systemctl stop ollama
sudo apt remove ollama
rm -rf ~/.ollama

# Remove source (optional)
rm -rf ~/LAT5150DRVMIL
```

---

## Safe Installation Practices

### Use Screen/Tmux for Long Operations

```bash
# Install tmux if not present
sudo apt install tmux

# Start tmux session
tmux new -s install

# Run installer
./install-complete.sh

# If disconnected, reattach with:
tmux attach -t install
```

### Monitor Resource Usage

**In separate terminal:**
```bash
# Watch disk space
watch -n 5 df -h

# Monitor RAM usage
htop

# Monitor network (during downloads)
iftop -i <interface>
```

### Preserve Existing Environment

```bash
# Before installation, export current environment
env > ~/environment-before-install.txt

# After installation, compare
env > ~/environment-after-install.txt
diff ~/environment-before-install.txt ~/environment-after-install.txt
```

---

## Post-Installation Configuration

### 1. Configure Server Binding

**For security, server binds to localhost by default.**

Edit `~/.config/dsmil/config.json`:

```json
{
    "server": {
        "host": "127.0.0.1",  // Localhost only (safe)
        "port": 9876
    }
}
```

**DO NOT change to `0.0.0.0` unless you have firewall protection!**

---

### 2. Add RAG Documents

```bash
# Add your research papers/docs
cp ~/Documents/*.pdf ~/.local/share/dsmil/rag_index/

# Or use the web interface:
# Open http://localhost:9876
# Click "RAG" → "Add Folder"
```

---

### 3. Configure API Keys (Optional)

**For cloud AI features (Claude, Gemini, OpenAI):**

Add to `~/.bashrc`:
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
export GOOGLE_API_KEY="AIza..."
export OPENAI_API_KEY="sk-..."
```

Then reload:
```bash
source ~/.bashrc
sudo systemctl restart dsmil-server
```

---

### 4. Enable Auto-Start (Optional)

**Already enabled by installer, but to verify:**

```bash
# Check if enabled
systemctl is-enabled dsmil-server

# Enable if not
sudo systemctl enable dsmil-server

# Disable if you don't want auto-start
sudo systemctl disable dsmil-server
```

---

## Integration with Existing Workflow

### Keep Both Your Old Setup and DSMIL

**Run DSMIL on different port:**

```bash
# Edit config
nano ~/.config/dsmil/config.json
# Change port to 8080 or 8888

# Restart
sudo systemctl restart dsmil-server
```

Now you can access:
- Your existing tools: `http://localhost:3000` (or whatever)
- DSMIL platform: `http://localhost:8080`

---

### Use DSMIL with Existing Projects

**The web interface works with any codebase:**

1. Open DSMIL interface: `http://localhost:9876`
2. Use auto-coding tools on your files
3. Add your project docs to RAG
4. Use web search for research

**Example:**
```
TOOLS → Edit File
Path: /home/john/my-project/src/api.py
Task: Add authentication middleware
```

---

### Integrate with Your IDE

**Access DSMIL from command line:**

```bash
# Query from terminal
curl "http://localhost:9876/ai/chat?msg=Explain%20this%20code"

# Add to RAG
curl "http://localhost:9876/rag/add-file?path=/path/to/file.pdf"

# Search RAG
curl "http://localhost:9876/rag/search?query=authentication"
```

**Create alias in `~/.bashrc`:**
```bash
alias dsmil-query='curl -s "http://localhost:9876/ai/chat?msg="'
alias dsmil-rag='curl -s "http://localhost:9876/rag/search?query="'
```

---

## Troubleshooting In-Place Installation

### Issue: Python Package Conflicts

**Symptom:** Import errors or version conflicts

**Solution 1: Use Virtual Environment**
```bash
python3 -m venv ~/.venv/dsmil
source ~/.venv/dsmil/bin/activate
pip install -r ~/LAT5150DRVMIL/requirements.txt
```

**Solution 2: Update Service to Use Venv**
```bash
sudo nano /etc/systemd/system/dsmil-server.service

# Change ExecStart to:
ExecStart=/home/john/.venv/dsmil/bin/python3 /home/john/LAT5150DRVMIL/03-web-interface/dsmil_unified_server.py

sudo systemctl daemon-reload
sudo systemctl restart dsmil-server
```

---

### Issue: Ollama Conflicts with Existing Service

**Symptom:** Port 11434 already in use

**Check what's using it:**
```bash
sudo lsof -i :11434
```

**Solution: Change Ollama port**
```bash
# Edit Ollama service
sudo systemctl edit ollama.service

# Add:
[Service]
Environment="OLLAMA_HOST=127.0.0.1:11435"

sudo systemctl daemon-reload
sudo systemctl restart ollama
```

**Update DSMIL config:**
```bash
nano ~/.config/dsmil/config.json
# Add: "ollama_url": "http://localhost:11435"
```

---

### Issue: Not Enough Disk Space

**Check space:**
```bash
df -h ~
df -h ~/.ollama
```

**Solutions:**

**Option 1: Move Ollama to larger drive**
```bash
sudo systemctl stop ollama

# Move models
sudo mv ~/.ollama /mnt/bigdrive/ollama
ln -s /mnt/bigdrive/ollama ~/.ollama

sudo systemctl start ollama
```

**Option 2: Clean old models**
```bash
ollama list
ollama rm <model-name>  # Remove unused models
```

**Option 3: Use external AI (no local models)**
```bash
# Just use API keys, skip Ollama
export ANTHROPIC_API_KEY="..."
# DSMIL will use cloud AI
```

---

### Issue: Permission Errors

**Symptom:** Cannot create directories or files

**Fix permissions:**
```bash
# Ensure you own your home directory subdirs
chown -R $USER:$USER ~/.config
chown -R $USER:$USER ~/.local

# Make install script executable
chmod +x ~/LAT5150DRVMIL/install-complete.sh
```

---

### Issue: DSMIL Module Won't Load

**Symptom:** `modprobe dsmil-72dev` fails

**This is normal if:**
- You're not on Dell Latitude 5450
- Hardware doesn't have DSMIL features
- Running in VM

**Platform still works in software mode!**

No action needed. Features that require hardware attestation will be disabled.

---

### Issue: Service Won't Start

**Check logs:**
```bash
sudo journalctl -u dsmil-server -n 50
```

**Common issues:**

**Port in use:**
```bash
sudo lsof -i :9876
# Change port in ~/.config/dsmil/config.json
```

**Missing dependencies:**
```bash
pip3 install --user requests flask
sudo systemctl restart dsmil-server
```

**Python not found:**
```bash
which python3
# Update ExecStart in /etc/systemd/system/dsmil-server.service
```

---

## Safe Installation Mode

### Install Without System Changes

**If you want minimal system changes:**

```bash
# Create isolated installation
mkdir ~/dsmil-isolated
cd ~/dsmil-isolated

# Clone here
git clone https://github.com/SWORDIntel/LAT5150DRVMIL
cd LAT5150DRVMIL

# Use virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python packages in venv
pip install requests anthropic flask sentence-transformers faiss-cpu

# Install Ollama (only system change needed)
curl -fsSL https://ollama.com/install.sh | sh

# Download models
ollama pull deepseek-r1:1.5b

# Run manually (no systemd)
python 03-web-interface/dsmil_unified_server.py
```

**This installation:**
- ✅ Uses virtual environment (isolated)
- ✅ Doesn't modify system Python
- ✅ Doesn't create systemd service
- ✅ Fully contained in one directory

---

## Coexistence with Other AI Tools

### Running Alongside Other LLM Tools

**DSMIL is designed to coexist with:**
- Existing Ollama installations
- Other Python AI projects
- Docker containers
- Local LLM UIs (like text-generation-webui)

**Tips:**
1. Use different ports
2. Share Ollama models (saves space)
3. Use shared RAG directory if desired

**Example: Share Ollama models**
```bash
# Both DSMIL and your other tool can use same Ollama
# Models are in ~/.ollama/models/ and shared automatically
```

---

### Using with Existing Development Environment

**DSMIL integrates seamlessly:**

```bash
# Your existing workflow:
code ~/my-project          # VSCode
npm run dev                # Your dev server on :3000

# DSMIL platform:
xdg-open http://localhost:9876  # DSMIL on :9876

# Both run simultaneously without conflict
```

---

## Updating In Place

### Update DSMIL Platform

```bash
cd ~/LAT5150DRVMIL

# Pull latest changes
git pull origin main

# Restart service
sudo systemctl restart dsmil-server
```

**No reinstallation needed!**

---

### Update AI Models

```bash
# List current models
ollama list

# Update a model
ollama pull deepseek-r1:1.5b  # Downloads latest version

# Remove old version
ollama rm deepseek-r1:1.5b:old
```

---

### Update Python Dependencies

```bash
pip3 install --user --upgrade \
    requests anthropic flask sentence-transformers faiss-cpu

sudo systemctl restart dsmil-server
```

---

## Monitoring During Installation

### Watch Installation Progress

**Terminal 1: Run installer**
```bash
./install-complete.sh
```

**Terminal 2: Monitor system**
```bash
# Watch disk space
watch -n 2 'df -h ~ | grep home'

# Watch downloads
watch -n 1 'ls -lh ~/.ollama/models/'

# Watch processes
htop
```

**Terminal 3: Check logs**
```bash
# If service starts during install
sudo journalctl -u dsmil-server -f
```

---

## Recovery Procedures

### If Installation Fails Mid-Way

**The installer uses `set -e` and will stop on first error.**

**Recovery:**
```bash
# Check what was installed
dpkg -l | grep dsmil
pip3 list | grep -E "anthropic|sentence"
command -v ollama

# Remove partial installation
./uninstall.sh

# Try again
./install-complete.sh
```

---

### If System Becomes Unstable

**Unlikely, but if it happens:**

```bash
# Stop all DSMIL services
sudo systemctl stop dsmil-server
sudo systemctl stop ollama

# Unload kernel module
sudo modprobe -r dsmil-72dev

# Restore from backup
tar -xzf ~/backup-*.tar.gz -C ~
```

---

## Best Practices

### 1. Install During Low-Activity Period

**Choose a time when:**
- You're not running critical processes
- You can afford 30-60 minutes
- You're not in the middle of important work

### 2. Keep Terminal Open

**Don't close terminal during installation!**
- Model downloads can take 10-30 minutes
- Keep terminal visible to see progress

### 3. Test Before Committing

**After installation:**
```bash
# Test web interface
xdg-open http://localhost:9876

# Try a simple query
curl "http://localhost:9876/ai/chat?msg=Hello"

# Check all services
sudo systemctl status dsmil-server
sudo systemctl status ollama
```

**Only enable auto-start after testing!**

---

## Quick Reference

### Installation Commands

| Task | Command |
|------|---------|
| Full install | `./install-complete.sh` |
| Basic install | `./install.sh` |
| Package install | `sudo dpkg -i packaging/*.deb` |
| Uninstall | `./uninstall.sh` |
| Check status | `systemctl status dsmil-server` |
| View logs | `journalctl -u dsmil-server -f` |

### Files Created

| File | Purpose |
|------|---------|
| `~/.config/dsmil/config.json` | Main configuration |
| `~/.local/share/dsmil/` | RAG index and data |
| `/etc/systemd/system/dsmil-server.service` | Service file |
| `~/.ollama/` | AI models |

### Disk Space Usage

| Component | Size |
|-----------|------|
| Python packages | ~500MB |
| Ollama binary | ~100MB |
| AI models (minimal) | ~2GB |
| AI models (full) | ~10GB |
| Source code | ~50MB |
| **Total (minimal)** | **~3GB** |
| **Total (full)** | **~11GB** |

---

## Support

**If you encounter issues:**

1. Check logs: `sudo journalctl -u dsmil-server -f`
2. Review `COMPLETE_INSTALLATION.md` troubleshooting section
3. Check GitHub issues: https://github.com/SWORDIntel/LAT5150DRVMIL/issues
4. Rollback using instructions above

---

## Summary

**In-place installation is safe when you:**
- ✅ Backup important files first
- ✅ Check disk space availability
- ✅ Review what will be installed
- ✅ Monitor installation progress
- ✅ Test before enabling auto-start
- ✅ Know how to rollback

**The installer is designed to:**
- ✅ Detect existing installations
- ✅ Use user directories (`--user` flag)
- ✅ Ask before making system changes
- ✅ Provide clear error messages
- ✅ Exit cleanly on errors

---

**Ready to install?** Run `./install-complete.sh` and follow the prompts! 🚀

**Questions?** See [COMPLETE_INSTALLATION.md](COMPLETE_INSTALLATION.md) for comprehensive guide.

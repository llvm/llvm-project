# Claude-Backups Installation Guide

**Purpose:** Install claude-backups system to run Claude Code locally (replace npx)

**Date:** 2025-10-29

---

## Quick Install

```bash
cd /home/john/claude-backups

# Run main installer
./installer

# Follow prompts
```

---

## What You're Installing

**From `/home/john/claude-backups/`:**

### Core Components

1. **Shadowgit** (AVX2/AVX512-accelerated git)
   - 3-10× faster git operations
   - Hardware-optimized diff engine
   - Location: `hooks/shadowgit/`

2. **98-Agent Coordination System**
   - Claude Agent Framework v7.0
   - Parallel agent execution
   - Location: `agents/`

3. **Local Opus/Claude**
   - Run Claude Code locally
   - Token-free inference
   - NPU/OpenVINO integration

4. **Hook System**
   - Pre-commit, post-task hooks
   - Performance monitoring
   - Crypto-POW optimization
   - Location: `hooks/`

5. **NPU/OpenVINO Optimizations**
   - Hardware-specific tuning
   - 40+ TFLOPS capability
   - Location: `openvino/`, `hardware/`

### Installation Components

**From `README.md`:**
- Python 3.11+ support
- Rust toolchain (for shadowgit)
- Docker integration
- OpenVINO runtime
- Various Python packages

---

## Installation Steps

### Step 1: Review Current System

```bash
cd /home/john/claude-backups

# Check README
cat README.md | head -100

# Check what modules are available
ls -la installers/
```

### Step 2: Run Installer

```bash
# Main installer (recommended)
./installer

# Or specific autonomous system
./install_autonomous_system.sh

# Or complete setup
./robust_setup.sh
```

**The installer will:**
1. Check dependencies
2. Install Python packages
3. Set up Rust toolchain (for shadowgit)
4. Configure NPU/OpenVINO
5. Install hooks
6. Set up 98-agent system
7. Configure local Opus

### Step 3: Integration with LAT5150DRVMIL

**After claude-backups installs:**

```bash
# Copy shadowgit to LAT repo
cp -r /home/john/claude-backups/hooks/shadowgit \
      /home/john/LAT5150DRVMIL/06-advanced/

# Copy agent system
cp -r /home/john/claude-backups/agents \
      /home/john/LAT5150DRVMIL/06-advanced/

# Copy NPU optimizations
cp -r /home/john/claude-backups/openvino \
      /home/john/LAT5150DRVMIL/06-advanced/

# Merge configurations
cat /home/john/claude-backups/.env >> ~/.claude/api_keys/keys.env
```

### Step 4: Test Integration

```bash
# Test shadowgit
cd /home/john/LAT5150DRVMIL
git status  # Should use accelerated shadowgit

# Test NPU optimization
python3 /home/john/claude-backups/hardware/npu_integration.py status

# Test local Claude
# (depends on how it's configured in claude-backups)
```

---

## Instead of npx

**Currently you're running:**
```bash
npx claude-code  # Cloud-based, runs via npm
```

**After claude-backups install, you'll run:**
```bash
# Local Claude Code (from claude-backups)
python3 /home/john/claude-backups/autonomous_claude_system.py

# Or integrated with LAT5150DRVMIL
python3 /home/john/LAT5150DRVMIL/02-ai-engine/unified_orchestrator.py query "your task"
```

**Benefits:**
- ✅ Runs locally (faster startup)
- ✅ NPU-accelerated (hardware optimizations)
- ✅ Shadowgit integration (faster git ops)
- ✅ 98-agent coordination (complex tasks)
- ✅ Can integrate with DSMIL attestation

---

## Expected File Structure After Install

```
/home/john/
├── LAT5150DRVMIL/              # Your unified platform
│   ├── 02-ai-engine/
│   │   ├── dsmil_ai_engine.py      # DSMIL attestation
│   │   ├── unified_orchestrator.py  # LOCAL-FIRST routing
│   │   └── sub_agents/              # Gemini, OpenAI
│   └── 06-advanced/                 # New folder
│       ├── shadowgit/               # From claude-backups
│       ├── agents/                  # 98-agent system
│       └── openvino/                # NPU optimizations
└── claude-backups/                 # Source system
    ├── hooks/shadowgit/
    ├── agents/
    └── installer
```

---

## Potential Issues & Solutions

### Issue 1: Python Version

**If installer requires Python 3.11:**
```bash
# Check version
python3 --version

# If < 3.11, might need to upgrade or use pyenv
```

### Issue 2: Dependency Conflicts

**If packages conflict:**
```bash
# Use separate venv for claude-backups
python3 -m venv ~/claude-backups/venv
source ~/claude-backups/venv/bin/activate
./installer
```

### Issue 3: Shadowgit Compilation

**If shadowgit fails to compile:**
```bash
# Check Rust installed
cargo --version

# Manually build
cd hooks/shadowgit/src
make
```

---

## Testing After Install

### 1. Test Shadowgit

```bash
cd /home/john/LAT5150DRVMIL
git status  # Should use accelerated version
time git diff  # Should be faster than normal git
```

### 2. Test Local Claude

```bash
# Run autonomous system
python3 /home/john/claude-backups/autonomous_claude_system.py

# Or bulletproof launcher
python3 /home/john/claude-backups/BULLETPROOF_LOCAL_LAUNCHER.py
```

### 3. Test Integration

```bash
# Unified orchestrator with all backends
python3 /home/john/LAT5150DRVMIL/02-ai-engine/unified_orchestrator.py status

# Should show:
# - Local DeepSeek (DSMIL-attested)
# - Gemini Pro (multimodal)
# - Claude-backups system (local coordination)
```

---

## Configuration Merge

**Combine configurations:**

```bash
# API keys (already done)
cat ~/.claude/api_keys/keys.env

# NPU settings (merge with claude-backups NPU config)
cat ~/.claude/npu-military.env
cat /home/john/claude-backups/config/npu-config.json  # If exists

# Shadowgit settings
# (follow claude-backups documentation)
```

---

## Rollback Plan

**If something breaks:**

```bash
# Uninstall claude-backups
cd /home/john/claude-backups
./uninstall  # If available

# Or manually remove
rm -rf ~/.claude-backups-config
# LAT5150DRVMIL remains unaffected
```

---

## Next Session Plan

**1. Reboot (5 min)**
- Apply NPU Covert Mode
- Verify 99.4 TOPS

**2. Install Claude-Backups (1 hour)**
- Run installer
- Integrate shadowgit
- Test local Claude

**3. Download Coding Models (30 min)**
- deepseek-coder:6.7b
- qwen2.5-coder:14b

**4. Merge Systems (1 hour)**
- Combine configurations
- Test unified platform
- Push to GitHub

**Total: ~3 hours to complete unified platform**

---

**Ready to install!** The installer script is at `/home/john/claude-backups/installer`

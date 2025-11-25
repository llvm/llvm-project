# LAT5150DRVMIL - How to Use Guide

**Quick operational guide for the three main entry points**

---

## 🎯 Three Ways to Use This System

### 1️⃣ dsmil.py - Interactive Driver & Package Manager

**What it does:** Builds kernel drivers, manages DEB packages, launches control centre

**Quick Start:**
```bash
sudo python3 dsmil.py
```

**You'll see a TEMPEST-themed menu with options:**

**QUICK START (Options 1-3):**
- `[1]` **Smart Build** - Auto-detects Rust, builds drivers, loads them (RECOMMENDED)
- `[2]` **Complete Setup** - Build → Load → Launch Control Centre
- `[3]` **Deploy & Activate** - Load drivers and auto-activate safe devices

**DRIVER OPERATIONS (Options 4-8):**
- `[4]` Build driver (standard)
- `[5]` Build driver (clean rebuild)
- `[6]` Load driver (auto-detect)
- `[7]` Unload driver
- `[8]` Driver status

**CONTROL CENTRE (Options 9-11):**
- `[9]` Launch control centre (interactive)
- `[10]` Control centre (AUTO-DISCOVER)
- `[11]` Control centre (AUTO-ACTIVATE SAFE DEVICES)

**AI & AUTOMATION (Options 15-16):**
- `[15]` Initialize Code-Mode
- `[16]` Launch AI Engine

**DEB PACKAGE SYSTEM (Options 17-19):**
- `[17]` **Build DEB packages** (4 packages: platform, tools, examples)
- `[18]` **Install DEB packages** (requires root)
- `[19]` **Verify installation** (10-point check)

**DIAGNOSTICS & INFO (Options 12-14):**
- `[12]` System diagnostics
- `[13]` Documentation
- `[14]` Command help

**Direct command-line usage:**
```bash
# Build drivers with auto-detection
sudo python3 dsmil.py build-auto

# Load drivers
sudo python3 dsmil.py load

# Check status
sudo python3 dsmil.py status

# Launch control centre with auto-discovery
sudo python3 dsmil.py control --auto
```

---

### 2️⃣ lat5150_entrypoint.sh - Complete Tmux Environment

**What it does:** Launches a 6-pane tmux session with all components

**Quick Start:**
```bash
./lat5150_entrypoint.sh
```

**On startup, you'll be prompted:**
```
Would you like to build/install DEB packages before launching?
  1) Build DEB packages (4 packages)
  2) Build and Install DEB packages (requires root)
  3) Skip - Launch environment now
```

**Select your option (default is 3 - skip):**
- Option 1: Builds packages only
- Option 2: Builds, installs, and verifies packages
- Option 3: Skips to tmux launch

**After selection, it launches 6 tmux panes:**

```
┌─────────────────┬─────────────────┬─────────────────┐
│                 │                 │                 │
│  MAIN           │  STATUS         │  LOGS           │
│  (dsmil.py)     │  (monitoring)   │  (real-time)    │
│                 │                 │                 │
├─────────────────┼─────────────────┼─────────────────┤
│                 │                 │                 │
│  TESTS          │  DEV SHELL      │  SYSTEM         │
│  (test runner)  │  (your work)    │  (monitoring)   │
│                 │                 │                 │
└─────────────────┴─────────────────┴─────────────────┘
```

**Tmux Controls:**
- `Ctrl+B` then `D` - Detach from session (keeps running)
- `Ctrl+B` then arrow keys - Navigate between panes
- `tmux attach -t lat5150` - Reattach to running session
- `tmux kill-session -t lat5150` - Kill the session

**What each pane does:**
- **MAIN**: Interactive dsmil.py menu
- **STATUS**: Live driver/system status
- **LOGS**: Real-time kernel logs
- **TESTS**: Test execution pane
- **DEV SHELL**: Your development workspace
- **SYSTEM**: System resource monitoring

---

### 3️⃣ dsmil_control_centre.py - Device Management TUI

**What it does:** Interactive TUI for discovering and managing 104 DSMIL devices

**Quick Start:**
```bash
./dsmil_control_centre.py
```

**Or from dsmil.py menu:**
```bash
sudo python3 dsmil.py
# Then select option 9, 10, or 11
```

**Or with command-line options:**
```bash
# Auto-discover devices
sudo ./dsmil_control_centre.py --auto-discover

# Auto-discover and activate safe devices
sudo ./dsmil_control_centre.py --auto-activate
```

**What you can do:**
1. **Device Discovery**
   - Scans for all 104 DSMIL devices
   - Shows device status (available, active, quarantined)
   - Displays security levels

2. **Device Activation**
   - Activate safe devices (99 devices available)
   - Safety guardrails prevent activating dangerous devices
   - 5 devices permanently quarantined:
     - 0x8009 - DATA_DESTRUCTION
     - 0x800A - SECURE_ERASE
     - 0x800B - PERMANENT_DISABLE
     - 0x8019 - NETWORK_KILL
     - 0x8029 - FILESYSTEM_WIPE

3. **Real-time Monitoring**
   - Live device status updates
   - TPM authentication status
   - System health metrics

4. **Security Features**
   - 4-layer safety enforcement
   - TPM 2.0 attestation
   - Quantum-resistant crypto (CSNA 2.0)

**Navigation:**
- Arrow keys to navigate
- Enter to select
- Q to quit
- Tab to switch between sections

---

## 🚀 Recommended Workflow

### First Time Setup
```bash
# Option 1: Install DEB packages (system-wide)
cd packaging
./build-all-debs.sh
sudo ./install-all-debs.sh
./verify-installation.sh

# Option 2: Or use dsmil.py menu
sudo python3 dsmil.py
# Select option 17 (build), then 18 (install), then 19 (verify)
```

### Daily Development
```bash
# Quick start with tmux environment
./lat5150_entrypoint.sh
# Select option 3 (skip packages) to launch immediately

# Or just the driver build system
sudo python3 dsmil.py
# Select option 1 (Smart Build)
```

### Device Management
```bash
# Launch control centre
sudo python3 dsmil.py
# Select option 11 (Auto-activate safe devices)

# Or directly
sudo ./dsmil_control_centre.py --auto-activate
```

---

## 📊 Component Comparison

| Component | Purpose | When to Use | Requires Root |
|-----------|---------|-------------|---------------|
| **dsmil.py** | Build drivers, manage packages | Building/loading drivers | Yes |
| **lat5150_entrypoint.sh** | Full dev environment | Development work | No* |
| **dsmil_control_centre.py** | Device management | Managing DSMIL devices | Yes |

*Root needed only if building/installing packages or drivers

---

## 🔧 Troubleshooting

### dsmil.py Issues

**"Permission denied"**
```bash
# Always run with sudo
sudo python3 dsmil.py
```

**"Kernel headers not found"**
```bash
# Install kernel headers
sudo apt-get install linux-headers-$(uname -r)
```

**"Rust not available"**
```
# This is OK! Build will use C stubs automatically
# Just select option 1 (Smart Build) and it handles it
```

### lat5150_entrypoint.sh Issues

**"tmux not found"**
```bash
sudo apt-get install tmux
```

**"Session already exists"**
```bash
# Kill existing session
tmux kill-session -t lat5150

# Or attach to it
tmux attach -t lat5150
```

### dsmil_control_centre.py Issues

**"No driver loaded"**
```bash
# Load driver first
sudo python3 dsmil.py
# Select option 6 (Load driver)
```

**"Permission denied"**
```bash
# Run with sudo
sudo ./dsmil_control_centre.py
```

---

## 📚 Documentation

**Start Here:**
- [QUICKSTART.md](QUICKSTART.md) - 5-minute setup guide
- [README.md](README.md) - Complete overview

**Build & Install:**
- [packaging/BUILD_INSTRUCTIONS.md](packaging/BUILD_INSTRUCTIONS.md) - DEB package guide
- [packaging/CHANGELOG.md](packaging/CHANGELOG.md) - Package system changes

**Reference:**
- [INDEX.md](INDEX.md) - Directory structure guide
- [SESSION_SUMMARY.md](SESSION_SUMMARY.md) - Complete session history

---

## 🎯 Quick Command Reference

```bash
# Build and install everything
cd packaging && ./build-all-debs.sh && sudo ./install-all-debs.sh

# Build kernel drivers
sudo python3 dsmil.py build-auto

# Load drivers
sudo python3 dsmil.py load

# Launch control centre with auto-activation
sudo python3 dsmil.py control --activate

# Launch full tmux environment
./lat5150_entrypoint.sh

# Verify DEB package installation
cd packaging && ./verify-installation.sh

# Check driver status
sudo python3 dsmil.py status

# Show demo of all components
./demo.sh
```

---

**Need help?** Run `sudo python3 dsmil.py` and select option 14 (Command help)

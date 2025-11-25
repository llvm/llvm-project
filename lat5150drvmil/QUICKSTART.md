# LAT5150DRVMIL Quick Start Guide

**Get up and running in 5 minutes**

## Option 1: DEB Package Installation (Recommended)

**One command to build and install everything:**

```bash
cd packaging && ./build-all-debs.sh && sudo ./install-all-debs.sh
```

This installs:
- DSMIL Platform (AI interface with 7 auto-coding tools)
- Dell MIL-SPEC management tools
- TPM2 acceleration examples
- Complete meta-package

**After installation, try:**
```bash
dsmil-status          # Check system status
milspec-monitor       # Monitor health
tpm2-accel-status     # Check TPM2 acceleration
```

---

## Option 2: Kernel Driver Only

**Build and load DSMIL kernel drivers:**

```bash
# Interactive menu
sudo python3 dsmil.py

# Or direct commands
sudo python3 dsmil.py build-auto   # Auto-detect Rust, build drivers
sudo python3 dsmil.py load-all     # Load both driver variants
sudo python3 dsmil.py status       # Check status
```

**Interactive Menu Includes:**
- Options 1-16: Driver operations, control centre, AI engine
- **Options 17-19: DEB Package System (NEW!)**
  - Build all DEB packages
  - Install DEB packages (requires root)
  - Verify installation

**Features:**
- Auto-detects Rust toolchain
- Falls back to C stubs if Rust unavailable
- Builds dsmil-104dev.ko and dsmil-84dev.ko
- Verbose diagnostic output
- **Integrated DEB package management**

---

## Option 3: Complete Development Suite

**Launch full tmux environment:**

```bash
./lat5150_entrypoint.sh
```

**Startup Prompt (NEW!):**
- `1)` Build DEB packages only
- `2)` Build + Install + Verify DEB packages
- `3)` Skip - Launch environment immediately (default)

**6-pane tmux session with:**
- Main menu (dsmil.py)
- Status monitoring
- Real-time logs
- Test runner
- Development shell
- System monitoring

---

## Option 4: Device Management

**Interactive device control:**

```bash
./dsmil_control_centre.py
```

TUI for:
- Discovering 104 DSMIL devices
- Safe device activation
- Real-time monitoring
- TPM authentication

---

## What You Get

### Core Components
- **104 DSMIL Devices** (99 usable, 5 quarantined for safety)
- **Kernel Drivers** (Rust safety layer or C stubs)
- **AI Platform** (ChatGPT-style interface, LOCAL-FIRST)
- **TPM 2.0** (88 cryptographic algorithms)
- **Quantum Crypto** (CSNA 2.0 post-quantum algorithms)

### DEB Packages (4 total)
1. **dsmil-platform** (2.5 MB) - Complete AI platform
2. **dell-milspec-tools** (24 KB) - Management tools
3. **tpm2-accel-examples** (19 KB) - Hardware acceleration examples
4. **dsmil-complete** (1.5 KB) - Meta-package

### Available Commands After Install
```bash
dsmil-status          # Device status
dsmil-test            # Test functionality
milspec-control       # Control features
milspec-monitor       # Health monitoring
tpm2-accel-status     # TPM2 acceleration
milspec-emergency-stop # Emergency shutdown
```

---

## System Requirements

### Minimum (Development/Docker)
- Linux (Ubuntu 22.04+, Debian 11+) or macOS
- 8 GB RAM
- 20 GB storage
- Python 3.10+

### Optimal (Dell MIL-SPEC Hardware)
- Dell Latitude 5450 Covert Edition
- TPM 2.0
- 64 GB RAM
- 4 TB NVMe
- Intel Core Ultra 7 with AI Boost

---

## Documentation

- **[README.md](README.md)** - Complete documentation
- **[packaging/BUILD_INSTRUCTIONS.md](packaging/BUILD_INSTRUCTIONS.md)** - DEB package details
- **[INDEX.md](INDEX.md)** - Directory guide
- **[00-documentation/](00-documentation/)** - Full docs

---

## Troubleshooting

### DEB Packages Won't Install
```bash
# Check if packages exist
ls -lh packaging/*.deb

# Rebuild if needed
cd packaging && ./build-all-debs.sh

# Install with dependency fixing
sudo ./install-all-debs.sh
sudo apt-get install -f
```

### Kernel Drivers Won't Build
```bash
# Check if kernel headers installed
ls /usr/src/linux-headers-$(uname -r)

# Install if missing
sudo apt-get install linux-headers-$(uname -r)

# Rebuild with verbose output
sudo python3 dsmil.py build-auto
```

### Rust Not Available
No problem! The build system auto-detects and falls back to C stubs:
```bash
# It will automatically use C stubs
sudo python3 dsmil.py build-auto
```

### Permission Errors
```bash
# All management commands need sudo
sudo python3 dsmil.py
sudo ./dsmil_control_centre.py
sudo ./install-all-debs.sh
```

---

## Next Steps

1. **Read the main README**: [README.md](README.md)
2. **Explore documentation**: [00-documentation/](00-documentation/)
3. **Check build instructions**: [packaging/BUILD_INSTRUCTIONS.md](packaging/BUILD_INSTRUCTIONS.md)
4. **Review directory structure**: [INDEX.md](INDEX.md)

---

**LAT5150DRVMIL v9.0.0** | Dell Latitude 5450 Covert Edition | LOCAL-FIRST AI

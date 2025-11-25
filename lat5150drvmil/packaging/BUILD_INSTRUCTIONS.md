# DEB Package Building Instructions

This directory contains the source structure and build scripts for all DSMIL .deb packages.

## Quick Start

### Build ALL packages at once:
```bash
cd packaging
./build-all-debs.sh
```

### Build individual packages:
```bash
./build-all-debs.sh dsmil-platform
./build-all-debs.sh dell-milspec-tools
./build-all-debs.sh tpm2-accel-examples
```

## Package Overview

### 1. dsmil-platform (8.3.1-1)
**Purpose**: DSMIL Unified AI Platform - LOCAL-FIRST AI development platform

**Contents**:
- Complete AI platform with ChatGPT-style interface
- 7 auto-coding tools (Edit, Create, Debug, Refactor, Review, Tests, Docs)
- Web search & crawling with PDF extraction
- RAG knowledge base
- Chat history persistence
- Hardware attestation via TPM 2.0

**Size**: ~2.5 MB
**Location**: `dsmil-platform_8.3.1-1/`
**Installs to**: `/opt/dsmil/`

**Dependencies**:
- python3 (>= 3.9)
- python3-pip
- git, curl, wget

**Build Command**:
```bash
dpkg-deb --build dsmil-platform_8.3.1-1 dsmil-platform_8.3.1-1.deb
```

---

### 2. dell-milspec-tools (1.0.0-1)
**Purpose**: Management and monitoring tools for Dell MIL-SPEC platform

**Contents**:
- Real-time monitoring dashboard for DSMIL device
- Safe token testing utilities for SMBIOS tokens
- Emergency stop mechanisms for safety
- Resource monitoring and thermal protection
- TPM2 hardware acceleration status tools
- System health monitoring and alerts

**Size**: ~24 KB
**Location**: `dell-milspec-tools/`
**Installs to**: `/usr/bin/`, `/usr/sbin/`, `/usr/share/dell-milspec/`

**Dependencies**:
- python3 (>= 3.10)
- python3-psutil
- bash (>= 4.4)
- coreutils, procps

**Executables Provided**:
- `dsmil-status` - Check DSMIL device status
- `dsmil-test` - Test DSMIL functionality
- `milspec-control` - Control MIL-SPEC features
- `milspec-monitor` - Monitor system health
- `tpm2-accel-status` - Check TPM2 acceleration
- `milspec-emergency-stop` - Emergency shutdown

**Build Command**:
```bash
dpkg-deb --build dell-milspec-tools dell-milspec-tools_1.0.0-1_amd64.deb
```

---

### 3. tpm2-accel-examples (1.0.0-1)
**Purpose**: Example programs for TPM2 hardware acceleration

**Contents**:
- SECRET level (security level 2) C example
- Complete workflow documentation
- Status checking script
- Makefile for compilation

**Size**: ~19 KB
**Location**: `tpm2-accel-examples_1.0.0-1/`
**Installs to**: `/usr/share/doc/tpm2-accel-examples/`

**Dependencies**:
- bash (>= 4.4)

**Recommends**:
- tpm2-accel-early-dkms (>= 1.0.0)
- gcc, make

**Examples Included**:
- `secret_level_crypto_example.c` - AES-256-GCM encryption
- `check_tpm2_acceleration.sh` - Verify hardware acceleration
- Makefile for easy compilation

**Build Command**:
```bash
dpkg-deb --build tpm2-accel-examples_1.0.0-1 tpm2-accel-examples_1.0.0-1.deb
```

---

### 4. dsmil-complete (8.3.2-1)
**Purpose**: Meta-package that depends on all other packages

**Dependencies**:
- dsmil-platform (>= 8.3.1)
- dell-milspec-tools (>= 1.0.0)
- tpm2-accel-examples (>= 1.0.0)

**Build Command**:
```bash
dpkg-deb --build dsmil-complete_8.3.2-1 dsmil-complete_8.3.2-1.deb
```

## Installation Order

**Important**: Install packages in this order to satisfy dependencies:

```bash
# 1. Install platform (foundation)
sudo dpkg -i dsmil-platform_8.3.1-1.deb

# 2. Install tools
sudo dpkg -i dell-milspec-tools_1.0.0-1_amd64.deb

# 3. Install examples
sudo dpkg -i tpm2-accel-examples_1.0.0-1.deb

# 4. Install meta-package (optional, pulls everything together)
sudo dpkg -i dsmil-complete_8.3.2-1.deb

# 5. Fix any missing dependencies
sudo apt-get install -f
```

**Or install all at once**:
```bash
sudo dpkg -i dsmil-platform_8.3.1-1.deb \
              dell-milspec-tools_1.0.0-1_amd64.deb \
              tpm2-accel-examples_1.0.0-1.deb \
              dsmil-complete_8.3.2-1.deb

sudo apt-get install -f
```

## Building from Scratch

### Prerequisites
```bash
sudo apt-get install dpkg-dev build-essential
```

### Manual Build Process

1. **Prepare the directory structure**:
   ```bash
   cd packaging/dsmil-platform_8.3.1-1
   ```

2. **Set proper permissions**:
   ```bash
   chmod -R 755 DEBIAN
   chmod 644 DEBIAN/control
   chmod 755 DEBIAN/postinst  # If exists
   ```

3. **Set ownership** (if running as root):
   ```bash
   sudo chown -R root:root .
   ```

4. **Build the package**:
   ```bash
   cd ..
   dpkg-deb --build dsmil-platform_8.3.1-1
   ```

5. **Verify the package**:
   ```bash
   dpkg-deb -I dsmil-platform_8.3.1-1.deb
   dpkg-deb -c dsmil-platform_8.3.1-1.deb | head -20
   ```

## Troubleshooting

### Permission Issues
If you get permission errors:
```bash
# Option 1: Build as root
sudo ./build-all-debs.sh

# Option 2: Build without root (will work but may show warnings)
./build-all-debs.sh

# Option 3: Fix permissions manually
chmod -R 755 dsmil-platform_8.3.1-1/DEBIAN
chmod 644 dsmil-platform_8.3.1-1/DEBIAN/control
```

### Dependency Issues
If installation fails due to missing dependencies:
```bash
# Auto-fix dependencies
sudo apt-get install -f

# Or install dependencies manually first
sudo apt-get install python3 python3-pip python3-psutil git curl wget
```

### Verification

**Automated verification** (recommended):
```bash
./verify-installation.sh
```

This comprehensive script checks:
- All 4 packages installed correctly
- All executables available in PATH
- Python environment configured
- Build prerequisites present
- Kernel headers available
- Rust toolchain (optional)
- Documentation installed
- System permissions

**Manual verification**:

Check if packages are installed:
```bash
dpkg -l | grep dsmil
dpkg -l | grep dell-milspec
dpkg -l | grep tpm2-accel
```

Check package contents:
```bash
dpkg -L dsmil-platform
dpkg -L dell-milspec-tools
dpkg -L tpm2-accel-examples
```

Test installed commands:
```bash
dsmil-status
tpm2-accel-status
milspec-monitor
```

## Package Maintenance

### Updating Package Version

To update package versions, edit the `DEBIAN/control` file in each package directory:

```bash
# Edit version number
nano dsmil-platform_8.3.1-1/DEBIAN/control

# Update Version: field
Version: 8.3.2-1  # Increment version

# Rebuild
./build-all-debs.sh
```

### Adding Files to Packages

1. Add files to the appropriate directory structure:
   ```bash
   # Example: Add new script to dell-milspec-tools
   cp my-new-script.sh dell-milspec-tools/usr/bin/
   chmod 755 dell-milspec-tools/usr/bin/my-new-script.sh
   ```

2. Rebuild the package:
   ```bash
   ./build-all-debs.sh dell-milspec-tools
   ```

### Removing Packages

```bash
# Remove individual package
sudo dpkg -r dsmil-complete

# Remove with config files
sudo dpkg -P dsmil-complete

# Remove all DSMIL packages
sudo dpkg -r dsmil-complete dell-milspec-tools tpm2-accel-examples dsmil-platform
```

## Directory Structure

```
packaging/
├── build-all-debs.sh                    # Master build script
├── BUILD_INSTRUCTIONS.md                # This file
│
├── dsmil-platform_8.3.1-1/              # Platform package
│   ├── DEBIAN/
│   │   ├── control                      # Package metadata
│   │   └── postinst                     # Post-installation script
│   ├── etc/systemd/system/              # Systemd services
│   └── opt/dsmil/                       # Main installation
│
├── dell-milspec-tools/                  # Tools package
│   ├── DEBIAN/
│   │   ├── control
│   │   ├── postinst
│   │   ├── prerm
│   │   └── postrm
│   ├── usr/bin/                         # User executables
│   ├── usr/sbin/                        # Admin executables
│   └── usr/share/dell-milspec/          # Shared files
│
├── tpm2-accel-examples_1.0.0-1/         # Examples package
│   ├── DEBIAN/
│   │   ├── control
│   │   └── postinst
│   └── usr/share/doc/tpm2-accel-examples/
│
└── dsmil-complete_8.3.2-1/              # Meta-package
    └── DEBIAN/
        └── control

```

## Quick Reference

| Package | Version | Size | Architecture | Type |
|---------|---------|------|--------------|------|
| dsmil-platform | 8.3.1-1 | 2.5 MB | amd64 | Application |
| dell-milspec-tools | 1.0.0-1 | 24 KB | amd64 | Tools |
| tpm2-accel-examples | 1.0.0-1 | 19 KB | all | Documentation |
| dsmil-complete | 8.3.2-1 | 1.5 KB | Meta-package | Meta |

## Support

For issues with package building or installation:

1. Check this documentation
2. Verify all dependencies are installed
3. Try building with sudo if permission issues occur
4. Check dpkg logs: `journalctl -xe` or `/var/log/dpkg.log`

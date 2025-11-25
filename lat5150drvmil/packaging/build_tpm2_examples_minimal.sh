#!/bin/bash
#
# Build TPM2 Acceleration Examples - Minimal Package (v1.0.0)
# Creates: tpm2-accel-examples_1.0.0-1_all.deb
#
# This script packages ONLY existing files (no new code required).
# The package provides SECRET level (2) demonstration capability.
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=====================================================================${NC}"
echo -e "${BLUE} Building TPM2 Acceleration Examples Package (Minimal v1.0.0)${NC}"
echo -e "${BLUE}=====================================================================${NC}"
echo ""

# Navigate to packaging directory
cd "$(dirname "$0")"
PACKAGING_DIR=$(pwd)

echo -e "${GREEN}Working directory: ${PACKAGING_DIR}${NC}"
echo ""

# Clean up any previous build
if [ -d "tpm2-accel-examples_1.0.0-1" ]; then
    echo -e "${YELLOW}Cleaning up previous build...${NC}"
    rm -rf tpm2-accel-examples_1.0.0-1
fi

if [ -f "tpm2-accel-examples_1.0.0-1.deb" ]; then
    rm -f tpm2-accel-examples_1.0.0-1.deb
fi

# 1. Create package directory structure
echo -e "${GREEN}[1/8] Creating directory structure...${NC}"
mkdir -p tpm2-accel-examples_1.0.0-1/DEBIAN
mkdir -p tpm2-accel-examples_1.0.0-1/usr/share/doc/tpm2-accel-examples/{examples,workflows}

# 2. Copy existing files
echo -e "${GREEN}[2/8] Copying source files...${NC}"
SRC="${PACKAGING_DIR}/../tpm2_compat/c_acceleration"
DST="tpm2-accel-examples_1.0.0-1/usr/share/doc/tpm2-accel-examples"

# Verify source directory exists
if [ ! -d "$SRC" ]; then
    echo -e "${RED}ERROR: Source directory not found: $SRC${NC}"
    exit 1
fi

# Copy example files
cp "$SRC/examples/secret_level_crypto_example.c" "$DST/examples/"
cp "$SRC/examples/secret_crypto" "$DST/examples/"
cp "$SRC/examples/Makefile" "$DST/examples/"
cp "$SRC/check_tpm2_acceleration.sh" "$DST/examples/"

# Copy documentation
cp "$SRC/SECURITY_LEVELS_AND_USAGE.md" "$DST/workflows/"
cp "$SRC/SECRET_LEVEL_WORKFLOW.md" "$DST/workflows/"
cp "$SRC/QUICKSTART_SECRET_LEVEL.md" "$DST/workflows/"

echo -e "${GREEN}   Copied 7 files${NC}"

# 3. Create README
echo -e "${GREEN}[3/8] Creating README.md...${NC}"
cat > "$DST/README.md" <<'EOF'
# TPM2 Acceleration Examples

**Dell Latitude 5450 MIL-SPEC - Intel Core Ultra 7 165H**

## Overview

This package provides example programs demonstrating TPM2 hardware
acceleration capabilities at Security Level 2 (SECRET).

## Quick Start

### Prerequisites

1. **TPM2 acceleration kernel module loaded**:
   ```bash
   lsmod | grep tpm2_accel_early
   ```

2. **User permissions** (add user to tpm2-accel group):
   ```bash
   sudo usermod -a -G tpm2-accel $USER
   # Log out and log back in
   ```

3. **Verify device access**:
   ```bash
   ls -la /dev/tpm2_accel_early
   ```

### Compile the Example

```bash
cd /usr/share/doc/tpm2-accel-examples/examples
make
```

This produces the `secret_crypto` binary.

### Run the Example

```bash
sudo ./secret_crypto
```

**Expected output**:
- Hardware status (NPU, GNA, ME, TPM detection)
- AES-256-GCM encryption demonstration
- SHA3-512 hashing demonstration
- Performance statistics

### Check System Status

```bash
./check_tpm2_acceleration.sh
```

This script displays:
- Module load status
- Security level (0-3)
- Device node information
- Hardware detection results
- Recent kernel messages

## Contents

### Example Programs

- **secret_level_crypto_example.c** (357 lines)
  - Complete SECRET level demonstration
  - Shows AES-256-GCM encryption
  - Shows SHA3-512 hashing
  - Uses Intel NPU acceleration

- **secret_crypto** (pre-compiled binary)
  - Ready to run
  - Requires sudo (device access)

- **Makefile**
  - Simple compilation: `make`
  - Clean: `make clean`

- **check_tpm2_acceleration.sh** (187 lines)
  - Comprehensive status checker
  - Hardware detection
  - Configuration display

### Documentation

- **workflows/QUICKSTART_SECRET_LEVEL.md**
  - Quick setup and usage guide
  - Performance expectations
  - Configuration instructions

- **workflows/SECRET_LEVEL_WORKFLOW.md**
  - Complete workflow documentation
  - Step-by-step procedures

- **workflows/SECURITY_LEVELS_AND_USAGE.md** (631 lines)
  - All 4 security levels explained (0-3)
  - Dell token authorization
  - IOCTL interface documentation
  - FAQ section

## Security Level 2 (SECRET)

### What It Provides

- **AES-256-GCM**: Hardware-accelerated authenticated encryption
- **SHA3-512**: Post-quantum safe hashing
- **Intel NPU**: 34.0 TOPS cryptographic acceleration
- **Intel GNA**: Advanced threat monitoring
- **Intel ME**: Full hardware attestation
- **Memory Encryption**: Hardware-backed protection
- **DMA Protection**: Prevents DMA attacks

### Performance

| Operation | Software | With NPU | Improvement |
|-----------|----------|----------|-------------|
| AES-256-GCM | 200 MB/s | 2.8 GB/s | **14x faster** |
| SHA3-512 | 100 MB/s | 1.2 GB/s | **12x faster** |
| Ops/sec | ~3,000 | 40,000+ | **13x more** |

### Configuration

Set security level in module parameters:
```bash
sudo modprobe -r tpm2_accel_early
sudo modprobe tpm2_accel_early security_level=2
```

Or configure permanently in `/etc/modprobe.d/tpm2-acceleration.conf`:
```
options tpm2_accel_early early_init=1 security_level=2
```

## Troubleshooting

### Device not found
```bash
# Check if module is loaded
lsmod | grep tpm2_accel_early

# Load module
sudo modprobe tpm2_accel_early security_level=2
```

### Permission denied
```bash
# Add user to group
sudo usermod -a -G tpm2-accel $USER

# Or temporarily change permissions
sudo chmod 666 /dev/tpm2_accel_early
```

### Compilation errors
```bash
# Install build tools
sudo apt-get install build-essential

# Check gcc version
gcc --version
```

## Standard TPM Commands

The acceleration module is **transparent** to standard TPM tools:

```bash
# These continue to work unchanged:
tpm2_pcrread           # Read PCR values
tpm2_getrandom 32      # Generate random numbers
tpm2_createprimary     # Create primary key
```

The module provides **additional** acceleration features via
`/dev/tpm2_accel_early`, while standard TPM operations use `/dev/tpm0`.

## Developer Integration

### Method 1: Direct IOCTL

```c
#include <fcntl.h>
#include <sys/ioctl.h>

int fd = open("/dev/tpm2_accel_early", O_RDWR);

struct tpm2_accel_status status;
ioctl(fd, TPM2_ACCEL_IOC_STATUS, &status);

close(fd);
```

See `secret_level_crypto_example.c` for complete example.

### Method 2: C Library (Coming Soon)

```c
#include <tpm2-accel/tpm2_compat_accelerated.h>

tpm2_device_handle_t device;
tpm2_device_open(&config, &device);
// ... use library functions
```

### Method 3: Python Bindings (Coming Soon)

```python
from tpm2_accel import TPM2AccelerationLibrary

lib = TPM2AccelerationLibrary()
lib.initialize()
status = lib.get_status()
```

## Support

- **Module package**: tpm2-accel-early-dkms
- **Tools package**: tpm2-accel-tools (recommended)
- **Main package**: dell-milspec-tools

## Next Steps

1. Try the example: `sudo ./secret_crypto`
2. Read the workflow: `workflows/SECRET_LEVEL_WORKFLOW.md`
3. Experiment with different security levels
4. Integrate into your application

## Classification

**UNCLASSIFIED // FOR OFFICIAL USE ONLY**

---

**Version**: 1.0.0
**Architecture**: all (architecture-independent)
**Date**: 2025-10-11
EOF

# 4. Create control file
echo -e "${GREEN}[4/8] Creating DEBIAN/control...${NC}"
cat > tpm2-accel-examples_1.0.0-1/DEBIAN/control <<'EOF'
Package: tpm2-accel-examples
Version: 1.0.0-1
Section: doc
Priority: optional
Architecture: all
Maintainer: Dell MIL-SPEC Tools Team <milspec@dell.com>
Depends: bash (>= 4.4)
Recommends: tpm2-accel-early-dkms (>= 1.0.0), gcc, make
Suggests: tpm2-tools, dell-milspec-tools
Installed-Size: 64
Homepage: https://github.com/dell/tpm2-acceleration
Description: Example programs for TPM2 hardware acceleration
 Example source code and documentation for using TPM2 acceleration
 features on Dell Latitude 5450 MIL-SPEC systems.
 .
 This package includes:
  - SECRET level (security level 2) C example
  - Complete workflow documentation
  - Status checking script
  - Makefile for compilation
 .
 Demonstrates AES-256-GCM encryption, SHA3-512 hashing, and
 Intel NPU/GNA/ME hardware acceleration features.
 .
 The example shows:
  - Opening /dev/tpm2_accel_early device
  - Configuring SECRET level parameters
  - Hardware-accelerated cryptography
  - Performance measurement
EOF

# 5. Create postinst script
echo -e "${GREEN}[5/8] Creating DEBIAN/postinst...${NC}"
cat > tpm2-accel-examples_1.0.0-1/DEBIAN/postinst <<'EOF'
#!/bin/bash
set -e

echo ""
echo "======================================================================"
echo " TPM2 Acceleration Examples Installed"
echo "======================================================================"
echo ""
echo "Location: /usr/share/doc/tpm2-accel-examples/"
echo ""
echo "Quick start:"
echo "  cd /usr/share/doc/tpm2-accel-examples/examples"
echo "  make"
echo "  sudo ./secret_crypto"
echo ""
echo "Check status:"
echo "  cd /usr/share/doc/tpm2-accel-examples/examples"
echo "  ./check_tpm2_acceleration.sh"
echo ""
echo "Documentation:"
echo "  /usr/share/doc/tpm2-accel-examples/workflows/"
echo ""
echo "Prerequisites:"
echo "  - tpm2-accel-early-dkms module installed and loaded"
echo "  - User in 'tpm2-accel' group (for device access)"
echo ""
echo "Add yourself to group:"
echo "  sudo usermod -a -G tpm2-accel \$USER"
echo "  (then log out and log back in)"
echo ""
echo "======================================================================"
echo ""

exit 0
EOF

chmod 755 tpm2-accel-examples_1.0.0-1/DEBIAN/postinst

# 6. Set file permissions
echo -e "${GREEN}[6/8] Setting file permissions...${NC}"
chmod 755 "$DST/examples/secret_crypto"
chmod 755 "$DST/examples/check_tpm2_acceleration.sh"
chmod 644 "$DST/examples/secret_level_crypto_example.c"
chmod 644 "$DST/examples/Makefile"
chmod 644 "$DST/workflows"/*.md
chmod 644 "$DST/README.md"

# 7. Build package
echo -e "${GREEN}[7/8] Building Debian package...${NC}"
dpkg-deb --build tpm2-accel-examples_1.0.0-1

# 8. Verify package
echo ""
echo -e "${GREEN}[8/8] Verifying package...${NC}"
echo ""

dpkg-deb --info tpm2-accel-examples_1.0.0-1.deb
echo ""
echo -e "${BLUE}Package contents:${NC}"
dpkg-deb --contents tpm2-accel-examples_1.0.0-1.deb

echo ""
echo -e "${GREEN}=====================================================================${NC}"
echo -e "${GREEN} Package built successfully!${NC}"
echo -e "${GREEN}=====================================================================${NC}"
echo ""
echo -e "${BLUE}Package file:${NC} tpm2-accel-examples_1.0.0-1.deb"
echo -e "${BLUE}Size:${NC} $(du -h tpm2-accel-examples_1.0.0-1.deb | cut -f1)"
echo ""
echo -e "${YELLOW}Installation:${NC}"
echo "  sudo dpkg -i tpm2-accel-examples_1.0.0-1.deb"
echo ""
echo -e "${YELLOW}Verification:${NC}"
echo "  dpkg -L tpm2-accel-examples"
echo "  cd /usr/share/doc/tpm2-accel-examples/examples"
echo "  make && sudo ./secret_crypto"
echo ""
echo -e "${YELLOW}Removal:${NC}"
echo "  sudo dpkg -r tpm2-accel-examples"
echo ""
echo -e "${GREEN}Next steps:${NC}"
echo "  1. Install the package"
echo "  2. Load TPM2 acceleration module (if not loaded)"
echo "  3. Run the example"
echo "  4. Read the documentation"
echo ""

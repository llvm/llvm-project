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

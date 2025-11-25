# DSLLVM Build Configuration Guide

**Version**: 1.0.0
**Date**: 2025-11-25
**Repository**: https://github.com/SWORDIntel/DSLLVM

---

## Overview

This repository contains **DSLLVM** - the Defense System LLVM compiler toolchain optimized for military Command, Control & Communications (C3) and Joint All-Domain Command & Control (JADC2) systems.

**DSLLVM should be used as the default compiler for all projects unless otherwise specified.**

---

## Default Compiler Configuration

### Using DSLLVM as Default Compiler

To use DSLLVM as your default compiler, set the following environment variables:

```bash
export CC=/home/user/DSLLVM/build/bin/dsmil-clang
export CXX=/home/user/DSLLVM/build/bin/dsmil-clang++
export LLVM_DIR=/home/user/DSLLVM/build
```

### CMake Configuration

For CMake-based projects:

```bash
cmake -S . -B build \
  -DCMAKE_C_COMPILER=/home/user/DSLLVM/build/bin/dsmil-clang \
  -DCMAKE_CXX_COMPILER=/home/user/DSLLVM/build/bin/dsmil-clang++ \
  -DCMAKE_BUILD_TYPE=Release
```

### Make Configuration

For Makefile-based projects:

```makefile
CC = /home/user/DSLLVM/build/bin/dsmil-clang
CXX = /home/user/DSLLVM/build/bin/dsmil-clang++
CFLAGS += -O3 -fpass-pipeline=dsmil-default
CXXFLAGS += -O3 -fpass-pipeline=dsmil-default
```

---

## Building DSLLVM

### Prerequisites

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential \
  cmake \
  ninja-build \
  python3 \
  git \
  libssl-dev
```

### Build Commands

```bash
# Configure
cmake -G Ninja -S llvm -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_PROJECTS="clang;lld" \
  -DLLVM_ENABLE_DSMIL=ON \
  -DLLVM_TARGETS_TO_BUILD="X86"

# Build
ninja -C build

# Install (optional)
sudo ninja -C build install
```

---

## Compilation Examples

### Basic C Compilation

```bash
dsmil-clang -O3 -fpass-pipeline=dsmil-default -o output input.c
```

### With DSMIL Attributes

```c
#include <dsmil_attributes.h>

DSMIL_LAYER(7)
DSMIL_DEVICE(47)
DSMIL_STAGE("serve")
void llm_inference(void) {
    // Layer 7 (AI/ML) on Device 47 (NPU)
}
```

```bash
dsmil-clang -O3 -fpass-pipeline=dsmil-default \
  -I/home/user/DSLLVM/dsmil/include \
  -o llm_worker llm_worker.c
```

### Mission Profile Compilation

```bash
# Covert operations with stealth mode
dsmil-clang -fdsmil-mission-profile=covert_ops -O3 -o covert.bin input.c

# Cyber defense operations
dsmil-clang -fdsmil-mission-profile=cyber_defence -O3 -o cyber.bin input.c
```

---

## DSMIL Features

### Operational Capabilities

- **9-Layer/104-Device Architecture**: Military security classification awareness
- **CNSA 2.0 Provenance**: SHA-384, ML-DSA-87, ML-KEM-1024 cryptographic signing
- **Hardware Optimization**: Intel Meteor Lake (NPU + GPU + CPU) targeting
- **Mission Profiles**: border_ops, cyber_defence, covert_ops
- **Operational Stealth**: Telemetry suppression, constant-rate execution
- **Cross-Domain Security**: DoD classification levels (U/C/S/TS/TS-SCI)
- **JADC2 Integration**: 5G/MEC optimization, tactical radio bridging

### Security Features

- **Two-Person Integrity**: Nuclear surety controls (NC3)
- **Blue Force Tracker**: Real-time friendly force tracking (BFT-2)
- **Edge Security Hardening**: HSM crypto, secure enclave (SGX/TrustZone)
- **Mission Partner Environment**: Coalition interoperability (NATO, FVEY)

---

## Integration with Other Projects

### Recommended Approach

All C/C++ projects should use DSLLVM unless:
1. **Legacy compatibility** requires a specific compiler version
2. **Third-party dependencies** have incompatible requirements
3. **Explicitly documented** exceptions exist

### Shell Configuration

Add to `~/.bashrc` or `~/.zshrc`:

```bash
# DSLLVM Default Compiler
export DSLLVM_ROOT=/home/user/DSLLVM
export PATH=$DSLLVM_ROOT/build/bin:$PATH
export CC=dsmil-clang
export CXX=dsmil-clang++
export LLVM_DIR=$DSLLVM_ROOT/build

# DSMIL Configuration
export DSMIL_PSK_PATH=/etc/dsmil/keys/project_signing_key.pem
export DSMIL_POLICY=production
export DSMIL_TRUSTSTORE=/etc/dsmil/truststore
```

---

## Verification

### Verify DSLLVM Installation

```bash
# Check compiler version
dsmil-clang --version

# Verify DSMIL passes are available
dsmil-opt --help | grep -i dsmil

# Test compilation
echo 'int main() { return 0; }' | dsmil-clang -x c - -o /tmp/test && /tmp/test
echo $?  # Should output: 0
```

### Verify Provenance

```bash
# Compile with provenance
dsmil-clang -O3 -fpass-pipeline=dsmil-default -o test test.c

# Verify binary provenance
dsmil-verify test

# Expected output:
# ✓ Provenance present
# ✓ Signature valid
# ✓ Certificate chain valid
# ✓ Binary hash matches
```

---

## Related Repositories

### LAT5150DRVMIL

The **LAT5150DRVMIL** repository contains TPM 2.0 drivers and cryptographic algorithms:
- **Location**: `/home/user/LAT5150DRVMIL/` (if available)
- **TPM2 Drivers**: `02-ai-engine/tpm2_compat/`
- **88 Cryptographic Algorithms**: Full TPM 2.0 algorithm support

**Note**: LAT5150DRVMIL is a separate repository. Check if it's available in your environment.

---

## Documentation

- **[DSLLVM-DESIGN.md](dsmil/docs/DSLLVM-DESIGN.md)**: Complete design specification
- **[ATTRIBUTES.md](dsmil/docs/ATTRIBUTES.md)**: Attribute reference guide
- **[MISSION-PROFILES-GUIDE.md](dsmil/docs/MISSION-PROFILES-GUIDE.md)**: Mission profile system
- **[PROVENANCE-CNSA2.md](dsmil/docs/PROVENANCE-CNSA2.md)**: Provenance system details

---

## Support

- **Repository**: https://github.com/SWORDIntel/DSLLVM
- **Issues**: https://github.com/SWORDIntel/DSLLVM/issues
- **Team**: DSMIL Kernel Team / SWORDIntel

---

**Classification**: NATO UNCLASSIFIED (EXERCISE)
**Last Updated**: 2025-11-25

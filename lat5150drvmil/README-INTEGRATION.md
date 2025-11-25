# LAT5150DRVMIL Integration with DSLLVM

**Version**: 1.0.0
**Date**: 2025-11-25
**Status**: Reference Copy

---

## Overview

This directory contains a **reference copy** of the LAT5150DRVMIL repository, providing the complete TPM2 implementation with kernel modules and hardware acceleration.

### Source Repository

**Original Repository**: https://github.com/SWORDIntel/LAT5150DRVMIL

This is a snapshot copy integrated into DSLLVM for reference and to ensure all TPM2 drivers and documentation are available locally.

---

## TPM2 Compatibility Layer

The complete TPM2 implementation with 88 algorithms is located at:

**`02-ai-engine/tpm2_compat/`**

This includes:
- ✅ **Rust implementation** with aws-lc backend
- ✅ **C kernel module** (`c_acceleration/tpm2_accel_early.c`)
- ✅ **Hardware NPU acceleration** support
- ✅ **Dell military token** authorization
- ✅ **Python management tools**
- ✅ **Complete documentation**

---

## Key Files

### TPM2 Implementation
- `02-ai-engine/tpm2_compat/c_acceleration/` - C kernel module and acceleration
- `02-ai-engine/tpm2_compat/TPM2_FULL_ALGORITHM_SUPPORT.md` - Algorithm specification
- `02-ai-engine/tpm2_compat/c_acceleration/tpm2_accel_early.c` - Kernel module (1,168 lines)
- `02-ai-engine/tpm2_compat/c_acceleration/tpm2_accel_early.h` - Header with IOCTL interface

### Documentation
- `00-documentation/` - Complete system documentation
- `BUILD_ORDER.md` - Build instructions
- `QUICKSTART.md` - Quick start guide

---

## Relationship to DSLLVM tpm2_compat

The DSLLVM `tpm2_compat/` directory (at repository root) provides a **simplified userspace C library** for portability.

This LAT5150DRVMIL copy provides the **complete system** with:
- Kernel-level integration
- Hardware acceleration (10-50× speedup)
- Production-ready deployment tools

**See**: `../tpm2_compat/INTEGRATION.md` for detailed comparison

---

## Usage

### Build TPM2 Kernel Module

```bash
cd lat5150drvmil/02-ai-engine/tpm2_compat/c_acceleration
./deploy.sh

# Verify
lsmod | grep tpm2_accel_early
```

### Build Rust Components

```bash
cd lat5150drvmil/02-ai-engine/tpm2_compat/c_acceleration
cargo build --release
```

### Deploy Full System

```bash
cd lat5150drvmil
make install
```

---

## 88 Cryptographic Algorithms

Both the DSLLVM userspace library and this full implementation support the same 88 algorithms:

- 10 Hash algorithms (SHA-256/384/512, SHA3-256/384/512, SM3, SHAKE-128/256)
- 16 AES modes (ECB, CBC, CTR, OFB, CFB, GCM, CCM, XTS)
- 6 Other symmetric ciphers (3DES, Camellia, SM4, ChaCha20, ChaCha20-Poly1305)
- 5 RSA key sizes (1024-8192 bits)
- 12 Elliptic curves (NIST P-curves, Curve25519/448, Ed25519/448, SM2, BN)
- 5 HMAC algorithms
- 11 Key derivation functions (HKDF, PBKDF2, scrypt, Argon2, SP800-108/56A)
- 8 Signature schemes
- 3 Key agreement protocols
- 4 Mask generation functions
- 8 Post-quantum algorithms (Kyber, Dilithium, Falcon)

---

## Integration Notes

This is a **reference copy** for local access. For the latest updates, refer to the original repository:

https://github.com/SWORDIntel/LAT5150DRVMIL

To update this copy:
```bash
git clone https://github.com/SWORDIntel/LAT5150DRVMIL.git /tmp/LAT5150DRVMIL-new
rm -rf lat5150drvmil
cp -r /tmp/LAT5150DRVMIL-new lat5150drvmil
rm -rf lat5150drvmil/.git
```

---

**Classification**: UNCLASSIFIED // FOR OFFICIAL USE ONLY
**Source**: https://github.com/SWORDIntel/LAT5150DRVMIL
**Integration Date**: 2025-11-25

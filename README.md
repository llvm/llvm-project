# DSLLVM - Defense System LLVM Compiler

**Version**: 1.6.0 (Phase 3: High-Assurance)
**Repository**: https://github.com/SWORDIntel/DSLLVM

---

## 🎯 Overview

Welcome to **DSLLVM** - a specialized LLVM-based compiler infrastructure for military Command, Control & Communications (C3) and Joint All-Domain Command & Control (JADC2) systems. This repository extends the standard LLVM project with:

- **DSMIL Compiler** (`dsmil/`): War-fighting compiler with classification-aware security
- **TPM2 Compatibility Layer** (`tpm2_compat/`): 88 cryptographic algorithms with hardware acceleration
- **Mission-Aware Compilation**: Optimized for contested operational environments

---

## 🚀 Quick Links

- **[DSLLVM Build Guide](DSLLVM-BUILD-GUIDE.md)**: How to use DSLLVM as your default compiler
- **[DSMIL Documentation](dsmil/README.md)**: DSMIL compiler features and usage
- **[TPM2 Algorithms](tpm2_compat/README.md)**: 88 cryptographic algorithms reference

---

## ⭐ Key Components

### 1. DSMIL Compiler (`dsmil/`)

War-fighting compiler with operational capabilities:
- 9-layer/104-device architecture awareness
- CNSA 2.0 cryptographic provenance (SHA-384, ML-DSA-87, ML-KEM-1024)
- Cross-domain security (DoD classification levels)
- JADC2 & 5G/MEC integration
- Operational stealth modes
- Mission profiles (border_ops, cyber_defence, covert_ops)

**Quick Start**:
```bash
export CC=/home/user/DSLLVM/build/bin/dsmil-clang
dsmil-clang -O3 -fpass-pipeline=dsmil-default -o output input.c
```

### 2. TPM2 Compatibility Layer (`tpm2_compat/`)

Complete TPM 2.0 cryptographic algorithm support (88 algorithms):
- 10 Hash algorithms (SHA-256/384/512, SHA3, SM3, SHAKE)
- 22 Symmetric ciphers (AES modes, ChaCha20, Camellia, SM4)
- 17 Asymmetric algorithms (RSA, ECC/NIST P-curves, Ed25519/448)
- 11 Key derivation functions (HKDF, PBKDF2, scrypt, Argon2)
- 8 Post-quantum algorithms (Kyber, Dilithium, Falcon)
- Hardware acceleration (Intel NPU, AES-NI, SHA-NI, AVX-512)

**Quick Start**:
```bash
cd tpm2_compat
cmake -S . -B build -DENABLE_HARDWARE_ACCEL=ON
cmake --build build
```

---

## 📦 Building DSLLVM

### Prerequisites
```bash
sudo apt-get install -y build-essential cmake ninja-build python3 git libssl-dev
```

### Build LLVM/Clang + DSMIL
```bash
cmake -G Ninja -S llvm -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_PROJECTS="clang;lld" \
  -DLLVM_ENABLE_DSMIL=ON \
  -DLLVM_TARGETS_TO_BUILD="X86"

ninja -C build
```

### Build TPM2 Library
```bash
cd tpm2_compat
cmake -S . -B build -DENABLE_HARDWARE_ACCEL=ON
cmake --build build -j$(nproc)
```

---

## 📚 Documentation

### DSLLVM-Specific
- **[DSLLVM-BUILD-GUIDE.md](DSLLVM-BUILD-GUIDE.md)**: Default compiler configuration
- **[dsmil/docs/DSLLVM-DESIGN.md](dsmil/docs/DSLLVM-DESIGN.md)**: DSMIL design specification
- **[dsmil/docs/MISSION-PROFILES-GUIDE.md](dsmil/docs/MISSION-PROFILES-GUIDE.md)**: Mission profiles
- **[tpm2_compat/README.md](tpm2_compat/README.md)**: TPM2 algorithms reference

### Upstream LLVM
- [Getting Started with LLVM](https://llvm.org/docs/GettingStarted.html)
- [Contributing to LLVM](https://llvm.org/docs/Contributing.html)

---

Welcome to the LLVM project!

This repository contains the source code for LLVM, a toolkit for the
construction of highly optimized compilers, optimizers, and run-time
environments.

The LLVM project has multiple components. The core of the project is
itself called "LLVM". This contains all of the tools, libraries, and header
files needed to process intermediate representations and convert them into
object files. Tools include an assembler, disassembler, bitcode analyzer, and
bitcode optimizer.

C-like languages use the [Clang](https://clang.llvm.org/) frontend. This
component compiles C, C++, Objective-C, and Objective-C++ code into LLVM bitcode
-- and from there into object files, using LLVM.

Other components include:
the [libc++ C++ standard library](https://libcxx.llvm.org),
the [LLD linker](https://lld.llvm.org), and more.

## Getting the Source Code and Building LLVM

Consult the
[Getting Started with LLVM](https://llvm.org/docs/GettingStarted.html#getting-the-source-code-and-building-llvm)
page for information on building and running LLVM.

For information on how to contribute to the LLVM project, please take a look at
the [Contributing to LLVM](https://llvm.org/docs/Contributing.html) guide.

## Getting in touch

Join the [LLVM Discourse forums](https://discourse.llvm.org/), [Discord
chat](https://discord.gg/xS7Z362),
[LLVM Office Hours](https://llvm.org/docs/GettingInvolved.html#office-hours) or
[Regular sync-ups](https://llvm.org/docs/GettingInvolved.html#online-sync-ups).

The LLVM project has adopted a [code of conduct](https://llvm.org/docs/CodeOfConduct.html) for
participants to all modes of communication within the project.

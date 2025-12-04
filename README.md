# DSLLVM - Defense System LLVM Compiler

**Version**: 1.6.0 (Phase 3: High-Assurance)
**Repository**: https://github.com/SWORDIntel/DSLLVM

---
## 🚀 Quick Links

- **[DSLLVM Build Guide](DSLLVM-BUILD-GUIDE.md)**: How to use DSLLVM as your default compiler
- **[DSMIL Documentation](dsmil/README.md)**: DSMIL compiler features and usage
- **[TPM2 Algorithms](tpm2_compat/README.md)**: 88 cryptographic algorithms reference
### Upstream LLVM
- [Getting Started with LLVM](https://llvm.org/docs/GettingStarted.html)
- [Contributing to LLVM](https://llvm.org/docs/Contributing.html)

### Security Articles
- [Constant-Time Support Lands in LLVM: Protecting Cryptographic Code at the Compiler Level](https://securityboulevard.com/2025/11/constant-time-support-lands-in-llvm-protecting-cryptographic-code-at-the-compiler-level/)

### DSLLVM-Specific
**Quick Start**:
```bash
cd tpm2_compat
cmake -S . -B build -DENABLE_HARDWARE_ACCEL=ON
cmake --build build
```

## 📦 Building DSLLVM

### Quick Install (Recommended)

**Automated installer** - Builds and installs DSLLVM, replacing system LLVM:

```bash
# System-wide installation (requires sudo)
sudo ./build-dsllvm.sh

# Install to custom prefix (no sudo needed)
./build-dsllvm.sh --prefix /opt/dsllvm

# See all options
./build-dsllvm.sh --help
```

The installer automatically:
- ✅ Checks prerequisites
- ✅ Backs up existing LLVM installation
- ✅ Builds DSLLVM with all DSMIL features
- ✅ Creates system symlinks (clang → dsmil-clang, etc.)
- ✅ Sets up environment configuration
- ✅ Verifies installation

**Build Types:**
- `Release` (default): Optimized production build, faster execution, larger binaries
- `Debug`: Full debug symbols, assertions enabled, slower execution, easier debugging
- `RelWithDebInfo`: Optimized with debug info, good for profiling
- `MinSizeRel`: Optimized for smallest binary size

### Manual Build

#### Prerequisites
```bash
sudo apt-get install -y build-essential cmake ninja-build python3 git libssl-dev
```

#### Build LLVM/Clang + DSMIL
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
SLLVM is a **DSMIL-aware build of LLVM** with a small set of targeted extensions:

- keeps the **standard LLVM/Clang toolchain behaviour**;
- adds **optional hooks** for a multi-layer DSMIL system (devices, clearances, and telemetry);
- exposes **AI and quantum-related metadata** to higher layers without changing normal compiler workflows.

If you already know LLVM, you can treat DSLLVM as “LLVM with an opinionated integration layer” rather than a new compiler.

> **Note**  
> This repository is intentionally vague about downstream systems.  

---

## Highlights

- ✅ **LLVM-first design**  
  - Tracks upstream LLVM closely; core passes and IR semantics are unchanged.  
  - Can be used as a regular `clang`/`lld` toolchain for non-DSMIL builds.

- 🛰️ **DSMIL integration points (optional)**  
  - Lightweight annotations and metadata channels to describe:
    - logical device / layer routing,
    - clearance tags,
    - build-time provenance and audit hints.  
  - All of this is **opt-in** and encoded as normal IR / object metadata.

- 🧠 **AI & telemetry hooks**  
  - Build artefacts can carry compact feature metadata for:
    - performance/size profiles,
    - security posture markers,
    - deployment hints to external AI advisors.  
  - No runtime is mandated; DSLLVM just **emits signals** higher layers may consume.

- ⚛️ **Quantum-aware, not quantum-dependent**  
  - Optional metadata path for handing small optimisation / search problems
    to external **Qiskit-based workflows**.  
  - From the compiler’s point of view, this is just structured metadata attached to IR.

- 🔐 **PQC-aligned security profile**  
  - Compiler options and metadata profiles intended to coexist with
    **CNSA 2.0 style suites** (e.g. ML-KEM-1024, ML-DSA-87, SHA-384) without hard-coding any crypto.  
  - DSLLVM does **not** ship cryptography; it exposes knobs and tags so
    downstream toolchains can enforce their own policies.

---

## What DSLLVM Is (and Is Not)

**Is:**

- A **minimally invasive** extension layer on top of LLVM/Clang/LLD.
- A way to **tag and describe** builds for a DSMIL-style multi-layer system.
- A place to keep **AI / quantum / PQC-relevant metadata** close to the code that produced the binaries.

**Is *not*:**

- Not a new IR or language.  
- Not a replacement for upstream security guidance or crypto libraries.  
- Not a mandatory runtime or kernel – it’s “just” the compiler side.

---

## Quantum & AI Integration

DSLLVM does **not** execute quantum workloads itself. Instead, it:

- lets you attach **“quantum candidate”** hints to selected optimisation or search problems;
- keeps those hints in IR / object metadata so an external Qiskit pipeline can pick them up;
- allows AI advisors to see **compiler-level features** (size, structure, call-graphs, annotations) without changing the generated machine code.

These features are entirely optional; standard builds can ignore them.

---

## Building & Using DSLLVM

**Recommended**: Use the automated installer (`./build-dsllvm.sh`) for a complete build and system integration. See the [Build Guide](DSLLVM-BUILD-GUIDE.md) for details.

DSLLVM follows the **standard LLVM build flow**:

1. Configure with CMake (out-of-tree build directory).
2. Build with Ninja or Make.
3. Use `clang`/`clang++`/`lld` as usual.

If you don’t enable any DSMIL/AI options, DSLLVM behaves like a regular LLVM toolchain.

---

## Status

- Core compiler functionality: ✅ usable
- DSMIL / AI / quantum metadata hooks: 🧪 experimental, evolving
- Downstream integrations (DSMIL runtime, advisory layers): out of scope for this repo

For most users, DSLLVM can be dropped in as **“LLVM with extra metadata channels”** and left at that.
## 📚 Documentation


- **[DSLLVM-BUILD-GUIDE.md](DSLLVM-BUILD-GUIDE.md)**: Default compiler configuration
- **[dsmil/docs/DSLLVM-DESIGN.md](dsmil/docs/DSLLVM-DESIGN.md)**: DSMIL design specification
- **[dsmil/docs/MISSION-PROFILES-GUIDE.md](dsmil/docs/MISSION-PROFILES-GUIDE.md)**: Mission profiles
- **[tpm2_compat/README.md](tpm2_compat/README.md)**: TPM2 algorithms reference


[![Upstream](https://img.shields.io/badge/LLVM-upstream%20aligned-262D3A?logo=llvm&logoColor=white)](https://llvm.org/)
[![DSMIL Stack](https://img.shields.io/badge/DSMIL-multi--layer%20architecture-0B8457.svg)](#what-is-dsmil)
[![Quantum Ready](https://img.shields.io/badge/quantum-Qiskit%20%7C%20hybrid-6C2DC7.svg)](#quantum--ai-integration)
[![PQC Profile](https://img.shields.io/badge/CNSA%202.0-ML--KEM--1024%20%E2%80%A2%20ML--DSA--87%20%E2%80%A2%20SHA--384-E67E22.svg)](#pqc--security-posture)
[![AI-Integrated](https://img.shields.io/badge/AI-instrumented%20toolchain-1F7A8C.svg)](#ai--telemetry-hooks)

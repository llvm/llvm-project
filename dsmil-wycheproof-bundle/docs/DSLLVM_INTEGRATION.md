# DSLLVM Integration Guide

This project is expected to be built and tested (where practical) with a **custom LLVM/Clang**
toolchain named **DSLLVM**.

- DSLLVM repository: https://github.com/SWORDIntel/DSLLVM

## 1. Purpose

DSLLVM provides:

- Hardened, PQC-aware LLVM/Clang builds.
- Additional instrumentation hooks suitable for:
  - Side-channel timing analysis.
  - Cryptographic implementation hardening.
- CPU tuning for Intel Meteor Lake (Core Ultra 7 165H) with aggressive optimization flags.

Your job is to **support DSLLVM** while keeping fallback to standard LLVM/GCC.

## 2. Expected Build Configurations

You should support at least:

1. **DSLLVM-Optimized Build**
   - `CC=/path/to/dsclang`
   - `CXX=/path/to/dsclang++`
   - CFLAGS (example):
     - `-O3 -pipe -fomit-frame-pointer -funroll-loops -fstrict-aliasing`
     - `-fno-plt -fdata-sections -ffunction-sections -flto=auto`
     - `-march=meteorlake -mtune=meteorlake`
   - Additional DSLLVM-specific flags (see DSLLVM docs).

2. **Baseline Clang/GCC Build**
   - For compatibility and CI verification.
   - Normal `clang` / `gcc` with moderate optimization (e.g. `-O2`).

## 3. Where DSLLVM Matters for Wycheproof

- **Crypto library builds** (OpenSSL, libsodium, DSMIL crypto modules):
  - Ensure you can:
    - Build these libraries with DSLLVM.
    - Tag the build metadata (compiler, flags, DSLLVM version) in the
      `CryptoLibraryUnderTest` records and `crypto_test_result` outputs.

- **Wycheproof Harness / Adapters:**
  - May be compiled with DSLLVM as well.
  - Clearly distinguish DSLLVM vs non-DSLLVM builds in build metadata.

## 4. Metadata Requirements

When running campaigns, always include in the result metadata:

- `compiler.name`: e.g. `dsclang`, `clang`, `gcc`.
- `compiler.version`: as reported by `--version`.
- `compiler.flags`: actual flags used.
- `hw_features`: e.g. `["avx2", "avx512", "amx", "aes", "sha"]`.

This metadata is critical for the AI/analytics layer to understand
**which build paths lead to which behaviors**.

## 5. Implementation Hints for the AI

- Add CMake/Gradle/Bazel flags or environment-variable-based configuration to select the toolchain.
- Provide sample config files (in `examples/` if needed) for:
  - DSLLVM build.
  - Standard LLVM/GCC build.
- Ensure any DSLLVM-specific behavior is **documented** and does not break the default build.

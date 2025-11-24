# DSLLVM - DSMIL-Optimized LLVM Toolchain

**Version**: 1.0
**Status**: Initial Development
**Owner**: SWORDIntel / DSMIL Kernel Team

---

## Overview

DSLLVM is a hardened LLVM/Clang toolchain specialized for the DSMIL kernel and userland stack on Intel Meteor Lake hardware (CPU + NPU + Arc GPU). It extends LLVM with:

- **DSMIL-aware hardware targeting** optimized for Meteor Lake
- **Semantic metadata** for 9-layer/104-device architecture
- **Bandwidth & memory-aware optimization**
- **MLOps stage-awareness** for AI/LLM workloads
- **CNSA 2.0 provenance** (SHA-384, ML-DSA-87, ML-KEM-1024)
- **Quantum optimization hooks** (Device 46)
- **Complete tooling** and pass pipelines

---

## Quick Start

### Building DSLLVM

```bash
# Configure with CMake
cmake -G Ninja -S llvm -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_PROJECTS="clang;lld" \
  -DLLVM_ENABLE_DSMIL=ON \
  -DLLVM_TARGETS_TO_BUILD="X86"

# Build
ninja -C build

# Install
ninja -C build install
```

### Using DSLLVM

```bash
# Compile with DSMIL default pipeline
dsmil-clang -O3 -fpass-pipeline=dsmil-default -o output input.c

# Use DSMIL attributes in source
cat > example.c << 'EOF'
#include <dsmil_attributes.h>

DSMIL_LLM_WORKER_MAIN
int main(int argc, char **argv) {
    return llm_worker_loop();
}
EOF

dsmil-clang -O3 -fpass-pipeline=dsmil-default -o llm_worker example.c
```

### Verifying Provenance

```bash
# Verify binary provenance
dsmil-verify /usr/bin/llm_worker

# Get detailed report
dsmil-verify --verbose --json /usr/bin/llm_worker > report.json
```

---

## Repository Structure

```
dsmil/
├── docs/                      # Documentation
│   ├── DSLLVM-DESIGN.md       # Main design specification
│   ├── ATTRIBUTES.md          # Attribute reference
│   ├── PROVENANCE-CNSA2.md    # Provenance system details
│   └── PIPELINES.md           # Pass pipeline configurations
│
├── include/                   # Public headers
│   ├── dsmil_attributes.h     # Source-level attribute macros
│   ├── dsmil_provenance.h     # Provenance structures/API
│   └── dsmil_sandbox.h        # Sandbox runtime support
│
├── lib/                       # Implementation
│   ├── Passes/                # DSMIL LLVM passes
│   │   ├── DsmilBandwidthPass.cpp
│   │   ├── DsmilDevicePlacementPass.cpp
│   │   ├── DsmilLayerCheckPass.cpp
│   │   ├── DsmilStagePolicyPass.cpp
│   │   ├── DsmilQuantumExportPass.cpp
│   │   ├── DsmilSandboxWrapPass.cpp
│   │   └── DsmilProvenancePass.cpp
│   │
│   ├── Runtime/               # Runtime support libraries
│   │   ├── dsmil_sandbox_runtime.c
│   │   └── dsmil_provenance_runtime.c
│   │
│   └── Target/X86/            # X86 target extensions
│       └── DSMILTarget.cpp    # Meteor Lake + DSMIL target
│
├── tools/                     # Toolchain wrappers & utilities
│   ├── dsmil-clang/           # Clang wrapper with DSMIL defaults
│   ├── dsmil-llc/             # LLC wrapper
│   ├── dsmil-opt/             # Opt wrapper with DSMIL passes
│   └── dsmil-verify/          # Provenance verification tool
│
├── test/                      # Test suite
│   └── dsmil/
│       ├── layer_policies/    # Layer enforcement tests
│       ├── stage_policies/    # Stage policy tests
│       ├── provenance/        # Provenance system tests
│       └── sandbox/           # Sandbox tests
│
├── cmake/                     # CMake integration
│   └── DSMILConfig.cmake      # DSMIL configuration
│
└── README.md                  # This file
```

---

## Key Features

### 1. DSMIL Target Integration

Custom target triple `x86_64-dsmil-meteorlake-elf` with Meteor Lake optimizations:

```bash
# AVX2, AVX-VNNI, AES, VAES, SHA, GFNI, BMI1/2, POPCNT, FMA, etc.
dsmil-clang -target x86_64-dsmil-meteorlake-elf ...
```

### 2. Source-Level Attributes

Annotate code with DSMIL metadata:

```c
#include <dsmil_attributes.h>

DSMIL_LAYER(7)
DSMIL_DEVICE(47)
DSMIL_STAGE("serve")
void llm_inference(void) {
    // Layer 7 (AI/ML) on Device 47 (NPU)
}
```

### 3. Compile-Time Verification

Layer boundary and policy enforcement:

```c
// ERROR: Upward layer transition without gateway
DSMIL_LAYER(7)
void user_function(void) {
    kernel_operation();  // Layer 1 function
}

// OK: With gateway
DSMIL_GATEWAY
DSMIL_LAYER(5)
int validated_entry(void *data) {
    return kernel_operation(data);
}
```

### 4. CNSA 2.0 Provenance

Every binary includes cryptographically-signed provenance:

```bash
$ dsmil-verify /usr/bin/llm_worker
✓ Provenance present
✓ Signature valid (PSK-2025-SWORDIntel-DSMIL)
✓ Certificate chain valid
✓ Binary hash matches
✓ DSMIL metadata:
    Layer: 7
    Device: 47
    Sandbox: l7_llm_worker
    Stage: serve
```

### 5. Automatic Sandboxing

Zero-code sandboxing via attributes:

```c
DSMIL_SANDBOX("l7_llm_worker")
int main(int argc, char **argv) {
    // Automatically sandboxed with:
    // - Minimal capabilities (libcap-ng)
    // - Seccomp filter
    // - Resource limits
    return run_inference_loop();
}
```

### 6. Bandwidth-Aware Optimization

Automatic memory tier recommendations:

```c
DSMIL_KV_CACHE
struct kv_cache_pool global_kv_cache;
// Recommended: ramdisk/tmpfs for high bandwidth

DSMIL_HOT_MODEL
const float weights[4096][4096];
// Recommended: large pages, NUMA pinning
```

---

## Pass Pipelines

### Production (`dsmil-default`)

Full optimization with strict enforcement:

```bash
dsmil-clang -O3 -fpass-pipeline=dsmil-default -o output input.c
```

- All DSMIL analysis and verification passes
- Layer/stage policy enforcement
- Provenance generation and signing
- Sandbox wrapping

### Development (`dsmil-debug`)

Fast iteration with warnings:

```bash
dsmil-clang -O2 -g -fpass-pipeline=dsmil-debug -o output input.c
```

- Relaxed enforcement (warnings only)
- Debug information preserved
- Faster compilation (no LTO)

### Lab/Research (`dsmil-lab`)

No enforcement, metadata only:

```bash
dsmil-clang -O1 -fpass-pipeline=dsmil-lab -o output input.c
```

- Metadata annotation only
- No policy checks
- Useful for experimentation

---

## Environment Variables

### Build-Time

- `DSMIL_PSK_PATH`: Path to Project Signing Key (required for provenance)
- `DSMIL_RDK_PUB_PATH`: Path to RDK public key (optional, for encrypted provenance)
- `DSMIL_BUILD_ID`: Unique build identifier
- `DSMIL_BUILDER_ID`: Builder hostname/ID
- `DSMIL_TSA_URL`: Timestamp authority URL (optional)

### Runtime

- `DSMIL_SANDBOX_MODE`: Override sandbox mode (`enforce`, `warn`, `disabled`)
- `DSMIL_POLICY`: Policy configuration (`production`, `development`, `lab`)
- `DSMIL_TRUSTSTORE`: Path to trust store directory (default: `/etc/dsmil/truststore/`)

---

## Documentation

- **[DSLLVM-DESIGN.md](docs/DSLLVM-DESIGN.md)**: Complete design specification
- **[ATTRIBUTES.md](docs/ATTRIBUTES.md)**: Attribute reference guide
- **[PROVENANCE-CNSA2.md](docs/PROVENANCE-CNSA2.md)**: Provenance system deep dive
- **[PIPELINES.md](docs/PIPELINES.md)**: Pass pipeline configurations

---

## Development Status

### ✅ Completed

- Design specification
- Documentation structure
- Header file definitions
- Directory layout

### 🚧 In Progress

- LLVM pass implementations
- Runtime library (sandbox, provenance)
- Tool wrappers (dsmil-clang, dsmil-verify)
- Test suite

### 📋 Planned

- CMake integration
- CI/CD pipeline
- Sample applications
- Performance benchmarks
- Security audit

---

## Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

### Key Areas for Contribution

1. **Pass Implementation**: Implement DSMIL analysis and transformation passes
2. **Target Integration**: Add Meteor Lake-specific optimizations
3. **Crypto Integration**: Integrate CNSA 2.0 libraries (ML-DSA, ML-KEM)
4. **Testing**: Expand test coverage
5. **Documentation**: Examples, tutorials, case studies

---

## License

DSLLVM is part of the LLVM Project and is licensed under the Apache License v2.0 with LLVM Exceptions. See [LICENSE.TXT](../LICENSE.TXT) for details.

---

## Contact

- **Project**: SWORDIntel/DSLLVM
- **Team**: DSMIL Kernel Team
- **Issues**: [GitHub Issues](https://github.com/SWORDIntel/DSLLVM/issues)

---

**DSLLVM**: Secure, Observable, Hardware-Optimized Compilation for DSMIL

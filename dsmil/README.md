# DSLLVM - War-Fighting Compiler for C3/JADC2 Systems

**Version**: 1.6.0 (Phase 3: High-Assurance)
**Status**: Active Development (v1.6 - High-Assurance Phase)
**Owner**: SWORDIntel / DSMIL Kernel Team

---

## Overview

DSLLVM is a **war-fighting compiler** specialized for military Command, Control & Communications (C3) and Joint All-Domain Command & Control (JADC2) systems. Built on LLVM/Clang, it extends the toolchain with classification-aware cross-domain security, 5G/MEC optimization, and operational features for contested environments.

### Core Capabilities

**Foundation (v1.0-v1.3)**
- **DSMIL-aware hardware targeting** optimized for Intel Meteor Lake (CPU + NPU + Arc GPU)
- **Semantic metadata** for 9-layer/104-device architecture
- **Bandwidth & memory-aware optimization**
- **MLOps stage-awareness** for AI/LLM workloads
- **CNSA 2.0 provenance** (SHA-384, ML-DSA-87, ML-KEM-1024)
- **Quantum optimization hooks** (Device 46)
- **Mission-aware compilation** with configurable profiles
- **AI-assisted compilation** (Layer 5/7/8 integration)

**Security Depth (v1.4)** ✅ COMPLETE
- **Operational Stealth Modes** (Feature 2.1): Telemetry suppression, constant-rate execution, network fingerprint reduction
- **Threat Signature Embedding** (Feature 2.2): CFG fingerprinting, supply chain verification, forensics-ready binaries
- **Blue vs Red Simulation** (Feature 2.3): Dual-build adversarial testing, scenario-based vulnerability injection

**Operational Deployment (v1.5)** - Phase 1 ✅ COMPLETE, Phase 2 ✅ COMPLETE
- **Cross-Domain Guards & Classification** (Feature 3.1): DoD classification levels (U/C/S/TS/TS-SCI), cross-domain security policies ✅
- **JADC2 & 5G/Edge Integration** (Feature 3.2): 5G/MEC optimization, latency budgets (5ms), bandwidth contracts (10Gbps) ✅
- **Blue Force Tracker** (Feature 3.3): Real-time friendly force tracking (BFT-2), AES-256 encrypted position updates, spoofing detection ✅
- **Radio Multi-Protocol Bridging** (Feature 3.7): Link-16, SATCOM, MUOS, SINCGARS tactical radio bridging ✅
- **5G Latency & Throughput Contracts** (Feature 3.9): Compile-time enforcement of 5G JADC2 requirements ✅

**High-Assurance (v1.6)** - Phase 3 ✅ COMPLETE
- **Two-Person Integrity** (Feature 3.4): Nuclear surety controls (NC3), ML-DSA-87 dual-signature authorization, DOE Sigma 14 ✅
- **Mission Partner Environment** (Feature 3.5): Coalition interoperability, releasability markings (REL NATO, REL FVEY, NOFORN) ✅
- **Edge Security Hardening** (Feature 3.8): HSM crypto, secure enclave (SGX/TrustZone), remote attestation, anti-tampering ✅
- **EM Spectrum Resilience** (Feature 3.6): BLOS fallback (5G→SATCOM), EMCON modes, jamming detection 🔜

### Military Network Support

- **NIPRNet**: UNCLASSIFIED operations, coalition sharing
- **SIPRNet**: SECRET operations (U/C/S), cross-domain guards
- **JWICS**: TOP SECRET/SCI operations, NOFORN enforcement
- **5G/MEC**: Edge computing for JADC2 (99.999% reliability, 5ms latency)
- **Tactical Radios**: Link-16, SATCOM, MUOS, SINCGARS multi-protocol bridging

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

### 1. Operational Stealth Mode (v1.4 - Feature 2.1) ⭐ NEW

Compiler-level transformations for low-signature execution in hostile environments:

```c
#include <dsmil_attributes.h>

// Aggressive stealth for covert operations
DSMIL_LOW_SIGNATURE("aggressive")
DSMIL_CONSTANT_RATE
DSMIL_LAYER(7)
void covert_data_collection(const uint8_t *data, size_t len) {
    // Compiler applies:
    // - Strip non-critical telemetry
    // - Constant-rate execution (prevents timing analysis)
    // - Jitter suppression (predictable timing)
    // - Network fingerprint reduction
    process_sensitive_data(data, len);
}
```

**Stealth Levels**:
- `minimal`: Basic telemetry reduction
- `standard`: Timing normalization + reduced telemetry
- `aggressive`: Maximum stealth (constant-rate, minimal signatures)

**Mission Profiles with Stealth**:
```bash
# Covert operations (aggressive stealth)
dsmil-clang -fdsmil-mission-profile=covert_ops -O3 -o covert.bin input.c

# Border operations with stealth
dsmil-clang -fdsmil-mission-profile=border_ops_stealth -O3 -o border.bin input.c
```

**Documentation**: [STEALTH-MODE.md](docs/STEALTH-MODE.md)

### 2. DSMIL Target Integration

Custom target triple `x86_64-dsmil-meteorlake-elf` with Meteor Lake optimizations:

```bash
# AVX2, AVX-VNNI, AES, VAES, SHA, GFNI, BMI1/2, POPCNT, FMA, etc.
dsmil-clang -target x86_64-dsmil-meteorlake-elf ...
```

### 3. Source-Level Attributes

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

### 4. Compile-Time Verification

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

### 5. CNSA 2.0 Provenance

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

### 6. Automatic Sandboxing

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

### 7. Bandwidth-Aware Optimization

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

### Runtime Path Configuration ⭐ NEW

DSLLVM supports dynamic path resolution for portable installations:

- `DSMIL_PREFIX`: Base installation prefix (default: `/opt/dsmil`)
- `DSMIL_CONFIG_DIR`: Configuration directory (default: `${DSMIL_PREFIX}/etc` or `/etc/dsmil`)
- `DSMIL_BIN_DIR`: Binary directory (default: `${DSMIL_PREFIX}/bin`)
- `DSMIL_LIB_DIR`: Library directory (default: `${DSMIL_PREFIX}/lib`)
- `DSMIL_TRUSTSTORE_DIR`: Trust store directory (default: `${DSMIL_CONFIG_DIR}/truststore`)
- `DSMIL_LOG_DIR`: Log directory (default: `${DSMIL_PREFIX}/var/log` or `/var/log/dsmil`)
- `DSMIL_RUNTIME_DIR`: Runtime directory (default: `${XDG_RUNTIME_DIR}/dsmil` or `/var/run/dsmil`)
- `DSMIL_CACHE_DIR`: Cache directory (default: `${XDG_CACHE_HOME}/dsmil` or `$HOME/.cache/dsmil`)
- `DSMIL_TMP_DIR`: Temporary directory (default: `${TMPDIR}` or `/tmp`)

See **[PATH-CONFIGURATION.md](docs/PATH-CONFIGURATION.md)** for complete path configuration guide.

### Runtime Behavior

- `DSMIL_SANDBOX_MODE`: Override sandbox mode (`enforce`, `warn`, `disabled`)
- `DSMIL_POLICY`: Policy configuration (`production`, `development`, `lab`)
- `DSMIL_MISSION_PROFILE_CONFIG`: Path to mission profile config (uses dynamic resolution)

---

## Documentation

### Core Documentation
- **[DSLLVM-DESIGN.md](docs/DSLLVM-DESIGN.md)**: Complete design specification
- **[DSLLVM-ROADMAP.md](docs/DSLLVM-ROADMAP.md)**: Strategic roadmap (v1.0 → v2.0)
- **[ATTRIBUTES.md](docs/ATTRIBUTES.md)**: Attribute reference guide
- **[PROVENANCE-CNSA2.md](docs/PROVENANCE-CNSA2.md)**: Provenance system deep dive
- **[PIPELINES.md](docs/PIPELINES.md)**: Pass pipeline configurations
- **[PATH-CONFIGURATION.md](docs/PATH-CONFIGURATION.md)**: Dynamic path configuration guide ⭐ NEW

### Feature Guides (v1.3+)
- **[MISSION-PROFILES-GUIDE.md](docs/MISSION-PROFILES-GUIDE.md)**: Mission profile system (Feature 1.1)
- **[FUZZ-HARNESS-SCHEMA.md](docs/FUZZ-HARNESS-SCHEMA.md)**: Auto-fuzz harness generation (Feature 1.2)
- **[TELEMETRY-ENFORCEMENT.md](docs/TELEMETRY-ENFORCEMENT.md)**: Minimum telemetry enforcement (Feature 1.3)
- **[STEALTH-MODE.md](docs/STEALTH-MODE.md)**: Operational stealth modes (Feature 2.1) ⭐ NEW

### Integration Guides
- **[AI-INTEGRATION.md](docs/AI-INTEGRATION.md)**: Layer 5/7/8 AI integration
- **[FUZZ-CICD-INTEGRATION.md](docs/FUZZ-CICD-INTEGRATION.md)**: CI/CD fuzzing integration

---

## Development Status

### ✅ Completed (v1.0-v1.2)

- ✅ Design specification
- ✅ Documentation structure
- ✅ Header file definitions (dsmil_attributes.h, dsmil_telemetry.h, dsmil_provenance.h)
- ✅ Directory layout
- ✅ CNSA 2.0 provenance framework
- ✅ AI integration (Layer 5/7/8)
- ✅ Constant-time enforcement (DSMIL_SECRET)
- ✅ ONNX cost models

### ✅ Completed (v1.3 - Operational Control)

- ✅ **Feature 1.1**: Mission Profiles (border_ops, cyber_defence, exercise_only)
- ✅ **Feature 1.2**: Auto-generated fuzz harnesses (dsmil-fuzz-export)
- ✅ **Feature 1.3**: Minimum telemetry enforcement (safety/mission critical)

### ✅ Completed (v1.4 - Security Depth)

- ✅ **Feature 2.1**: Operational Stealth Modes
  - ✅ Stealth attributes (DSMIL_LOW_SIGNATURE, DSMIL_CONSTANT_RATE, etc.)
  - ✅ DsmilStealthPass implementation
  - ✅ Stealth runtime support (timing, network batching)
  - ✅ Mission profile integration (covert_ops, border_ops_stealth)
  - ✅ Examples and test cases
  - ✅ Comprehensive documentation
- ✅ **Feature 2.2**: Threat Signature Embedding for Forensics
  - ✅ Threat signature structures (CFG hash, crypto patterns, protocol schemas)
  - ✅ DsmilThreatSignaturePass implementation
  - ✅ JSON signature generation for Layer 62 forensics/SIEM
  - ✅ Non-identifying fingerprints for imposter detection
- ✅ **Feature 2.3**: Blue vs Red Scenario Simulation
  - ✅ Blue/red attributes (DSMIL_RED_TEAM_HOOK, DSMIL_ATTACK_SURFACE, etc.)
  - ✅ DsmilBlueRedPass implementation
  - ✅ Red build runtime support (logging, scenario control)
  - ✅ Dual-build mission profiles (blue_production, red_stress_test)
  - ✅ Example code and integration guide

### 🎯 v1.4 Security Depth Phase Complete!

All three features from Phase 2 (v1.4) are now implemented:
- Feature 2.1: Operational Stealth Modes ✅
- Feature 2.2: Threat Signature Embedding ✅
- Feature 2.3: Blue vs Red Scenario Simulation ✅

### 🚧 In Progress
- 🚧 LLVM pass implementations (remaining passes)
- 🚧 Runtime library completion (sandbox, provenance)
- 🚧 Tool wrappers (dsmil-clang, dsmil-verify)

### 📋 Planned (v1.5 - System Intelligence)

- 📋 **Feature 3.1**: Schema compiler for exotic devices (104 devices)
- 📋 **Feature 3.2**: Cross-binary invariant checking
- 📋 **Feature 3.3**: Temporal profiles (bootstrap → stabilize → production)
- 📋 CMake integration
- 📋 CI/CD pipeline
- 📋 Performance benchmarks

### 🔬 Research (v2.0 - Adaptive Optimization)

- 🔬 **Feature 4.1**: Compiler-level RL loop on real hardware
- 🔬 Hardware-specific learned profiles
- 🔬 Continuous improvement via RL

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

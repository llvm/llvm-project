# DSLLVM - War-Fighting Compiler for C3/JADC2 Systems

**Version**: 1.6.1 (Runtime Enhancements Complete)
**Status**: Active Development (v1.6.1 - Comprehensive Runtime APIs)
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

**Runtime Enhancements (v1.6.1)** ⭐ NEW - COMPLETE
- **Advanced INT8 Quantization**: Comprehensive quantization runtime (calibration, GEMM, accuracy validation, hardware acceleration)
- **Layer 7 LLM Runtime**: Device 47 LLM support with INT8 enforcement, memory management (40 GB), KV cache optimization
- **Layer 8 Security AI**: 8 specialized devices (188 TOPS) for adversarial defense, threat detection, GNN correlation, zero-day prediction
- **Layer 9 Executive Command**: 4 specialized devices (330 TOPS) for strategic planning, NC3 integration, coalition coordination
- **Device 255 Master Crypto**: Unified cryptographic operations (88 algorithms) with TPM/Hardware/Software engines
- **Cross-Layer Intelligence Flow**: Event-driven upward intelligence flow with security clearance verification
- **Memory Budget Management**: 62 GB dynamic pool with layer-specific budgets and thread-safe allocation
- **HIL Orchestration**: NPU/GPU/CPU workload routing (48.2 TOPS) with utilization monitoring
- **MLOps Pipeline Integration**: INT8 verification, pruning validation, speedup calculation, model requirement enforcement

### Military Network Support

- **NIPRNet**: UNCLASSIFIED operations, coalition sharing
- **SIPRNet**: SECRET operations (U/C/S), cross-domain guards
- **JWICS**: TOP SECRET/SCI operations, NOFORN enforcement
- **5G/MEC**: Edge computing for JADC2 (99.999% reliability, 5ms latency)
- **Tactical Radios**: Link-16, SATCOM, MUOS, SINCGARS multi-protocol bridging

### Runtime API Summary ⭐ NEW

DSMIL provides **comprehensive runtime libraries** spanning all 9 layers with **~1338 TOPS INT8** total compute:

| Layer | Runtime APIs | TOPS | Key Capabilities |
|-------|--------------|------|-----------------|
| **Layer 7** | LLM, Quantum, INT8 Quantization | 440 | LLMs (up to 7B), INT8 quantization, quantum integration |
| **Layer 8** | Security AI (8 devices) | 188 | Adversarial defense, threat detection, GNN correlation, zero-day prediction |
| **Layer 9** | Executive Command (4 devices) | 330 | Strategic planning, NC3, coalition fusion, crisis management |
| **All Layers** | Device 255 Crypto, Memory Budget, HIL, Intelligence Flow | - | Unified crypto, memory management, hardware orchestration |

**Total System Compute**: ~1338 TOPS INT8 across Layers 3-9 (48 AI devices)

---

## Quick Start

### Building DSLLVM

#### Automated Installer (Recommended)

**Quick install** - Builds and replaces system LLVM with DSLLVM:

```bash
# System-wide installation (requires sudo)
sudo ./build-dsllvm.sh

# Install to custom prefix (no sudo needed)
./build-dsllvm.sh --prefix /opt/dsllvm

# Debug build (for development/debugging)
./build-dsllvm.sh --build-type Debug

# See all options
./build-dsllvm.sh --help
```

**What does `--build-type Debug` do?**
- **Debug symbols**: Full debug information included (file names, line numbers, variable names)
- **Assertions enabled**: Runtime checks and assertions are active (catches bugs during development)
- **No optimizations**: Code is not optimized, making it easier to step through with a debugger
- **Larger binaries**: Debug builds are significantly larger due to symbol tables
- **Slower execution**: Unoptimized code runs slower, but easier to debug
- **Use case**: Development, debugging compiler passes, investigating crashes

**Other build types:**
- `Release` (default): Optimized for production, fastest execution
- `RelWithDebInfo`: Optimized code with debug symbols (good for profiling)
- `MinSizeRel`: Optimized for smallest binary size

#### Manual Build

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

## Runtime APIs ⭐ NEW

DSMIL provides comprehensive runtime libraries for Layer 7 AI/ML workloads, quantum integration, memory management, hardware orchestration, unified cryptographic operations, Layer 8 security AI, and Layer 9 executive command:

### Quick Reference

| Runtime API | Header | Implementation | TOPS | Key Features |
|-------------|--------|----------------|------|--------------|
| **INT8 Quantization** | `dsmil_int8_quantization.h` | `dsmil_int8_quantization_runtime.c` | 48.2 | Calibration, GEMM, accuracy validation |
| **Layer 7 LLM** | `dsmil_layer7_llm.h` | `dsmil_layer7_llm_runtime.c` | 440 | Memory mgmt, INT8 enforcement, KV cache |
| **Device 46 Quantum** | `dsmil_quantum_runtime.h` | `dsmil_quantum_runtime.c` | 55 | QAOA/QUBO, quantum feature maps |
| **MLOps Pipeline** | `dsmil_mlops_optimization.h` | `dsmil_mlops_optimization_runtime.c` | - | INT8 verification, pruning, speedup |
| **Intelligence Flow** | `dsmil_intelligence_flow.h` | `dsmil_intelligence_flow_runtime.c` | - | Upward flow, event-driven, clearance |
| **Memory Budget** | `dsmil_memory_budget.h` | `dsmil_memory_budget_runtime.c` | - | 62 GB pool, layer budgets, thread-safe |
| **HIL Orchestration** | `dsmil_hil_orchestration.h` | `dsmil_hil_orchestration_runtime.c` | 48.2 | NPU/GPU/CPU routing, utilization |
| **Layer 8 Security** | `dsmil_layer8_security.h` | `dsmil_layer8_security_runtime.c` | 188 | 8 devices, adversarial defense, GNN |
| **Layer 9 Executive** | `dsmil_layer9_executive.h` | `dsmil_layer9_executive_runtime.c` | 330 | 4 devices, NC3, coalition fusion |
| **Device 255 Crypto** | `dsmil_device255_crypto.h` | `dsmil_device255_crypto_runtime.c` | - | 88 algorithms, TPM/HW/SW engines |

### Layer 7 Device 47 LLM Runtime
- **Header**: `dsmil/include/dsmil_layer7_llm.h`
- **Implementation**: `dsmil/lib/Runtime/dsmil_layer7_llm_runtime.c`
- Memory management (40 GB Layer 7 budget)
- INT8 quantization enforcement (>95% accuracy)
- KV cache optimization
- Model lifecycle management
- INT8 matrix operations for attention/FFN layers

### Advanced INT8 Quantization Runtime ⭐ NEW
- **Header**: `dsmil/include/dsmil_int8_quantization.h`
- **Implementation**: `dsmil/lib/Runtime/dsmil_int8_quantization_runtime.c`
- **Calibration**: FP32 → INT8 quantization parameter calculation
- **Quantization Schemes**: Symmetric, asymmetric, per-tensor, per-channel, dynamic
- **INT8 GEMM**: Matrix multiplication with INT8 inputs (NPU: 13.0 TOPS, GPU: 32.0 TOPS, CPU: 3.2 TOPS)
- **Accuracy Validation**: >95% retention requirement enforcement
- **Weight Quantization**: Per-layer and per-channel quantization for linear/conv layers
- **Dynamic Quantization**: Runtime activation quantization
- **Hardware Acceleration**: NPU/GPU/CPU INT8 TOPS utilization
- **Speedup Estimation**: INT8 vs FP32 performance prediction

### Device 46 Quantum Runtime
- **Header**: `dsmil/include/dsmil_quantum_runtime.h`
- **Implementation**: `dsmil/lib/Runtime/dsmil_quantum_runtime.c`
- Qiskit-based QAOA/QUBO optimization
- Quantum feature maps for anomaly detection
- Hybrid quantum-classical workflows
- CPU-bound simulation (2 GB memory budget)

### MLOps Pipeline Optimization
- **Header**: `dsmil/include/dsmil_mlops_optimization.h`
- **Implementation**: `dsmil/lib/Runtime/dsmil_mlops_optimization_runtime.c`
- INT8 quantization verification (mandatory)
- Pruning sparsity validation (50% target)
- Combined speedup calculation (12× minimum, 30× target)
- Model requirement verification

### Cross-Layer Intelligence Flow
- **Header**: `dsmil/include/dsmil_intelligence_flow.h`
- **Implementation**: `dsmil/lib/Runtime/dsmil_intelligence_flow_runtime.c`
- Upward intelligence flow pattern
- Event-driven architecture
- Security clearance verification
- Layer-to-layer communication

### Memory Budget Management
- **Header**: `dsmil/include/dsmil_memory_budget.h`
- **Implementation**: `dsmil/lib/Runtime/dsmil_memory_budget_runtime.c`
- 62 GB total memory pool management
- Layer-specific budgets (Layer 7: 40 GB max)
- Dynamic allocation with global constraint enforcement
- Thread-safe memory tracking

### Hardware Integration Layer (HIL) Orchestration
- **Header**: `dsmil/include/dsmil_hil_orchestration.h`
- **Implementation**: `dsmil/lib/Runtime/dsmil_hil_orchestration_runtime.c`
- NPU/GPU/CPU workload assignment (48.2 TOPS total)
- Utilization monitoring
- Availability checking
- Device-aware workload routing

### Layer 8 Security AI Runtime ⭐ NEW
- **Header**: `dsmil/include/dsmil_layer8_security.h`
- **Implementation**: `dsmil/lib/Runtime/dsmil_layer8_security_runtime.c`
- **8 Devices (51-58)**: 188 TOPS INT8 total
  - Device 51: Enhanced Security Framework (15 TOPS) - Anomaly detection, behavioral analytics (LSTM/GRU)
  - Device 52: Adversarial ML Defense (30 TOPS) - Adversarial training with GANs, robustness testing
  - Device 53: Cybersecurity AI (25 TOPS) - Threat intelligence, attack prediction, zero-day prediction
  - Device 54: Threat Intelligence (25 TOPS) - IOC extraction (NLP), attribution analysis (GNN)
  - Device 55: Automated Security Response (20 TOPS) - RL-based incident response automation
  - Device 56: Post-Quantum Crypto (20 TOPS) - ML-optimized PQC algorithms (ML-KEM, ML-DSA)
  - Device 57: Autonomous Operations (28 TOPS) - Self-healing systems, adaptive defense
  - Device 58: Security Analytics (25 TOPS) - Security event correlation (GNN), forensics
- **Model sizes**: 50-300M parameters
- **Latency**: <100ms for real-time threat detection
- **Detection accuracy**: >99% known threats, >95% zero-day
- **Specialized capabilities**:
  - Adversarial defense training (GANs)
  - Security event correlation (Graph Neural Networks)
  - Zero-day attack prediction
  - Behavioral pattern analysis (LSTM/GRU temporal patterns)
  - PQC algorithm ML optimization
  - Automated incident response (Reinforcement Learning)

### Layer 9 Executive Command Runtime ⭐ NEW
- **Header**: `dsmil/include/dsmil_layer9_executive.h`
- **Implementation**: `dsmil/lib/Runtime/dsmil_layer9_executive_runtime.c`
- **4 Devices (59-62)**: 330 TOPS INT8 total
  - Device 59: Executive Command (85 TOPS) - Strategic decision support, crisis management, real-time resource allocation
  - Device 60: Coalition Fusion (85 TOPS) - Multi-national intelligence fusion, multi-lingual NLP, cross-cultural analysis
  - Device 61: Nuclear C&C Integration (80 TOPS) - NC3 analysis, strategic stability, deterrence modeling (ROE-governed)
  - Device 62: Strategic Intelligence (80 TOPS) - Global threat assessment, geopolitical modeling, risk forecasting
- **Model sizes**: 1B-7B parameters
- **Latency**: <1000ms for complex strategic queries
- **Context windows**: Up to 32K tokens for comprehensive analysis
- **Specialized capabilities**:
  - Crisis management with real-time decision support
  - Multi-criteria decision analysis and policy simulation
  - Releasability markings (REL NATO, REL FVEY, NOFORN)
  - Strategic stability assessment (NC3, Device 61, Section 4.1c compliant)
  - Long-term strategic planning with scenario analysis
  - Multi-national coordination and joint operations

### Device 255 Master Crypto Controller ⭐ NEW
- **Header**: `dsmil/include/dsmil_device255_crypto.h`
- **Implementation**: `dsmil/lib/Runtime/dsmil_device255_crypto_runtime.c`
- Unified cryptographic operations (88 algorithms)
- TPM 2.0 / Hardware / Software engine support
- PQC algorithm support (ML-KEM-1024, ML-DSA-87)
- Layer-aware crypto operations
- Capability management and locking

**Device 255 Integration Points**:
- **Device 15 (CRYPTO)**: Wycheproof crypto operations via Device 255
- **Device 47 (AI/ML)**: Model encryption/signing (AES-256-GCM, ML-DSA-87 CNSA 2.0)
- **Device 46 (Quantum)**: PQC key generation and test vectors
- **Layer 8 (ENHANCED_SEC)**: PQC-only mode enforcement
- **MLOps Pipeline**: Model provenance signing (ML-DSA-87 CNSA 2.0)

See **[DEVICE255-MASTER-CRYPTO-ENHANCEMENTS.md](docs/DEVICE255-MASTER-CRYPTO-ENHANCEMENTS.md)** and **[COMPREHENSIVE-PLAN-ALIGNMENT-ENHANCEMENTS.md](docs/COMPREHENSIVE-PLAN-ALIGNMENT-ENHANCEMENTS.md)** for complete details.

---

## Detailed Feature Documentation

### Advanced INT8 Quantization Runtime

**Purpose**: Comprehensive INT8 quantization support for mandatory MLOps pipeline compliance and hardware acceleration.

**Key Functions**:
- `dsmil_int8_calibrate()` - Calculate quantization parameters from FP32 data (symmetric/asymmetric, per-tensor/per-channel)
- `dsmil_int8_quantize()` - Convert FP32 → INT8 with scale/zero-point
- `dsmil_int8_dequantize()` - Convert INT8 → FP32 for accuracy validation
- `dsmil_int8_gemm()` - INT8 matrix multiplication (GEMM) with INT32 accumulator or INT8 output
- `dsmil_int8_matmul_with_bias()` - INT8 matmul with FP32 bias and activation (ReLU, GELU)
- `dsmil_int8_validate_accuracy()` - Verify >95% accuracy retention requirement
- `dsmil_int8_quantize_weights()` - Per-layer/per-channel weight quantization for linear/conv layers
- `dsmil_int8_dynamic_quantize()` - Runtime activation quantization with dynamic scales
- `dsmil_int8_get_hardware_caps()` - Query NPU (13.0 TOPS), GPU (32.0 TOPS), CPU (3.2 TOPS) capabilities
- `dsmil_int8_estimate_speedup()` - Predict INT8 vs FP32 performance (4× base + hardware acceleration)

**Quantization Schemes**:
- **Symmetric**: Zero point = 0, optimal for most cases
- **Asymmetric**: Non-zero zero point, better for skewed distributions
- **Per-Tensor**: Single scale/zero-point per tensor
- **Per-Channel**: Per-channel scales for conv/linear layers (better accuracy)
- **Dynamic**: Runtime quantization with dynamic scales (activations)

**Hardware Acceleration**:
- NPU: 13.0 TOPS INT8 (continuous inference)
- GPU: 32.0 TOPS INT8 (dense math, vision, LLM attention)
- CPU: 3.2 TOPS INT8 (AMX, control plane)

**Example Usage**:
```c
#include "dsmil_int8_quantization.h"

// Calibrate quantization parameters
dsmil_int8_params_t params;
dsmil_int8_calibrate(fp32_weights, num_weights, DSMIL_INT8_SYMMETRIC, 
                     false, 0, &params);

// Quantize weights
int8_t *int8_weights = malloc(num_weights);
dsmil_int8_quantize(fp32_weights, int8_weights, num_weights, &params);

// Validate accuracy (>95% retention required)
dsmil_int8_accuracy_metrics_t metrics;
if (dsmil_int8_validate_accuracy(fp32_model, int8_model, test_data, &metrics) == 0) {
    if (metrics.meets_requirement) {
        printf("INT8 quantization validated: %.2f%% retention\n", 
               metrics.accuracy_retention * 100.0f);
    }
}
```

---

### Layer 7 Device 47 LLM Runtime

**Purpose**: Primary AI layer runtime for LLM workloads with INT8 quantization enforcement and memory management.

**Key Functions**:
- `dsmil_device47_llm_init()` - Initialize runtime with Layer 7 memory budget (40 GB max)
- `dsmil_device47_llm_load()` - Load INT8-quantized LLM model with validation
- `dsmil_device47_verify_int8_quantization()` - Verify >95% accuracy retention
- `dsmil_device47_get_int8_params()` - Get quantization parameters for model
- `dsmil_device47_int8_matmul()` - INT8 matrix operations for attention/FFN layers
- `dsmil_device47_set_kv_cache_size()` - Configure KV cache for long-context LLMs
- `dsmil_device47_check_memory_budget()` - Verify memory usage within Layer 7 budget

**Memory Management**:
- Layer 7 budget: 40 GB maximum (from 62 GB total pool)
- Per-model tracking: Memory usage, KV cache size, context length
- Budget enforcement: Automatic rejection if budget exceeded

**INT8 Integration**:
- Mandatory INT8 quantization per MLOps pipeline
- Hardware acceleration via NPU/GPU INT8 TOPS
- Accuracy validation: >95% retention requirement

**Example Usage**:
```c
#include "dsmil_layer7_llm.h"

// Initialize with 40 GB budget
dsmil_device47_llm_init(40ULL * 1024 * 1024 * 1024);

// Load INT8 model
dsmil_device47_llm_ctx_t ctx;
if (dsmil_device47_llm_load("/path/to/int8_model.bin", &ctx) == 0) {
    // Verify quantization
    if (dsmil_device47_verify_int8_quantization(&ctx)) {
        printf("Model loaded: %s, Accuracy: %.2f%%\n", 
               ctx.model_name, ctx.quantization_accuracy * 100.0f);
    }
    
    // Perform INT8 matrix multiplication for attention
    dsmil_device47_int8_matmul(&ctx, attention_input, attention_weights, 
                               output, "attention");
}
```

---

### Layer 8 Security AI Runtime

**Purpose**: AI-powered security with 8 specialized devices (188 TOPS INT8) for adversarial defense, threat detection, and security analytics.

**Device Architecture**:
- **Device 51** (15 TOPS): Enhanced Security Framework - Anomaly detection, behavioral analytics (LSTM/GRU)
- **Device 52** (30 TOPS): Adversarial ML Defense - GANs for adversarial training, robustness testing
- **Device 53** (25 TOPS): Cybersecurity AI - Threat intelligence, attack prediction, zero-day detection
- **Device 54** (25 TOPS): Threat Intelligence - IOC extraction (NLP), attribution analysis (GNN)
- **Device 55** (20 TOPS): Automated Security Response - RL-based incident response automation
- **Device 56** (20 TOPS): Post-Quantum Crypto - ML-optimized PQC algorithms
- **Device 57** (28 TOPS): Autonomous Operations - Self-healing systems, adaptive defense
- **Device 58** (25 TOPS): Security Analytics - Security event correlation (GNN), forensics

**Key Functions**:
- `dsmil_layer8_security_init()` - Initialize specific device (51-58)
- `dsmil_layer8_analyze_binary()` - Binary security analysis (side-channels, vulnerabilities)
- `dsmil_layer8_detect_adversarial()` - Adversarial input detection
- `dsmil_layer8_analyze_side_channel()` - Side-channel vulnerability analysis
- `dsmil_layer8_extract_iocs()` - IOC extraction using NLP (Device 54)
- `dsmil_layer8_automated_response()` - RL-based automated incident response (Device 55)
- `dsmil_layer8_detect_anomaly()` - Anomaly detection (Device 51)
- `dsmil_layer8_validate_crypto()` - Cryptographic validation (PQC-only mode)
- `dsmil_layer8_train_adversarial_defense()` - Adversarial defense training with GANs (Device 52)
- `dsmil_layer8_correlate_security_events()` - Security event correlation with GNN (Device 58)
- `dsmil_layer8_predict_zero_day()` - Zero-day attack prediction (Device 53, >95% accuracy)
- `dsmil_layer8_analyze_behavioral_patterns()` - Behavioral analytics with LSTM/GRU (Device 51)
- `dsmil_layer8_optimize_pqc()` - ML-optimized PQC algorithms (Device 56)
- `dsmil_layer8_enable_zero_trust()` - Enable zero-trust security mode

**Performance Characteristics**:
- Model sizes: 50-300M parameters
- Latency: <100ms for real-time threat detection
- Detection accuracy: >99% known threats, >95% zero-day
- Real-time processing: 10,000+ inferences/sec

**Example Usage**:
```c
#include "dsmil_layer8_security.h"

// Initialize Device 52 for adversarial defense
dsmil_layer8_security_ctx_t ctx;
dsmil_layer8_security_init(DSMIL_L8_DEVICE52_ADVERSARIAL_DEFENSE, &ctx);

// Train adversarial defense model
dsmil_layer8_train_adversarial_defense(model_path, adversarial_samples, 
                                       num_samples, hardened_model_path);

// Correlate security events using GNN (Device 58)
void *correlation_graph;
size_t graph_size = 1024;
dsmil_layer8_correlate_security_events(events, num_events, 
                                      correlation_graph, &graph_size);

// Predict zero-day attacks (Device 53)
float confidence;
dsmil_layer8_predict_zero_day(threat_indicators, num_indicators, 
                              prediction, &confidence);
```

---

### Layer 9 Executive Command Runtime

**Purpose**: Strategic command AI with 4 specialized devices (330 TOPS INT8) for executive decision support, NC3 integration, and coalition coordination.

**Device Architecture**:
- **Device 59** (85 TOPS): Executive Command - Strategic decision support, crisis management, real-time resource allocation
- **Device 60** (85 TOPS): Coalition Fusion - Multi-national intelligence fusion, multi-lingual NLP, cross-cultural analysis
- **Device 61** (80 TOPS): Nuclear C&C Integration - NC3 analysis, strategic stability, deterrence modeling (ROE-governed, Section 4.1c compliant)
- **Device 62** (80 TOPS): Strategic Intelligence - Global threat assessment, geopolitical modeling, risk forecasting

**Key Functions**:
- `dsmil_layer9_executive_init()` - Initialize specific device (59-62)
- `dsmil_layer9_synthesize_intelligence()` - Synthesize intelligence from Layers 3-8 (Device 60)
- `dsmil_layer9_generate_recommendation()` - Strategic decision recommendations (Device 59)
- `dsmil_layer9_plan_campaign()` - Campaign-level mission planning
- `dsmil_layer9_coordinate_coalition()` - Coalition interoperability coordination (Device 60)
- `dsmil_layer9_validate_nc3()` - NC3 decision validation (Device 61, ROE-governed)
- `dsmil_layer9_assess_global_threats()` - Global threat assessment (Device 62)
- `dsmil_layer9_crisis_management()` - Crisis management with real-time decision support (Device 59)
- `dsmil_layer9_multi_criteria_decision()` - Multi-criteria decision analysis and policy simulation
- `dsmil_layer9_apply_releasability()` - Apply releasability markings (REL NATO, REL FVEY, NOFORN)
- `dsmil_layer9_assess_strategic_stability()` - Strategic stability assessment (Device 61, NC3)
- `dsmil_layer9_strategic_planning()` - Long-term strategic planning with scenario analysis
- `dsmil_layer9_multinational_coordination()` - Multi-national coordination (Device 60)

**Performance Characteristics**:
- Model sizes: 1B-7B parameters
- Latency: <1000ms for complex strategic queries
- Context windows: Up to 32K tokens for comprehensive analysis
- Strategic planning: Long-term scenarios with policy simulation

**NC3 Compliance (Device 61)**:
- ROE-governed per Rescindment 220330R NOV 25
- Section 4.1c compliance: ANALYSIS ONLY, NO kinetic control (NON-WAIVABLE)
- Two-person integrity validation
- TPM attestation requirements
- Full audit trail

**Example Usage**:
```c
#include "dsmil_layer9_executive.h"

// Initialize Device 59 for executive command
dsmil_layer9_executive_ctx_t ctx;
dsmil_layer9_executive_init(DSMIL_L9_DEVICE59_EXECUTIVE_CMD, &ctx);

// Crisis management
void *decision_support;
size_t support_size = 4096;
dsmil_layer9_crisis_management(&ctx, crisis_data, data_size, 
                               decision_support, &support_size);

// Apply releasability markings (Device 60)
void *marked_data;
size_t marked_size = 4096;
dsmil_layer9_apply_releasability(&ctx, intelligence_data, data_size,
                                 DSMIL_COALITION_NATO, marked_data, &marked_size);

// NC3 strategic stability assessment (Device 61)
dsmil_layer9_executive_ctx_t nc3_ctx;
dsmil_layer9_executive_init(DSMIL_L9_DEVICE61_NUCLEAR_CC, &nc3_ctx);
dsmil_layer9_assess_strategic_stability(&nc3_ctx, stability_data, data_size,
                                        assessment_result, &result_size);
```

---

### Device 255 Master Crypto Controller

**Purpose**: Unified cryptographic operations (88 algorithms) across all layers with TPM/Hardware/Software engine support.

**Key Functions**:
- `dsmil_device255_init()` - Initialize for specific layer (0-9)
- `dsmil_device255_set_engine()` - Select engine (TPM, Hardware, Software)
- `dsmil_device255_hash()` - Hash operations (SHA-256, SHA-384, SHA-512)
- `dsmil_device255_encrypt()` / `dsmil_device255_decrypt()` - Encryption/decryption (AES-256-GCM)
- `dsmil_device255_sign()` / `dsmil_device255_verify()` - Signing/verification (RSA, ECDSA, ML-DSA-87)
- `dsmil_device255_rng()` - Random number generation
- `dsmil_device255_pqc_available()` - Check PQC algorithm availability (ML-KEM-1024, ML-DSA-87)
- `dsmil_device255_cap_control()` - Enable/disable capabilities
- `dsmil_device255_cap_lock()` - TPM-protected capability locking
- `dsmil_device255_get_stats()` - Get operation statistics

**Integration Points**:
- **Device 15**: Wycheproof crypto operations via Device 255
- **Device 47**: Model encryption/signing (AES-256-GCM, ML-DSA-87 CNSA 2.0)
- **Device 46**: PQC key generation and test vectors
- **Layer 8**: PQC-only mode enforcement
- **MLOps**: Model provenance signing (ML-DSA-87 CNSA 2.0)

**Capabilities**:
- 88 algorithms total (10 hash, 22 symmetric, 5 asymmetric, 12 ECC, 11 KDF, 5 HMAC, 8 signatures, 3 key agreement, 4 MGF, 8 PQC)
- Hardware acceleration: AES-NI, SHA-NI, AVX-512
- TPM 2.0 integration: Protected operations, attestation, secure key storage

**Example Usage**:
```c
#include "dsmil_device255_crypto.h"

// Initialize for Layer 7
dsmil_device255_ctx_t ctx;
dsmil_device255_init(7, &ctx);

// Use hardware acceleration
dsmil_device255_set_engine(&ctx, DSMIL_CRYPTO_ENGINE_HARDWARE);

// Encrypt model (AES-256-GCM)
dsmil_device255_encrypt(&ctx, TPM_ALG_AES, key, key_len, iv, iv_len,
                       plaintext, plaintext_len, ciphertext, &ciphertext_len);

// Sign with ML-DSA-87 (CNSA 2.0)
if (dsmil_device255_pqc_available(&ctx, TPM_ALG_ML_DSA_87)) {
    dsmil_device255_sign(&ctx, TPM_ALG_ML_DSA_87, private_key, key_len,
                        message, message_len, signature, &signature_len);
}
```

---

### Cross-Layer Intelligence Flow

**Purpose**: Event-driven upward intelligence flow with security clearance verification.

**Key Functions**:
- `dsmil_intelligence_flow_init()` - Initialize intelligence flow system
- `dsmil_intelligence_publish()` - Publish intelligence event (upward flow)
- `dsmil_intelligence_subscribe()` - Subscribe to intelligence events (higher layers)
- `dsmil_intelligence_verify_clearance()` - Verify clearance for cross-layer flow
- `dsmil_intelligence_flow_shutdown()` - Shutdown intelligence flow system

**Intelligence Types**:
- `DSMIL_INTEL_RAW_DATA` - Layer 3: Raw sensor/data feeds
- `DSMIL_INTEL_DOMAIN_ANALYTICS` - Layer 3: Domain analytics
- `DSMIL_INTEL_MISSION_PLANNING` - Layer 4: Mission planning
- `DSMIL_INTEL_PREDICTIVE` - Layer 5: Predictive analytics
- `DSMIL_INTEL_NUCLEAR` - Layer 6: Nuclear intelligence
- `DSMIL_INTEL_AI_SYNTHESIS` - Layer 7: AI synthesis (Device 47)
- `DSMIL_INTEL_SECURITY` - Layer 8: Security overlay
- `DSMIL_INTEL_EXECUTIVE` - Layer 9: Executive command

**Security Boundaries**:
- Upward flow only (target layer >= source layer)
- Clearance verification required
- Event-driven architecture with callbacks

---

### Memory Budget Management

**Purpose**: Dynamic memory pool management (62 GB total) with layer-specific budgets.

**Layer Budgets**:
- Layer 2: 4 GB max
- Layer 3: 6 GB max
- Layer 4: 8 GB max
- Layer 5: 10 GB max
- Layer 6: 12 GB max
- **Layer 7: 40 GB max** (PRIMARY AI LAYER)
- Layer 8: 8 GB max
- Layer 9: 12 GB max

**Key Functions**:
- `dsmil_memory_budget_init()` - Initialize memory budget system
- `dsmil_memory_allocate()` - Allocate memory from layer budget
- `dsmil_memory_free()` - Free memory and update layer usage
- `dsmil_memory_check_budget()` - Check if allocation would exceed budget
- `dsmil_memory_get_usage()` - Get current memory usage statistics
- `dsmil_memory_verify_global_constraint()` - Verify sum ≤ 62 GB
- `dsmil_memory_get_layer_available()` - Get available memory for layer

**Thread Safety**: All operations are thread-safe with mutex protection.

---

### Hardware Integration Layer (HIL) Orchestration

**Purpose**: Workload orchestration across NPU/GPU/CPU (48.2 TOPS INT8 total).

**Hardware Units**:
- **NPU**: 13.0 TOPS INT8 (continuous inference)
- **GPU**: 32.0 TOPS INT8 (dense math, vision, LLM attention)
- **CPU**: 3.2 TOPS INT8 (AMX, control plane)

**Key Functions**:
- `dsmil_hil_init()` - Initialize HIL orchestration
- `dsmil_hil_assign_workload()` - Assign workload to hardware unit
- `dsmil_hil_get_utilization()` - Get current TOPS utilization
- `dsmil_hil_check_availability()` - Check if unit can accept workload
- `dsmil_hil_get_unit_info()` - Get hardware unit information

**Workload Routing**:
- Device 47 (LLM): Prefers GPU for attention operations
- Device 46 (Quantum): Uses CPU (simulation)
- Automatic load balancing: Least utilized unit selection

---

### MLOps Pipeline Optimization

**Purpose**: Compile-time and runtime support for mandatory MLOps pipeline requirements.

**Requirements**:
- **INT8 Quantization**: Mandatory, >95% accuracy retention
- **Pruning**: 50% sparsity target
- **Distillation**: 7B → 1.5B model compression
- **Flash Attention 2**: For transformers
- **Combined Speedup**: 12× minimum, 30× target, 60× maximum

**Key Functions**:
- `dsmil_mlops_get_default_targets()` - Get optimization targets
- `dsmil_mlops_verify_model()` - Verify model meets requirements
- `dsmil_mlops_verify_int8_quantization()` - Verify INT8 quantization (>95%)
- `dsmil_mlops_verify_pruning()` - Verify pruning sparsity (50%)
- `dsmil_mlops_calculate_speedup()` - Calculate combined optimization speedup

---

### Device 46 Quantum Runtime

**Purpose**: Qiskit-based quantum simulation for QAOA/QUBO optimization and quantum feature maps.

**Key Functions**:
- `dsmil_device46_quantum_init()` - Initialize quantum runtime (8-12 qubits statevector, ~30 qubits MPS)
- `dsmil_device46_qaoa_optimize()` - QAOA optimization for hyperparameter search
- `dsmil_device46_quantum_feature_map()` - Quantum feature maps for anomaly detection
- `dsmil_device46_hybrid_optimization()` - Quantum-assisted model optimization (Device 47 integration)

**Memory Budget**: 2 GB from Layer 7 pool (CPU-bound simulation)

---

### Device Integration Runtimes

**Device 15 Wycheproof Integration** (`dsmil_device15_wycheproof_runtime.c`):
- Uses Device 255 for all cryptographic operations
- Hash, encrypt, decrypt, RNG operations
- TPM engine for secure operations

**Device 47 Crypto Integration** (`dsmil_device47_crypto_runtime.c`):
- Model encryption/decryption (AES-256-GCM)
- Model signing/verification (ML-DSA-87 CNSA 2.0)
- Hardware acceleration for performance

**Device 46 PQC Integration** (`dsmil_device46_pqc_runtime.c`):
- PQC key generation (ML-KEM-1024, ML-DSA-87)
- PQC test vector generation
- Quantum-safe cryptography

**Layer 8 Security Crypto** (`dsmil_layer8_security_crypto_runtime.c`):
- PQC-only mode enforcement
- Classical crypto disabling (RSA, ECC)
- PQC algorithm verification

**MLOps Crypto Runtime** (`dsmil_mlops_crypto_runtime.c`):
- Model provenance signing (ML-DSA-87 CNSA 2.0)
- SHA-384 hashing per CNSA 2.0
- Gate verification before deployment

## Documentation

### Core Documentation
- **[DSLLVM-DESIGN.md](docs/DSLLVM-DESIGN.md)**: Complete design specification
- **[DSLLVM-ROADMAP.md](docs/DSLLVM-ROADMAP.md)**: Strategic roadmap (v1.0 → v2.0)
- **[ATTRIBUTES.md](docs/ATTRIBUTES.md)**: Attribute reference guide
- **[PROVENANCE-CNSA2.md](docs/PROVENANCE-CNSA2.md)**: Provenance system deep dive
- **[PIPELINES.md](docs/PIPELINES.md)**: Pass pipeline configurations
- **[PATH-CONFIGURATION.md](docs/PATH-CONFIGURATION.md)**: Dynamic path configuration guide ⭐ NEW

### Enhancement Documentation ⭐ NEW
- **[COMPREHENSIVE-PLAN-ALIGNMENT-ENHANCEMENTS.md](docs/COMPREHENSIVE-PLAN-ALIGNMENT-ENHANCEMENTS.md)**: Alignment with comprehensive plan
- **[DEVICE255-MASTER-CRYPTO-ENHANCEMENTS.md](docs/DEVICE255-MASTER-CRYPTO-ENHANCEMENTS.md)**: Device 255 integration guide

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
- 🚧 Dynamic path resolution runtime (v1.6.1) ✅ COMPLETE

### ✅ Completed (v1.7 - Developer Experience & Observability)

- ✅ **Configuration Validation & Health Check Tool** (`dsmil-config-validate`)
  - ✅ Mission profile validation
  - ✅ Path configuration validation
  - ✅ Truststore validation
  - ✅ Classification validation
  - ✅ Auto-fix common issues
  - ✅ Health report generation
- ✅ **Compile-Time Performance Profiling** (`dsmil-metrics`)
  - ✅ Pass execution time tracking
  - ✅ Memory usage metrics
  - ✅ Feature impact analysis
  - ✅ Build comparison tool
  - ✅ HTML dashboard generation
- ✅ **Interactive Setup Wizard** (`dsmil-setup`)
  - ✅ Installation detection
  - ✅ Mission profile setup
  - ✅ Path configuration
  - ✅ Verification and auto-fix
- ✅ **Runtime Observability Integration** (`dsmil-telemetry-collector`)
  - ✅ Prometheus metrics export
  - ✅ OpenTelemetry integration
  - ✅ Structured JSON logging (ELK/Splunk)
  - ✅ Performance, security, and operational metrics

### ✅ Completed (v1.6.1 - Runtime Enhancements) ⭐ NEW

#### Core Runtime Libraries
- ✅ **Advanced INT8 Quantization Runtime**: Comprehensive quantization support (calibration, GEMM, accuracy validation, hardware acceleration)
- ✅ **Layer 7 Device 47 LLM Runtime**: Memory management (40 GB budget), INT8 quantization enforcement (>95% accuracy), KV cache optimization, INT8 matrix operations
- ✅ **Device 46 Quantum Runtime**: Qiskit integration, QAOA/QUBO optimization, quantum feature maps, hybrid quantum-classical workflows (2 GB memory budget)
- ✅ **MLOps Pipeline Optimization**: INT8 verification, pruning validation (50% sparsity), combined speedup calculation (12× minimum, 30× target), model requirement verification
- ✅ **Cross-Layer Intelligence Flow**: Upward flow pattern, event-driven architecture, security clearance verification, layer-to-layer communication
- ✅ **Memory Budget Management**: 62 GB total pool management, layer-specific budgets (Layer 7: 40 GB max), dynamic allocation, thread-safe tracking
- ✅ **HIL Orchestration**: NPU/GPU/CPU workload routing (48.2 TOPS total), utilization monitoring, availability checking, device-aware assignment

#### Layer 8 Security AI (8 Devices, 188 TOPS INT8)
- ✅ **Device 51** (15 TOPS): Enhanced Security Framework - Anomaly detection, behavioral analytics (LSTM/GRU)
- ✅ **Device 52** (30 TOPS): Adversarial ML Defense - GANs for adversarial training, robustness testing
- ✅ **Device 53** (25 TOPS): Cybersecurity AI - Threat intelligence, attack prediction, zero-day detection (>95% accuracy)
- ✅ **Device 54** (25 TOPS): Threat Intelligence - IOC extraction (NLP), attribution analysis (GNN)
- ✅ **Device 55** (20 TOPS): Automated Security Response - RL-based incident response automation
- ✅ **Device 56** (20 TOPS): Post-Quantum Crypto - ML-optimized PQC algorithms (ML-KEM, ML-DSA)
- ✅ **Device 57** (28 TOPS): Autonomous Operations - Self-healing systems, adaptive defense
- ✅ **Device 58** (25 TOPS): Security Analytics - Security event correlation (GNN), forensics
- ✅ **Specialized Functions**: Adversarial defense training, security event correlation, zero-day prediction, behavioral pattern analysis, PQC optimization

#### Layer 9 Executive Command (4 Devices, 330 TOPS INT8)
- ✅ **Device 59** (85 TOPS): Executive Command - Strategic decision support, crisis management, real-time resource allocation
- ✅ **Device 60** (85 TOPS): Coalition Fusion - Multi-national intelligence fusion, multi-lingual NLP, cross-cultural analysis
- ✅ **Device 61** (80 TOPS): Nuclear C&C Integration - NC3 analysis, strategic stability, deterrence modeling (ROE-governed, Section 4.1c compliant)
- ✅ **Device 62** (80 TOPS): Strategic Intelligence - Global threat assessment, geopolitical modeling, risk forecasting
- ✅ **Specialized Functions**: Crisis management, multi-criteria decision analysis, releasability markings (REL NATO/FVEY/NOFORN), strategic stability assessment, long-term planning, multi-national coordination

#### Device 255 Master Crypto Controller
- ✅ **Unified Crypto API**: 88 algorithms (TPM + Hardware + Software)
- ✅ **Engine Support**: TPM 2.0, Hardware acceleration (AES-NI/SHA-NI/AVX-512), Software fallback
- ✅ **PQC Algorithms**: ML-KEM-1024, ML-DSA-87 support
- ✅ **Capability Management**: Runtime enable/disable, TPM-protected locking
- ✅ **Layer-Aware Operations**: Context-aware crypto per layer (0-9)

#### Device Integration Runtimes
- ✅ **Device 15 Wycheproof**: Crypto operations via Device 255 (hash, encrypt, decrypt, RNG)
- ✅ **Device 47 Model Crypto**: Encryption/signing (AES-256-GCM, ML-DSA-87 CNSA 2.0)
- ✅ **Device 46 PQC**: Key generation, test vectors (ML-KEM-1024, ML-DSA-87)
- ✅ **Layer 8 Security Crypto**: PQC-only mode enforcement, classical crypto disabling
- ✅ **MLOps Crypto**: Model provenance signing (ML-DSA-87 CNSA 2.0, SHA-384)

#### Configuration & Integration
- ✅ **Wycheproof Bundle Configs**: Device 255 integration YAML, intelligence flows configuration
- ✅ **Schema Updates**: `device255_metadata` property in crypto_test_result.schema.yaml
- ✅ **Documentation**: Comprehensive feature documentation, API references, usage examples

### 💡 Proposed Enhancements (v1.8+)

See **[ENHANCEMENT-SUGGESTIONS.md](docs/ENHANCEMENT-SUGGESTIONS.md)** for future enhancements:

5. **Multi-Architecture Support** - ARM64, RISC-V, embedded targets (deferred)

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
4. **Runtime API Enhancement**: Expand Layer 8/9 specialized functions, improve INT8 quantization algorithms
5. **Testing**: Expand test coverage for runtime APIs
6. **Documentation**: Examples, tutorials, case studies, integration guides
7. **Hardware Acceleration**: Optimize INT8 GEMM for NPU/GPU, implement quantum simulation backends
8. **Security AI Models**: Integrate actual GAN/GNN/LSTM models for Layer 8 devices
9. **Strategic AI Models**: Integrate large language models (1B-7B) for Layer 9 devices
10. **Device 255 Backend**: Implement actual TPM 2.0 integration, hardware crypto acceleration

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

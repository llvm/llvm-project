# DSMIL Optimization Pipelines
**Pass Ordering and Pipeline Configurations for DSLLVM**

Version: v1.0
Last Updated: 2025-11-24

---

## Overview

DSLLVM provides several pre-configured pass pipelines optimized for different DSMIL deployment scenarios. These pipelines integrate standard LLVM optimization passes with DSMIL-specific analysis, verification, and transformation passes.

---

## 1. Pipeline Presets

### 1.1 `dsmil-default` (Production)

**Use Case**: Production DSMIL binaries with full enforcement

**Invocation**:
```bash
dsmil-clang -O3 -fpass-pipeline=dsmil-default -o output input.c
```

**Pass Sequence**:

```
Module Pipeline:
  ├─ Standard Frontend (Parsing, Sema, CodeGen)
  │
  ├─ Early Optimizations
  │  ├─ Inlining
  │  ├─ SROA (Scalar Replacement of Aggregates)
  │  ├─ Early CSE
  │  └─ Instcombine
  │
  ├─ DSMIL Metadata Propagation
  │  └─ dsmil-metadata-propagate
  │      Purpose: Propagate dsmil_* attributes from source to IR metadata
  │      Ensures all functions/globals have complete DSMIL context
  │
  ├─ Mid-Level Optimizations (-O3)
  │  ├─ Loop optimizations (unroll, vectorization)
  │  ├─ Aggressive instcombine
  │  ├─ GVN (Global Value Numbering)
  │  ├─ Dead code elimination
  │  └─ Function specialization
  │
  ├─ DSMIL Analysis Passes
  │  ├─ dsmil-bandwidth-estimate
  │  │   Purpose: Analyze memory bandwidth requirements
  │  │   Outputs: !dsmil.bw_bytes_read, !dsmil.bw_gbps_estimate
  │  │
  │  ├─ dsmil-device-placement
  │  │   Purpose: Recommend CPU/NPU/GPU placement
  │  │   Inputs: Bandwidth estimates, dsmil_layer/device metadata
  │  │   Outputs: !dsmil.placement metadata, *.dsmilmap sidecar
  │  │
  │  └─ dsmil-quantum-export
  │      Purpose: Extract QUBO problems from dsmil_quantum_candidate functions
  │      Outputs: *.quantum.json sidecar
  │
  ├─ DSMIL Verification Passes
  │  ├─ dsmil-layer-check
  │  │   Purpose: Enforce layer boundary policies
  │  │   Errors: On disallowed transitions without dsmil_gateway
  │  │
  │  └─ dsmil-stage-policy
  │      Purpose: Validate MLOps stage usage (no debug in production)
  │      Errors: On policy violations (configurable strictness)
  │
  ├─ Link-Time Optimization (LTO)
  │  ├─ Whole-program analysis
  │  ├─ Dead function elimination
  │  ├─ Cross-module inlining
  │  └─ Final optimization rounds
  │
  └─ DSMIL Link-Time Transforms
     ├─ dsmil-sandbox-wrap
     │   Purpose: Inject sandbox setup wrapper around main()
     │   Renames: main → main_real
     │   Injects: Capability + seccomp setup in new main()
     │
     └─ dsmil-provenance-emit
         Purpose: Generate CNSA 2.0 provenance, sign, embed in ELF
         Outputs: .note.dsmil.provenance section
```

**Configuration**:
```yaml
dsmil_default_config:
  enforcement: strict
  layer_policy: enforce
  stage_policy: production  # No debug/experimental
  bandwidth_model: meteorlake_64gbps
  provenance: cnsa2_sha384_mldsa87
  sandbox: enabled
  quantum_export: enabled
```

**Typical Compile Time Overhead**: 8-12%

---

### 1.2 `dsmil-debug` (Development)

**Use Case**: Development builds with relaxed enforcement

**Invocation**:
```bash
dsmil-clang -O2 -g -fpass-pipeline=dsmil-debug -o output input.c
```

**Pass Sequence**:

```
Module Pipeline:
  ├─ Standard Frontend with debug info
  ├─ Moderate Optimizations (-O2)
  ├─ DSMIL Metadata Propagation
  ├─ DSMIL Analysis (bandwidth, placement, quantum)
  ├─ DSMIL Verification (WARNING mode only)
  │  ├─ dsmil-layer-check --warn-only
  │  └─ dsmil-stage-policy --allow-debug
  ├─ NO LTO (faster iteration)
  ├─ dsmil-sandbox-wrap (OPTIONAL via flag)
  └─ dsmil-provenance-emit (test signing key)
```

**Configuration**:
```yaml
dsmil_debug_config:
  enforcement: warn
  layer_policy: warn_only  # Emit warnings, don't fail build
  stage_policy: development  # Allow debug/experimental
  bandwidth_model: generic
  provenance: test_key  # Development signing key
  sandbox: optional  # Only if --enable-sandbox passed
  quantum_export: disabled  # Skip in debug
  debug_info: dwarf5
```

**Typical Compile Time Overhead**: 4-6%

---

### 1.3 `dsmil-lab` (Research/Experimentation)

**Use Case**: Research, experimentation, no enforcement

**Invocation**:
```bash
dsmil-clang -O1 -fpass-pipeline=dsmil-lab -o output input.c
```

**Pass Sequence**:

```
Module Pipeline:
  ├─ Standard Frontend
  ├─ Basic Optimizations (-O1)
  ├─ DSMIL Metadata Propagation
  ├─ DSMIL Analysis (annotation only, no enforcement)
  │  ├─ dsmil-bandwidth-estimate
  │  ├─ dsmil-device-placement --suggest-only
  │  └─ dsmil-quantum-export
  ├─ NO verification (layer-check, stage-policy skipped)
  ├─ NO sandbox-wrap
  └─ OPTIONAL provenance (--enable-provenance to opt-in)
```

**Configuration**:
```yaml
dsmil_lab_config:
  enforcement: none
  layer_policy: disabled
  stage_policy: disabled
  bandwidth_model: generic
  provenance: disabled  # Opt-in via flag
  sandbox: disabled
  quantum_export: enabled  # Always useful for research
  annotations_only: true  # Just add metadata, no checks
```

**Typical Compile Time Overhead**: 2-3%

---

### 1.4 `dsmil-kernel` (Kernel Mode)

**Use Case**: DSMIL kernel, drivers, layer 0-2 code

**Invocation**:
```bash
dsmil-clang -O3 -fpass-pipeline=dsmil-kernel -ffreestanding -o module.ko input.c
```

**Pass Sequence**:

```
Module Pipeline:
  ├─ Frontend (freestanding mode)
  ├─ Kernel-specific optimizations
  │  ├─ No red-zone assumptions
  │  ├─ Stack protector (strong)
  │  └─ Retpoline/IBRS for Spectre mitigation
  ├─ DSMIL Metadata Propagation
  ├─ DSMIL Analysis
  │  ├─ dsmil-bandwidth-estimate (crucial for DMA ops)
  │  └─ dsmil-device-placement
  ├─ DSMIL Verification
  │  ├─ dsmil-layer-check (enforced, kernel ≤ layer 2)
  │  └─ dsmil-stage-policy --kernel-mode
  ├─ Kernel LTO (partial, per-module)
  └─ dsmil-provenance-emit (kernel module signing key)
     Note: NO sandbox-wrap (kernel space)
```

**Configuration**:
```yaml
dsmil_kernel_config:
  enforcement: strict
  layer_policy: enforce_kernel  # Only allow layer 0-2
  stage_policy: kernel_production
  max_layer: 2
  provenance: kernel_module_key
  sandbox: disabled  # N/A in kernel
  kernel_hardening: enabled
```

---

## 2. Pass Details

### 2.1 `dsmil-metadata-propagate`

**Type**: Module pass (early)

**Purpose**: Ensure DSMIL attributes are consistently represented as IR metadata

**Actions**:
1. Walk all functions with `dsmil_*` attributes
2. Create corresponding IR metadata nodes
3. Propagate metadata to inlined callees
4. Handle defaults (e.g., layer 0 if unspecified)

**Example IR Transformation**:

Before:
```llvm
define void @foo() #0 {
  ; ...
}
attributes #0 = { "dsmil_layer"="7" "dsmil_device"="47" }
```

After:
```llvm
define void @foo() !dsmil.layer !1 !dsmil.device_id !2 {
  ; ...
}
!1 = !{i32 7}
!2 = !{i32 47}
```

---

### 2.2 `dsmil-bandwidth-estimate`

**Type**: Function pass (analysis)

**Purpose**: Estimate memory bandwidth requirements

**Algorithm**:
```
For each function:
  1. Walk all load/store instructions
  2. Classify access patterns:
     - Sequential: stride = element_size
     - Strided: stride > element_size
     - Random: gather/scatter or unpredictable
  3. Account for vectorization:
     - AVX2 (256-bit): 4x throughput
     - AVX-512 (512-bit): 8x throughput
  4. Compute:
     bytes_read = Σ(load_size × trip_count)
     bytes_written = Σ(store_size × trip_count)
  5. Estimate GB/s assuming 64 GB/s peak bandwidth:
     bw_gbps = (bytes_read + bytes_written) / execution_time_estimate
  6. Classify memory class:
     - kv_cache: >20 GB/s, random access
     - model_weights: >10 GB/s, sequential
     - hot_ram: >5 GB/s
     - cold_storage: <1 GB/s
```

**Output Metadata**:
```llvm
!dsmil.bw_bytes_read = !{i64 1048576000}      ; 1 GB
!dsmil.bw_bytes_written = !{i64 524288000}    ; 512 MB
!dsmil.bw_gbps_estimate = !{double 23.5}
!dsmil.memory_class = !{!"kv_cache"}
```

---

### 2.3 `dsmil-device-placement`

**Type**: Module pass (analysis + annotation)

**Purpose**: Recommend execution target (CPU/NPU/GPU) and memory tier

**Decision Logic**:

```python
def recommend_placement(function):
    layer = function.metadata['dsmil.layer']
    device = function.metadata['dsmil.device_id']
    bw_gbps = function.metadata['dsmil.bw_gbps_estimate']

    # Device-specific hints
    if device == 47:  # NPU primary
        target = 'npu'
    elif device in [40, 41, 42]:  # GPU accelerators
        target = 'gpu'
    elif device in [30..39]:  # Crypto accelerators
        target = 'cpu_crypto'
    else:
        target = 'cpu'

    # Bandwidth-based memory tier
    if bw_gbps > 30:
        memory_tier = 'ramdisk'  # Fastest
    elif bw_gbps > 15:
        memory_tier = 'tmpfs'
    elif bw_gbps > 5:
        memory_tier = 'local_ssd'
    else:
        memory_tier = 'remote_minio'  # Network storage OK

    # Stage-specific overrides
    if function.metadata['dsmil.stage'] == 'pretrain':
        memory_tier = 'local_ssd'  # Checkpoints

    return {
        'target': target,
        'memory_tier': memory_tier
    }
```

**Output**:
- IR metadata: `!dsmil.placement = !{!"target: npu, memory: ramdisk"}`
- Sidecar: `binary_name.dsmilmap` with per-function recommendations

---

### 2.4 `dsmil-layer-check`

**Type**: Module pass (verification)

**Purpose**: Enforce DSMIL layer boundary policies

**Algorithm**:
```
For each call edge (caller → callee):
  1. Extract layer_caller, clearance_caller, roe_caller
  2. Extract layer_callee, clearance_callee, roe_callee

  3. Check layer transition:
     If layer_caller > layer_callee:
       // Downward call (safer, usually allowed)
       OK
     Else if layer_caller < layer_callee:
       // Upward call (privileged, requires gateway)
       If NOT callee.has_attribute('dsmil_gateway'):
         ERROR: "Upward layer transition without gateway"
     Else:
       // Same layer
       OK

  4. Check clearance:
     If clearance_caller < clearance_callee:
       If NOT callee.has_attribute('dsmil_gateway'):
         ERROR: "Insufficient clearance to call function"

  5. Check ROE escalation:
     If roe_caller == "ANALYSIS_ONLY" AND roe_callee == "LIVE_CONTROL":
       If NOT callee.has_attribute('dsmil_gateway'):
         ERROR: "ROE escalation requires gateway"
```

**Example Error**:
```
input.c:45:5: error: layer boundary violation
    kernel_write(data);
    ^~~~~~~~~~~~~~~
note: caller 'user_function' is at layer 7 (user)
note: callee 'kernel_write' is at layer 1 (kernel)
note: add __attribute__((dsmil_gateway)) to 'kernel_write' or use a gateway function
```

---

### 2.5 `dsmil-stage-policy`

**Type**: Module pass (verification)

**Purpose**: Enforce MLOps stage policies

**Policy Rules** (configurable):

```yaml
production_policy:
  allowed_stages: [pretrain, finetune, quantized, distilled, serve]
  forbidden_stages: [debug, experimental]
  min_layer_for_quantized: 3  # Layer ≥3 must use quantized models

development_policy:
  allowed_stages: [pretrain, finetune, quantized, distilled, serve, debug, experimental]
  forbidden_stages: []
  warnings_only: true

kernel_policy:
  allowed_stages: [serve, production_kernel]
  forbidden_stages: [debug, experimental, pretrain, finetune]
```

**Example Error**:
```
input.c:12:1: error: stage policy violation
__attribute__((dsmil_stage("debug")))
^
note: production binaries cannot link dsmil_stage("debug") code
note: build configuration: DSMIL_POLICY=production
```

---

### 2.6 `dsmil-quantum-export`

**Type**: Function pass (analysis + export)

**Purpose**: Extract optimization problems for quantum offload

**Process**:
1. Identify functions with `dsmil_quantum_candidate` attribute
2. Analyze function body:
   - Extract integer variables (candidates for QUBO variables)
   - Identify optimization loops (for/while with min/max objectives)
   - Detect constraint patterns (if statements, bounds checks)
3. Attempt QUBO/Ising mapping:
   - Binary decision variables → qubits
   - Objective function → Q matrix (quadratic terms)
   - Constraints → penalty terms in Q matrix
4. Export to `*.quantum.json`

**Example Input**:
```c
__attribute__((dsmil_quantum_candidate("placement")))
int placement_solver(struct model models[], struct device devices[], int n) {
    int cost = 0;
    int placement[n];  // placement[i] = device index for model i

    // Minimize communication cost
    for (int i = 0; i < n; i++) {
        for (int j = i+1; j < n; j++) {
            if (models[i].depends_on[j] && placement[i] != placement[j]) {
                cost += communication_cost(devices[placement[i]], devices[placement[j]]);
            }
        }
    }

    return cost;
}
```

**Example Output** (`*.quantum.json`):
```json
{
  "schema": "dsmil-quantum-v1",
  "functions": [
    {
      "name": "placement_solver",
      "kind": "placement",
      "representation": "qubo",
      "variables": 16,  // n=4 models × 4 devices
      "qubo": {
        "Q": [[/* 16×16 matrix */]],
        "variable_names": [
          "model_0_device_0", "model_0_device_1", ...,
          "model_3_device_3"
        ],
        "constraints": {
          "one_hot": "each model assigned to exactly one device"
        }
      }
    }
  ]
}
```

---

### 2.7 `dsmil-sandbox-wrap`

**Type**: Link-time transform

**Purpose**: Inject sandbox setup wrapper around `main()`

**Transformation**:

Before:
```c
__attribute__((dsmil_sandbox("l7_llm_worker")))
int main(int argc, char **argv) {
    return llm_worker_loop();
}
```

After (conceptual):
```c
// Original main renamed
int main_real(int argc, char **argv) __asm__("main_real");
int main_real(int argc, char **argv) {
    return llm_worker_loop();
}

// New main injected
int main(int argc, char **argv) {
    // 1. Load sandbox profile
    const struct dsmil_sandbox_profile *profile =
        dsmil_get_sandbox_profile("l7_llm_worker");

    // 2. Drop capabilities (libcap-ng)
    capng_clear(CAPNG_SELECT_BOTH);
    capng_updatev(CAPNG_ADD, CAPNG_EFFECTIVE | CAPNG_PERMITTED,
                  CAP_NET_BIND_SERVICE, -1);  // Example: only allow binding ports
    capng_apply(CAPNG_SELECT_BOTH);

    // 3. Install seccomp filter
    struct sock_fprog prog = {
        .len = profile->seccomp_filter_len,
        .filter = profile->seccomp_filter
    };
    prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0);
    prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, &prog);

    // 4. Set resource limits
    struct rlimit rlim = {
        .rlim_cur = 4UL * 1024 * 1024 * 1024,  // 4 GB
        .rlim_max = 4UL * 1024 * 1024 * 1024
    };
    setrlimit(RLIMIT_AS, &rlim);

    // 5. Call real main
    return main_real(argc, argv);
}
```

**Profiles** (defined in `/etc/dsmil/sandbox/`):
- `l7_llm_worker.profile`: Minimal capabilities, restricted syscalls
- `l5_network_daemon.profile`: Network I/O, no filesystem write
- `l3_crypto_worker.profile`: Crypto operations, no network

---

### 2.8 `dsmil-provenance-emit`

**Type**: Link-time transform

**Purpose**: Generate, sign, and embed CNSA 2.0 provenance

**Process**:
1. **Collect metadata**:
   - Compiler version, target triple, commit hash
   - Git repo, commit, dirty status
   - Build timestamp, builder ID, flags
   - DSMIL layer/device/role assignments
2. **Compute hashes**:
   - Binary hash (SHA-384 over all PT_LOAD segments)
   - Section hashes (per ELF section)
3. **Canonicalize provenance**:
   - Serialize to deterministic JSON or CBOR
4. **Sign**:
   - Hash canonical provenance with SHA-384
   - Sign hash with ML-DSA-87 using PSK
5. **Embed**:
   - Create `.note.dsmil.provenance` section
   - Add NOTE program header

**Configuration**:
```bash
export DSMIL_PSK_PATH=/secure/keys/psk_2025.pem
export DSMIL_BUILD_ID=$(uuidgen)
export DSMIL_BUILDER_ID=$(hostname)
```

---

## 3. Custom Pipeline Configuration

### 3.1 Override Default Pipeline

```bash
# Use custom pass order
dsmil-clang -O3 \
  -fpass-plugin=/opt/dsmil/lib/DsmilPasses.so \
  -fpass-order=inline,dsmil-metadata-propagate,sroa,instcombine,gvn,... \
  -o output input.c
```

### 3.2 Skip Specific Passes

```bash
# Skip stage policy check (development override)
dsmil-clang -O3 -fpass-pipeline=dsmil-default \
  -mllvm -dsmil-skip-stage-policy \
  -o output input.c

# Disable provenance (testing)
dsmil-clang -O3 -fpass-pipeline=dsmil-default \
  -mllvm -dsmil-no-provenance \
  -o output input.c
```

### 3.3 Pass Flags

```bash
# Layer check: warn instead of error
-mllvm -dsmil-layer-check-mode=warn

# Bandwidth estimate: use custom memory model
-mllvm -dsmil-bandwidth-model=custom \
-mllvm -dsmil-bandwidth-peak-gbps=128

# Device placement: force CPU target
-mllvm -dsmil-device-placement-override=cpu

# Provenance: use test signing key
-mllvm -dsmil-provenance-test-key=/tmp/test_psk.pem
```

---

## 4. Integration with Build Systems

### 4.1 CMake

```cmake
# Enable DSMIL toolchain
set(CMAKE_C_COMPILER ${DSMIL_ROOT}/bin/dsmil-clang)
set(CMAKE_CXX_COMPILER ${DSMIL_ROOT}/bin/dsmil-clang++)

# Set default pipeline for target
add_executable(llm_worker llm_worker.c)
target_compile_options(llm_worker PRIVATE -fpass-pipeline=dsmil-default)
target_link_options(llm_worker PRIVATE -fpass-pipeline=dsmil-default)

# Development build: use debug pipeline
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  target_compile_options(llm_worker PRIVATE -fpass-pipeline=dsmil-debug)
endif()

# Kernel module: use kernel pipeline
add_library(dsmil_driver MODULE driver.c)
target_compile_options(dsmil_driver PRIVATE -fpass-pipeline=dsmil-kernel)
```

### 4.2 Makefile

```makefile
CC = dsmil-clang
CXX = dsmil-clang++
CFLAGS = -O3 -fpass-pipeline=dsmil-default

# Per-target override
llm_worker: llm_worker.c
	$(CC) $(CFLAGS) -fpass-pipeline=dsmil-default -o $@ $<

debug_tool: debug_tool.c
	$(CC) -O2 -g -fpass-pipeline=dsmil-debug -o $@ $<

kernel_module.ko: kernel_module.c
	$(CC) -O3 -fpass-pipeline=dsmil-kernel -ffreestanding -o $@ $<
```

### 4.3 Bazel

```python
# BUILD file
cc_binary(
    name = "llm_worker",
    srcs = ["llm_worker.c"],
    copts = [
        "-fpass-pipeline=dsmil-default",
    ],
    linkopts = [
        "-fpass-pipeline=dsmil-default",
    ],
    toolchains = ["@dsmil_toolchain//:cc"],
)
```

---

## 5. Performance Tuning

### 5.1 Compilation Speed

**Faster Builds** (development):
```bash
# Use dsmil-debug (no LTO, less optimization)
dsmil-clang -O2 -fpass-pipeline=dsmil-debug -o output input.c

# Skip expensive passes
dsmil-clang -O3 -fpass-pipeline=dsmil-default \
  -mllvm -dsmil-skip-quantum-export \  # Skip QUBO extraction
  -mllvm -dsmil-skip-bandwidth-estimate \  # Skip bandwidth analysis
  -o output input.c
```

**Faster LTO**:
```bash
# Use ThinLTO instead of full LTO
dsmil-clang -O3 -flto=thin -fpass-pipeline=dsmil-default -o output input.c
```

### 5.2 Runtime Performance

**Aggressive Optimization**:
```bash
# Enable PGO (Profile-Guided Optimization)
# 1. Instrumented build
dsmil-clang -O3 -fpass-pipeline=dsmil-default -fprofile-generate -o llm_worker input.c

# 2. Training run
./llm_worker < training_workload.txt

# 3. Optimized build with profile
dsmil-clang -O3 -fpass-pipeline=dsmil-default -fprofile-use=default.profdata -o llm_worker input.c
```

**Tuning for Meteor Lake**:
```bash
# Already included in dsmil-default, but can be explicit:
dsmil-clang -O3 -march=meteorlake -mtune=meteorlake \
  -mavx2 -mfma -maes -msha \  # Explicitly enable features
  -fpass-pipeline=dsmil-default \
  -o output input.c
```

---

## 6. Troubleshooting

### Issue: "Pass 'dsmil-layer-check' not found"

**Solution**: Ensure DSMIL pass plugin is loaded:
```bash
export DSMIL_PASS_PLUGIN=/opt/dsmil/lib/DsmilPasses.so
dsmil-clang -fpass-plugin=$DSMIL_PASS_PLUGIN -fpass-pipeline=dsmil-default ...
```

### Issue: "Cannot find PSK for provenance signing"

**Solution**: Set `DSMIL_PSK_PATH`:
```bash
export DSMIL_PSK_PATH=/secure/keys/psk_2025.pem
# OR use test key for development:
export DSMIL_PSK_PATH=/opt/dsmil/keys/test_psk.pem
```

### Issue: Compilation very slow with `dsmil-default`

**Solution**: Use `dsmil-debug` for development iteration:
```bash
dsmil-clang -O2 -fpass-pipeline=dsmil-debug -o output input.c
```

---

## See Also

- [DSLLVM-DESIGN.md](DSLLVM-DESIGN.md) - Main specification
- [ATTRIBUTES.md](ATTRIBUTES.md) - DSMIL attribute reference
- [PROVENANCE-CNSA2.md](PROVENANCE-CNSA2.md) - Provenance system details

---

**End of Pipeline Documentation**

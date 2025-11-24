# DSMIL Attributes Reference
**Comprehensive Guide to DSMIL Source-Level Annotations**

Version: v1.0
Last Updated: 2025-11-24

---

## Overview

DSLLVM extends Clang with a set of custom attributes that encode DSMIL-specific semantics directly in C/C++ source code. These attributes are lowered to LLVM IR metadata and consumed by DSMIL-specific optimization and verification passes.

All DSMIL attributes use the `dsmil_` prefix and are available via `__attribute__((...))` syntax.

---

## Layer & Device Attributes

### `dsmil_layer(int layer_id)`

**Purpose**: Assign a function or global to a specific DSMIL architectural layer.

**Parameters**:
- `layer_id` (int): Layer index, typically 0-8 or 1-9 depending on naming convention.

**Applies to**: Functions, global variables

**Example**:
```c
__attribute__((dsmil_layer(7)))
void llm_inference_worker(void) {
    // Layer 7 (AI/ML) operations
}
```

**IR Lowering**:
```llvm
!dsmil.layer = !{i32 7}
```

**Backend Effects**:
- Function placed in `.text.dsmil.layer7` section
- Entry added to `*.dsmilmap` sidecar file
- Used by `dsmil-layer-check` pass for boundary validation

**Notes**:
- Invalid layer transitions are caught at compile-time by `dsmil-layer-check`
- Functions without this attribute default to layer 0 (kernel/hardware)

---

### `dsmil_device(int device_id)`

**Purpose**: Assign a function or global to a specific DSMIL device.

**Parameters**:
- `device_id` (int): Device index, 0-103 per DSMIL architecture.

**Applies to**: Functions, global variables

**Example**:
```c
__attribute__((dsmil_device(47)))
void npu_workload(void) {
    // Runs on Device 47 (NPU/AI accelerator)
}
```

**IR Lowering**:
```llvm
!dsmil.device_id = !{i32 47}
```

**Backend Effects**:
- Function placed in `.text.dsmil.dev47` section
- Metadata used by `dsmil-device-placement` for optimization hints

**Device Categories** (partial list):
- 0-9: Core kernel devices
- 10-19: Storage subsystem
- 20-29: Network subsystem
- 30-39: Security/crypto devices
- 40-49: AI/ML devices (46 = quantum integration, 47 = NPU primary)
- 50-59: Telemetry/observability
- 60-69: Power management
- 70-103: Application/user-defined

---

## Security & Policy Attributes

### `dsmil_clearance(uint32_t clearance_mask)`

**Purpose**: Specify security clearance level and compartments for a function.

**Parameters**:
- `clearance_mask` (uint32): 32-bit bitmask encoding clearance level and compartments.

**Applies to**: Functions

**Example**:
```c
__attribute__((dsmil_clearance(0x07070707)))
void sensitive_operation(void) {
    // Requires specific clearance
}
```

**IR Lowering**:
```llvm
!dsmil.clearance = !{i32 0x07070707}
```

**Clearance Format** (proposed):
- Bits 0-7: Base clearance level (0-255)
- Bits 8-15: Compartment A
- Bits 16-23: Compartment B
- Bits 24-31: Compartment C

**Verification**:
- `dsmil-layer-check` ensures lower-clearance code cannot call higher-clearance code without gateway

---

### `dsmil_roe(const char *rules)`

**Purpose**: Specify Rules of Engagement for a function (authorization to perform specific actions).

**Parameters**:
- `rules` (string): ROE policy identifier

**Applies to**: Functions

**Example**:
```c
__attribute__((dsmil_roe("ANALYSIS_ONLY")))
void analyze_data(const void *data) {
    // Read-only analysis operations
}

__attribute__((dsmil_roe("LIVE_CONTROL")))
void actuate_hardware(int device_id, int value) {
    // Can control physical hardware
}
```

**Common ROE Values**:
- `"ANALYSIS_ONLY"`: Read-only, no side effects
- `"LIVE_CONTROL"`: Can modify hardware/system state
- `"NETWORK_EGRESS"`: Can send data externally
- `"CRYPTO_SIGN"`: Can sign data with system keys
- `"ADMIN_OVERRIDE"`: Emergency administrative access

**IR Lowering**:
```llvm
!dsmil.roe = !{!"ANALYSIS_ONLY"}
```

**Verification**:
- Enforced by `dsmil-layer-check` and runtime policy engine
- Transitions from weaker to stronger ROE require explicit gateway

---

### `dsmil_gateway`

**Purpose**: Mark a function as an authorized boundary crossing point.

**Parameters**: None

**Applies to**: Functions

**Example**:
```c
__attribute__((dsmil_gateway))
__attribute__((dsmil_layer(5)))
__attribute__((dsmil_clearance(0x05050505)))
int validated_syscall_handler(int syscall_num, void *args) {
    // Can safely transition from layer 7 userspace to layer 5 kernel
    return do_syscall(syscall_num, args);
}
```

**IR Lowering**:
```llvm
!dsmil.gateway = !{i1 true}
```

**Semantics**:
- Without this attribute, `dsmil-layer-check` rejects cross-layer or cross-clearance calls
- Gateway functions must implement proper validation and sanitization
- Audit events generated at runtime for all gateway transitions

---

### `dsmil_sandbox(const char *profile_name)`

**Purpose**: Specify sandbox profile for program entry point.

**Parameters**:
- `profile_name` (string): Name of predefined sandbox profile

**Applies to**: `main` function

**Example**:
```c
__attribute__((dsmil_sandbox("l7_llm_worker")))
int main(int argc, char **argv) {
    // Runs with l7_llm_worker sandbox restrictions
    return run_inference_loop();
}
```

**IR Lowering**:
```llvm
!dsmil.sandbox = !{!"l7_llm_worker"}
```

**Link-Time Transformation**:
- `dsmil-sandbox-wrap` pass renames `main` → `main_real`
- Injects wrapper `main` that:
  - Sets up libcap-ng capability restrictions
  - Installs seccomp-bpf filter
  - Configures resource limits
  - Calls `main_real()`

**Predefined Profiles**:
- `"l7_llm_worker"`: AI inference sandbox
- `"l5_network_daemon"`: Network service restrictions
- `"l3_crypto_worker"`: Cryptographic operations
- `"l1_device_driver"`: Kernel driver restrictions

---

## MLOps Stage Attributes

### `dsmil_stage(const char *stage_name)`

**Purpose**: Encode MLOps lifecycle stage for functions and binaries.

**Parameters**:
- `stage_name` (string): MLOps stage identifier

**Applies to**: Functions, binaries (via main)

**Example**:
```c
__attribute__((dsmil_stage("quantized")))
void model_inference_int8(const int8_t *input, int8_t *output) {
    // Quantized inference path
}

__attribute__((dsmil_stage("debug")))
void verbose_diagnostics(void) {
    // Debug-only code
}
```

**Common Stage Values**:
- `"pretrain"`: Pre-training phase
- `"finetune"`: Fine-tuning operations
- `"quantized"`: Quantized models (INT8/INT4)
- `"distilled"`: Distilled/compressed models
- `"serve"`: Production serving/inference
- `"debug"`: Debug/diagnostic code
- `"experimental"`: Research/non-production

**IR Lowering**:
```llvm
!dsmil.stage = !{!"quantized"}
```

**Policy Enforcement**:
- `dsmil-stage-policy` pass validates stage usage per deployment target
- Production binaries (layer ≥3) may prohibit `debug` and `experimental` stages
- Automated MLOps pipelines use stage metadata to route workloads

---

## Memory & Performance Attributes

### `dsmil_kv_cache`

**Purpose**: Mark storage for key-value cache in LLM inference.

**Parameters**: None

**Applies to**: Functions, global variables

**Example**:
```c
__attribute__((dsmil_kv_cache))
struct kv_cache_pool {
    float *keys;
    float *values;
    size_t capacity;
} global_kv_cache;

__attribute__((dsmil_kv_cache))
void allocate_kv_cache(size_t tokens) {
    // KV cache allocation routine
}
```

**IR Lowering**:
```llvm
!dsmil.memory_class = !{!"kv_cache"}
```

**Optimization Effects**:
- `dsmil-bandwidth-estimate` prioritizes KV cache bandwidth
- `dsmil-device-placement` suggests high-bandwidth memory tier (ramdisk/tmpfs)
- Backend may use specific cache line prefetch strategies

---

### `dsmil_hot_model`

**Purpose**: Mark frequently accessed model weights.

**Parameters**: None

**Applies to**: Global variables, functions that access hot paths

**Example**:
```c
__attribute__((dsmil_hot_model))
const float attention_weights[4096][4096] = { /* ... */ };

__attribute__((dsmil_hot_model))
void attention_forward(const float *query, const float *key, float *output) {
    // Hot path in transformer model
}
```

**IR Lowering**:
```llvm
!dsmil.memory_class = !{!"hot_model"}
!dsmil.sensitivity = !{!"MODEL_WEIGHTS"}
```

**Optimization Effects**:
- May be placed in large pages (2MB/1GB)
- Prefetch optimizations
- Pinned in high-speed memory tier

---

## Quantum Integration Attributes

### `dsmil_quantum_candidate(const char *problem_type)`

**Purpose**: Mark a function as candidate for quantum-assisted optimization.

**Parameters**:
- `problem_type` (string): Type of optimization problem

**Applies to**: Functions

**Example**:
```c
__attribute__((dsmil_quantum_candidate("placement")))
int optimize_model_placement(struct model *m, struct device *devices, int n) {
    // Classical placement solver
    // Will be analyzed for quantum offload potential
    return classical_solver(m, devices, n);
}

__attribute__((dsmil_quantum_candidate("schedule")))
void job_scheduler(struct job *jobs, int count) {
    // Scheduling problem suitable for quantum annealing
}
```

**Problem Types**:
- `"placement"`: Device/model placement optimization
- `"routing"`: Network path selection
- `"schedule"`: Job/task scheduling
- `"hyperparam_search"`: Hyperparameter tuning

**IR Lowering**:
```llvm
!dsmil.quantum_candidate = !{!"placement"}
```

**Processing**:
- `dsmil-quantum-export` pass analyzes function
- Attempts to extract QUBO/Ising formulation
- Emits `*.quantum.json` sidecar for Device 46 quantum orchestrator

---

## Attribute Compatibility Matrix

| Attribute | Functions | Globals | main |
|-----------|-----------|---------|------|
| `dsmil_layer` | ✓ | ✓ | ✓ |
| `dsmil_device` | ✓ | ✓ | ✓ |
| `dsmil_clearance` | ✓ | ✗ | ✓ |
| `dsmil_roe` | ✓ | ✗ | ✓ |
| `dsmil_gateway` | ✓ | ✗ | ✗ |
| `dsmil_sandbox` | ✗ | ✗ | ✓ |
| `dsmil_stage` | ✓ | ✗ | ✓ |
| `dsmil_kv_cache` | ✓ | ✓ | ✗ |
| `dsmil_hot_model` | ✓ | ✓ | ✗ |
| `dsmil_quantum_candidate` | ✓ | ✗ | ✗ |

---

## Best Practices

### 1. Always Specify Layer & Device for Critical Code

```c
// Good
__attribute__((dsmil_layer(7)))
__attribute__((dsmil_device(47)))
void inference_critical(void) { /* ... */ }

// Bad - implicit layer 0
void inference_critical(void) { /* ... */ }
```

### 2. Use Gateway Functions for Boundary Crossings

```c
// Good
__attribute__((dsmil_gateway))
__attribute__((dsmil_layer(5)))
int validated_entry(void *user_data) {
    if (!validate(user_data)) return -EINVAL;
    return kernel_operation(user_data);
}

// Bad - implicit boundary crossing will fail verification
__attribute__((dsmil_layer(7)))
void user_function(void) {
    kernel_operation(data);  // ERROR: layer 7 → layer 5 without gateway
}
```

### 3. Tag Debug Code Appropriately

```c
// Good - won't be included in production
__attribute__((dsmil_stage("debug")))
void verbose_trace(void) { /* ... */ }

// Good - production path
__attribute__((dsmil_stage("serve")))
void fast_inference(void) { /* ... */ }
```

### 4. Combine Attributes for Full Context

```c
__attribute__((dsmil_layer(7)))
__attribute__((dsmil_device(47)))
__attribute__((dsmil_stage("quantized")))
__attribute__((dsmil_sandbox("l7_llm_worker")))
__attribute__((dsmil_clearance(0x07000000)))
__attribute__((dsmil_roe("ANALYSIS_ONLY")))
int main(int argc, char **argv) {
    // Fully annotated entry point
    return llm_worker_loop();
}
```

---

## Troubleshooting

### Error: "Layer boundary violation"

```
error: function 'foo' (layer 7) calls 'bar' (layer 3) without dsmil_gateway
```

**Solution**: Add `dsmil_gateway` to the callee or refactor to avoid cross-layer call.

### Error: "Stage policy violation"

```
error: production binary cannot link dsmil_stage("debug") code
```

**Solution**: Remove debug code from production build or use conditional compilation.

### Warning: "Missing layer attribute"

```
warning: function 'baz' has no dsmil_layer attribute, defaulting to layer 0
```

**Solution**: Add explicit `__attribute__((dsmil_layer(N)))` to function.

---

## Header File Reference

Include `<dsmil_attributes.h>` for convenient macro definitions:

```c
#include <dsmil_attributes.h>

DSMIL_LAYER(7)
DSMIL_DEVICE(47)
DSMIL_STAGE("serve")
void my_function(void) {
    // Equivalent to __attribute__((dsmil_layer(7))) etc.
}
```

---

## See Also

- [DSLLVM-DESIGN.md](DSLLVM-DESIGN.md) - Main design specification
- [PROVENANCE-CNSA2.md](PROVENANCE-CNSA2.md) - Security and provenance details
- [PIPELINES.md](PIPELINES.md) - Optimization pass pipelines

---

**End of Attributes Reference**

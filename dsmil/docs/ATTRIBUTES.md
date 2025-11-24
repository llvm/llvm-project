# DSMIL Attributes Reference
**Comprehensive Guide to DSMIL Source-Level Annotations**

Version: v1.2
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

### `dsmil_untrusted_input`

**Purpose**: Mark function parameters or globals that ingest untrusted data.

**Parameters**: None

**Applies to**: Function parameters, global variables

**Example**:
```c
// Mark parameter as untrusted
__attribute__((dsmil_untrusted_input))
void process_network_input(const char *user_data, size_t len) {
    // Must validate user_data before use
    if (!validate_input(user_data, len)) {
        return;
    }
    // Safe processing
}

// Mark global as untrusted
__attribute__((dsmil_untrusted_input))
char network_buffer[4096];
```

**IR Lowering**:
```llvm
!dsmil.untrusted_input = !{i1 true}
```

**Integration with AI Advisors**:
- Layer 8 Security AI can trace data flows from `dsmil_untrusted_input` sources
- Automatically detect flows into sensitive sinks (crypto operations, exec functions)
- Suggest additional validation or sandboxing for risky paths
- Combined with `dsmil-layer-check` to enforce information flow control

**Common Patterns**:
```c
// Network input
__attribute__((dsmil_untrusted_input))
ssize_t recv_from_network(void *buf, size_t len);

// File input
__attribute__((dsmil_untrusted_input))
void *load_config_file(const char *path);

// IPC input
__attribute__((dsmil_untrusted_input))
struct message *receive_ipc_message(void);
```

**Security Best Practices**:
1. Always validate untrusted input before use
2. Use sandboxed functions (`dsmil_sandbox`) to process untrusted data
3. Combine with `dsmil_gateway` for controlled transitions
4. Enable L8 security scan (`--ai-mode=advisor`) to detect flow violations

---

### `dsmil_secret`

**Purpose**: Mark cryptographic secrets and functions requiring constant-time execution to prevent side-channel attacks.

**Parameters**: None

**Applies to**: Function parameters, function return values, functions (entire body constant-time)

**Example**:
```c
// Mark function for constant-time enforcement
__attribute__((dsmil_secret))
void aes_encrypt(const uint8_t *key, const uint8_t *plaintext, uint8_t *ciphertext) {
    // All operations on key and derived values are constant-time
    // No secret-dependent branches or memory accesses allowed
}

// Mark specific parameters as secrets
void hmac_compute(
    __attribute__((dsmil_secret)) const uint8_t *key,
    size_t key_len,
    const uint8_t *message,
    size_t msg_len,
    uint8_t *mac
) {
    // Only 'key' parameter is tainted as secret
    // Branches on msg_len are allowed (public)
}

// Constant-time comparison
__attribute__((dsmil_secret))
int crypto_compare(const uint8_t *a, const uint8_t *b, size_t len) {
    int result = 0;
    for (size_t i = 0; i < len; i++) {
        result |= a[i] ^ b[i];  // Constant-time
    }
    return result;
}
```

**IR Lowering**:
```llvm
; On SSA values derived from secret parameters
!dsmil.secret = !{i1 true}

; After verification pass succeeds
!dsmil.ct_verified = !{i1 true}
```

**Constant-Time Enforcement**:

The `dsmil-ct-check` pass enforces strict constant-time guarantees:

1. **No Secret-Dependent Branches**:
   - ❌ `if (secret_byte & 0x01) { ... }`
   - ✓ `mask = -(secret_byte & 0x01); result = (result & ~mask) | (alternative & mask);`

2. **No Secret-Dependent Memory Access**:
   - ❌ `value = table[secret_index];`
   - ✓ Use constant-time lookup via masking or SIMD gather with fixed-time fallback

3. **No Variable-Time Instructions**:
   - ❌ `quotient = secret / divisor;` (division is variable-time)
   - ❌ `remainder = secret % modulus;` (modulo is variable-time)
   - ✓ Use whitelisted intrinsics: `__builtin_constant_time_select()`
   - ✓ Hardware AES-NI: `_mm_aesenc_si128()` is constant-time

**Violation Examples**:
```c
__attribute__((dsmil_secret))
void bad_crypto(const uint8_t *key) {
    // ERROR: secret-dependent branch
    if (key[0] == 0x00) {
        fast_path();
    } else {
        slow_path();
    }

    // ERROR: secret-dependent array indexing
    uint8_t sbox_value = sbox[key[1]];

    // ERROR: variable-time division
    uint32_t derived = key[2] / key[3];
}
```

**Allowed Patterns**:
```c
__attribute__((dsmil_secret))
void good_crypto(const uint8_t *key, const uint8_t *plaintext, size_t len) {
    // OK: Branching on public data (len)
    if (len < 16) {
        return;
    }

    // OK: Constant-time operations
    for (size_t i = 0; i < len; i++) {
        // XOR is constant-time
        plaintext[i] ^= key[i % 16];
    }

    // OK: Hardware crypto intrinsics (whitelisted)
    __m128i state = _mm_loadu_si128((__m128i*)plaintext);
    __m128i round_key = _mm_loadu_si128((__m128i*)key);
    state = _mm_aesenc_si128(state, round_key);
}
```

**AI Integration**:

* **Layer 8 Security AI** performs deep analysis of `dsmil_secret` functions:
  - Identifies potential cache-timing vulnerabilities
  - Detects power analysis risks
  - Suggests constant-time alternatives for flagged patterns
  - Validates that suggested mitigations are side-channel resistant

* **Layer 5 Performance AI** balances security with performance:
  - Recommends AVX-512 constant-time implementations where beneficial
  - Suggests hardware-accelerated options (AES-NI, SHA extensions)
  - Provides performance estimates for constant-time vs variable-time implementations

**Policy Enforcement**:

* Functions in **Layers 8–9** (Security/Executive) with `dsmil_sandbox("crypto_worker")` **must** use `dsmil_secret` for:
  - All key material (symmetric keys, private keys)
  - Key derivation operations
  - Signature generation (not verification, which can be variable-time)
  - Decryption operations (encryption can be variable-time for some schemes)

* **Production builds** (`DSMIL_PRODUCTION=1`):
  - Violations trigger **compile-time errors**
  - No binary generated if constant-time check fails

* **Lab builds** (`--ai-mode=lab`):
  - Violations emit **warnings only**
  - Binary generated with metadata marking unverified functions

**Metadata**:

After successful verification:
```json
{
  "symbol": "aes_encrypt",
  "layer": 8,
  "device_id": 80,
  "security": {
    "constant_time": true,
    "verified_by": "dsmil-ct-check v1.2",
    "verification_date": "2025-11-24T10:30:00Z",
    "l8_scan_score": 0.95,
    "side_channel_resistant": true
  }
}
```

**Common Use Cases**:

```c
// Cryptographic primitives (Layer 8)
DSMIL_LAYER(8) DSMIL_DEVICE(80)
__attribute__((dsmil_secret))
void sha384_compress(const uint8_t *key, uint8_t *state);

// Key exchange (Layer 8)
DSMIL_LAYER(8) DSMIL_DEVICE(81)
__attribute__((dsmil_secret))
int ml_kem_1024_decapsulate(const uint8_t *sk, const uint8_t *ct, uint8_t *shared);

// Signature generation (Layer 9)
DSMIL_LAYER(9) DSMIL_DEVICE(90)
__attribute__((dsmil_secret))
int ml_dsa_87_sign(const uint8_t *sk, const uint8_t *msg, size_t len, uint8_t *sig);

// Constant-time string comparison
DSMIL_LAYER(8)
__attribute__((dsmil_secret))
int secure_memcmp(const void *a, const void *b, size_t n);
```

**Relationship with Other Attributes**:

* Combine with `dsmil_sandbox("crypto_worker")` for defense-in-depth:
  ```c
  DSMIL_LAYER(8) DSMIL_DEVICE(80) DSMIL_SANDBOX("crypto_worker")
  __attribute__((dsmil_secret))
  int main(void) {
      // Sandboxed + constant-time enforced
      return crypto_service_loop();
  }
  ```

* Orthogonal to `dsmil_untrusted_input`:
  - `dsmil_secret`: Protects secrets from leaking via timing
  - `dsmil_untrusted_input`: Tracks untrusted data to prevent injection attacks
  - Combined: Safe handling of secrets in presence of untrusted input

**Performance Considerations**:

* Constant-time enforcement typically adds **5-15% overhead** for crypto operations
* Hardware-accelerated paths (AES-NI, SHA-NI) remain **near-zero overhead**
* Layer 5 AI can identify cases where constant-time is unnecessary (e.g., already using hardware crypto)

**Debugging**:

Enable verbose constant-time checking:
```bash
dsmil-clang -mllvm -dsmil-ct-check-verbose=1 \
  -mllvm -dsmil-ct-show-violations=1 \
  crypto.c -o crypto.o
```

Output shows detailed taint propagation and violation locations with suggested fixes.

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
| `dsmil_untrusted_input` | ✓ (params) | ✓ | ✗ |
| `dsmil_secret` (v1.2) | ✓ (params/return) | ✗ | ✓ |
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

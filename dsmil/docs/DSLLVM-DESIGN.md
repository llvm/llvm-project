# DSLLVM Design Specification
**DSMIL-Optimized LLVM Toolchain for Intel Meteor Lake**

Version: v1.0
Status: Draft
Owner: SWORDIntel / DSMIL Kernel Team

---

## 0. Scope & Intent

DSLLVM is a hardened LLVM/Clang toolchain specialized for the **DSMIL kernel + userland stack** on Intel Meteor Lake (CPU + NPU + Arc GPU), with:

1. **DSMIL-aware hardware target & optimal flags** tuned for Meteor Lake.
2. **DSMIL semantic metadata** baked into LLVM IR (layers, devices, ROE, clearance).
3. **Bandwidth & memory-aware optimization** tailored to realistic hardware limits.
4. **MLOps stage-awareness** for AI/LLM workloads (pretrain/finetune/serve, quantized/distilled, etc.).
5. **CNSA 2.0–compatible provenance & sandbox integration**
   - **SHA-384** (hash), **ML-DSA-87** (signature), **ML-KEM-1024** (KEM).
6. **Quantum-assisted optimization hooks** (Device 46, Qiskit-compatible side outputs).
7. **Complete packaging & tooling**: wrappers, pass pipelines, repo layout, and CI integration.

DSLLVM does *not* invent a new language. It extends LLVM/Clang with attributes, metadata, passes, and ELF/sidecar outputs that align with the DSMIL 9-layer / 104-device architecture and its MLOps pipeline.

---

## 1. DSMIL Hardware Target Integration

### 1.1 Target Triple & Subtarget

Introduce a dedicated target triple:

- `x86_64-dsmil-meteorlake-elf`

Characteristics:

- Base ABI: x86-64 SysV (compatible with mainstream Linux).
- Default CPU: `meteorlake`.
- Default features (`+dsmil-optimal`):

  - AVX2, AVX-VNNI
  - AES, VAES, SHA, GFNI
  - BMI1/2, POPCNT, FMA
  - MOVDIRI, WAITPKG
  - Other Meteor Lake–specific micro-optimizations when available.

This matches and centralizes the "optimal flags" we otherwise would repeat in `CFLAGS/LDFLAGS`.

### 1.2 Frontend Wrappers

Provide thin wrappers that always select the DSMIL target:

- `dsmil-clang`
- `dsmil-clang++`
- `dsmil-llc`

Default options baked into wrappers:

- `-target x86_64-dsmil-meteorlake-elf`
- `-march=meteorlake -mtune=meteorlake`
- `-O3 -pipe -fomit-frame-pointer -funroll-loops -fstrict-aliasing -fno-plt`
- `-ffunction-sections -fdata-sections -flto=auto`

These wrappers become the **canonical toolchain** for DSMIL kernel, drivers, and userland components.

### 1.3 Device-Aware Code Model

DSMIL defines 9 layers and 104 devices. DSLLVM integrates this via a **DSMIL code model**:

- Each function may carry:

  - `device_id` (0–103)
  - `layer` (0–8 or 1–9)
  - `role` (e.g. `control`, `llm_worker`, `crypto`, `telemetry`)

- Backend uses this to:

  - Place functions in per-device/ per-layer sections:
    - `.text.dsmil.dev47`, `.text.dsmil.layer7`, `.data.dsmil.dev12`, …
  - Emit a sidecar mapping file (`*.dsmilmap`) describing symbol → layer/device/role.

This enables the runtime, scheduler, and observability stack to understand code placement without extra scanning.

---

## 2. DSMIL Semantic Metadata in IR

### 2.1 Source-Level Attributes

Expose portable C/C++ attributes to encode DSMIL semantics at the source level:

```c
__attribute__((dsmil_layer(7)))
__attribute__((dsmil_device(47)))
__attribute__((dsmil_clearance(0x07070707)))
__attribute__((dsmil_roe("ANALYSIS_ONLY")))
__attribute__((dsmil_gateway))
__attribute__((dsmil_sandbox("l7_llm_worker")))
__attribute__((dsmil_stage("quantized")))
__attribute__((dsmil_kv_cache))
__attribute__((dsmil_hot_model))
```

Key attributes:

* `dsmil_layer(int)` – DSMIL layer index (0–8 or 1–9).
* `dsmil_device(int)` – DSMIL device id (0–103).
* `dsmil_clearance(uint32)` – 32-bit clearance / compartment mask.
* `dsmil_roe(string)` – Rules of Engagement (e.g. `ANALYSIS_ONLY`, `LIVE_CONTROL`).
* `dsmil_gateway` – function is authorized to cross layer or device boundaries.
* `dsmil_sandbox(string)` – role-based sandbox profile name.
* `dsmil_stage(string)` – MLOps stage (`pretrain`, `finetune`, `quantized`, `distilled`, `serve`, `debug`, etc.).
* `dsmil_kv_cache` – marks KV-cache storage.
* `dsmil_hot_model` – marks hot-path model weights.

### 2.2 IR Metadata Schema

Front-end lowers attributes to LLVM metadata:

For functions:

* `!dsmil.layer = i32 7`
* `!dsmil.device_id = i32 47`
* `!dsmil.clearance = i32 0x07070707`
* `!dsmil.roe = !"ANALYSIS_ONLY"`
* `!dsmil.gateway = i1 true`
* `!dsmil.sandbox = !"l7_llm_worker"`
* `!dsmil.stage = !"quantized"`
* `!dsmil.memory_class = !"kv_cache"` (for `dsmil_kv_cache`)

For globals:

* `!dsmil.sensitivity = !"MODEL_WEIGHTS"`
* `!dsmil.memory_class = !"hot_model"`

### 2.3 Verification Pass: `dsmil-layer-check`

Add a module pass: **`dsmil-layer-check`** that:

* Walks the call graph and verifies:

  * Disallowed layer transitions (e.g. low → high without `dsmil_gateway`) are rejected.
  * Functions with lower `dsmil_clearance` cannot call higher-clearance functions unless flagged as an explicit gateway with ROE.
  * ROE transitions follow a policy (e.g. `ANALYSIS_ONLY` cannot escalate into `LIVE_CONTROL` code without explicit exemption metadata).

* On violation:

  * Emit detailed diagnostics (file, function, caller→callee, layer/clearance values).
  * Optionally generate a JSON report (`*.dsmilviolations.json`) for CI.

This ensures DSMIL layering and clearance policies are enforced **at compile-time**, not just at runtime.

---

## 3. Bandwidth & Memory-Aware Optimization

### 3.1 Bandwidth Cost Model: `dsmil-bandwidth-estimate`

Introduce mid-end analysis pass **`dsmil-bandwidth-estimate`**:

* For each function, compute:

  * Approximate `bytes_read`, `bytes_written` (per invocation).
  * Vectorization characteristics (SSE, AVX2, AVX-VNNI use).
  * Access patterns (contiguous vs strided, gather/scatter hints).

* Derive:

  * `bw_gbps_estimate` under an assumed memory model (e.g. 64 GB/s).
  * `memory_class` labels such as:

    * `kv_cache`
    * `model_weights`
    * `hot_ram`
    * `cold_storage`

* Attach metadata:

  * `!dsmil.bw_bytes_read`
  * `!dsmil.bw_bytes_written`
  * `!dsmil.bw_gbps_estimate`
  * `!dsmil.memory_class`

### 3.2 Placement & Hints: `dsmil-device-placement`

Add pass **`dsmil-device-placement`** (mid-end or LTO):

* Uses:

  * DSMIL semantic metadata (layer, device, sensitivity).
  * Bandwidth estimates.

* Computes:

  * Recommended execution target per function:

    * `"cpu"`, `"npu"`, `"gpu"`, `"hybrid"`
  * Recommended memory tier:

    * `"ramdisk"`, `"tmpfs"`, `"local_ssd"`, `"remote_minio"`, etc.

* Encodes this in:

  * IR metadata: `!dsmil.placement` = !"{target: npu, memory: ramdisk}"
  * Sidecar file (see next section).

### 3.3 Sidecar Mapping File: `*.dsmilmap`

For each linked binary, emit `binary_name.dsmilmap` (JSON or CBOR):

Example entry:

```json
{
  "symbol": "llm_decode_step",
  "layer": 7,
  "device_id": 47,
  "clearance": "0x07070707",
  "stage": "serve",
  "bw_gbps_estimate": 23.5,
  "memory_class": "kv_cache",
  "placement": {
    "target": "npu",
    "memory_tier": "ramdisk"
  }
}
```

This file is consumed by:

* DSMIL orchestrator / scheduler.
* MLOps stack.
* Observability and audit tooling.

---

## 4. MLOps Stage-Aware Compilation

### 4.1 Stage Semantics: `dsmil_stage`

`__attribute__((dsmil_stage("...")))` encodes MLOps lifecycle information:

Examples:

* `"pretrain"` – Pre-training phase code/artifacts.
* `"finetune"` – Fine-tuning for specific tasks.
* `"quantized"` – Quantized model code (INT8/INT4, etc.).
* `"distilled"` – Distilled/compact models.
* `"serve"` – Serving / inference path.
* `"debug"` – Debug-only diagnostics.
* `"experimental"` – Non-production experiments.

### 4.2 Policy Pass: `dsmil-stage-policy`

Add pass **`dsmil-stage-policy`** that validates stage usage:

Policy examples (configurable):

* **Production binaries (`DSMIL_PRODUCTION`):**

  * No `debug` or `experimental` stages allowed.
  * L≥3 must not link untagged or `pretrain` code.
  * L≥3 LLM workloads must be `quantized` or `distilled`.

* **Sandbox / lab binaries:**

  * Allow more flexibility but log stage mixes.

On violation:

* Emit compile-time errors or warnings depending on policy strictness.
* Generate `*.dsmilstage-report.json` for CI.

### 4.3 Pipeline Integration

The `*.dsmilmap` entries include `stage` per symbol. MLOps uses it to:

* Select deployment targets (training cluster vs serving edge).
* Enforce that only compliant artifacts are deployed to production.
* Drive automated quantization/optimization pipelines (if `stage != quantized`, schedule quantization job).

---

## 5. CNSA 2.0 Provenance & Sandbox Integration

**Objectives:**

* Provide strong, CNSA 2.0–aligned provenance for each binary:

  * **Hash:** SHA-384
  * **Signature:** ML-DSA-87
  * **KEM:** ML-KEM-1024 (for optional confidentiality of provenance/policy data).
* Provide standardized, attribute-driven sandboxing using libcap-ng + seccomp.

### 5.1 Cryptographic Roles & Keys

Logical key roles:

1. **Toolchain Signing Key (TSK)**

   * Algorithm: ML-DSA-87
   * Used to sign:

     * DSLLVM release manifests (optional).
     * Toolchain provenance if desired.

2. **Project Signing Key (PSK)**

   * Algorithm: ML-DSA-87
   * One per project/product line.
   * Used to sign each binary's provenance.

3. **Runtime Decryption Key (RDK)**

   * Algorithm: ML-KEM-1024
   * Used by DSMIL runtime components (kernel/LSM/loader) to decapsulate symmetric keys for decrypting sensitive provenance/policy blobs.

All hashing: **SHA-384**.

### 5.2 Provenance Record Lifecycle

At link-time, DSLLVM produces a **provenance record**:

1. Construct logical object:

   ```json
   {
     "schema": "dsmil-provenance-v1",
     "compiler": {
       "name": "dsmil-clang",
       "version": "X.Y.Z",
       "target": "x86_64-dsmil-meteorlake-elf"
     },
     "source": {
       "vcs": "git",
       "repo": "https://github.com/SWORDIntel/...",
       "commit": "abcd1234...",
       "dirty": false
     },
     "build": {
       "timestamp": "...",
       "builder_id": "build-node-01",
       "flags": ["-O3", "-march=meteorlake", "..."]
     },
     "dsmil": {
       "default_layer": 7,
       "default_device": 47,
       "roles": ["llm_worker", "control_plane"]
     },
     "hashes": {
       "binary_sha384": "…",
       "sections": {
         ".text": "…",
         ".rodata": "…"
       }
     }
   }
   ```

2. Canonicalize structure → `prov_canonical` (e.g., deterministic JSON or CBOR).

3. Compute `H = SHA-384(prov_canonical)`.

4. Sign `H` using ML-DSA-87 with PSK → signature `σ`.

5. Produce final record:

   ```json
   {
     "prov": { ... },
     "hash_alg": "SHA-384",
     "sig_alg": "ML-DSA-87",
     "sig": "…"
   }
   ```

6. Embed in ELF:

   * `.note.dsmil.provenance` (compact format, possibly CBOR)
   * Optionally a dedicated loadable segment `.dsmil_prov`.

### 5.3 Optional Confidentiality With ML-KEM-1024

For high-sensitivity environments:

1. Generate symmetric key `K`.

2. Encrypt `prov` (or part of it) using AEAD (e.g., AES-256-GCM) with key `K`.

3. Encapsulate `K` using ML-KEM-1024 RDK public key → ciphertext `ct`.

4. Wrap structure:

   ```json
   {
     "enc_prov": "…",            // AEAD ciphertext + tag
     "kem_alg": "ML-KEM-1024",
     "kem_ct": "…",
     "hash_alg": "SHA-384",
     "sig_alg": "ML-DSA-87",
     "sig": "…"
   }
   ```

5. Embed into ELF sections as above.

This ensures only entities that hold the **RDK private key** can decrypt provenance while validation remains globally verifiable.

### 5.4 Runtime Validation

On `execve` or kernel module load, DSMIL loader/LSM:

1. Extract `.note.dsmil.provenance` / `.dsmil_prov`.

2. If encrypted:

   * Decapsulate `K` using ML-KEM-1024.
   * Decrypt AEAD payload.

3. Recompute SHA-384 hash over canonicalized provenance.

4. Verify ML-DSA-87 signature against PSK (and optionally TSK trust chain).

5. If validation fails:

   * Deny execution or require explicit emergency override.

6. If validation succeeds:

   * Expose trusted provenance to:

     * Policy engine for layer/role enforcement.
     * Audit/forensics systems.

### 5.5 Sandbox Wrapping: `dsmil_sandbox`

Attribute:

```c
__attribute__((dsmil_sandbox("l7_llm_worker")))
int main(int argc, char **argv);
```

Link-time pass **`dsmil-sandbox-wrap`**:

* Renames original `main` → `main_real`.
* Injects wrapper `main` that:

  * Applies a role-specific **capability profile** using libcap-ng.
  * Installs a role-specific **seccomp** filter (predefined profile tied to sandbox name).
  * Optionally loads runtime policy derived from provenance (which may have been decrypted via ML-KEM-1024).
  * Calls `main_real()`.

Provenance record includes:

* `sandbox_profile = "l7_llm_worker"`

This provides standardized, role-based sandbox behavior across DSMIL binaries with **minimal developer burden**.

---

## 6. Quantum-Assisted Optimization Hooks (Device 46)

Device 46 is reserved for **quantum integration / experimental optimization**. DSLLVM provides hooks without coupling production code to quantum tooling.

### 6.1 Quantum Candidate Tagging

Attribute:

```c
__attribute__((dsmil_quantum_candidate("placement")))
void placement_solver(...);
```

Semantics:

* Marks a function as a **candidate for quantum optimization / offload**.
* Optional string differentiates class of problem:

  * `"placement"` (model/device placement).
  * `"routing"` (network path selection).
  * `"schedule"` (job scheduling).
  * `"hyperparam_search"` (hyperparameter tuning).

Lowered metadata:

* `!dsmil.quantum_candidate = !"placement"`

### 6.2 Problem Extraction Pass: `dsmil-quantum-export`

Pass **`dsmil-quantum-export`**:

* For each `dsmil_quantum_candidate`:

  * Analyze function and extract:

    * Variables and constraints representing optimization problem, where feasible.
    * Map to QUBO/Ising style formulation when patterns match known templates.

* Emit sidecar files per binary:

  * `binary_name.quantum.json` (or `.yaml` / `.qubo`) describing problem instances.

Example structure:

```json
{
  "schema": "dsmil-quantum-v1",
  "binary": "scheduler.bin",
  "functions": [
    {
      "name": "placement_solver",
      "kind": "placement",
      "representation": "qubo",
      "qubo": {
        "Q": [[0, 1], [1, 0]],
        "variables": ["model_1_device_47", "model_1_device_12"]
      }
    }
  ]
}
```

### 6.3 External Quantum Flow

* DSLLVM itself remains classical.
* External **Quantum Orchestrator (Device 46)**:

  * Consumes `*.quantum.json` / `.qubo`.
  * Maps problems into Qiskit/other frameworks.
  * Runs VQE/QAOA/other routines.
  * Writes back improved parameters / mappings as:

    * `*.quantum_solution.json` that DSMIL runtime or next build can ingest.

This allows iterative improvement of placement/scheduling/hyperparameters using quantum tooling without destabilizing the core toolchain.

---

## 7. Tooling, Packaging & Repo Layout

### 7.1 CLI Tools & Wrappers

Provide the following user-facing tools:

* `dsmil-clang`, `dsmil-clang++`, `dsmil-llc`

  * Meteor Lake + DSMIL defaults baked in.

* `dsmil-opt`

  * Wrapper around `opt` with DSMIL pass pipeline presets.

* `dsmil-verify`

  * High-level command that:

    * Runs provenance verification on binaries.
    * Checks DSMIL layer policy, stage policy, and sandbox config.
    * Outputs human-readable and JSON summaries.

### 7.2 Standard Pass Pipelines

Recommended default pass pipeline for **production DSMIL binary**:

1. Standard LLVM optimization pipeline (`-O3`).
2. DSMIL passes (order approximate):

   * `dsmil-bandwidth-estimate`
   * `dsmil-device-placement`
   * `dsmil-layer-check`
   * `dsmil-stage-policy`
   * `dsmil-quantum-export` (for tagged functions)
   * `dsmil-sandbox-wrap` (LTO / link stage)
   * `dsmil-provenance-emit` (CNSA 2.0 record generation)

Expose as shorthand:

* `-fpass-pipeline=dsmil-default`
* `-fpass-pipeline=dsmil-debug` (less strict)
* `-fpass-pipeline=dsmil-lab` (no enforcement, just annotation).

### 7.3 Repository Layout (Proposed)

```text
DSLLVM/
├─ dsmil/
│  ├─ cmake/                      # CMake integration, target definitions
│  ├─ docs/
│  │  ├─ DSLLVM-DESIGN.md         # This specification
│  │  ├─ PROVENANCE-CNSA2.md      # Deep dive on CNSA 2.0 crypto flows
│  │  ├─ ATTRIBUTES.md            # Reference for dsmil_* attributes
│  │  └─ PIPELINES.md             # Pass pipeline presets
│  ├─ include/
│  │  ├─ dsmil_attributes.h       # C/C++ attribute macros / annotations
│  │  ├─ dsmil_provenance.h       # Structures / helpers for provenance
│  │  └─ dsmil_sandbox.h          # Role-based sandbox helper declarations
│  ├─ lib/
│  │  ├─ Target/
│  │  │  └─ X86/
│  │  │     └─ DSMILTarget.cpp    # meteorlake+dsmil target integration
│  │  ├─ Passes/
│  │  │  ├─ DsmilBandwidthPass.cpp
│  │  │  ├─ DsmilDevicePlacementPass.cpp
│  │  │  ├─ DsmilLayerCheckPass.cpp
│  │  │  ├─ DsmilStagePolicyPass.cpp
│  │  │  ├─ DsmilQuantumExportPass.cpp
│  │  │  ├─ DsmilSandboxWrapPass.cpp
│  │  │  └─ DsmilProvenancePass.cpp
│  │  └─ Runtime/
│  │     ├─ dsmil_sandbox_runtime.c
│  │     └─ dsmil_provenance_runtime.c
│  ├─ tools/
│  │  ├─ dsmil-clang/             # Wrapper frontends
│  │  ├─ dsmil-llc/
│  │  ├─ dsmil-opt/
│  │  └─ dsmil-verify/
│  └─ test/
│     ├─ dsmil/
│     │  ├─ layer_policies/
│     │  ├─ stage_policies/
│     │  ├─ provenance/
│     │  └─ sandbox/
│     └─ lit.cfg.py
```

### 7.4 CI / CD & Policy Enforcement

* **Build matrix**:

  * `Release`, `RelWithDebInfo` for DSMIL target.
  * Linux x86-64 builders with Meteor Lake-like flags.

* **CI checks**:

  1. Build DSLLVM and run internal test suite.
  2. Compile sample DSMIL workloads:

     * Kernel module sample.
     * L7 LLM worker.
     * Crypto worker.
     * Telemetry agent.
  3. Run `dsmil-verify` against produced binaries:

     * Confirm provenance is valid (CNSA 2.0).
     * Confirm layer/stage policies pass.
     * Confirm sandbox profiles present for configured roles.

* **Artifacts**:

  * Publish:

    * Toolchain tarballs / packages.
    * Reference `*.dsmilmap` and `.quantum.json` outputs for sample binaries.

---

## Appendix A – Attribute Summary

Quick reference:

* `dsmil_layer(int)`
* `dsmil_device(int)`
* `dsmil_clearance(uint32)`
* `dsmil_roe(const char*)`
* `dsmil_gateway`
* `dsmil_sandbox(const char*)`
* `dsmil_stage(const char*)`
* `dsmil_kv_cache`
* `dsmil_hot_model`
* `dsmil_quantum_candidate(const char*)`

---

## Appendix B – DSMIL Pass Summary

* `dsmil-bandwidth-estimate`

  * Estimate data movement and bandwidth per function.

* `dsmil-device-placement`

  * Suggest CPU/NPU/GPU target + memory tier.

* `dsmil-layer-check`

  * Enforce DSMIL layer/clearance/ROE constraints.

* `dsmil-stage-policy`

  * Enforce MLOps stage policies for binaries.

* `dsmil-quantum-export`

  * Export QUBO/Ising-style problems for quantum optimization.

* `dsmil-sandbox-wrap`

  * Insert sandbox setup wrappers around `main` based on `dsmil_sandbox`.

* `dsmil-provenance-pass`

  * Generate CNSA 2.0 provenance with SHA-384 + ML-DSA-87, optional ML-KEM-1024.

---

## Appendix C – Integration Roadmap

### Phase 1: Foundation (Weeks 1-4)

1. **Target Integration**
   * Add `x86_64-dsmil-meteorlake-elf` target triple to LLVM
   * Configure Meteor Lake feature flags
   * Create basic wrapper scripts

2. **Attribute Framework**
   * Implement C/C++ attribute parsing in Clang
   * Define IR metadata schema
   * Add metadata emission in CodeGen

### Phase 2: Core Passes (Weeks 5-10)

1. **Analysis Passes**
   * Implement `dsmil-bandwidth-estimate`
   * Implement `dsmil-device-placement`

2. **Verification Passes**
   * Implement `dsmil-layer-check`
   * Implement `dsmil-stage-policy`

### Phase 3: Advanced Features (Weeks 11-16)

1. **Provenance System**
   * Integrate CNSA 2.0 cryptographic libraries
   * Implement `dsmil-provenance-pass`
   * Add ELF section emission

2. **Sandbox Integration**
   * Implement `dsmil-sandbox-wrap`
   * Create runtime library components

### Phase 4: Quantum & Tooling (Weeks 17-20)

1. **Quantum Hooks**
   * Implement `dsmil-quantum-export`
   * Define output formats

2. **User Tools**
   * Implement `dsmil-verify`
   * Create comprehensive test suite
   * Documentation and examples

### Phase 5: Hardening & Deployment (Weeks 21-24)

1. **Testing & Validation**
   * Comprehensive integration tests
   * Performance benchmarking
   * Security audit

2. **CI/CD Integration**
   * Automated builds
   * Policy validation
   * Release packaging

---

## Appendix D – Security Considerations

### Threat Model

1. **Supply Chain Attacks**
   * Mitigation: CNSA 2.0 provenance with ML-DSA-87 signatures
   * All binaries must have valid signatures from trusted PSK

2. **Layer Boundary Violations**
   * Mitigation: Compile-time `dsmil-layer-check` enforcement
   * Runtime validation via provenance

3. **Privilege Escalation**
   * Mitigation: `dsmil-sandbox-wrap` with libcap-ng + seccomp
   * ROE policy enforcement

4. **Side-Channel Attacks**
   * Consideration: Constant-time crypto operations in provenance system
   * Metadata encryption via ML-KEM-1024 for sensitive deployments

### Compliance

* **CNSA 2.0**: SHA-384, ML-DSA-87, ML-KEM-1024
* **FIPS 140-3**: When using approved crypto implementations
* **Common Criteria**: EAL4+ target for provenance system

---

## Appendix E – Performance Considerations

### Compilation Overhead

* **Metadata Emission**: <1% overhead
* **Analysis Passes**: 2-5% compilation time increase
* **Provenance Generation**: 1-3% link time increase
* **Total**: <10% increase in build times

### Runtime Overhead

* **Provenance Validation**: One-time cost at program load (~10-50ms)
* **Sandbox Setup**: One-time cost at program start (~5-20ms)
* **Metadata Access**: Zero runtime overhead (compile-time only)

### Memory Overhead

* **Binary Size**: +5-15% (metadata, provenance sections)
* **Sidecar Files**: ~1-5 KB per binary (`.dsmilmap`, `.quantum.json`)

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| v1.0 | 2025-11-24 | SWORDIntel/DSMIL Team | Initial specification |

---

**End of Specification**

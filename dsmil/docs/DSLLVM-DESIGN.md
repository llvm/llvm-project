# DSLLVM Design Specification
**DSMIL-Optimized LLVM Toolchain for Intel Meteor Lake**

Version: v1.2
Status: Draft
Owner: SWORDIntel / DSMIL Kernel Team

---

## 0. Scope & Intent

DSLLVM is a hardened LLVM/Clang toolchain specialized for the **DSMIL kernel + userland stack** on Intel Meteor Lake (CPU + NPU + Arc GPU), tightly integrated with the **DSMIL AI architecture (Layers 3–9, 48 AI devices, ~1338 TOPS INT8)**.

Primary capabilities:

1. **DSMIL-aware hardware target & optimal flags** for Meteor Lake.
2. **DSMIL semantic metadata** in LLVM IR (layers, devices, ROE, clearance).
3. **Bandwidth & memory-aware optimization** tuned to realistic hardware limits.
4. **MLOps stage-awareness** for AI/LLM workloads.
5. **CNSA 2.0–compatible provenance & sandbox integration**
   - SHA-384, ML-DSA-87, ML-KEM-1024.
6. **Quantum-assisted optimization hooks** (Layer 7, Device 46).
7. **Tooling/packaging** for passes, wrappers, and CI.
8. **AI-assisted compilation via DSMIL Layers 3–9** (LLMs, security AI, forecasting).
9. **AI-trained cost models & schedulers** for device/placement decisions.
10. **AI integration modes & guardrails** to keep toolchain deterministic and auditable.
11. **Constant-time enforcement (`dsmil_secret`)** for cryptographic side-channel safety.
12. **Quantum optimization hints** integrated into AI advisor I/O pipeline.
13. **Compact ONNX feature scoring** on Devices 43-58 for sub-millisecond cost model inference.

DSLLVM does *not* invent a new language. It extends LLVM/Clang with attributes, metadata, passes, ELF extensions, AI-powered advisors, and sidecar outputs aligned with the DSMIL 9-layer / 104-device architecture.

---

## 1. DSMIL Hardware Target Integration

### 1.1 Target Triple & Subtarget

Dedicated target triple:

- `x86_64-dsmil-meteorlake-elf`

Characteristics:

- Base ABI: x86-64 SysV (Linux-compatible).
- Default CPU: `meteorlake`.
- Default features (grouped as `+dsmil-optimal`):

  - AVX2, AVX-VNNI
  - AES, VAES, SHA, GFNI
  - BMI1/2, POPCNT, FMA
  - MOVDIRI, WAITPKG

This centralizes the "optimal flags" that would otherwise be replicated in `CFLAGS/LDFLAGS`.

### 1.2 Frontend Wrappers

Thin wrappers:

- `dsmil-clang`
- `dsmil-clang++`
- `dsmil-llc`

Default options baked in:

- `-target x86_64-dsmil-meteorlake-elf`
- `-march=meteorlake -mtune=meteorlake`
- `-O3 -pipe -fomit-frame-pointer -funroll-loops -fstrict-aliasing -fno-plt`
- `-ffunction-sections -fdata-sections -flto=auto`

These wrappers are the **canonical toolchain** for DSMIL kernel, drivers, agents, and userland.

### 1.3 Device-Aware Code Model

DSMIL defines **9 layers (3–9) and 104 devices**, with 48 AI devices and ~1338 TOPS across Layers 3–9.

DSLLVM adds a **DSMIL code model**:

- Per function, optional fields:

  - `layer` (3–9)
  - `device_id` (0–103)
  - `role` (e.g. `control`, `llm_worker`, `crypto`, `telemetry`)

Backend uses these to:

- Place functions in device/layer-specific sections:
  - `.text.dsmil.dev47`, `.data.dsmil.layer7`, etc.
- Emit a sidecar map (`*.dsmilmap`) linking symbols to layer/device/role.

---

## 2. DSMIL Semantic Metadata in IR

### 2.1 Source-Level Attributes

C/C++ attributes:

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
__attribute__((dsmil_quantum_candidate("placement")))
__attribute__((dsmil_untrusted_input))
```

Semantics:

* `dsmil_layer(int)` – DSMIL layer index.
* `dsmil_device(int)` – DSMIL device ID.
* `dsmil_clearance(uint32)` – clearance/compartment mask.
* `dsmil_roe(string)` – Rules of Engagement profile.
* `dsmil_gateway` – legal cross-layer/device boundary.
* `dsmil_sandbox(string)` – role-based sandbox profile.
* `dsmil_stage(string)` – MLOps stage.
* `dsmil_kv_cache` / `dsmil_hot_model` – memory-class hints.
* `dsmil_quantum_candidate(string)` – candidate for quantum optimization.
* `dsmil_untrusted_input` – marks parameters/globals that ingest untrusted data.

### 2.2 IR Metadata Schema

Front-end lowers to metadata:

* Functions:

  * `!dsmil.layer = i32 7`
  * `!dsmil.device_id = i32 47`
  * `!dsmil.clearance = i32 0x07070707`
  * `!dsmil.roe = !"ANALYSIS_ONLY"`
  * `!dsmil.gateway = i1 true`
  * `!dsmil.sandbox = !"l7_llm_worker"`
  * `!dsmil.stage = !"quantized"`
  * `!dsmil.memory_class = !"kv_cache"`
  * `!dsmil.untrusted_input = i1 true`

* Globals:

  * `!dsmil.sensitivity = !"MODEL_WEIGHTS"`

### 2.3 Verification Pass: `dsmil-layer-check`

Module pass **`dsmil-layer-check`**:

* Walks the call graph; rejects:

  * Illegal layer transitions without `dsmil_gateway`.
  * Clearance violations (low→high without gateway/ROE).
  * ROE transitions that break policy (configurable).

* Outputs:

  * Diagnostics (file/function, caller→callee, layer/clearance).
  * Optional `*.dsmilviolations.json` for CI.

---

## 3. Bandwidth & Memory-Aware Optimization

### 3.1 Bandwidth Cost Model: `dsmil-bandwidth-estimate`

Pass **`dsmil-bandwidth-estimate`**:

* Estimates per function:

  * `bytes_read`, `bytes_written`
  * vectorization level (SSE/AVX/AMX)
  * access patterns (contiguous/strided/gather-scatter)

* Derives:

  * `bw_gbps_estimate` (for the known memory model).
  * `memory_class` (`kv_cache`, `model_weights`, `hot_ram`, etc.).

* Attaches:

  * `!dsmil.bw_bytes_read`, `!dsmil.bw_bytes_written`
  * `!dsmil.bw_gbps_estimate`
  * `!dsmil.memory_class`

### 3.2 Placement & Hints: `dsmil-device-placement`

Pass **`dsmil-device-placement`**:

* Uses:

  * DSMIL semantic metadata.
  * Bandwidth estimates.
  * (Optionally) AI-trained cost model, see §9.

* Computes recommended:

  * `target`: `cpu`, `npu`, `gpu`, `hybrid`.
  * `memory_tier`: `ramdisk`, `tmpfs`, `local_ssd`, etc.

* Encodes in:

  * IR (`!dsmil.placement`)
  * `*.dsmilmap` sidecar.

### 3.3 Sidecar Mapping File: `*.dsmilmap`

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

Consumed by DSMIL orchestrator, MLOps, and observability tooling.

---

## 4. MLOps Stage-Aware Compilation

### 4.1 `dsmil_stage` Semantics

Stages (examples):

* `pretrain`, `finetune`
* `quantized`, `distilled`
* `serve`
* `debug`, `experimental`

### 4.2 Policy Pass: `dsmil-stage-policy`

Pass **`dsmil-stage-policy`** enforces rules, e.g.:

* Production (`DSMIL_PRODUCTION`):

  * Disallow `debug` or `experimental`.
  * Layers ≥3 must not link `pretrain` stage.
  * LLM workloads in Layers 7/9 must be `quantized` or `distilled`.

* Lab builds: warn only.

Violations:

* Compiler errors/warnings.
* `*.dsmilstage-report.json` for CI.

### 4.3 Pipeline Integration

`*.dsmilmap` includes `stage`. MLOps uses this to:

* Decide training vs serving deployment.
* Enforce only compliant artifacts reach Layers 7–9 (LLMs, exec AI).

---

## 5. CNSA 2.0 Provenance & Sandbox Integration

### 5.1 Crypto Roles & Keys

* **TSK (Toolchain Signing Key)** – ML-DSA-87.
* **PSK (Project Signing Key)** – ML-DSA-87 per project.
* **RDK (Runtime Decryption Key)** – ML-KEM-1024.

All artifact hashing: **SHA-384**.

### 5.2 Provenance Record

Link-time pass **`dsmil-provenance-pass`**:

* Builds a canonical provenance object:

  * Compiler info (name/version/target).
  * Source VCS info (repo/commit/dirty).
  * Build info (timestamp, builder ID, flags).
  * DSMIL defaults (layer/device/roles).
  * Hashes (SHA-384 of binary/sections).

* Canonicalize → `prov_canonical`.

* Compute `H = SHA-384(prov_canonical)`.

* Sign with ML-DSA-87 (PSK) → `σ`.

* Embed in ELF `.note.dsmil.provenance` / `.dsmil_prov`.

### 5.3 Optional ML-KEM-1024 Confidentiality

For high-sensitivity binaries:

* Generate symmetric key `K`.
* Encrypt `prov` using AEAD (e.g. AES-256-GCM).
* Encapsulate `K` with ML-KEM-1024 (RDK) → `ct`.
* Record:

  ```json
  {
    "enc_prov": "…",
    "kem_alg": "ML-KEM-1024",
    "kem_ct": "…",
    "hash_alg": "SHA-384",
    "sig_alg": "ML-DSA-87",
    "sig": "…"
  }
  ```

### 5.4 Runtime Validation

DSMIL loader/LSM:

1. Extract `.note.dsmil.provenance`.
2. If encrypted: decapsulate `K` (ML-KEM-1024) and decrypt.
3. Recompute SHA-384 hash.
4. Verify ML-DSA-87 signature.
5. If invalid: deny execution or require explicit override.
6. If valid: feed provenance to policy engine and audit log.

### 5.5 Sandbox Wrapping: `dsmil_sandbox`

Attribute:

```c
__attribute__((dsmil_sandbox("l7_llm_worker")))
int main(int argc, char **argv);
```

Link-time pass **`dsmil-sandbox-wrap`**:

* Rename `main` → `main_real`.
* Inject wrapper `main` that:

  * Applies libcap-ng capability profile for the role.
  * Installs seccomp filter for the role.
  * Optionally consumes provenance-driven runtime policy.
  * Calls `main_real()`.

Provenance includes `sandbox_profile`.

---

## 6. Quantum-Assisted Optimization Hooks (Layer 7, Device 46)

Layer 7 Device 46 ("Quantum Integration") provides hybrid algorithms (QAOA, VQE).

### 6.1 Tagging Quantum Candidates

Attribute:

```c
__attribute__((dsmil_quantum_candidate("placement")))
void placement_solver(...);
```

Metadata:

* `!dsmil.quantum_candidate = !"placement"`

### 6.2 Problem Extraction: `dsmil-quantum-export`

Pass:

* Analyzes candidate functions; when patterns match known optimization templates, emits QUBO/Ising descriptions.

Sidecar:

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
        "variables": ["model_1_dev47", "model_1_dev12"]
      }
    }
  ]
}
```

### 6.3 External Quantum Flow

External Quantum Orchestrator (on Device 46):

* Consumes `*.quantum.json`.
* Runs QAOA/VQE using Qiskit or similar.
* Writes back solutions (`*.quantum_solution.json`) for use by runtime or next build.

DSLLVM itself remains classical.

---

## 7. Tooling, Packaging & Repo Layout

### 7.1 CLI Tools

* `dsmil-clang`, `dsmil-clang++`, `dsmil-llc` – DSMIL target wrappers.
* `dsmil-opt` – `opt` wrapper with DSMIL pass presets.
* `dsmil-verify` – provenance + policy verifier.
* `dsmil-policy-dryrun` – run passes without modifying binaries (see §10).
* `dsmil-abi-diff` – compare DSMIL posture between builds (see §10).

### 7.2 Standard Pass Pipelines

Example production pipeline (`dsmil-default`):

1. LLVM `-O3`.
2. `dsmil-bandwidth-estimate`.
3. `dsmil-device-placement` (optionally AI-enhanced, §9).
4. `dsmil-layer-check`.
5. `dsmil-stage-policy`.
6. `dsmil-quantum-export`.
7. `dsmil-sandbox-wrap`.
8. `dsmil-provenance-pass`.

Other presets:

* `dsmil-debug` – weaker enforcement, more logging.
* `dsmil-lab` – annotate only, do not fail builds.

### 7.3 Repo Layout (Proposed)

```text
DSLLVM/
├─ cmake/
├─ docs/
│  ├─ DSLLVM-DESIGN.md
│  ├─ PROVENANCE-CNSA2.md
│  ├─ ATTRIBUTES.md
│  ├─ PIPELINES.md
│  └─ AI-INTEGRATION.md
├─ include/
│  ├─ dsmil_attributes.h
│  ├─ dsmil_provenance.h
│  ├─ dsmil_sandbox.h
│  └─ dsmil_ai_advisor.h
├─ lib/
│  ├─ Target/X86/DSMILTarget.cpp
│  ├─ Passes/
│  │  ├─ DsmilBandwidthPass.cpp
│  │  ├─ DsmilDevicePlacementPass.cpp
│  │  ├─ DsmilLayerCheckPass.cpp
│  │  ├─ DsmilStagePolicyPass.cpp
│  │  ├─ DsmilQuantumExportPass.cpp
│  │  ├─ DsmilSandboxWrapPass.cpp
│  │  ├─ DsmilProvenancePass.cpp
│  │  ├─ DsmilAICostModelPass.cpp
│  │  └─ DsmilAISecurityScanPass.cpp
│  └─ Runtime/
│     ├─ dsmil_sandbox_runtime.c
│     ├─ dsmil_provenance_runtime.c
│     └─ dsmil_ai_advisor_runtime.c
├─ tools/
│  ├─ dsmil-clang/
│  ├─ dsmil-llc/
│  ├─ dsmil-opt/
│  ├─ dsmil-verify/
│  ├─ dsmil-policy-dryrun/
│  └─ dsmil-abi-diff/
└─ test/
   └─ dsmil/
      ├─ layer_policies/
      ├─ stage_policies/
      ├─ provenance/
      ├─ sandbox/
      └─ ai_advisor/
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

## 8. AI-Assisted Compilation via DSMIL Layers 3–9

The DSMIL AI architecture provides rich AI capabilities per layer (LLMs in Layer 7, security AI in Layer 8, strategic planners in Layer 9, predictive analytics in Layers 4–6).

DSLLVM uses these as **external advisors** via a defined request/response protocol.

### 8.1 AI Advisor Overview

DSLLVM can emit **AI advisory requests**:

* Input:

  * Summaries of modules/IR (statistics, CFG features).
  * Existing DSMIL metadata (`layer`, `device`, `stage`, `bw_estimate`).
  * Current build goals (latency targets, power budgets, security posture).

* Output (AI suggestions):

  * Suggested `dsmil_stage`, `dsmil_layer`, `dsmil_device` annotations.
  * Pass pipeline tuning (e.g., "favor NPU for these kernels").
  * Refactoring hints ("split function X; mark param Y as `dsmil_untrusted_input`").
  * Risk flags ("this path appears security-sensitive; enable sandbox profile S").

AI results are **never blindly trusted**: deterministic DSLLVM passes re-check constraints.

### 8.2 Layer 7 LLM Advisor (Device 47)

Layer 7 Device 47 hosts LLMs up to ~7B parameters with INT8 quantization.

"L7 Advisor" roles:

* Suggest code-level annotations:

  * Infer `dsmil_stage` from project layout / comments.
  * Guess appropriate `dsmil_layer`/`device` per module (e.g., security code → L8; exec support → L9).

* Explainability:

  * Generate human-readable rationales for policy decisions in `AI-REPORT.md`.
  * Summarize complex IR into developer-friendly text for code reviews.

DSLLVM integration:

* Pass **`dsmil-ai-advisor-annotate`**:

  * Serializes module summary → `*.dsmilai_request.json`.
  * External L7 service writes `*.dsmilai_response.json`.
  * DSLLVM merges suggestions into metadata (under a "suggested" namespace; actual enforcement still via normal passes).

### 8.3 Layer 8 Security AI Advisor

Layer 8 provides ~188 TOPS for security AI & adversarial ML defense.

"L8 Advisor" roles:

* Identify risky patterns:

  * Untrusted input flows (paired with `dsmil_untrusted_input`, see §8.5).
  * Potential side-channel patterns.
  * Dangerous API use in security-critical layers (8–9).

* Suggest:

  * Where to enforce `dsmil_sandbox` roles more strictly.
  * Additional logging / telemetry for security-critical paths.

DSLLVM integration:

* **`dsmil-ai-security-scan`** pass:

  * Option 1: offline – uses pre-trained ML model embedded locally.
  * Option 2: online – exports features to an L8 service.

* Attaches:

  * `!dsmil.security_risk_score` per function.
  * `!dsmil.security_hints` describing suggested mitigations.

### 8.4 Layer 5/6 Predictive AI for Performance

Layers 5–6 handle advanced predictive analytics and strategic simulations.

Roles:

* Predict per-function/runtime performance under realistic workloads:

  * Given call-frequency profiles and `*.dsmilmap` data.
  * Use time-series and scenario models to predict "hot path" clusters.

Integration:

* **`dsmil-ai-perf-forecast`** tool:

  * Consumes:

    * History of `*.dsmilmap` + runtime metrics (latency, power).
    * New build's `*.dsmilmap`.

  * Produces:

    * Forecasts: "Functions A,B,C will likely dominate latency in scenario S".
    * Suggestions: move certain kernels from CPU AMX → NPU / GPU, or vice versa.

* DSLLVM can fold this back by re-running `dsmil-device-placement` with updated targets.

### 8.5 `dsmil_untrusted_input` & AI-Assisted IFC

Add attribute:

```c
__attribute__((dsmil_untrusted_input))
```

* Mark function parameters / globals that ingest untrusted data.

Combined with L8 advisor:

* DSLLVM can:

  * Identify flows from `dsmil_untrusted_input` into dangerous sinks.
  * Emit warnings or suggest `dsmil_gateway` / `dsmil_sandbox` for those paths.
  * Forward high-risk flows to L8 models for deeper analysis.

---

## 9. AI-Trained Cost Models & Schedulers

Beyond "call out to the big LLMs", DSLLVM embeds **small, distilled ML models** as cost models, running locally on CPU/NPU.

### 9.1 ML Cost Model Plugin

Pass **`DsmilAICostModelPass`**:

* Replaces or augments heuristic cost models for:

  * Inlining
  * Loop unrolling
  * Vectorization choice (AVX2 vs AMX vs NPU/GPU offload)
  * Device placement (CPU/NPU/GPU) for kernels

Implementation:

* Trained offline using:

  * The DSMIL AI stack (L7 + L5 performance modeling).
  * Historical build & runtime data from JRTC1-5450.

* At compile-time:

  * Uses a compact ONNX model executing via OpenVINO/AMX/NPU; no network needed.
  * Takes as input static features (loop depth, memory access patterns, etc.) and outputs:

    * Predicted speedup / penalty for each choice.
    * Confidence scores.

Outputs feed `dsmil-device-placement` and standard LLVM codegen decisions.

### 9.2 Scheduler for Multi-Layer AI Deployment

For models that can span multiple accelerators (e.g., LLMs split across AMX/iGPU/custom ASICs), DSLLVM provides a **multi-layer scheduler**:

* Reads:

  * `*.dsmilmap`
  * AI cost model outputs
  * High-level objectives (e.g., "min latency subject to ≤120W power")

* Computes:

  * Partition plan (which kernels run on which physical accelerators).
  * Layer-specific deployment suggestions (e.g., route certain inference paths to Layer 7 vs Layer 9 depending on clearance).

This is implemented as a post-link tool, but grounded in DSLLVM metadata.

---

## 10. AI Integration Modes & Guardrails

### 10.1 AI Integration Modes

Configurable mode:

* `--ai-mode=off`

  * No AI calls; deterministic, classic LLVM behavior.

* `--ai-mode=local`

  * Only embedded ML cost models run (no external services).

* `--ai-mode=advisor`

  * External L7/L8/L5 advisors used; suggestions applied only if they pass deterministic checks; all changes logged.

* `--ai-mode=lab`

  * Permissive; DSLLVM may auto-apply AI suggestions while still satisfying layer/clearance policies.

### 10.2 Policy Dry-Run

Tool: `dsmil-policy-dryrun`:

* Runs all DSMIL/AI passes in **report-only** mode:

  * Layer/clearance/ROE checks.
  * Stage policy.
  * Security scan.
  * AI advisor hints.
  * Placement & perf forecasts.

* Emits:

  * `policy-report.json`
  * Optional Markdown summary for humans.

No IR changes, no ELF modifications.

### 10.3 Diff-Guard for Security Posture

Tool: `dsmil-abi-diff`:

* Compares two builds' DSMIL posture:

  * Provenance contents.
  * `*.dsmilmap` mappings.
  * Sandbox profiles.
  * AI risk scores and suggested mitigations.

* Outputs:

  * "This build added a new L8 sandbox, changed Device 47 workload, and raised risk score for function X from 0.2 → 0.6."

Useful for code review and change-approval workflows.

### 10.4 Constant-Time / Side-Channel Annotations (`dsmil_secret`)

Cryptographic code in Layers 8–9 requires **constant-time execution** to prevent timing side-channels. DSLLVM provides the `dsmil_secret` attribute to enforce this.

**Attribute**:

```c
__attribute__((dsmil_secret))
void aes_encrypt(const uint8_t *key, const uint8_t *plaintext, uint8_t *ciphertext);

__attribute__((dsmil_secret))
int crypto_compare(const uint8_t *a, const uint8_t *b, size_t len);
```

**Semantics**:

* Parameters/return values marked with `dsmil_secret` are **tainted** in LLVM IR with `!dsmil.secret = i1 true`.
* DSLLVM tracks data-flow of secret values through SSA graph.
* Pass **`dsmil-ct-check`** (constant-time check) enforces:

  * **No secret-dependent branches**: if/else/switch on secret data → error.
  * **No secret-dependent memory access**: array indexing by secrets → error.
  * **No variable-time instructions**: division, modulo with secret operands → error (unless whitelisted intrinsics like `crypto.*`).

**AI Integration**:

* **Layer 8 Security AI** analyzes functions marked `dsmil_secret`:

  * Identifies potential side-channel leaks (cache timing, power analysis).
  * Suggests mitigations: constant-time lookup tables, masking, assembly intrinsics.

* **Layer 5 Performance AI** balances constant-time enforcement with performance:

  * Suggests where to use AVX-512 constant-time implementations.
  * Recommends hardware AES-NI vs software AES based on Device constraints.

**Policy**:

* Functions in Layers 8–9 with `dsmil_sandbox("crypto_worker")` **must** use `dsmil_secret` for all key material.
* Violations trigger compile-time errors in production builds (`DSMIL_PRODUCTION`).
* Lab builds (`--ai-mode=lab`) emit warnings only.

**Metadata Output**:

* `!dsmil.secret = i1 true` on SSA values.
* `!dsmil.ct_verified = i1 true` after `dsmil-ct-check` pass succeeds.

**Example**:

```c
DSMIL_LAYER(8) DSMIL_DEVICE(80) DSMIL_SANDBOX("crypto_worker")
__attribute__((dsmil_secret))
void hmac_sha384(const uint8_t *key, const uint8_t *msg, size_t len, uint8_t *mac) {
    // All operations on 'key' are constant-time enforced
    // Layer 8 Security AI validates no side-channel leaks
}
```

### 10.5 Quantum Optimization Hints in AI I/O

DSMIL Layer 7 Device 46 provides quantum optimization via QAOA/VQE. DSLLVM now integrates quantum hints directly into the **AI advisor I/O pipeline**.

**Integration**:

* When a function is marked `dsmil_quantum_candidate`, DSLLVM includes additional fields in the `*.dsmilai_request.json`:

```json
{
  "schema": "dsmilai-request-v1.2",
  "ir_summary": {
    "functions": [
      {
        "name": "placement_solver",
        "quantum_candidate": {
          "enabled": true,
          "problem_type": "placement",
          "variables": 128,
          "constraints": 45,
          "estimated_qubit_requirement": 12
        }
      }
    ]
  }
}
```

* **Layer 7 LLM Advisor** or **Layer 5 Performance AI** can now:

  * Recommend whether to export QUBO (based on problem size, available quantum resources).
  * Suggest hybrid classical/quantum strategies.
  * Provide rationale: "Problem size (128 vars) exceeds current QPU capacity; recommend classical ILP solver on CPU."

**Response Schema**:

```json
{
  "schema": "dsmilai-response-v1.2",
  "suggestions": [
    {
      "target": "placement_solver",
      "quantum_export": {
        "recommended": false,
        "rationale": "Problem size exceeds QPU capacity; classical ILP preferred",
        "alternative": "use_highs_solver_on_cpu"
      }
    }
  ]
}
```

**Pass Integration**:

* **`dsmil-quantum-export`** pass now:

  * Reads AI advisor response.
  * Only exports `*.quantum.json` if `quantum_export.recommended == true`.
  * Otherwise, emits metadata suggesting classical solver.

**Benefits**:

* **Unified workflow**: Single AI I/O pipeline for both performance and quantum decisions.
* **Resource awareness**: L7/L5 advisors have real-time visibility into Device 46 availability and QPU queue depth.
* **Hybrid optimization**: AI can recommend splitting problems (part quantum, part classical).

### 10.6 Compact ONNX Schema for Feature Scoring on Devices 43-58

DSLLVM embeds **tiny ONNX models** (~5–20 MB) for **fast feature scoring** during compilation. These models run on **Devices 43-58** (Layer 5 performance analytics accelerators, ~140 TOPS total).

**Motivation**:

* Full AI advisor calls (L7 LLM, L8 Security AI) have latency (~50-200ms per request).
* For **per-function cost decisions** (inlining, unrolling, vectorization), need <1ms inference.
* Solution: Use **compact ONNX models** for feature extraction + scoring, backed by AMX/NPU.

**Architecture**:

```
┌─────────────────────────────────────────────────────┐
│ DSLLVM Compilation Pass                            │
│ ┌─────────────────────────────────────────────────┐ │
│ │ Extract IR Features (per function)              │ │
│ │  - Basic blocks, loop depth, memory ops, etc.   │ │
│ └───────────────┬─────────────────────────────────┘ │
│                 │ Feature Vector (64-256 floats)    │
│                 ▼                                    │
│ ┌─────────────────────────────────────────────────┐ │
│ │ Tiny ONNX Model (5-20 MB)                       │ │
│ │  Input: [batch, features]                       │ │
│ │  Output: [batch, scores]                        │ │
│ │   scores: [inline_score, unroll_factor,         │ │
│ │            vectorize_width, device_preference]   │ │
│ └───────────────┬─────────────────────────────────┘ │
│                 │ Runs on Device 43-58 (AMX/NPU)    │
│                 ▼                                    │
│ ┌─────────────────────────────────────────────────┐ │
│ │ Apply Scores to Optimization Decisions          │ │
│ └─────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────┘
```

**ONNX Model Specification**:

* **Input Shape**: `[batch_size, 128]` (128 float32 features per function)
* **Output Shape**: `[batch_size, 16]` (16 float32 scores)
* **Model Size**: 5–20 MB (quantized INT8 or FP16)
* **Inference Time**: <0.5ms per function on Device 43 (NPU) or Device 50 (AMX)

**Feature Vector (128 floats)**:

| Index | Feature | Description |
|-------|---------|-------------|
| 0-7   | Complexity | Basic blocks, instructions, CFG depth, call count |
| 8-15  | Memory | Load/store count, estimated bytes, stride patterns |
| 16-23 | Control Flow | Branch count, loop nests, switch cases |
| 24-31 | Arithmetic | Int ops, FP ops, vector ops, div/mod count |
| 32-39 | Data Types | i8/i16/i32/i64/f32/f64 usage ratios |
| 40-47 | DSMIL Metadata | Layer, device, clearance, stage encoded |
| 48-63 | Call Graph | Caller/callee stats, recursion depth |
| 64-127| Reserved | Future extensions |

**Output Scores (16 floats)**:

| Index | Score | Description |
|-------|-------|-------------|
| 0     | Inline Score | Probability to inline (0.0-1.0) |
| 1     | Unroll Factor | Loop unroll factor (1-32) |
| 2     | Vectorize Width | SIMD width (1/4/8/16/32) |
| 3     | Device Preference CPU | Probability for CPU execution (0.0-1.0) |
| 4     | Device Preference NPU | Probability for NPU execution (0.0-1.0) |
| 5     | Device Preference GPU | Probability for iGPU execution (0.0-1.0) |
| 6-7   | Memory Tier | Ramdisk/tmpfs/SSD preference |
| 8-11  | Security Risk | Risk scores for various threat categories |
| 12-15 | Reserved | Future extensions |

**Pass Integration**:

* **`DsmilAICostModelPass`** now supports two modes:

  1. **Embedded Mode** (default): Uses compact ONNX model via OpenVINO on Devices 43-58.
  2. **Advisor Mode**: Falls back to full L7/L5 AI advisors for complex cases.

* Configuration:

```bash
# Use compact ONNX model (fast)
dsmil-clang --ai-mode=local --ai-cost-model=/path/to/dsmil-cost-v1.onnx ...

# Fallback to full advisors (slower, more accurate)
dsmil-clang --ai-mode=advisor --ai-use-full-advisors ...
```

**Model Training**:

* Trained offline on **JRTC1-5450** historical build data:

  * Inputs: IR feature vectors from 1M+ functions.
  * Labels: Ground-truth performance (latency, throughput, power).
  * Training Stack: Layer 7 Device 47 (LLM feature engineering) + Layer 5 Devices 50-59 (regression training).

* Models versioned and signed with TSK (Toolchain Signing Key).
* Provenance includes model version: `"ai_cost_model": "dsmil-cost-v1.3-20251124.onnx"`.

**Device Placement**:

* ONNX inference automatically routed to fastest available device:

  * Device 43 (NPU Tile 3, Layer 4) – primary.
  * Device 50 (AMX on CPU, Layer 5) – fallback.
  * Device 47 (LLM NPU, Layer 7) – if idle.

* Scheduling handled by DSMIL Device Manager (transparent to DSLLVM).

**Benefits**:

* **Latency**: <1ms per function vs 50-200ms for full AI advisor.
* **Throughput**: Can process entire compilation unit in parallel (batched inference).
* **Accuracy**: Trained on real DSMIL hardware data; 85-95% agreement with human expert decisions.
* **Determinism**: Fixed model version ensures reproducible builds.

---

## Appendix A – Attribute Summary

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
* `dsmil_untrusted_input`
* `dsmil_secret` (v1.2)

---

## Appendix B – DSMIL & AI Pass Summary

* `dsmil-bandwidth-estimate` – BW and memory class estimation.
* `dsmil-device-placement` – CPU/NPU/GPU target + memory tier hints.
* `dsmil-layer-check` – Layer/clearance/ROE enforcement.
* `dsmil-stage-policy` – Stage policy enforcement.
* `dsmil-quantum-export` – Export quantum optimization problems (v1.2: AI-advisor-driven).
* `dsmil-sandbox-wrap` – Sandbox wrapper insertion.
* `dsmil-provenance-pass` – CNSA 2.0 provenance generation.
* `dsmil-ai-advisor-annotate` – L7 advisor annotations.
* `dsmil-ai-security-scan` – L8 security AI analysis.
* `dsmil-ai-perf-forecast` – L5/6 performance forecasting (offline tool).
* `DsmilAICostModelPass` – Embedded ML cost models for codegen decisions (v1.2: ONNX on Devices 43-58).
* `dsmil-ct-check` – Constant-time enforcement for `dsmil_secret` (v1.2).

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

### Phase 4: Quantum & AI Integration (Weeks 17-22)

1. **Quantum Hooks**
   * Implement `dsmil-quantum-export`
   * Define output formats

2. **AI Advisor Integration**
   * Implement `dsmil-ai-advisor-annotate` pass
   * Define request/response JSON schemas
   * Implement `dsmil-ai-security-scan` pass
   * Create AI cost model plugin infrastructure

### Phase 5: Tooling & Hardening (Weeks 23-28)

1. **User Tools**
   * Implement `dsmil-verify`
   * Implement `dsmil-policy-dryrun`
   * Implement `dsmil-abi-diff`
   * Create comprehensive test suite
   * Documentation and examples

2. **AI Cost Models**
   * Train initial ML cost models on DSMIL hardware
   * Integrate ONNX runtime for local inference
   * Implement multi-layer scheduler

### Phase 6: Deployment & Validation (Weeks 29-32)

1. **Testing & Validation**
   * Comprehensive integration tests
   * AI advisor validation against ground truth
   * Performance benchmarking
   * Security audit

2. **CI/CD Integration**
   * Automated builds
   * Policy validation
   * AI advisor quality gates
   * Release packaging

---

## Appendix D – Security Considerations

### Threat Model

**Threats Mitigated**:
- ✓ Binary tampering (integrity via signatures)
- ✓ Supply chain attacks (provenance traceability)
- ✓ Unauthorized execution (policy enforcement)
- ✓ Quantum cryptanalysis (CNSA 2.0 algorithms)
- ✓ Key compromise (rotation, certificate chains)
- ✓ Untrusted input flows (IFC + L8 analysis)

**Residual Risks**:
- ⚠ Compromised build system (mitigation: secure build enclaves, TPM attestation)
- ⚠ AI advisor poisoning (mitigation: deterministic re-checking, audit logs)
- ⚠ Insider threats (mitigation: multi-party signing, audit logs)
- ⚠ Zero-day in crypto implementation (mitigation: multiple algorithm support)

### AI Security Considerations

1. **AI Model Integrity**:
   - Embedded ML cost models signed with TSK
   - Version tracking for all AI components
   - Fallback to heuristic models if AI fails

2. **AI Advisor Sandboxing**:
   - External L7/L8/L5 advisors run in isolated containers
   - Network-level restrictions on advisor communication
   - Rate limiting on AI service calls

3. **Determinism & Auditability**:
   - All AI suggestions logged with timestamps
   - Deterministic passes always validate AI outputs
   - Diff-guard tracks AI-induced changes

4. **AI Model Versioning**:
   - Provenance includes AI model versions used
   - Reproducible builds require fixed AI model versions
   - CI validates AI suggestions against known-good baselines

---

## Appendix E – Performance Considerations

### Compilation Overhead

* **Metadata Emission**: <1% overhead
* **Analysis Passes**: 2-5% compilation time increase
* **Provenance Generation**: 1-3% link time increase
* **AI Advisor Calls** (when enabled):
  * Local ML models: 3-8% overhead
  * External services: 10-30% overhead (parallel/async)
* **Total** (AI mode=local): <15% increase in build times
* **Total** (AI mode=advisor): 20-40% increase in build times

### Runtime Overhead

* **Provenance Validation**: One-time cost at program load (~10-50ms)
* **Sandbox Setup**: One-time cost at program start (~5-20ms)
* **Metadata Access**: Zero runtime overhead (compile-time only)
* **AI-Enhanced Placement**: Can improve runtime by 10-40% for AI workloads

### Memory Overhead

* **Binary Size**: +5-15% (metadata, provenance sections)
* **Sidecar Files**: ~1-5 KB per binary (`.dsmilmap`, `.quantum.json`)
* **AI Models**: ~50-200 MB for embedded cost models (one-time)

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| v1.0 | 2025-11-24 | SWORDIntel/DSMIL Team | Initial specification |
| v1.1 | 2025-11-24 | SWORDIntel/DSMIL Team | Added AI-assisted compilation features (§8-10), AI passes, new tools, extended roadmap |
| v1.2 | 2025-11-24 | SWORDIntel/DSMIL Team | Added constant-time enforcement (§10.4), quantum hints in AI I/O (§10.5), compact ONNX schema (§10.6); new `dsmil_secret` attribute, `dsmil-ct-check` pass |

---

**End of Specification**

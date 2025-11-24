# DSLLVM Strategic Roadmap
**Evolution of DSMIL-Optimized LLVM Toolchain as AI Grid Control Plane**

Version: 1.0
Date: 2025-11-24
Owner: SWORDIntel / DSMIL Kernel Team
Status: Strategic Planning Document

---

## Executive Summary

DSLLVM v1.2 established the **foundation**: a hardened LLVM/Clang toolchain with DSMIL hardware integration, AI-assisted compilation (Layers 3-9), CNSA 2.0 provenance, constant-time enforcement, and compact ONNX cost models.

**The Next Frontier:** Treat DSLLVM as the **control law** for the entire DSMIL AI grid (9 layers, 104 devices, ~1338 TOPS). This roadmap extends DSLLVM from "compiler with AI features" to "compiler-as-orchestrator" for a war-grade AI system.

**Core Philosophy:**
- DSLLVM is the **single source of truth** for system-wide security policy
- Compilation becomes a **mission-aware** process (border ops, cyber defense, exercises)
- The toolchain **learns from hardware** via RL and embedded ML models
- Security/forensics/testing become **compiler-native** features

This roadmap adds **10 major capabilities** across **4 strategic phases** (v1.3 → v2.0), organized by operational impact and technical dependencies.

---

## Table of Contents

1. [Foundation Review: v1.0-v1.2](#foundation-review-v10-v12)
2. [Phase 1: Operational Control (v1.3)](#phase-1-operational-control-v13)
3. [Phase 2: Security Depth (v1.4)](#phase-2-security-depth-v14)
4. [Phase 3: System Intelligence (v1.5)](#phase-3-system-intelligence-v15)
5. [Phase 4: Adaptive Optimization (v2.0)](#phase-4-adaptive-optimization-v20)
6. [Feature Dependency Graph](#feature-dependency-graph)
7. [Risk Assessment & Mitigations](#risk-assessment--mitigations)
8. [Resource Requirements](#resource-requirements)
9. [Success Metrics](#success-metrics)

---

## Foundation Review: v1.0-v1.2

### v1.0: Core Infrastructure (Completed)
**Delivered:**
- DSMIL hardware target (`x86_64-dsmil-meteorlake-elf`)
- 9-layer/104-device semantic metadata system
- CNSA 2.0 provenance (SHA-384, ML-DSA-87, ML-KEM-1024)
- Bandwidth/memory-aware optimization
- Quantum-assisted optimization hooks (Device 46)
- Sandbox integration (libcap-ng + seccomp-bpf)
- Complete tooling: `dsmil-clang`, `dsmil-verify`, `dsmil-opt`

**Key Passes:**
- `dsmil-bandwidth-estimate`, `dsmil-device-placement`, `dsmil-layer-check`, `dsmil-stage-policy`, `dsmil-quantum-export`, `dsmil-sandbox-wrap`, `dsmil-provenance-pass`

### v1.1: AI-Assisted Compilation (Completed)
**Delivered:**
- Layer 7 LLM Advisor integration (Device 47, Llama-3-7B-INT8)
- Layer 8 Security AI for vulnerability detection (~188 TOPS)
- Layer 5/6 Performance forecasting
- AI integration modes: `off`, `local`, `advisor`, `lab`
- Request/response JSON protocol (`dsmilai-request-v1`, `dsmilai-response-v1`)
- `dsmil_untrusted_input` attribute for IFC tracking

**Key Passes:**
- `dsmil-ai-advisor-annotate`, `dsmil-ai-security-scan`, `dsmil-ai-perf-forecast`, `DsmilAICostModelPass`

### v1.2: Security Hardening & Performance (Completed)
**Delivered:**
- **Constant-time enforcement:** `dsmil_secret` attribute + `dsmil-ct-check` pass
  - No secret-dependent branches/memory access/variable-time instructions
  - Layer 8 Security AI validates side-channel resistance
- **Quantum hints in AI I/O:** Integrated quantum candidate metadata into advisor protocol
  - AI-driven QUBO export decisions based on QPU availability
- **Compact ONNX feature scoring:** Tiny models (5-20 MB) on Devices 43-58
  - <0.5ms per-function inference (100-400× faster than full AI advisor)
  - 26,667 functions/s throughput on Device 43 (NPU, batch=32)

**Foundation Capabilities (v1.0-v1.2):**
- ✅ Hardware integration (9 layers, 104 devices)
- ✅ AI advisor pipeline (L5/7/8 integration)
- ✅ Security enforcement (constant-time, sandboxing, provenance)
- ✅ Performance optimization (ONNX cost models, quantum hooks)
- ✅ Policy framework (layer/clearance/ROE/stage checking)

---

## Phase 1: Operational Control (v1.3)

**Theme:** Make DSLLVM **mission-aware** and **operationally flexible**

**Target Date:** Q1 2026 (12-16 weeks)
**Priority:** **HIGH** (Immediate operational value)
**Risk:** **LOW** (Leverages existing v1.2 infrastructure)

### Feature 1.1: Mission Profiles as First-Class Compile Targets ⭐⭐⭐

**Motivation:** Replace "debug/release" with **mission-specific build configurations** (`border_ops`, `cyber_defence`, `exercise_only`).

**Design:**

```bash
# Compile for border operations mission
dsmil-clang -fdsmil-mission-profile=border_ops -O3 sensor.c -o sensor.bin

# Compile for exercise (relaxed constraints)
dsmil-clang -fdsmil-mission-profile=exercise_only -O3 test_harness.c
```

**Mission Profile Configuration** (`/etc/dsmil/mission-profiles.json`):

```json
{
  "border_ops": {
    "description": "Border operations: max security, minimal telemetry",
    "pipeline": "dsmil-hardened",
    "ai_mode": "local",  // No external AI calls
    "sandbox_default": "l8_strict",
    "allow_stages": ["quantized", "serve"],
    "deny_stages": ["debug", "experimental"],
    "quantum_export": false,  // No QUBO export in field
    "ct_enforcement": "strict",  // All crypto must be constant-time
    "telemetry_level": "minimal",  // Low-signature mode
    "provenance_required": true,
    "max_deployment_days": null,  // No time limit
    "clearance_floor": "0xFF080000"  // Minimum L8 clearance
  },
  "cyber_defence": {
    "description": "Cyber defense: AI-enhanced, full telemetry",
    "pipeline": "dsmil-default",
    "ai_mode": "advisor",  // Full L7/L8 AI advisors
    "sandbox_default": "l8_standard",
    "allow_stages": ["quantized", "serve", "distilled"],
    "deny_stages": ["debug"],
    "quantum_export": true,  // Use Device 46 if available
    "ct_enforcement": "strict",
    "telemetry_level": "full",  // Max observability
    "provenance_required": true,
    "layer_5_forecasting": true  // Enable perf prediction
  },
  "exercise_only": {
    "description": "Training exercise: relaxed constraints, verbose logging",
    "pipeline": "dsmil-lab",
    "ai_mode": "lab",  // Permissive AI mode
    "sandbox_default": "permissive",
    "allow_stages": ["*"],  // All stages allowed
    "deny_stages": [],
    "quantum_export": true,
    "ct_enforcement": "warn",  // Warnings only, no errors
    "telemetry_level": "verbose",
    "provenance_required": false,  // Optional for exercises
    "max_deployment_days": 30,  // Time-bomb: expires after 30 days
    "clearance_floor": "0x00000000"  // No clearance required
  },
  "lab_research": {
    "description": "Lab research: experimental features enabled",
    "pipeline": "dsmil-lab",
    "ai_mode": "lab",
    "sandbox_default": "lab_isolated",
    "allow_stages": ["*"],
    "ct_enforcement": "off",  // No enforcement for research
    "telemetry_level": "debug",
    "provenance_required": false,
    "experimental_features": ["rl_tuning", "novel_devices"]
  }
}
```

**Provenance Impact:**

```json
{
  "compiler_version": "dsmil-clang 19.0.0-v1.3",
  "mission_profile": "border_ops",
  "mission_profile_hash": "sha384:a1b2c3d4...",
  "mission_profile_version": "2025-11-24",
  "mission_constraints_verified": true,
  "build_date": "2025-12-01T10:30:00Z",
  "expiry_date": null,  // No expiry for border_ops
  "deployment_restrictions": {
    "max_deployment_days": null,
    "clearance_floor": "0xFF080000",
    "approved_networks": ["SIPRNET", "JWICS"]
  }
}
```

**New Attribute:**

```c
// Tag source code with mission requirements
__attribute__((dsmil_mission_profile("border_ops")))
int main(void) {
    // Must compile with border_ops profile or fail
}
```

**Pass Integration:**

**New pass:** `dsmil-mission-policy`
- Reads mission profile from CLI flag or source attribute
- Enforces mission-specific constraints:
  - Stage whitelist/blacklist
  - AI mode restrictions
  - Telemetry level
  - Clearance floor
- Validates all passes run with mission-appropriate config
- Fails build if violations detected

**CI/CD Integration:**

```yaml
# .github/workflows/dsmil-build.yml
jobs:
  build-border-ops:
    runs-on: meteor-lake
    steps:
      - name: Compile for border operations
        run: |
          dsmil-clang -fdsmil-mission-profile=border_ops \
            -O3 src/*.c -o border_ops.bin
      - name: Verify provenance
        run: |
          dsmil-verify --check-mission-profile=border_ops border_ops.bin
```

**Benefits:**
- ✅ **Single codebase, multiple missions:** No #ifdef hell
- ✅ **Policy enforcement:** Impossible to deploy wrong profile
- ✅ **Audit trail:** Provenance records mission intent
- ✅ **Operational flexibility:** Flip between max-security/max-tempo without code changes

**Implementation Effort:** **2-3 weeks** (90% reuses existing v1.2 pass infrastructure)

**Risks:**
- ⚠ **Accidental deployment of wrong profile:** Mitigation: `dsmil-verify` enforces profile checks at load time
- ⚠ **Profile proliferation:** Mitigation: Limit to 5-7 well-defined profiles; require governance approval for new profiles

---

### Feature 1.2: Auto-Generated Fuzz & Chaos Harnesses from IR ⭐⭐⭐

**Motivation:** Leverage existing `dsmil_untrusted_input` tracking (v1.2) to **automatically generate fuzz harnesses** for critical components.

**Design:**

**New pass:** `dsmil-fuzz-export`
- Scans IR for functions with `dsmil_untrusted_input` parameters
- Extracts:
  - API boundaries
  - Argument domains (types, ranges, constraints)
  - State machines / protocol parsers
  - Invariants (from assertions, comments, prior analysis)
- Emits `*.dsmilfuzz.json` describing harness requirements

**Output:** `*.dsmilfuzz.json`

```json
{
  "schema": "dsmil-fuzz-v1",
  "binary": "network_daemon.bin",
  "fuzz_targets": [
    {
      "function": "parse_network_packet",
      "location": "net.c:127",
      "untrusted_params": ["packet_data", "length"],
      "parameter_domains": {
        "packet_data": {
          "type": "bytes",
          "length_ref": "length",
          "constraints": ["non-null"]
        },
        "length": {
          "type": "size_t",
          "min": 0,
          "max": 65535,
          "special_values": [0, 1, 16, 1500, 65535]
        }
      },
      "invariants": [
        "length <= 65535",
        "packet_data[0] == MAGIC_BYTE (0x42)"
      ],
      "state_machine": {
        "states": ["IDLE", "HEADER_PARSED", "PAYLOAD_PARSED"],
        "transitions": [
          {"from": "IDLE", "to": "HEADER_PARSED", "condition": "valid_header"},
          {"from": "HEADER_PARSED", "to": "PAYLOAD_PARSED", "condition": "valid_payload"}
        ]
      },
      "suggested_harness": {
        "input_generation": {
          "strategy": "grammar-based",
          "grammar": "packet_format.bnf"
        },
        "coverage_goals": [
          "all_branches",
          "boundary_conditions",
          "state_machine_exhaustive"
        ],
        "chaos_scenarios": [
          "partial_packet (50% complete)",
          "malformed_header",
          "oversized_payload",
          "null_terminator_missing"
        ]
      },
      "l8_risk_score": 0.87,  // From Layer 8 Security AI
      "priority": "high"
    }
  ]
}
```

**Layer 7 LLM Advisor Integration:**

Send `*.dsmilfuzz.json` to L7 advisor → generates harness skeleton:

```c
// Auto-generated by DSLLVM v1.3 dsmil-fuzz-export + L7 Advisor
// Target: parse_network_packet (net.c:127)
// Priority: HIGH (L8 risk score: 0.87)

#include <stdint.h>
#include <stddef.h>
#include "net.h"

// LibFuzzer entry point
int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    // Boundary check (from invariants)
    if (size < 1) return 0;
    if (size > 65535) return 0;

    // State machine check (L7 inferred from analysis)
    if (data[0] != MAGIC_BYTE) {
        // Invalid magic byte - still test parser error handling
    }

    // Call target function
    int result = parse_network_packet(data, size);

    // Optional: Check postconditions
    // assert(global_state == EXPECTED_STATE);

    return 0;
}

// Chaos scenarios (from L8 Security AI suggestions)
#ifdef DSMIL_FUZZ_CHAOS

// Scenario 1: Partial packet (50% complete, then connection drops)
void chaos_partial_packet(void) {
    uint8_t packet[1000];
    init_packet(packet, 1000);
    parse_network_packet(packet, 500);  // Truncated
}

// Scenario 2: Malformed header (corrupt but valid checksum)
void chaos_malformed_header(void) {
    uint8_t packet[100];
    craft_malformed_header(packet);
    parse_network_packet(packet, 100);
}

#endif // DSMIL_FUZZ_CHAOS
```

**CI/CD Integration:**

```yaml
jobs:
  fuzz-test:
    runs-on: fuzz-cluster
    steps:
      - name: Extract fuzz targets
        run: |
          dsmil-clang --emit-fuzz-spec src/*.c -o network_daemon.dsmilfuzz.json

      - name: Generate harnesses (L7 advisor)
        run: |
          dsmil-ai-fuzz-gen network_daemon.dsmilfuzz.json \
            --advisor=l7_llm \
            --output=fuzz/

      - name: Run fuzzing (24 hours)
        run: |
          libfuzzer-parallel fuzz/ --max-time=86400 --jobs=64

      - name: Report crashes
        run: |
          dsmil-fuzz-report --crashes=crashes/ --l8-severity
```

**Layer 8 Chaos Integration:**

L8 Security AI suggests **chaos behaviors** for dependencies:

```json
{
  "chaos_scenarios": [
    {
      "name": "slow_io",
      "description": "Simulate slow I/O (network latency 1000ms)",
      "inject_at": ["socket_recv", "file_read"],
      "parameters": {"latency_ms": 1000}
    },
    {
      "name": "partial_failure",
      "description": "50% of allocations fail",
      "inject_at": ["malloc", "mmap"],
      "parameters": {"failure_rate": 0.5}
    },
    {
      "name": "corrupt_but_valid",
      "description": "Corrupt input but valid checksum/signature",
      "inject_at": ["crypto_verify"],
      "parameters": {"corruption_type": "bit_flip_small"}
    }
  ]
}
```

**Benefits:**
- ✅ **Compiler-native fuzzing:** No manual harness writing
- ✅ **AI-enhanced:** L7 generates smart harnesses; L8 suggests chaos scenarios
- ✅ **Security-first:** Prioritizes high-risk functions (L8 risk scores)
- ✅ **CI integration:** Automated fuzz testing in pipeline

**Implementation Effort:** **3-4 weeks**
- Week 1: `dsmil-fuzz-export` pass (IR analysis)
- Week 2: JSON schema + L7 advisor integration (harness generation)
- Week 3: L8 chaos scenario generation
- Week 4: CI/CD integration + testing

**Risks:**
- ⚠ **Harness isolation:** Fuzz harnesses must not ship in production
  - Mitigation: Separate build target (`--emit-fuzz-spec` flag); CI checks for accidental inclusion
- ⚠ **False negatives:** AI-generated harnesses might miss edge cases
  - Mitigation: Combine with manual review; track coverage metrics; iterate based on findings

---

### Feature 1.3: Minimum Telemetry Enforcement ⭐⭐

**Motivation:** Prevent "dark functions" that fail silently with no forensic trail.

**Design:**

**New attributes:**

```c
__attribute__((dsmil_safety_critical))
__attribute__((dsmil_mission_critical))
```

**Policy:**
- Functions marked `dsmil_safety_critical` or `dsmil_mission_critical` **must** have at least one telemetry hook:
  - Structured logging (syslog, journald)
  - Performance counters (`dsmil_counter_inc()`)
  - Trace points (eBPF, ftrace)
  - Health check registration

**New pass:** `dsmil-telemetry-check`
- Scans for critical functions
- Checks for presence of telemetry calls
- Fails build if zero observability hooks found
- L5/L8 advisors suggest: "Add metric at function entry/exit?"

**Example:**

```c
DSMIL_LAYER(8) DSMIL_DEVICE(80)
__attribute__((dsmil_safety_critical))  // NEW: Requires telemetry
__attribute__((dsmil_secret))
void ml_kem_1024_decapsulate(const uint8_t *sk, const uint8_t *ct, uint8_t *shared) {
    // DSLLVM enforces: must have at least one telemetry hook

    dsmil_counter_inc("ml_kem_decapsulate_calls");  // ✅ Satisfies requirement

    // ... crypto operations (constant-time enforced) ...

    if (error_condition) {
        dsmil_log_error("ml_kem_decapsulate_failed", "reason=%s", reason);
    }
}
```

**Compiler Error if Missing:**

```
error: function 'ml_kem_1024_decapsulate' is marked dsmil_safety_critical
       but has no telemetry hooks

note: add at least one of: dsmil_counter_inc(), dsmil_log_*(),
      dsmil_trace_point(), dsmil_health_register()

suggestion: add 'dsmil_counter_inc("ml_kem_decapsulate_calls");' at function entry
```

**Telemetry API** (`dsmil_telemetry.h`):

```c
// Counters (low-overhead, atomic)
void dsmil_counter_inc(const char *name);
void dsmil_counter_add(const char *name, uint64_t value);

// Structured logging (rate-limited)
void dsmil_log_info(const char *event, const char *fmt, ...);
void dsmil_log_warning(const char *event, const char *fmt, ...);
void dsmil_log_error(const char *event, const char *fmt, ...);

// Trace points (eBPF/ftrace integration)
void dsmil_trace_point(const char *name, const void *data, size_t len);

// Health checks (periodic validation)
void dsmil_health_register(const char *component, dsmil_health_fn fn);
```

**Layer 5/8 Advisor Integration:**

L5/L8 analyze critical functions and suggest:

```json
{
  "telemetry_suggestions": [
    {
      "function": "ml_kem_1024_decapsulate",
      "missing_telemetry": true,
      "suggestions": [
        {
          "type": "counter",
          "location": "function_entry",
          "code": "dsmil_counter_inc(\"ml_kem_decapsulate_calls\");",
          "rationale": "Track invocation rate for capacity planning"
        },
        {
          "type": "latency_histogram",
          "location": "function_exit",
          "code": "dsmil_histogram_observe(\"ml_kem_latency_us\", latency);",
          "rationale": "Monitor performance degradation"
        }
      ]
    }
  ]
}
```

**Benefits:**
- ✅ **Post-incident learning:** Always have data to understand failures
- ✅ **Capacity planning:** Track invocation rates for critical paths
- ✅ **Performance monitoring:** Detect degradation early
- ✅ **Security forensics:** Audit trail for crypto operations

**Implementation Effort:** **2 weeks**
- Week 1: Telemetry API design + runtime library
- Week 2: `dsmil-telemetry-check` pass + L5/L8 suggestion integration

**Risks:**
- ⚠ **PII/secret leakage in logs:** L8 must validate log contents
  - Mitigation: `dsmil-log-scan` pass checks for patterns like keys, tokens, PIIs
- ⚠ **Performance overhead:** Too much telemetry slows critical paths
  - Mitigation: Counters are atomic (low-overhead); structured logs are rate-limited

---

## Phase 1 Summary

**Deliverables (v1.3):**
1. ✅ Mission Profiles (#1.1)
2. ✅ Auto-Generated Fuzz Harnesses (#1.2)
3. ✅ Minimum Telemetry Enforcement (#1.3)

**Timeline:** 12-16 weeks (Q1 2026)

**Impact:**
- **Operational:** Mission-aware compilation; automated security testing
- **Security:** Fuzz-first development; enforced observability
- **Usability:** Single codebase for multiple missions

**Dependencies:**
- Requires v1.2 foundation (AI advisors, `dsmil_untrusted_input`, provenance)
- Requires mission profile governance (5-7 approved profiles)
- Requires telemetry infrastructure (syslog/journald/eBPF integration)

---

## Phase 2: Security Depth (v1.4)

**Theme:** Make DSLLVM **adversary-aware** and **forensically prepared**

**Target Date:** Q2 2026 (12-16 weeks)
**Priority:** **MEDIUM-HIGH** (Enhances security posture)
**Risk:** **MEDIUM** (Requires operational coordination)

### Feature 2.1: "Operational Stealth" Modes for AI-Laden Binaries ⭐⭐

**Motivation:** Binaries deployed in hostile net-space need **minimal telemetry/sideband signature** to avoid detection.

**Design:**

**New attribute/flag:**

```c
__attribute__((dsmil_low_signature))
void forward_observer_loop(void) {
    // Compiler optimizes for low detectability
}
```

Or via mission profile:

```json
{
  "covert_ops": {
    "description": "Covert operations: minimal signature",
    "telemetry_level": "stealth",  // NEW: stealth mode
    "ai_mode": "local",  // No external calls
    "behavioral_constraints": {
      "constant_rate_ops": true,  // Avoid bursty patterns
      "jitter_suppression": true,  // Minimize timing variance
      "network_fingerprint": "minimal"  // Reduce detectability
    }
  }
}
```

**DSLLVM Optimizations:**

**New pass:** `dsmil-stealth-transform`
- **Strips optional logging/metrics:** Removes non-critical telemetry
- **Constant-rate execution:** Pads operations to fixed time intervals
- **Jitter suppression:** Minimizes timing variance (crypto already constant-time via `dsmil_secret`)
- **Network fingerprint reduction:** Batches/delays network I/O to avoid patterns

**Layer 5/8 AI Integration:**

L5 models **detectability** based on:
- Timing patterns (bursty vs constant-rate)
- Network traffic (packet sizes, intervals)
- CPU patterns (predictable vs erratic)

L8 balances **detectability vs debugging**:
- Suggests which logs can be safely removed
- Warns about critical telemetry (safety-critical functions still need minimal hooks)

**Trade-offs:**

| Aspect | Normal Build | Stealth Build |
|--------|--------------|---------------|
| Telemetry | Full (counters, logs, traces) | Minimal (critical only) |
| Network I/O | Immediate | Batched/delayed |
| CPU patterns | Optimized for perf | Optimized for consistency |
| Debugging | Easy (verbose logs) | Hard (minimal hooks) |
| Detectability | High | Low |

**Guardrails:**

- ⚠ **Safety-critical functions still require minimum telemetry** (from Feature 1.3)
- ⚠ **Stealth builds must be paired with high-fidelity test mode elsewhere**
- ⚠ **Forensics capability reduced** → only deploy in hostile environments

**Benefits:**
- ✅ **Reduced signature:** Harder to detect via timing/network/CPU patterns
- ✅ **Mission-appropriate:** Can flip between stealth/observable modes
- ✅ **AI-optimized:** L5/L8 advisors model detectability

**Implementation Effort:** **3-4 weeks**

**Risks:**
- ⚠ **Lower observability makes forensics harder**
  - Mitigation: Require companion high-fidelity test build; mandate post-mission data exfiltration
- ⚠ **Constant-rate execution may degrade performance**
  - Mitigation: L5 advisor finds balance; only apply to covert mission profiles

---

### Feature 2.2: "Threat Signature" Embedding for Future Forensics ⭐

**Motivation:** Enable **future AI-driven forensics** by embedding latent threat descriptors in binaries.

**Design:**

**For high-risk modules, DSLLVM embeds:**
- Minimal, non-identifying **fingerprints** of:
  - Control-flow structure (CFG hash)
  - Serialization formats (protocol schemas)
  - Crypto usage patterns (algorithm + mode combinations)
- **Purpose:** Layer 62 (Forensics/SIEM) can correlate observed malware with known-good templates

**Example:**

```json
{
  "threat_signature": {
    "version": "1.0",
    "binary_hash": "sha384:...",
    "control_flow_fingerprint": {
      "algorithm": "CFG-Merkle-Hash",
      "hash": "0x1a2b3c4d...",
      "functions_included": ["main", "crypto_init", "network_send"]
    },
    "protocol_schemas": [
      {
        "protocol": "TLS-1.3",
        "extensions": ["ALPN", "SNI"],
        "ciphersuites": ["TLS_AES_256_GCM_SHA384"]
      }
    ],
    "crypto_patterns": {
      "algorithms": ["ML-KEM-1024", "ML-DSA-87", "AES-256-GCM"],
      "key_derivation": "HKDF-SHA384",
      "constant_time_enforced": true
    }
  }
}
```

**Use Case:**

1. **Known-good binary** compiled with DSLLVM v1.4 → embeds threat signature
2. **Months later:** Forensics team finds **suspicious binary** on network
3. **Layer 62 forensics AI** extracts CFG fingerprint from suspicious binary
4. **Correlation:** Matches against known-good signatures → "This is a tampered version of our sensor.bin"

**Security Considerations:**

- ⚠ **Risk:** Reverse-engineering threat signatures could leak internal structure
  - **Mitigation:** Signatures are **non-identifying** (hashes, not raw CFGs); only stored in secure SIEM
- ⚠ **Risk:** False positives/negatives in correlation
  - **Mitigation:** Use multiple features (CFG + protocol + crypto); require human review

**Benefits:**
- ✅ **Imposter detection:** Spot tampered/malicious versions of own binaries
- ✅ **Supply chain security:** Detect unauthorized modifications
- ✅ **AI-powered forensics:** Layer 62 can correlate at scale

**Implementation Effort:** **2-3 weeks**

**Risks:**
- ⚠ **Leakage of internal structure**
  - Mitigation: Store signatures in secure SIEM only; encrypt with ML-KEM-1024
- ⚠ **Storage overhead:** Signatures add ~5-10 KB per binary
  - Mitigation: Optional feature; only enable for high-value targets

---

### Feature 2.3: Compiler-Level "Blue vs Red" Scenario Simulation ⭐

**Motivation:** Structured way to test "how this code would look from the other side."

**Design:**

**Two parallel builds of same system:**

```bash
# Blue team build (defender view)
dsmil-clang -fdsmil-role=blue -O3 src/*.c -o defender.bin

# Red team build (attacker stress-test view)
dsmil-clang -fdsmil-role=red -O3 src/*.c -o attacker_test.bin
```

**Blue Build (Normal):**
- CNSA 2.0 provenance
- Strict sandbox
- Full telemetry
- Constant-time enforcement

**Red Build (Stress-Test):**
- **Same logic**, but:
  - **Extra instrumentation:** See how it could be abused
  - **L8 "what if" analysis hooks:** Not shipped in prod
  - **Vulnerability injection points:** For testing defenses
  - **Attack surface mapping:** Which functions are exposed

**Example:**

```c
// Blue build: Normal
DSMIL_LAYER(7) DSMIL_DEVICE(47)
void process_user_input(const char *input) {
    validate_and_process(input);
}

// Red build: Instrumented
DSMIL_LAYER(7) DSMIL_DEVICE(47)
void process_user_input(const char *input) {
    #ifdef DSMIL_RED_BUILD
        // Log: potential injection point
        dsmil_red_log("injection_point", "function=%s param=%s",
                     __func__, "input");

        // L8 analysis: what if validation bypassed?
        if (dsmil_red_scenario("bypass_validation")) {
            // Simulate attacker bypassing validation
            raw_process(input);  // Vulnerable path
        } else
    #endif

    validate_and_process(input);  // Normal path
}
```

**Layer 5/9 Campaign-Level Analysis:**

L5/L9 advisors simulate **campaign-level effects**:
- "If attacker compromises 3 binaries in this deployment, what's the blast radius?"
- "Which binaries, if tampered, would bypass Layer 8 defenses?"

**Guardrails:**

- ⚠ **Red build must be aggressively confined**
  - Sandboxed in isolated test environment only
  - Never deployed to production
  - Signed with separate key (not TSK)

**Benefits:**
- ✅ **Adversarial thinking:** Test defenses from attacker perspective
- ✅ **Campaign-level modeling:** L5/L9 simulate multi-binary compromise
- ✅ **Structured stress-testing:** No need for separate tooling

**Implementation Effort:** **4-5 weeks**

**Risks:**
- ⚠ **Red build must never cross into ops**
  - Mitigation: Separate provenance key; CI enforces isolation; runtime checks reject red builds
- ⚠ **Complexity:** Maintaining two build flavors
  - Mitigation: Share 95% of code; only instrumentation differs

---

## Phase 2 Summary

**Deliverables (v1.4):**
1. ✅ Operational Stealth Modes (#2.1)
2. ✅ Threat Signature Embedding (#2.2)
3. ✅ Blue vs Red Scenario Simulation (#2.3)

**Timeline:** 12-16 weeks (Q2 2026)

**Impact:**
- **Security:** Stealth mode for hostile environments; forensics-ready binaries; adversarial testing
- **Operational:** Mission-specific detectability tuning
- **Forensics:** AI-powered correlation via threat signatures

**Dependencies:**
- Requires v1.3 (mission profiles, telemetry enforcement)
- Requires Layer 62 (forensics/SIEM) integration for threat signatures
- Requires secure test infrastructure for blue/red builds

---

## Phase 3: System Intelligence (v1.5)

**Theme:** Treat DSLLVM as **system-wide orchestrator** for distributed security

**Target Date:** Q3 2026 (16-20 weeks)
**Priority:** **MEDIUM** (System-level capabilities)
**Risk:** **MEDIUM-HIGH** (Requires build system integration)

### Feature 3.1: DSLLVM as "Schema Compiler" for Exotic Devices ⭐⭐

**Motivation:** Auto-generate type-safe bindings for 104 DSMIL devices from single source of truth.

**Design:**

**Device Specification** (YAML/JSON):

```yaml
# /etc/dsmil/devices/device-51.yaml
device_id: 51
sku: "ADV-ML-ASIC-51"
name: "Adversarial ML Defense Engine"
layer: 8
clearance: "0xFF080808"
firmware_version: "3.2.1-DSMIL"

bars:
  BAR0:
    size: "4 MB"
    purpose: "Control/Status registers + OpCode FIFO"
  BAR1:
    size: "256 MB"
    purpose: "Model weight/bias storage (encrypted)"

opcodes:
  - code: 0x01
    name: SELF_TEST
    requires: operator
    args: []
    returns: status_t
    notes: "Runs BIST; no model access"

  - code: 0x02
    name: LOAD_DEFENSE_MODEL
    requires: 2PI
    args: [model_payload_t*, size_t]
    returns: status_t
    notes: "Accepts signed payload; rejects unsigned"

  - code: 0x05
    name: ZEROIZE
    requires: 2PI_HSM
    args: []
    returns: void
    notes: "Zeroes SRAM/keys; transitions to ZEROIZED"

states: [OFF, STANDBY, ARMED, ACTIVE, QUARANTINE, ZEROIZED]

allowed_transitions:
  - from: STANDBY
    to: ARMED
    condition: "2PI + signed_image"
  - from: ARMED
    to: ACTIVE
    condition: "policy_loaded + runtime_attested"

security_constraints:
  - "2PI required for opcodes 0x02/0x05"
  - "Firmware payloads must be signed (RSA-3072/SHA3-384)"
  - "QUARANTINE enforces read-only logs and disables DMA"
```

**Tool:** `dsmil-devicegen`

```bash
# Generate type-safe C++ bindings from device spec
dsmil-devicegen --input=/etc/dsmil/devices/ --output=generated/

# Output:
#   generated/device_51.h         (C++ bindings)
#   generated/device_51_verify.h  (LLVM pass for static verification)
```

**Generated Code** (`generated/device_51.h`):

```cpp
// Auto-generated by dsmil-devicegen from device-51.yaml
// DO NOT EDIT

#pragma once
#include <dsmil_device_base.h>

namespace dsmil::device51 {

// Type-safe opcode wrappers
class AdversarialMLDefenseEngine : public DSMILDevice {
public:
    AdversarialMLDefenseEngine() : DSMILDevice(51) {}

    // Opcode 0x01: SELF_TEST
    // Requires: operator clearance
    __attribute__((dsmil_device(51)))
    __attribute__((dsmil_clearance(0xFF080808)))
    status_t self_test() {
        check_clearance(OPERATOR);
        return invoke_opcode(0x01);
    }

    // Opcode 0x02: LOAD_DEFENSE_MODEL
    // Requires: 2PI clearance
    __attribute__((dsmil_device(51)))
    __attribute__((dsmil_clearance(0xFF080808)))
    __attribute__((dsmil_2pi_required))  // NEW: 2PI enforcement
    status_t load_defense_model(const model_payload_t *payload, size_t size) {
        check_clearance(TWO_PERSON_INTEGRITY);
        verify_signature(payload, size);  // Auto-inserted
        return invoke_opcode(0x02, payload, size);
    }

    // Opcode 0x05: ZEROIZE
    // Requires: 2PI + HSM token
    __attribute__((dsmil_device(51)))
    __attribute__((dsmil_clearance(0xFF080808)))
    __attribute__((dsmil_2pi_hsm_required))
    void zeroize() {
        check_clearance(TWO_PERSON_INTEGRITY_HSM);
        invoke_opcode(0x05);
        // Auto-inserted state transition
        transition_to_state(ZEROIZED);
    }

private:
    // State machine enforcement
    enum State { OFF, STANDBY, ARMED, ACTIVE, QUARANTINE, ZEROIZED };
    State current_state = OFF;

    void transition_to_state(State new_state) {
        // Auto-generated from allowed_transitions
        if (!is_valid_transition(current_state, new_state)) {
            throw std::runtime_error("Invalid state transition");
        }
        current_state = new_state;
    }
};

} // namespace dsmil::device51
```

**Generated LLVM Pass** (`generated/device_51_verify.cpp`):

```cpp
// Auto-generated LLVM pass for static verification
class Device51VerifyPass : public PassInfoMixin<Device51VerifyPass> {
public:
    PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM) {
        for (auto &F : M) {
            // Check: Only functions with clearance >= 0xFF080808 can call device 51
            if (accesses_device(F, 51)) {
                uint32_t clearance = get_clearance(F);
                if (clearance < 0xFF080808) {
                    errs() << "ERROR: Function " << F.getName()
                          << " accesses Device 51 without sufficient clearance\n";
                    return PreservedAnalyses::none();
                }
            }

            // Check: load_defense_model requires 2PI attribute
            if (calls_function(F, "load_defense_model")) {
                if (!has_attribute(F, "dsmil_2pi_required")) {
                    errs() << "ERROR: Function " << F.getName()
                          << " calls load_defense_model without 2PI enforcement\n";
                    return PreservedAnalyses::none();
                }
            }
        }
        return PreservedAnalyses::all();
    }
};
```

**Benefits:**
- ✅ **No hand-rolled wrappers:** Single device spec generates all bindings
- ✅ **Type-safe:** Compile-time checks for clearance, state transitions
- ✅ **Static verification:** LLVM pass enforces device constraints
- ✅ **Maintainability:** Update device spec → regenerate bindings

**Implementation Effort:** **4-5 weeks**

**Risks:**
- ⚠ **Device spec becomes security-critical:** Bad spec = bad guarantees
  - Mitigation: Device specs require governance approval; signed with TSK
- ⚠ **Spec proliferation:** 104 devices = 104 specs
  - Mitigation: Templating for similar devices; automated validation

---

### Feature 3.2: Cross-Binary Invariant Checking ⭐⭐

**Motivation:** Treat multiple binaries as a **single distributed system** and enforce invariants across them.

**Design:**

**System-Level Invariants** (`/etc/dsmil/system-invariants.yaml`):

```yaml
# System-wide security invariants
invariants:
  - name: "Only crypto workers can access Device 30"
    constraint: |
      forall binary B in system:
        if B.accesses(device_30) then B.sandbox == "crypto_worker"
    severity: critical

  - name: "At most 3 binaries can bypass Layer 7"
    constraint: |
      count(binaries where has_gateway(layer=7)) <= 3
    severity: high

  - name: "No debug stage in production layer >= 7"
    constraint: |
      forall binary B in system:
        if B.layer >= 7 and B.deployed_to == "production"
        then B.stage != "debug"
    severity: critical

  - name: "All L8 crypto must be constant-time"
    constraint: |
      forall binary B in system:
        if B.layer == 8 and B.role == "crypto_worker"
        then forall function F in B:
          if F.is_crypto() then F.has_attribute("dsmil_secret")
    severity: critical
```

**Build Orchestrator:** `dsmil-system-build`

```bash
# Build entire system with invariant checking
dsmil-system-build --config=deployment.yaml \
  --invariants=/etc/dsmil/system-invariants.yaml \
  --output=dist/

# Output:
#   dist/sensor_1.bin
#   dist/sensor_2.bin
#   dist/crypto_worker.bin
#   dist/network_gateway.bin
#   dist/system-validation-report.json
```

**Orchestrator Workflow:**

1. **Build all binaries** → collect `*.dsmilmap` from each
2. **Load system invariants** from `/etc/dsmil/system-invariants.yaml`
3. **Check invariants** across all `*.dsmilmap` files
4. **Fail build if violated:**

```
ERROR: System invariant violated

Invariant: "Only crypto workers can access Device 30"
Violation: Binary 'sensor_1.bin' (sandbox: 'l7_sensor') accesses Device 30

Fix: Either:
  1. Change sensor_1 sandbox to 'crypto_worker', OR
  2. Remove Device 30 access from sensor_1.c

Affected files:
  - src/sensor_1.c:127 (function: read_crypto_data)
```

**Integration with CI:**

```yaml
jobs:
  system-build:
    runs-on: build-cluster
    steps:
      - name: Build entire system
        run: |
          dsmil-system-build --config=deployment.yaml \
            --invariants=/etc/dsmil/system-invariants.yaml

      - name: Validate invariants
        run: |
          if [ $? -ne 0 ]; then
            echo "System invariant violation detected. See logs."
            exit 1
          fi

      - name: Deploy
        run: |
          kubectl apply -f dist/manifests/
```

**Benefits:**
- ✅ **System-level security:** Enforce constraints across entire deployment
- ✅ **Architectural enforcement:** "The system is the unit of security, not the binary"
- ✅ **Early detection:** Catch violations at build time, not runtime

**Implementation Effort:** **5-6 weeks**

**Risks:**
- ⚠ **Build system integration:** Requires coordination across repos
  - Mitigation: Start with single-repo systems; extend to multi-repo
- ⚠ **Brittleness:** Infra drift breaks invariants
  - Mitigation: Keep invariants minimal (5-10 critical rules); validate against deployment reality

---

### Feature 3.3: "Temporal Profiles" – Compiling for Phase of Operation ⭐

**Motivation:** **Day-0 deployment, Day-30 hardened, Day-365 long-term maintenance** – all as compile profiles.

**Design:**

**Temporal Profiles** (combines with Mission Profiles from v1.3):

```json
{
  "bootstrap": {
    "description": "Day 0-30: Initial deployment, experimentation",
    "pipeline": "dsmil-debug",
    "ct_enforcement": "warn",
    "telemetry_level": "verbose",
    "ai_mode": "advisor",  // Full AI for learning
    "experimental_features": true,
    "max_deployment_days": 30,  // Time-bomb: expires after 30 days
    "next_required_profile": "stabilize"
  },
  "stabilize": {
    "description": "Day 31-90: Tighten security, collect data",
    "pipeline": "dsmil-default",
    "ct_enforcement": "strict",
    "telemetry_level": "standard",
    "ai_mode": "advisor",
    "experimental_features": false,
    "max_deployment_days": 60,
    "next_required_profile": "production"
  },
  "production": {
    "description": "Day 91+: Long-term hardened production",
    "pipeline": "dsmil-hardened",
    "ct_enforcement": "strict",
    "telemetry_level": "minimal",
    "ai_mode": "local",  // No external AI calls
    "experimental_features": false,
    "max_deployment_days": null,  // No expiry
    "upgrade_required_from": "stabilize"  // Must recompile from stabilize
  }
}
```

**Provenance Tracks Lifecycle:**

```json
{
  "temporal_profile": "bootstrap",
  "build_date": "2025-12-01T00:00:00Z",
  "expiry_date": "2025-12-31T00:00:00Z",  // 30 days
  "next_required_profile": "stabilize",
  "deployment_phase": "initial"
}
```

**Runtime Enforcement:**

DSMIL loader checks provenance:
- If `expiry_date` passed → refuse to run
- Emit: "Binary expired. Recompile with 'stabilize' profile."

**Layer 5/9 Advisor Integration:**

L5/L9 project **risk/benefit of moving between phases:**
- "System X is ready to move from bootstrap → stabilize (30 days stable, <5 incidents)"
- "System Y should stay in stabilize (12 critical bugs in last 60 days)"

**Benefits:**
- ✅ **Lifecycle awareness:** Early/mature systems have different priorities
- ✅ **Time-based enforcement:** Prevents stale bootstrap builds in prod
- ✅ **Smooth transitions:** Explicit upgrade path (bootstrap → stabilize → production)

**Implementation Effort:** **3-4 weeks**

**Risks:**
- ⚠ **Must track "no bootstrap binaries remain in production"**
  - Mitigation: CI enforces; runtime loader rejects expired binaries
- ⚠ **Ops complexity:** Managing multiple lifecycle phases
  - Mitigation: Automate phase transitions based on L5/L9 recommendations

---

## Phase 3 Summary

**Deliverables (v1.5):**
1. ✅ Schema Compiler for Exotic Devices (#3.1)
2. ✅ Cross-Binary Invariant Checking (#3.2)
3. ✅ Temporal Profiles (#3.3)

**Timeline:** 16-20 weeks (Q3 2026)

**Impact:**
- **System Intelligence:** Device schema automation; cross-binary security; lifecycle-aware builds
- **Operational:** Reduced manual work; automated invariant enforcement
- **Security:** System-wide guarantees; time-based expiry

**Dependencies:**
- Requires v1.3 (mission profiles)
- Requires device specifications for all 104 devices (governance process)
- Requires build orchestrator integration (multi-binary builds)

---

## Phase 4: Adaptive Optimization (v2.0)

**Theme:** DSLLVM **learns from hardware** and **adapts to operational reality**

**Target Date:** Q4 2026 (20-24 weeks)
**Priority:** **RESEARCH** (Long-term investment)
**Risk:** **HIGH** (Requires ML infrastructure + operational separation)

### Feature 4.1: Compiler-Level RL Loop on Real Hardware ⭐⭐⭐

**Motivation:** Use **reinforcement learning** to tune compiler "knobs" per hardware configuration.

**Design:**

**Small Parameter Vector:**

```python
θ = {
    inline_limit: int,          # [10, 500]
    npu_threshold: float,       # [0.0, 1.0]
    gpu_threshold: float,       # [0.0, 1.0]
    sandbox_aggressiveness: int,# [1, 5]
    vectorize_preference: str,  # ["SSE", "AVX2", "AVX-512", "AMX"]
    unroll_factor_base: int     # [1, 32]
}
```

**RL Training Loop** (Lab-only, Devices 43-58):

```
1. Initialize θ randomly
2. For N iterations:
   a. Compile workload W with parameters θ
   b. Deploy to sandboxed lab hardware
   c. Measure:
      - Latency (ms)
      - Throughput (ops/s)
      - Power (watts)
      - Security violations (count)
   d. Compute reward:
      R = -latency - 0.5*power + 100*throughput - 1000*violations
   e. Update θ using policy gradient (PPO, A3C, etc.)
3. Select best θ → freeze as static profile for production
```

**Architecture:**

```
┌─────────────────────────────────────────────────┐
│ RL Training Loop (Lab Environment)              │
│ ┌─────────────────────────────────────────────┐ │
│ │ 1. DSLLVM compiles with parameters θ        │ │
│ └──────────────┬──────────────────────────────┘ │
│                │ Binary artifact                 │
│                ▼                                 │
│ ┌─────────────────────────────────────────────┐ │
│ │ 2. Deploy to sandboxed lab hardware         │ │
│ │    (Isolated Meteor Lake testbed)           │ │
│ └──────────────┬──────────────────────────────┘ │
│                │ Metrics (latency, power, etc.) │
│                ▼                                 │
│ ┌─────────────────────────────────────────────┐ │
│ │ 3. RL Agent (Devices 43-58, Layer 5)       │ │
│ │    Computes reward R(θ, metrics)            │ │
│ │    Updates policy: θ ← θ + ∇R               │ │
│ └──────────────┬──────────────────────────────┘ │
│                │ New parameters θ'               │
│                └─────────────┐                   │
│                              ↓                   │
│ ┌─────────────────────────────────────────────┐ │
│ │ 4. Repeat until convergence                 │ │
│ │    Select best θ* → freeze as profile       │ │
│ └─────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────┐
│ Production Deployment (Static Profile)          │
│ ┌─────────────────────────────────────────────┐ │
│ │ DSLLVM uses learned θ* (no live RL)        │ │
│ │ Provenance records: θ* + training metadata  │ │
│ └─────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────┘
```

**Layer 5/7/8 Integration:**

- **Layer 5:** RL agent runs on Devices 43-58
- **Layer 7:** LLM advisor suggests feature engineering for θ
- **Layer 8:** Security AI validates: "Does θ introduce vulnerabilities?"

**Learned Profiles** (Example Output):

```json
{
  "profile_name": "meteor_lake_llm_inference",
  "hardware": {
    "cpu": "Intel Meteor Lake",
    "npu": "NPU Tile 3 (Device 43)",
    "gpu": "Intel Arc iGPU"
  },
  "learned_parameters": {
    "inline_limit": 342,
    "npu_threshold": 0.73,
    "gpu_threshold": 0.21,
    "sandbox_aggressiveness": 3,
    "vectorize_preference": "AMX",
    "unroll_factor_base": 16
  },
  "training_metadata": {
    "workload": "llm_inference_7b_int8",
    "iterations": 5000,
    "final_reward": 87.3,
    "performance": {
      "avg_latency_ms": 23.1,
      "throughput_qps": 234,
      "power_watts": 87
    }
  },
  "provenance": {
    "rl_algorithm": "PPO",
    "training_date": "2026-09-15",
    "validated_by": "L8_Security_AI",
    "signature": "ML-DSA-87:..."
  }
}
```

**Production Usage:**

```bash
# Use learned profile for Meteor Lake LLM inference
dsmil-clang --rl-profile=meteor_lake_llm_inference -O3 llm.c -o llm.bin
```

**Provenance:**

```json
{
  "compiler_version": "dsmil-clang 20.0.0-v2.0",
  "rl_profile": "meteor_lake_llm_inference",
  "rl_profile_hash": "sha384:...",
  "rl_training_date": "2026-09-15",
  "parameters_used": {
    "inline_limit": 342,
    "npu_threshold": 0.73,
    ...
  }
}
```

**Guardrails:**

- ⚠ **RL system is lab-only:** Never live exploration in production
- ⚠ **Results brought into prod as static profiles:** No runtime adaptation
- ⚠ **L8 validation required:** RL-learned profiles must pass security scan
- ⚠ **Determinism preserved:** Fixed profile → reproducible builds

**Benefits:**
- ✅ **Hardware-specific tuning:** Learns optimal θ for each DSMIL platform
- ✅ **Better than heuristics:** RL discovers non-obvious optimization strategies
- ✅ **Continuous improvement:** Retrain as hardware/workloads evolve

**Implementation Effort:** **8-10 weeks**

**Risks:**
- ⚠ **RL agent could learn unsafe parameters**
  - Mitigation: L8 Security AI validates all learned profiles; reject if violations detected
- ⚠ **Lab/prod separation critical**
  - Mitigation: RL training runs in isolated sandbox; prod uses frozen profiles only
- ⚠ **Exploration overhead:** RL training expensive (1000s of compile-deploy-measure cycles)
  - Mitigation: Run overnight on dedicated lab hardware; amortize over many workloads

---

## Phase 4 Summary

**Deliverables (v2.0):**
1. ✅ Compiler-Level RL Loop on Real Hardware (#4.1)

**Timeline:** 20-24 weeks (Q4 2026)

**Impact:**
- **Adaptive Optimization:** Hardware-specific learned profiles
- **Performance:** Better than heuristic tuning
- **Future-Proof:** Continuously improve as hardware evolves

**Dependencies:**
- Requires isolated lab hardware (Meteor Lake testbed)
- Requires Devices 43-58 (Layer 5) for RL agent
- Requires L8 Security AI for profile validation
- Requires operational separation (lab vs prod)

---

## Feature Dependency Graph

```
v1.0-v1.2 Foundation
    │
    ├─> v1.3 Phase 1: Operational Control
    │    ├─> Feature 1.1: Mission Profiles ⭐⭐⭐
    │    │    └─> Enables Feature 1.3 (mission-specific telemetry)
    │    │    └─> Enables Feature 2.1 (stealth mission profile)
    │    │    └─> Enables Feature 3.3 (temporal profiles)
    │    │
    │    ├─> Feature 1.2: Auto-Fuzz Harnesses ⭐⭐⭐
    │    │    └─> Depends on: v1.2 (dsmil_untrusted_input, L8 Security AI)
    │    │
    │    └─> Feature 1.3: Minimum Telemetry ⭐⭐
    │         └─> Enables Feature 2.1 (stealth mode balances telemetry)
    │
    ├─> v1.4 Phase 2: Security Depth
    │    ├─> Feature 2.1: Operational Stealth ⭐⭐
    │    │    └─> Depends on: Feature 1.1 (mission profiles), Feature 1.3 (telemetry)
    │    │
    │    ├─> Feature 2.2: Threat Signatures ⭐
    │    │    └─> Requires: Layer 62 (forensics/SIEM) integration
    │    │
    │    └─> Feature 2.3: Blue vs Red Builds ⭐
    │         └─> Depends on: L8 Security AI (v1.1)
    │
    ├─> v1.5 Phase 3: System Intelligence
    │    ├─> Feature 3.1: Schema Compiler ⭐⭐
    │    │    └─> Independent (can implement anytime after v1.0)
    │    │
    │    ├─> Feature 3.2: Cross-Binary Invariants ⭐⭐
    │    │    └─> Depends on: Build orchestrator, *.dsmilmap (v1.0)
    │    │
    │    └─> Feature 3.3: Temporal Profiles ⭐
    │         └─> Depends on: Feature 1.1 (mission profiles)
    │
    └─> v2.0 Phase 4: Adaptive Optimization
         └─> Feature 4.1: RL Loop ⭐⭐⭐
              └─> Depends on: Devices 43-58 (v1.2 ONNX), L8 Security AI (v1.1)
```

**Critical Path:**
```
v1.0-v1.2 → Feature 1.1 (Mission Profiles) → Feature 1.3 (Telemetry) → Feature 2.1 (Stealth) → v1.4
                                           → Feature 3.3 (Temporal) → v1.5
```

**Independent Features:**
- Feature 1.2 (Auto-Fuzz): Can implement anytime after v1.2
- Feature 2.2 (Threat Signatures): Independent, requires Layer 62
- Feature 2.3 (Blue/Red): Independent, requires L8 AI
- Feature 3.1 (Schema Compiler): Independent, can implement anytime

---

## Risk Assessment & Mitigations

### High-Risk Features

| Feature | Risk | Mitigation |
|---------|------|------------|
| **2.1 Stealth** | Lower observability → harder forensics | Require companion high-fidelity test build; mandate post-mission data exfiltration |
| **2.3 Blue/Red** | Red build leaks into production | Separate provenance key; CI enforces isolation; runtime rejects red builds |
| **3.2 Cross-Binary** | Brittle if infra drifts | Keep invariants minimal (5-10 rules); validate against deployment reality |
| **4.1 RL Loop** | RL learns unsafe parameters | L8 Security AI validates all profiles; reject if violations; lab-only training |

### Medium-Risk Features

| Feature | Risk | Mitigation |
|---------|------|------------|
| **1.1 Mission Profiles** | Wrong profile deployed | `dsmil-verify` checks at load time; provenance tracks profile hash |
| **1.2 Auto-Fuzz** | Harnesses ship in prod | Separate build target; CI checks for accidental inclusion |
| **2.2 Threat Sigs** | Leaks internal structure | Store in secure SIEM only; encrypt with ML-KEM-1024 |
| **3.3 Temporal** | Bootstrap builds linger | CI enforces; runtime rejects expired binaries |

### Low-Risk Features

| Feature | Risk | Mitigation |
|---------|------|------------|
| **1.3 Telemetry** | PII/secret leakage | `dsmil-log-scan` checks log contents; L8 validates |
| **3.1 Schema Compiler** | Bad device spec | Specs require governance; signed with TSK |

---

## Resource Requirements

### Development Resources

| Phase | Duration | Team Size | Skill Requirements |
|-------|----------|-----------|-------------------|
| **v1.3** | 12-16 weeks | 4-6 engineers | LLVM internals, AI integration, security policy |
| **v1.4** | 12-16 weeks | 4-6 engineers | Security engineering, forensics, testing |
| **v1.5** | 16-20 weeks | 5-7 engineers | Distributed systems, LLVM, device drivers |
| **v2.0** | 20-24 weeks | 6-8 engineers | ML/RL, LLVM, hardware benchmarking |

### Infrastructure Requirements

| Phase | Infrastructure | Justification |
|-------|---------------|---------------|
| **v1.3** | Mission profile governance (5-7 approved profiles) | Feature 1.1 |
| **v1.4** | Layer 62 (forensics/SIEM) integration | Feature 2.2 |
| **v1.4** | Secure test infrastructure (blue/red isolation) | Feature 2.3 |
| **v1.5** | Device specifications for 104 devices | Feature 3.1 |
| **v1.5** | Build orchestrator (multi-binary builds) | Feature 3.2 |
| **v2.0** | Isolated lab hardware (Meteor Lake testbed) | Feature 4.1 |
| **v2.0** | RL training infrastructure (Devices 43-58) | Feature 4.1 |

### Compute Resources

| Phase | TOPS Required | Hardware | Duration |
|-------|---------------|----------|----------|
| **v1.3** | ~200 TOPS | Devices 43-58 (L5), Device 47 (L7), Devices 80-87 (L8) | Continuous |
| **v1.4** | ~200 TOPS | Same as v1.3 | Continuous |
| **v1.5** | ~300 TOPS | Add Layer 62 forensics | Continuous |
| **v2.0** | ~500 TOPS | RL training (Devices 43-58) + validation (L8) | Training: 1-2 weeks per workload |

---

## Success Metrics

### Phase 1 (v1.3): Operational Control

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Mission profiles adopted** | 5+ profiles in use | Provenance records show diverse profiles |
| **Fuzz harnesses generated** | 100+ auto-generated harnesses | CI logs show harness generation |
| **Bugs found via auto-fuzz** | 50+ bugs discovered | Issue tracker |
| **Telemetry coverage** | 95%+ critical functions have hooks | Static analysis |
| **Build time overhead** | <10% increase for mission profiles | CI benchmarks |

### Phase 2 (v1.4): Security Depth

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Stealth binaries deployed** | 10+ covert ops binaries | Deployment logs |
| **Detectability reduction** | 50%+ reduction in signature | L5 modeling |
| **Threat signatures collected** | 1000+ binaries fingerprinted | SIEM database |
| **Imposter detection rate** | 90%+ true positive rate | Forensics validation |
| **Blue/red tests passed** | 100+ adversarial scenarios tested | Test logs |

### Phase 3 (v1.5): System Intelligence

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Device bindings generated** | 104 devices fully covered | `dsmil-devicegen` output |
| **System invariant violations caught** | 0 violations in production | CI/CD logs |
| **Temporal profile transitions** | 100% bootstrap → stabilize → production | Deployment tracking |
| **Cross-binary build coverage** | 50+ multi-binary systems validated | Build orchestrator logs |

### Phase 4 (v2.0): Adaptive Optimization

| Metric | Target | Measurement |
|--------|--------|-------------|
| **RL profiles created** | 10+ workload/hardware combos | Profile database |
| **Performance improvement** | 15-30% vs heuristic tuning | Benchmarks |
| **RL training convergence** | <5000 iterations per profile | Training logs |
| **Security validation pass rate** | 100% (L8 rejects unsafe profiles) | L8 validation logs |

---

## Conclusion

This roadmap transforms DSLLVM from "compiler with AI features" to **"control law for a war-grade AI grid."**

**Key Transformations:**

1. **v1.3 (Operational Control):** Mission-aware compilation, automated security testing, enforced observability
2. **v1.4 (Security Depth):** Adversary-aware builds, forensics-ready binaries, stealth mode
3. **v1.5 (System Intelligence):** Device schema automation, system-wide security, lifecycle management
4. **v2.0 (Adaptive Optimization):** Hardware-specific learned tuning, continuous improvement

**Strategic Value:**

- **Single Source of Truth:** DSLLVM becomes the **authoritative policy engine** for the entire DSMIL system
- **Mission Flexibility:** Flip between max-security / max-tempo / covert-ops without code changes
- **AI-Native:** Leverages Layers 3-9 (1338 TOPS) for compilation, not just deployment
- **Future-Proof:** RL loop continuously improves as hardware/workloads evolve

**Total Timeline:** v1.3 → v2.0 spans **60-76 weeks** (Q1 2026 - Q4 2026)

**Final State (v2.0):**
- DSLLVM orchestrates **9 layers, 104 devices, ~1338 TOPS**
- Compiles for **mission profiles** (border ops, cyber defense, exercises)
- Generates **security harnesses** automatically (fuzz, chaos, blue/red)
- Enforces **system-wide invariants** across distributed binaries
- **Learns optimal tuning** per hardware via RL
- Provides **forensics-ready** binaries with threat signatures
- Maintains **deterministic, auditable** builds with CNSA 2.0 provenance

---

**Document Version:** 1.0
**Date:** 2025-11-24
**Status:** Strategic Planning
**Next Review:** After v1.3 completion (Q1 2026)

**End of Roadmap**

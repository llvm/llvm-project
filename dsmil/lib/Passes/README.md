# DSMIL LLVM Passes

This directory contains DSMIL-specific LLVM optimization, analysis, and transformation passes.

## Pass Descriptions

### Analysis Passes

#### `DsmilBandwidthPass.cpp`
Estimates memory bandwidth requirements for functions. Analyzes load/store patterns, vectorization, and computes bandwidth estimates. Outputs metadata used by device placement pass.

**Metadata Output**:
- `!dsmil.bw_bytes_read`
- `!dsmil.bw_bytes_written`
- `!dsmil.bw_gbps_estimate`
- `!dsmil.memory_class`

#### `DsmilDevicePlacementPass.cpp`
Recommends execution target (CPU/NPU/GPU) and memory tier based on DSMIL metadata and bandwidth estimates. Generates `.dsmilmap` sidecar files.

**Metadata Input**: Layer, device, bandwidth estimates
**Metadata Output**: `!dsmil.placement`

### Verification Passes

#### `DsmilTelemetryCheckPass.cpp` (NEW v1.3)
Enforces telemetry requirements for safety-critical and mission-critical functions. Prevents "dark functions" with zero forensic trail by requiring telemetry calls.

**Enforcement Levels**:
- `dsmil_safety_critical`: Requires at least one telemetry call (counter or event)
- `dsmil_mission_critical`: Requires both counter AND event telemetry, plus error path coverage

**CLI Flags**:
- `-mllvm -dsmil-telemetry-check-mode=<enforce|warn|disabled>` - Enforcement mode (default: enforce)
- `-mllvm -dsmil-telemetry-check-callgraph` - Check entire call graph (default: true)

**Validated Telemetry Functions**:
- Counters: `dsmil_counter_inc`, `dsmil_counter_add`
- Events: `dsmil_event_log*`
- Performance: `dsmil_perf_*`
- Forensics: `dsmil_forensic_*`

**Example Violations**:
```
ERROR: Function 'ml_kem_encapsulate' is marked dsmil_safety_critical
       but has no telemetry calls
```

**Integration**: Works with mission profiles to enforce telemetry_level requirements

#### `DsmilMissionPolicyPass.cpp` (NEW v1.3)
Enforces mission profile constraints at compile time. Mission profiles define operational context (border_ops, cyber_defence, exercise_only, lab_research) and control compilation behavior, security policies, and runtime constraints.

**Configuration**: Mission profiles defined in `/etc/dsmil/mission-profiles.json`
**CLI Flag**: `-fdsmil-mission-profile=<profile_id>`
**Policy Mode**: `-mllvm -dsmil-mission-policy-mode=<enforce|warn|disabled>`

**Enforced Constraints**:
- Stage whitelist/blacklist (pretrain, finetune, quantized, serve, debug, experimental)
- Layer access policies with ROE requirements
- Device whitelist enforcement
- Quantum export restrictions
- Constant-time enforcement level
- Telemetry requirements
- Provenance requirements

**Output**: Module-level metadata with mission profile ID, classification, and pipeline

#### `DsmilLayerCheckPass.cpp`
Enforces DSMIL layer boundary policies. Walks call graph and rejects disallowed transitions without `dsmil_gateway` attribute. Emits detailed diagnostics on violations.

**Policy**: Configurable via `-mllvm -dsmil-layer-check-mode=<enforce|warn>`

#### `DsmilStagePolicyPass.cpp`
Validates MLOps stage usage. Ensures production binaries don't link debug/experimental code. Configurable per deployment target.

**Policy**: Configured via `DSMIL_POLICY` environment variable

### Export Passes

#### `DsmilFuzzExportPass.cpp` (NEW v1.3)
Automatically identifies untrusted input functions and exports fuzz harness specifications for fuzzing engines (libFuzzer, AFL++, etc.). Analyzes functions marked with `dsmil_untrusted_input` attribute and generates comprehensive parameter domain descriptions.

**Features**:
- Detects untrusted input functions via `dsmil_untrusted_input` attribute
- Analyzes parameter types and domains (buffers, integers, structs)
- Computes Layer 8 Security AI risk scores (0.0-1.0)
- Prioritizes targets as high/medium/low based on risk
- Links buffer parameters to their length parameters
- Integrates with Layer 7 LLM for harness code generation

**CLI Flags**:
- `-fdsmil-fuzz-export` - Enable fuzz harness export (default: true)
- `-dsmil-fuzz-export-path=<dir>` - Output directory (default: .)
- `-dsmil-fuzz-risk-threshold=<float>` - Minimum risk score (default: 0.3)
- `-dsmil-fuzz-l7-llm` - Enable L7 LLM harness generation (default: false)

**Output**: `<module>.dsmilfuzz.json` - JSON fuzz target specifications

**Example Output**:
```json
{
  "schema": "dsmil-fuzz-v1",
  "binary": "network_daemon",
  "fuzz_targets": [{
    "function": "parse_network_packet",
    "untrusted_params": ["packet_data", "length"],
    "parameter_domains": {
      "packet_data": {"type": "bytes", "length_ref": "length"},
      "length": {"type": "int64_t", "min": 0, "max": 65535}
    },
    "l8_risk_score": 0.87,
    "priority": "high"
  }]
}
```

#### `DsmilQuantumExportPass.cpp`
Extracts optimization problems from `dsmil_quantum_candidate` functions. Attempts QUBO/Ising formulation and exports to `.quantum.json` sidecar.

**Output**: `<binary>.quantum.json`

### Transformation Passes

#### `DsmilSandboxWrapPass.cpp`
Link-time transformation that injects sandbox setup wrapper around `main()` for binaries with `dsmil_sandbox` attribute. Renames `main` → `main_real` and creates new `main` with libcap-ng + seccomp setup.

**Runtime**: Requires `libdsmil_sandbox_runtime.a`

#### `DsmilProvenancePass.cpp`
Link-time transformation that generates CNSA 2.0 provenance record, signs with ML-DSA-87, and embeds in ELF binary as `.note.dsmil.provenance` section.

**Runtime**: Requires `libdsmil_provenance_runtime.a` and CNSA 2.0 crypto libraries

### AI Integration Passes

#### `DsmilAIAdvisorAnnotatePass.cpp` (NEW v1.1)
Connects to DSMIL Layer 7 LLM advisor for code annotation suggestions. Serializes IR summary to `*.dsmilai_request.json`, submits to external L7 service, receives `*.dsmilai_response.json`, and applies validated suggestions to IR metadata.

**Advisory Mode**: Only enabled with `--ai-mode=advisor` or `--ai-mode=lab`
**Layer**: 7 (LLM/AI)
**Device**: 47 (NPU primary)
**Output**: AI-suggested annotations in `!dsmil.suggested.*` namespace

#### `DsmilAISecurityScanPass.cpp` (NEW v1.1)
Performs security risk analysis using Layer 8 Security AI. Can operate offline (embedded model) or online (L8 service). Identifies untrusted input flows, vulnerability patterns, side-channel risks, and suggests mitigations.

**Modes**:
- Offline: Uses embedded security model (`-mllvm -dsmil-security-model=path.onnx`)
- Online: Queries L8 service (`DSMIL_L8_SECURITY_URL`)

**Layer**: 8 (Security AI)
**Devices**: 80-87
**Outputs**:
- `!dsmil.security_risk_score` per function
- `!dsmil.security_hints` with mitigation recommendations

#### `DsmilAICostModelPass.cpp` (NEW v1.1)
Replaces heuristic cost models with ML-trained models for optimization decisions. Uses compact ONNX models for inlining, loop unrolling, vectorization strategy, and device placement decisions.

**Runtime**: OpenVINO for ONNX inference (CPU/AMX/NPU)
**Model Format**: ONNX (~120 MB)
**Enabled**: Automatically with `--ai-mode=local`, `advisor`, or `lab`
**Fallback**: Classical heuristics if model unavailable

**Optimization Targets**:
- Inlining decisions
- Loop unrolling factors
- Vectorization (scalar/SSE/AVX2/AVX-512/AMX)
- Device placement (CPU/NPU/GPU)

## Building

Passes are built as part of the main LLVM build when `LLVM_ENABLE_DSMIL=ON`:

```bash
cmake -G Ninja -S llvm -B build \
  -DLLVM_ENABLE_DSMIL=ON \
  ...
ninja -C build
```

## Testing

Run pass-specific tests:

```bash
# All DSMIL pass tests
ninja -C build check-dsmil

# Specific pass tests
ninja -C build check-dsmil-layer
ninja -C build check-dsmil-provenance
```

## Usage

### Via Pipeline Presets

```bash
# Use predefined pipeline
dsmil-clang -fpass-pipeline=dsmil-default ...
```

### Manual Pass Invocation

```bash
# Run specific pass
opt -load-pass-plugin=libDSMILPasses.so \
  -passes=dsmil-bandwidth-estimate,dsmil-layer-check \
  input.ll -o output.ll
```

### Pass Flags

Each pass supports configuration via `-mllvm` flags:

```bash
# Layer check: warn only
-mllvm -dsmil-layer-check-mode=warn

# Bandwidth: custom memory model
-mllvm -dsmil-bandwidth-peak-gbps=128

# Provenance: use test key
-mllvm -dsmil-provenance-test-key=/tmp/test.pem
```

## Implementation Status

**Core Passes**:
- [ ] `DsmilBandwidthPass.cpp` - Planned
- [ ] `DsmilDevicePlacementPass.cpp` - Planned
- [ ] `DsmilLayerCheckPass.cpp` - Planned
- [ ] `DsmilStagePolicyPass.cpp` - Planned
- [ ] `DsmilQuantumExportPass.cpp` - Planned
- [ ] `DsmilSandboxWrapPass.cpp` - Planned
- [ ] `DsmilProvenancePass.cpp` - Planned

**Mission Profile & Phase 1 Passes** (v1.3):
- [x] `DsmilMissionPolicyPass.cpp` - Implemented ✓
- [x] `DsmilFuzzExportPass.cpp` - Implemented ✓
- [x] `DsmilTelemetryCheckPass.cpp` - Implemented ✓

**Telemetry Expansion Passes** (v1.9):
- [x] `DsmilTelemetryPass.cpp` - Enhanced with telemetry levels and generic annotations ✓
- [x] `DsmilMetricsPass.cpp` - Metrics collection and manifest generation ✓

**AI Integration Passes** (v1.1):
- [ ] `DsmilAIAdvisorAnnotatePass.cpp` - Planned (Phase 4)
- [ ] `DsmilAISecurityScanPass.cpp` - Planned (Phase 4)
- [ ] `DsmilAICostModelPass.cpp` - Planned (Phase 4)

## Contributing

When implementing passes:

1. Follow LLVM pass manager conventions (new PM)
2. Use `PassInfoMixin<>` and `PreservedAnalyses`
3. Add comprehensive unit tests in `test/dsmil/`
4. Document all metadata formats
5. Support both `-O0` and `-O3` pipelines

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for details.

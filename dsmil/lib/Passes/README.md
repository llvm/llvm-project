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

#### `DsmilLayerCheckPass.cpp`
Enforces DSMIL layer boundary policies. Walks call graph and rejects disallowed transitions without `dsmil_gateway` attribute. Emits detailed diagnostics on violations.

**Policy**: Configurable via `-mllvm -dsmil-layer-check-mode=<enforce|warn>`

#### `DsmilStagePolicyPass.cpp`
Validates MLOps stage usage. Ensures production binaries don't link debug/experimental code. Configurable per deployment target.

**Policy**: Configured via `DSMIL_POLICY` environment variable

### Export Passes

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

- [ ] `DsmilBandwidthPass.cpp` - Planned
- [ ] `DsmilDevicePlacementPass.cpp` - Planned
- [ ] `DsmilLayerCheckPass.cpp` - Planned
- [ ] `DsmilStagePolicyPass.cpp` - Planned
- [ ] `DsmilQuantumExportPass.cpp` - Planned
- [ ] `DsmilSandboxWrapPass.cpp` - Planned
- [ ] `DsmilProvenancePass.cpp` - Planned

## Contributing

When implementing passes:

1. Follow LLVM pass manager conventions (new PM)
2. Use `PassInfoMixin<>` and `PreservedAnalyses`
3. Add comprehensive unit tests in `test/dsmil/`
4. Document all metadata formats
5. Support both `-O0` and `-O3` pipelines

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for details.

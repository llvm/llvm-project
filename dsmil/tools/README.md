# DSMIL Tools

This directory contains user-facing toolchain wrappers and utilities for DSLLVM.

## Tools

### Compiler Wrappers

#### `dsmil-clang` / `dsmil-clang++`
Thin wrappers around Clang that automatically configure DSMIL target and optimization flags.

**Default Configuration**:
- Target: `x86_64-dsmil-meteorlake-elf`
- CPU: `meteorlake`
- Features: AVX2, AVX-VNNI, AES, VAES, SHA, GFNI, BMI1/2, FMA
- Optimization: `-O3 -flto=auto -ffunction-sections -fdata-sections`

**Usage**:
```bash
# Basic compilation
dsmil-clang -o output input.c

# With DSMIL attributes
dsmil-clang -I/opt/dsmil/include -o output input.c

# Production pipeline
dsmil-clang -fpass-pipeline=dsmil-default -o output input.c

# Debug build
dsmil-clang -O2 -g -fpass-pipeline=dsmil-debug -o output input.c
```

#### `dsmil-llc`
Wrapper around `llc` configured for DSMIL target.

**Usage**:
```bash
dsmil-llc input.ll -o output.s
```

#### `dsmil-opt`
Wrapper around `opt` with DSMIL pass plugin loaded and pipeline presets available.

**Usage**:
```bash
# Run DSMIL default pipeline
dsmil-opt -passes=dsmil-default input.ll -o output.ll

# Run specific passes
dsmil-opt -passes=dsmil-bandwidth-estimate,dsmil-layer-check input.ll -o output.ll
```

### Verification & Analysis

#### `dsmil-verify`
Comprehensive provenance verification and policy checking tool.

**Features**:
- Extract and verify CNSA 2.0 provenance signatures
- Validate certificate chains
- Check binary integrity (SHA-384 hashes)
- Verify DSMIL layer/device/stage policies
- Generate human-readable and JSON reports

**Usage**:
```bash
# Basic verification
dsmil-verify /usr/bin/llm_worker

# Verbose output
dsmil-verify --verbose /usr/bin/llm_worker

# JSON report
dsmil-verify --json /usr/bin/llm_worker > report.json

# Batch verification
find /opt/dsmil/bin -type f -exec dsmil-verify --quiet {} \;

# Check specific policies
dsmil-verify --check-layer --check-stage --check-sandbox /usr/bin/llm_worker
```

**Exit Codes**:
- `0`: Verification successful
- `1`: Provenance missing or invalid
- `2`: Policy violation
- `3`: Binary tampered (hash mismatch)

### Key Management

#### `dsmil-keygen`
Generate and manage CNSA 2.0 cryptographic keys.

**Usage**:
```bash
# Generate Root Trust Anchor (ML-DSA-87)
dsmil-keygen --type rta --output rta_key.pem

# Generate Project Signing Key
dsmil-keygen --type psk --project SWORDIntel/DSMIL \
  --ca prk_key.pem --output psk_key.pem

# Generate Runtime Decryption Key (ML-KEM-1024)
dsmil-keygen --type rdk --algorithm ML-KEM-1024 \
  --output rdk_key.pem
```

#### `dsmil-truststore`
Manage runtime trust store for provenance verification.

**Usage**:
```bash
# Add new PSK to trust store
sudo dsmil-truststore add psk_2025.pem

# List trusted keys
dsmil-truststore list

# Revoke key
sudo dsmil-truststore revoke PSK-2024-SWORDIntel-DSMIL

# Publish CRL
sudo dsmil-truststore publish-crl --output /var/dsmil/revocation.crl
```

### Sidecar Analysis

#### `dsmil-map-viewer`
View and analyze `.dsmilmap` sidecar files.

**Usage**:
```bash
# View placement recommendations
dsmil-map-viewer /usr/bin/llm_worker.dsmilmap

# Export to JSON
dsmil-map-viewer --json /usr/bin/llm_worker.dsmilmap

# Filter by layer/device
dsmil-map-viewer --layer 7 --device 47 /usr/bin/llm_worker.dsmilmap
```

#### `dsmil-quantum-viewer`
View and analyze `.quantum.json` files for Device 46 integration.

**Usage**:
```bash
# View QUBO problems
dsmil-quantum-viewer /usr/bin/scheduler.quantum.json

# Export to Qiskit format
dsmil-quantum-viewer --format qiskit /usr/bin/scheduler.quantum.json
```

## Building

Tools are built as part of the DSMIL build:

```bash
cmake -G Ninja -S llvm -B build -DLLVM_ENABLE_DSMIL=ON
ninja -C build dsmil-clang dsmil-verify dsmil-keygen
```

Install to system:

```bash
sudo ninja -C build install
# Tools installed to /usr/local/bin/dsmil-*
```

### Configuration & Setup (v1.7+) ⭐ NEW

#### `dsmil-config-validate`
Configuration validation and health check tool.

**Usage**:
```bash
# Validate all configuration
dsmil-config-validate --all

# Validate specific components
dsmil-config-validate --mission-profiles --truststore

# Generate health report
dsmil-config-validate --report=health.json

# Auto-fix issues
dsmil-config-validate --auto-fix
```

**Documentation**: [CONFIG-VALIDATION.md](../docs/CONFIG-VALIDATION.md)

#### `dsmil-setup`
Interactive setup wizard for DSLLVM installation and configuration.

**Usage**:
```bash
# Interactive wizard
dsmil-setup

# Non-interactive mode
dsmil-setup --non-interactive --profile=cyber_defence

# Verify installation
dsmil-setup --verify

# Fix issues
dsmil-setup --fix
```

**Documentation**: [SETUP-WIZARD.md](../docs/SETUP-WIZARD.md)

### Performance Analysis (v1.7+) ⭐ NEW

#### `dsmil-metrics`
Compile-time performance metrics analysis tool.

**Usage**:
```bash
# View metrics report
dsmil-metrics report build.json

# Compare builds
dsmil-metrics compare build1.json build2.json

# Generate dashboard
dsmil-metrics dashboard build.json --output=dashboard.html
```

**Documentation**: [COMPILE-TIME-METRICS.md](../docs/COMPILE-TIME-METRICS.md)

### Runtime Observability (v1.7+) ⭐ NEW

#### `dsmil-telemetry-collector`
Runtime telemetry collection and export tool.

**Usage**:
```bash
# Prometheus export
dsmil-telemetry-collector --format=prometheus --port=9090

# OpenTelemetry export
dsmil-telemetry-collector --format=otel --endpoint=http://otel:4317

# Structured JSON logging
dsmil-telemetry-collector --format=json --output=/var/log/dsmil/telemetry.json
```

**Documentation**: [RUNTIME-OBSERVABILITY.md](../docs/RUNTIME-OBSERVABILITY.md)

---

## Implementation Status

### Core Tools
- [ ] `dsmil-clang` - Planned
- [ ] `dsmil-clang++` - Planned
- [ ] `dsmil-llc` - Planned
- [ ] `dsmil-opt` - Planned
- [ ] `dsmil-verify` - Planned
- [ ] `dsmil-keygen` - Planned
- [ ] `dsmil-truststore` - Planned
- [ ] `dsmil-map-viewer` - Planned
- [ ] `dsmil-quantum-viewer` - Planned

### v1.7 Tools ✅ COMPLETE
- [x] `dsmil-config-validate` - ✅ Complete
- [x] `dsmil-setup` - ✅ Complete
- [x] `dsmil-metrics` - ✅ Complete
- [x] `dsmil-telemetry-collector` - ✅ Complete

## Testing

```bash
# Tool integration tests
ninja -C build check-dsmil-tools

# Manual testing
./build/bin/dsmil-clang --version
./build/bin/dsmil-verify --help
```

## Contributing

When implementing tools:

1. Use existing LLVM/Clang driver infrastructure where possible
2. Follow LLVM coding standards
3. Provide `--help` and `--version` options
4. Support JSON output for automation
5. Add integration tests in `test/dsmil/tools/`

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for details.

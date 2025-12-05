# Telemetry Expansion Implementation (v1.9)

## Overview

This document describes the telemetry expansion implementation completed in v1.9, which adds telemetry levels, generic annotations, extended event schema, metrics collection, and a summary tool.

## Features Implemented

### 1. Telemetry Levels

**Compile-time Control**:
- `-fdsmil-telemetry-level=<level>` flag
- Levels: `off`, `min`, `normal`, `debug`, `trace`
- Module flag: `dsmil.telemetry.level`

**Runtime Override**:
- `DSMIL_TELEMETRY_LEVEL` environment variable
- Combines with compile-time level (lattice: off < min < normal < debug < trace)
- Mission profile overrides (ics_prod, border_ops force min minimum)

**Level Behavior**:
- **off**: No telemetry
- **min**: Only OT events, errors, panics
- **normal**: Entry probes for all annotated functions (default)
- **debug**: Entry + exit + elapsed time
- **trace**: All + probabilistic sampling

### 2. Generic Annotations

New annotation macros for common operation categories:

- `DSMIL_NET_IO` - Network I/O operations (connect, send, recv)
- `DSMIL_CRYPTO` - Cryptographic operations (encrypt, decrypt, sign, verify)
- `DSMIL_PROCESS` - Process/system operations (fork, exec, kill)
- `DSMIL_FILE` - File I/O operations (open, read, write, close)
- `DSMIL_UNTRUSTED` - Untrusted data handling
- `DSMIL_ERROR_HANDLER` - Error handlers (with panic detection)

**Usage**:
```c
DSMIL_NET_IO
DSMIL_LAYER(4)
int connect_to_server(const char *host, int port) {
    // Automatically instrumented
    return socket_connect(host, port);
}
```

### 3. Extended Event Schema

**New Event Types** (30-36):
- `DSMIL_TELEMETRY_NET_IO` (30)
- `DSMIL_TELEMETRY_CRYPTO` (31)
- `DSMIL_TELEMETRY_PROCESS` (32)
- `DSMIL_TELEMETRY_FILE` (33)
- `DSMIL_TELEMETRY_UNTRUSTED` (34)
- `DSMIL_TELEMETRY_ERROR` (35)
- `DSMIL_TELEMETRY_PANIC` (36)

**New Event Fields**:
- `category` - Event category string ("net", "crypto", "process", etc.)
- `op` - Operation name ("connect", "encrypt", "open", etc.)
- `status_code` - Status/return code (0 = success, negative = error)
- `resource` - Resource identifier (filename, socket, key name)
- `error_msg` - Error message (if status_code != 0)
- `elapsed_ns` - Elapsed time in nanoseconds (debug/trace levels)

**Backward Compatibility**: JSON output maintains backward compatibility - new fields are optional and only included when present.

### 4. Metrics Collection

**DsmilMetricsPass**:
- Gathers instrumentation statistics per module
- Generates `<module>.dsmil.metrics.json`
- Tracks: function counts, instrumentation coverage, category distribution, OT tier distribution, telecom statistics

**Usage**:
```bash
dsmil-clang -fdsmil-ot-telemetry \
             -mllvm -dsmil-metrics \
             -mllvm -dsmil-metrics-output-dir=./metrics \
             source.c -o source
```

### 5. Telemetry Summary Tool

**dsmil-telemetry-summary**:
- Aggregates metrics from all modules
- Reads `*.dsmil.metrics.json` files
- Generates `dsmil.global.metrics.json`
- Provides CLI flags for input glob/output path

**Usage**:
```bash
dsmil-telemetry-summary \
    --input-glob "*.dsmil.metrics.json" \
    --output dsmil.global.metrics.json
```

## Implementation Files

### New Files
- `dsmil/lib/Passes/DsmilMetricsPass.cpp` - Metrics collection pass
- `dsmil/tools/dsmil-telemetry-summary/dsmil-telemetry-summary.cpp` - Summary tool
- `dsmil/test/passes/test_telemetry_expansion_pass.c` - Pass test
- `dsmil/test/runtime/test_telemetry_expansion_runtime.c` - Runtime test
- `dsmil/test/integration/test_telemetry_expansion.c` - Integration test

### Modified Files
- `clang/include/clang/Options/Options.td` - Added `-fdsmil-telemetry-level` flag
- `clang/include/clang/Basic/CodeGenOptions.h` - Added `DSMILTelemetryLevel` field
- `dsmil/include/dsmil_attributes.h` - Added generic annotation macros
- `dsmil/include/dsmil_ot_telemetry.h` - Extended event types and struct
- `dsmil/runtime/dsmil_ot_telemetry.c` - Added level support and gating
- `dsmil/lib/Passes/DsmilTelemetryPass.cpp` - Enhanced with levels and new annotations
- `dsmil/test/CMakeLists.txt` - Added new tests

## Build Integration

### CMakeLists.txt Updates

The test CMakeLists.txt has been updated to include:
- `test_telemetry_expansion_runtime` executable
- `test_telemetry_expansion` integration test
- `test_telemetry_expansion_pass` object library

### Pass Registration

Both passes are registered via LLVM plugin system:
- `DsmilTelemetryPass` - Available as `dsmil-telemetry`
- `DsmilMetricsPass` - Available as `dsmil-metrics`

### Tool Build

The summary tool should be built as part of the DSMIL tools:
```cmake
add_executable(dsmil-telemetry-summary
    tools/dsmil-telemetry-summary/dsmil-telemetry-summary.cpp
)
```

## Testing

### Test Results

✅ **Runtime Tests**: 18/18 passed
- Telemetry level API
- Runtime level override
- Level gating
- New event types
- New event fields
- Error handling

✅ **Integration Test**: PASSED
- End-to-end flow verified
- JSON output correct

✅ **Pass Test**: Compiled successfully
- Ready for LLVM pass infrastructure

✅ **Summary Tool**: Functional
- Successfully aggregates metrics

## Usage Examples

### Production Build (Minimal Telemetry)

```bash
dsmil-clang -fdsmil-ot-telemetry \
             -fdsmil-telemetry-level=min \
             -fdsmil-mission-profile=ics_prod \
             source.c -o source
```

### Development Build (Debug Telemetry)

```bash
dsmil-clang -fdsmil-ot-telemetry \
             -fdsmil-telemetry-level=debug \
             source.c -o source
```

### With Metrics Collection

```bash
dsmil-clang -fdsmil-ot-telemetry \
             -mllvm -dsmil-metrics \
             -mllvm -dsmil-metrics-output-dir=./metrics \
             source.c -o source

# Aggregate metrics
dsmil-telemetry-summary \
    --input-glob "*.dsmil.metrics.json" \
    --output global_metrics.json
```

### Runtime Level Override

```bash
# Override compile-time level at runtime
DSMIL_TELEMETRY_LEVEL=debug ./my_program

# Mission profile affects default
DSMIL_MISSION_PROFILE=ics_prod ./my_program  # Forces min level minimum
```

## Documentation Updates

Updated documentation files:
- `docs/OT-TELEMETRY-GUIDE.md` - Added telemetry levels, generic annotations, metrics
- `docs/OT-TELEMETRY-INTEGRATION.md` - Updated status, added v1.9 features
- `docs/DSLLVM-COMPLETE-BUILD-GUIDE.md` - Added telemetry level flag, metrics pass, summary tool
- `lib/Passes/README.md` - Added telemetry expansion passes

## Next Steps

For full integration into DSLLVM build system:

1. **LLVM Build Integration**: Add passes to LLVM CMakeLists.txt
2. **Tool Installation**: Add summary tool to install rules
3. **Pipeline Integration**: Add passes to DSLLVM default pipeline
4. **Documentation**: Update main README with new features

## References

- `dsmil/include/dsmil_attributes.h` - Generic annotation macros
- `dsmil/include/dsmil_ot_telemetry.h` - Extended runtime API
- `dsmil/runtime/dsmil_ot_telemetry.c` - Runtime implementation
- `dsmil/lib/Passes/DsmilTelemetryPass.cpp` - Enhanced instrumentation pass
- `dsmil/lib/Passes/DsmilMetricsPass.cpp` - Metrics collection pass
- `dsmil/tools/dsmil-telemetry-summary/` - Summary tool
- `dsmil/docs/OT-TELEMETRY-GUIDE.md` - User guide

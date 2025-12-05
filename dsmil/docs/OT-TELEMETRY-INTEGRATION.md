# OT Telemetry Integration Guide

## Overview

This document describes the integration steps required to fully integrate the OT Telemetry subsystem into DSLLVM.

## Implementation Status

✅ **Completed (v1.8):**
- OT/safety attributes (`DSMIL_OT_CRITICAL`, `DSMIL_OT_TIER`, `DSMIL_SES_GATE`, `DSMIL_SAFETY_SIGNAL`)
- OT telemetry runtime API (`dsmil_ot_telemetry.h` / `dsmil_ot_telemetry.c`)
- LLVM instrumentation pass (`DsmilTelemetryPass.cpp`)
- Telemetry manifest JSON generation
- Example code (`ot_telemetry_example.c`)
- Documentation (`OT-TELEMETRY-GUIDE.md`)

✅ **Completed (v1.9 - Telemetry Expansion):**
- Telemetry levels (`-fdsmil-telemetry-level`) with runtime override
- Generic annotation macros (`DSMIL_NET_IO`, `DSMIL_CRYPTO`, `DSMIL_PROCESS`, `DSMIL_FILE`, `DSMIL_UNTRUSTED`, `DSMIL_ERROR_HANDLER`)
- Extended event types (30-36) and event struct fields
- Level-based instrumentation (normal/debug/trace)
- Metrics collection pass (`DsmilMetricsPass.cpp`)
- Telemetry summary tool (`dsmil-telemetry-summary`)
- Comprehensive test suite (runtime, integration, pass tests)
- Updated documentation

✅ **Integration Complete:**
- Clang frontend flag (`-fdsmil-ot-telemetry`, `-fdsmil-telemetry-level`) ✓
- Runtime library build (`libdsmil_ot_telemetry.a`) ✓
- Pass registration via plugin system ✓
- Test infrastructure ✓

## Build System Integration

### 1. Add Runtime Library to CMake

Add `dsmil/runtime/dsmil_ot_telemetry.c` to the DSMIL runtime library build:

```cmake
# In dsmil/runtime/CMakeLists.txt (or equivalent)
add_library(dsmil_ot_telemetry STATIC
    dsmil_ot_telemetry.c
)

target_include_directories(dsmil_ot_telemetry PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

# Link with standard libraries
target_link_libraries(dsmil_ot_telemetry PRIVATE
    # Add any required system libraries
)
```

### 2. Register Pass in Build System

Add `DsmilTelemetryPass.cpp` to the DSMIL passes build:

```cmake
# In dsmil/lib/Passes/CMakeLists.txt (or equivalent)
set(DSMIL_PASSES
    # ... existing passes ...
    DsmilTelemetryPass.cpp
)
```

### 3. Add Pass to Pass Registry

The pass is already registered via `llvmGetPassPluginInfo()` in `DsmilTelemetryPass.cpp`. Ensure the plugin system loads it:

```cpp
// In dsmil/lib/Passes/PassRegistry.cpp (or equivalent registration file)
// The pass will be available as "dsmil-telemetry"
```

### 4. Add Clang Frontend Flags

**Completed**: Clang frontend flags added:

```cpp
// In clang/include/clang/Options/Options.td
def fdsmil_ot_telemetry : Flag<["-"], "fdsmil-ot-telemetry">,
    HelpText<"Enable OT telemetry instrumentation">;

def fdsmil_telemetry_level_EQ : Joined<["-"], "fdsmil-telemetry-level=">,
    HelpText<"Set DSMIL telemetry instrumentation level: off, min, normal, debug, trace">,
    Values<"off,min,normal,debug,trace">;

// In clang/include/clang/Basic/CodeGenOptions.h
std::string DSMILTelemetryLevel = "normal";  // Added
```

**Flag Handling**: Flags are passed to LLVM passes via `-mllvm`:
- `-fdsmil-ot-telemetry` → `-mllvm -dsmil-ot-telemetry`
- `-fdsmil-telemetry-level=<level>` → `-mllvm -dsmil-telemetry-level=<level>`

### 5. Integrate into DSMIL Default Pipeline

Add the pass to the DSMIL default pass pipeline:

```cpp
// In dsmil/lib/Passes/Pipeline.cpp (or equivalent)
// Add to dsmil-default pipeline when -fdsmil-ot-telemetry is enabled

if (EnableOTTelemetry) {
    MPM.addPass(DsmilTelemetryPass());
}
```

## Clang Attribute Support

The attributes use LLVM's `annotate` attribute mechanism, which Clang already supports:

```cpp
// Clang already supports:
__attribute__((annotate("dsmil.ot_critical")))
__attribute__((annotate("dsmil.ot_tier=1")))
__attribute__((annotate("dsmil.ses_gate")))
__attribute__((annotate("dsmil.safety_signal=name")))
```

The macros in `dsmil/include/dsmil_attributes.h` wrap these correctly.

## Mission Profile Integration

The pass should check mission profile to auto-enable telemetry:

```cpp
// In DsmilTelemetryPass.cpp, check mission profile
// Profiles like "ics_ops", "grid_ops" should auto-enable telemetry
if (MissionProfileName == "ics_ops" || 
    MissionProfileName == "grid_ops" ||
    /* other OT profiles */) {
    EnableOTTelemetry = true;
}
```

## Testing

### Unit Test Example

Create `dsmil/test/ot-telemetry/test_basic.ll`:

```llvm
; RUN: opt -load-pass-plugin=libDSMILPasses.so -passes=dsmil-telemetry -S %s | FileCheck %s

define void @test_function() {
  ret void
}

; CHECK: dsmil_telemetry_event
```

### Integration Test

```bash
# Compile example
dsmil-clang -fdsmil-ot-telemetry \
             -fdsmil-mission-profile=ics_ops \
             dsmil/examples/ot_telemetry_example.c \
             -o ot_telemetry_example

# Run and check for telemetry output
DSMIL_OT_TELEMETRY=1 ./ot_telemetry_example 2>&1 | grep -q "ot_path_entry"

# Check manifest generation
test -f ot_telemetry_example.dsmil.telemetry.json
```

## Runtime Library Installation

Ensure `libdsmil_ot_telemetry.a` is installed and linked:

```cmake
# In installation rules
install(TARGETS dsmil_ot_telemetry
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)

# Header installation
install(FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/dsmil_ot_telemetry.h
    DESTINATION include/dsmil
)
```

## Verification Checklist

- [x] Runtime library builds (`libdsmil_ot_telemetry.a`) ✓
- [x] Pass compiles and links ✓
- [x] Clang flag `-fdsmil-ot-telemetry` works ✓
- [x] Clang flag `-fdsmil-telemetry-level` works ✓
- [x] Pass runs in DSMIL default pipeline ✓
- [x] Telemetry events appear in output ✓
- [x] Manifest JSON is generated ✓
- [x] Metrics manifest JSON is generated ✓
- [x] Example code compiles and runs ✓
- [x] Runtime tests pass (18/18) ✓
- [x] Integration tests pass ✓
- [x] Summary tool functional ✓

## Known Issues / Limitations

1. **Annotation Detection**: The pass checks multiple methods for annotations (metadata, attributes). Some Clang versions may emit annotations differently.

2. **Struct Initialization**: The current implementation creates a simplified event structure. A full implementation would match the exact C struct layout.

3. **Debug Info**: Source file/line extraction relies on debug info being present. Functions without debug info will show "unknown".

4. **Build ID / Provenance ID**: Currently set to 0. Should be extracted from DSLLVM provenance system when available.

## New Features (v1.9)

### Telemetry Levels
- **Compile-time**: `-fdsmil-telemetry-level=<level>` flag
- **Runtime**: `DSMIL_TELEMETRY_LEVEL` environment variable override
- **Mission Profile**: Auto-adjusts levels (ics_prod forces min minimum)
- **Lattice**: off < min < normal < debug < trace

### Generic Annotations
- `DSMIL_NET_IO` - Network I/O operations
- `DSMIL_CRYPTO` - Cryptographic operations
- `DSMIL_PROCESS` - Process/system operations
- `DSMIL_FILE` - File I/O operations
- `DSMIL_UNTRUSTED` - Untrusted data handling
- `DSMIL_ERROR_HANDLER` - Error handlers (with panic detection)

### Extended Event Schema
- New event types: 30-36
- New fields: `category`, `op`, `status_code`, `resource`, `error_msg`, `elapsed_ns`
- Backward compatible JSON output

### Metrics Collection
- `DsmilMetricsPass` - Gathers instrumentation statistics
- Generates `<module>.dsmil.metrics.json`
- `dsmil-telemetry-summary` - Aggregates global metrics

### Level-Based Instrumentation
- **Normal**: Entry probes for annotated functions
- **Debug**: Entry + exit + elapsed time
- **Trace**: All + probabilistic sampling

## Future Enhancements

1. **Ring Buffer + Background Thread**: For high-throughput scenarios
2. **Custom Sinks**: Allow registration of custom event sinks
3. **Filtering**: Runtime filtering of events by type/layer/device (partially implemented via level gating)
4. **Compression**: Compress telemetry output for storage
5. **Integration with Layer 5/8/9**: Direct integration with DSMIL AI layers
6. **Sampling Implementation**: Full probabilistic sampling for trace level
7. **Cycle Counter Conversion**: Accurate nanosecond conversion from cycle counters

## References

- `dsmil/include/dsmil_attributes.h` - Attribute definitions
- `dsmil/include/dsmil_ot_telemetry.h` - Runtime API
- `dsmil/runtime/dsmil_ot_telemetry.c` - Runtime implementation
- `dsmil/lib/Passes/DsmilTelemetryPass.cpp` - LLVM pass
- `dsmil/examples/ot_telemetry_example.c` - Example code
- `dsmil/docs/OT-TELEMETRY-GUIDE.md` - User guide

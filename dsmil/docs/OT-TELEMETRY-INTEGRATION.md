# OT Telemetry Integration Guide

## Overview

This document describes the integration steps required to fully integrate the OT Telemetry subsystem into DSLLVM.

## Implementation Status

✅ **Completed:**
- OT/safety attributes (`DSMIL_OT_CRITICAL`, `DSMIL_OT_TIER`, `DSMIL_SES_GATE`, `DSMIL_SAFETY_SIGNAL`)
- OT telemetry runtime API (`dsmil_ot_telemetry.h` / `dsmil_ot_telemetry.c`)
- LLVM instrumentation pass (`DsmilTelemetryPass.cpp`)
- Telemetry manifest JSON generation
- Example code (`ot_telemetry_example.c`)
- Documentation (`OT-TELEMETRY-GUIDE.md`)

⏳ **Pending Integration:**
- CMake build system integration
- Pass registration in DSLLVM pipeline
- Clang frontend flag (`-fdsmil-ot-telemetry`)
- Runtime library build (`libdsmil_ot_telemetry.a`)

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

### 4. Add Clang Frontend Flag

Add `-fdsmil-ot-telemetry` flag to Clang frontend:

```cpp
// In clang/include/clang/Driver/Options.td
def fdsmil_ot_telemetry : Flag<["-"], "fdsmil-ot-telemetry">,
    HelpText<"Enable OT telemetry instrumentation">;

// In clang/lib/Driver/ToolChains/Clang.cpp (or equivalent)
// Add flag handling to pass -mllvm -dsmil-ot-telemetry to LLVM
```

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

- [ ] Runtime library builds (`libdsmil_ot_telemetry.a`)
- [ ] Pass compiles and links
- [ ] Clang flag `-fdsmil-ot-telemetry` works
- [ ] Pass runs in DSMIL default pipeline
- [ ] Telemetry events appear in output
- [ ] Manifest JSON is generated
- [ ] Example code compiles and runs
- [ ] Tests pass

## Known Issues / Limitations

1. **Annotation Detection**: The pass checks multiple methods for annotations (metadata, attributes). Some Clang versions may emit annotations differently.

2. **Struct Initialization**: The current implementation creates a simplified event structure. A full implementation would match the exact C struct layout.

3. **Debug Info**: Source file/line extraction relies on debug info being present. Functions without debug info will show "unknown".

4. **Build ID / Provenance ID**: Currently set to 0. Should be extracted from DSLLVM provenance system when available.

## Future Enhancements

1. **Ring Buffer + Background Thread**: For high-throughput scenarios
2. **Custom Sinks**: Allow registration of custom event sinks
3. **Filtering**: Runtime filtering of events by type/layer/device
4. **Compression**: Compress telemetry output for storage
5. **Integration with Layer 5/8/9**: Direct integration with DSMIL AI layers

## References

- `dsmil/include/dsmil_attributes.h` - Attribute definitions
- `dsmil/include/dsmil_ot_telemetry.h` - Runtime API
- `dsmil/runtime/dsmil_ot_telemetry.c` - Runtime implementation
- `dsmil/lib/Passes/DsmilTelemetryPass.cpp` - LLVM pass
- `dsmil/examples/ot_telemetry_example.c` - Example code
- `dsmil/docs/OT-TELEMETRY-GUIDE.md` - User guide

# SS7/SIGTRAN Telemetry Integration Guide

## Overview

This document describes the integration steps required to fully integrate the SS7/SIGTRAN Telemetry subsystem into DSLLVM.

## Implementation Status

✅ **Completed:**
- Telecom attributes (`DSMIL_TELECOM_STACK`, `DSMIL_SS7_ROLE`, `DSMIL_SIGTRAN_ROLE`, etc.)
- Telemetry event structure extensions (telecom fields)
- LLVM pass (`DsmilTelecomPass.cpp`) for annotation discovery and manifest generation
- Runtime helper macros (`dsmil_telecom_log.h`)
- Example code (`telecom_ss7_example.c`)
- Documentation (`TELECOM-SS7-GUIDE.md`)

⏳ **Pending Integration:**
- CMake build system integration
- Pass registration in DSLLVM pipeline
- Clang frontend flag (`-fdsmil-telecom-flags`)
- Mission profile auto-enable logic

## Build System Integration

### 1. Add Pass to Build System

Add `DsmilTelecomPass.cpp` to the DSMIL passes build:

```cmake
# In dsmil/lib/Passes/CMakeLists.txt (or equivalent)
set(DSMIL_PASSES
    # ... existing passes ...
    DsmilTelecomPass.cpp
)
```

### 2. Register Pass in Pass Registry

The pass is already registered via `llvmGetPassPluginInfo()` in `DsmilTelecomPass.cpp`. Ensure the plugin system loads it:

```cpp
// In dsmil/lib/Passes/PassRegistry.cpp (or equivalent registration file)
// The pass will be available as "dsmil-telecom"
```

### 3. Add Clang Frontend Flag

Add `-fdsmil-telecom-flags` flag to Clang frontend:

```cpp
// In clang/include/clang/Driver/Options.td
def fdsmil_telecom_flags : Flag<["-"], "fdsmil-telecom-flags">,
    HelpText<"Enable telecom annotation discovery and manifest generation">;

// In clang/lib/Driver/ToolChains/Clang.cpp (or equivalent)
// Add flag handling to pass -mllvm -dsmil-telecom-flags to LLVM
```

### 4. Integrate into DSMIL Default Pipeline

Add the pass to the DSMIL default pass pipeline:

```cpp
// In dsmil/lib/Passes/Pipeline.cpp (or equivalent)
// Add to dsmil-default pipeline when -fdsmil-telecom-flags is enabled
// or when mission profile indicates telecom usage

if (EnableTelecomFlags || isTelecomProfile(MissionProfile)) {
    MPM.addPass(DsmilTelecomPass());
}
```

### 5. Mission Profile Auto-Enable

Add logic to auto-enable telecom flagging for telecom-related mission profiles:

```cpp
// In DsmilTelecomPass or pipeline logic
bool isTelecomProfile(const std::string &Profile) {
    return Profile.find("ss7") != std::string::npos ||
           Profile.find("telco") != std::string::npos ||
           Profile.find("sigtran") != std::string::npos ||
           Profile.find("telecom") != std::string::npos;
}
```

## Clang Attribute Support

The attributes use LLVM's `annotate` attribute mechanism, which Clang already supports:

```cpp
// Clang already supports:
__attribute__((annotate("dsmil.telecom_stack=ss7")))
__attribute__((annotate("dsmil.ss7_role=STP")))
__attribute__((annotate("dsmil.telecom_env=honeypot")))
// etc.
```

The macros in `dsmil/include/dsmil_attributes.h` wrap these correctly.

## Environment Variable Control

Add environment variable support for runtime control:

```cpp
// In DsmilTelecomPass or runtime
const char *env = getenv("DSMIL_TELECOM_FLAGS");
if (env && (env[0] == '0' || env[0] == 'f' || env[0] == 'F')) {
    EnableTelecomFlags = false;
}
```

## Testing

### Unit Test Example

Create `dsmil/test/telecom/test_basic.ll`:

```llvm
; RUN: opt -load-pass-plugin=libDSMILPasses.so -passes=dsmil-telecom -S %s | FileCheck %s

define void @ss7_function() {
  ret void
}

; CHECK: Generated manifest
```

### Integration Test

```bash
# Compile example
dsmil-clang -fdsmil-telecom-flags \
             -fdsmil-mission-profile=ss7_lab \
             dsmil/examples/telecom_ss7_example.c \
             -o telecom_ss7_example

# Run and check for manifest
test -f telecom_ss7_example.dsmil.telecom.json

# Verify manifest content
cat telecom_ss7_example.dsmil.telecom.json | grep -q "ss7"
```

### Security Policy Test

```bash
# Should error: honeypot code with production profile
dsmil-clang -fdsmil-telecom-flags \
             -fdsmil-mission-profile=prod \
             honeypot_code.c 2>&1 | grep -q "Error.*honeypot"
```

## Integration with miltop_ss7

### Step 1: Annotate Functions

Add telecom attributes to miltop_ss7 functions:

```c
// In miltop_ss7 codebase
DSMIL_TELECOM_STACK("ss7")
DSMIL_SS7_ROLE("STP")
DSMIL_TELECOM_ENV("honeypot")
DSMIL_SIG_SECURITY("defense_lab")
void miltop_ss7_handler(const uint8_t *msg, size_t len) {
    // Handler code
}
```

### Step 2: Add Telemetry Logging

Use helper macros for telemetry:

```c
#include "dsmil/include/dsmil_telecom_log.h"

void process_ss7_message(const uint8_t *msg) {
    uint32_t opc = extract_opc(msg);
    uint32_t dpc = extract_dpc(msg);
    uint8_t sio = extract_sio(msg);
    
    DSMIL_LOG_SS7_RX(opc, dpc, sio, 1, 2);
    
    // Process message...
}
```

### Step 3: Compile with Flags

```bash
dsmil-clang -fdsmil-telecom-flags \
             -fdsmil-mission-profile=ss7_honeypot \
             miltop_ss7.c -o miltop_ss7
```

### Step 4: Use Manifest

Layer 8/9 can ingest the manifest for network awareness:

```json
{
  "module_id": "miltop_ss7",
  "telecom": {
    "stacks": ["ss7"],
    "default_env": "honeypot"
  },
  "functions": [
    {
      "name": "miltop_ss7_handler",
      "telecom_stack": "ss7",
      "ss7_role": "STP",
      "telecom_env": "honeypot"
    }
  ]
}
```

## Verification Checklist

- [ ] Pass compiles and links
- [ ] Clang flag `-fdsmil-telecom-flags` works
- [ ] Pass runs in DSMIL default pipeline
- [ ] Telecom manifest JSON is generated
- [ ] Mission profile auto-enable works
- [ ] Security policy validation works
- [ ] Example code compiles and runs
- [ ] Tests pass
- [ ] Integration with miltop_ss7 verified

## Known Issues / Limitations

1. **Annotation Detection**: The pass checks multiple methods for annotations (metadata, attributes) for compatibility across Clang versions.

2. **Build ID / Provenance ID**: Currently set to "0". Should be integrated with DSLLVM provenance system.

3. **Telemetry Integration**: Currently manifest-only. Full runtime telemetry requires integration with OT telemetry pass.

4. **Mission Profile Parsing**: Simple string matching for telecom profiles. Could be enhanced with structured profile definitions.

## Future Enhancements

1. **Full Telemetry Integration**: Automatically instrument telecom functions with telemetry calls
2. **Network Topology**: Build network topology graph from manifests
3. **Anomaly Detection**: Layer 8/9 integration for signaling anomaly detection
4. **Policy Engine**: More sophisticated security policy enforcement
5. **Multi-Stack Support**: Enhanced support for SIP, Diameter, etc.

## References

- `dsmil/include/dsmil_attributes.h` - Attribute definitions
- `dsmil/include/dsmil_telecom_log.h` - Telemetry helper macros
- `dsmil/include/dsmil_ot_telemetry.h` - Telemetry API
- `dsmil/lib/Passes/DsmilTelecomPass.cpp` - LLVM pass
- `dsmil/examples/telecom_ss7_example.c` - Example code
- `dsmil/docs/TELECOM-SS7-GUIDE.md` - User guide

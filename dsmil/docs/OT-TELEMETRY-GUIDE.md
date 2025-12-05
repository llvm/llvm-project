# DSLLVM OT Telemetry Guide

## Overview

The DSLLVM OT Telemetry subsystem provides high-value safety and Operational Technology (OT) visibility with minimal runtime overhead. It focuses on:

1. **OT/AI safety boundaries** - Tracking functions that interact with OT/ICS control paths
2. **Layer/device/mission profile awareness** - Context-aware telemetry with full DSMIL metadata
3. **Binary provenance + authority levels** - Integration with DSLLVM provenance and CNSA2.0 signing

## Features

- Automatic instrumentation of OT-critical functions
- Safety signal update tracking (pressure, flow, current, speed, etc.)
- SES (Safety Envelope Supervisor) gate intent logging
- Telemetry manifest JSON generation for build-time analysis
- Async-safe runtime implementation with minimal overhead

## Attributes

### Function-Level Attributes

#### `DSMIL_OT_CRITICAL`

Marks functions that interact with OT/ICS control paths or the Safety Envelope Supervisor (SES).

```c
DSMIL_OT_CRITICAL
DSMIL_LAYER(3)
DSMIL_DEVICE(12)
void pump_control_update(double setpoint) {
    // Automatically instrumented with entry/exit telemetry
}
```

#### `DSMIL_OT_TIER(level)`

Marks the authority tier for a function (0-3):

- **0**: Safety kernel / Safety Instrumented System (SIS) - highest authority
- **1**: High-impact control - direct control of critical processes
- **2**: Optimization/scheduling - operational optimization
- **3**: Analytics/advisory only - read-only analysis, no control

```c
DSMIL_OT_TIER(1)  // High-impact control
DSMIL_OT_CRITICAL
void critical_valve_control(int valve_id, double position) {
    // Tier 1: Direct control
}
```

#### `DSMIL_SES_GATE`

Marks functions that send intents to the Safety Envelope Supervisor.

```c
DSMIL_SES_GATE
DSMIL_OT_CRITICAL
int request_pump_start(int pump_id) {
    // Sends intent to SES (automatically logged)
    return ses_send_intent("pump_start", pump_id);
}
```

### Variable-Level Attributes

#### `DSMIL_SAFETY_SIGNAL(name)`

Marks variables that represent safety-relevant setpoints or signals.

```c
DSMIL_SAFETY_SIGNAL("line7_pressure_setpoint")
static double pressure_setpoint = 100.0;  // PSI

void update_pressure(double new_value) {
    pressure_setpoint = new_value;  // Automatically logged
}
```

## Compilation

### Basic Usage

Enable OT telemetry instrumentation with the `-fdsmil-ot-telemetry` flag:

```bash
dsmil-clang -fdsmil-ot-telemetry \
             -fdsmil-mission-profile=ics_ops \
             -c example.c -o example.o
```

### Mission Profile Integration

OT telemetry is automatically enabled when:
- `-fdsmil-ot-telemetry` is explicitly set, OR
- Mission profile implies OT/ICS usage (e.g., `ics_ops`, `grid_ops`)

```bash
# Automatically enables telemetry for ICS operations
dsmil-clang -fdsmil-mission-profile=ics_ops example.c
```

### Manifest Generation

Telemetry manifests are automatically generated as `<module>.dsmil.telemetry.json`:

```bash
dsmil-clang -fdsmil-ot-telemetry example.c
# Generates: example.dsmil.telemetry.json
```

Custom manifest path:

```bash
dsmil-clang -fdsmil-ot-telemetry \
             -mllvm -dsmil-telemetry-manifest-path=telemetry/manifest.json \
             example.c
```

## Runtime Configuration

### Environment Variables

- `DSMIL_OT_TELEMETRY=0/1` - Enable/disable telemetry at runtime (default: ON in production)

```bash
# Disable telemetry for testing
DSMIL_OT_TELEMETRY=0 ./my_program

# Enable telemetry (default)
DSMIL_OT_TELEMETRY=1 ./my_program
```

### Runtime API

```c
#include "dsmil/include/dsmil_ot_telemetry.h"

// Initialize telemetry (called automatically, but can be called manually)
dsmil_ot_telemetry_init();

// Check if telemetry is enabled
if (dsmil_ot_telemetry_is_enabled()) {
    // Telemetry is active
}

// Shutdown telemetry (flushes pending events)
dsmil_ot_telemetry_shutdown();
```

## Telemetry Events

### Event Types

1. `DSMIL_TELEMETRY_OT_PATH_ENTRY` - OT-critical function entry
2. `DSMIL_TELEMETRY_OT_PATH_EXIT` - OT-critical function exit
3. `DSMIL_TELEMETRY_SES_INTENT` - SES intent sent
4. `DSMIL_TELEMETRY_SES_ACCEPT` - SES intent accepted
5. `DSMIL_TELEMETRY_SES_REJECT` - SES intent rejected
6. `DSMIL_TELEMETRY_INVARIANT_HIT` - Safety invariant checked (passed)
7. `DSMIL_TELEMETRY_INVARIANT_FAIL` - Safety invariant violation

### Event Format

Events are logged as JSON lines to stderr (default) or via ring buffer:

```json
{
  "type": "ot_path_entry",
  "ts": 1234567890123456789,
  "module": "pump_controller",
  "func": "pump_control_update",
  "file": "pump.c",
  "line": 42,
  "layer": 3,
  "device": 12,
  "stage": "control",
  "profile": "ics_ops",
  "tier": 1,
  "build_id": 12345678,
  "provenance_id": 87654321
}
```

Safety signal updates include additional fields:

```json
{
  "type": "invariant_hit",
  "signal": "line7_pressure_setpoint",
  "value": 125.5,
  "min": 50.0,
  "max": 200.0,
  "layer": 3,
  "device": 12
}
```

## Telemetry Manifest

The telemetry manifest (`<module>.dsmil.telemetry.json`) provides build-time analysis of instrumented functions and signals:

```json
{
  "module_id": "pump_controller",
  "build_id": "0x12345678",
  "provenance_id": "0xabcdef00",
  "mission_profile": "ics_ops",
  "functions": [
    {
      "name": "pump_control_update",
      "layer": 3,
      "device": 12,
      "stage": "control",
      "ot_critical": true,
      "authority_tier": 1,
      "ses_gate": true
    }
  ],
  "safety_signals": [
    {
      "name": "line7_pressure_setpoint",
      "type": "double",
      "layer": 3,
      "device": 12
    }
  ]
}
```

## Example

See `dsmil/examples/ot_telemetry_example.c` for a complete example:

```c
#include "dsmil/include/dsmil_attributes.h"
#include "dsmil/include/dsmil_ot_telemetry.h"

// Safety signal
DSMIL_SAFETY_SIGNAL("line7_pressure_setpoint")
static double pressure_setpoint = 100.0;

// OT-critical function
DSMIL_OT_CRITICAL
DSMIL_OT_TIER(1)
DSMIL_SES_GATE
DSMIL_LAYER(3)
DSMIL_DEVICE(12)
DSMIL_STAGE("control")
int pump_control_update(int pump_id, double new_pressure) {
    pressure_setpoint = new_pressure;  // Automatically logged
    return 0;
}

int main(void) {
    dsmil_ot_telemetry_init();
    pump_control_update(1, 125.5);
    dsmil_ot_telemetry_shutdown();
    return 0;
}
```

Compile and run:

```bash
dsmil-clang -fdsmil-ot-telemetry \
             -fdsmil-mission-profile=ics_ops \
             ot_telemetry_example.c -o ot_telemetry_example
./ot_telemetry_example
```

## Integration with DSLLVM Pipeline

The OT telemetry pass is automatically included in the DSMIL default pipeline when `-fdsmil-ot-telemetry` is enabled. It runs after inlining to avoid duplicate instrumentation.

### Manual Pass Invocation

```bash
opt -load-pass-plugin=libDSMILPasses.so \
    -passes=dsmil-telemetry \
    input.ll -o output.ll
```

## Performance Considerations

- **Minimal overhead**: Telemetry calls are async-safe and use simple logging
- **Zero-cost when disabled**: `DSMIL_OT_TELEMETRY=0` eliminates all overhead
- **Ring buffer option**: For high-throughput scenarios, ring buffer + background thread can be used
- **No heap allocation**: Hot path avoids dynamic memory allocation

## Best Practices

1. **Mark all OT-critical functions** with `DSMIL_OT_CRITICAL`
2. **Set appropriate authority tiers** to reflect safety impact
3. **Use `DSMIL_SES_GATE`** for all SES interactions
4. **Annotate safety signals** with `DSMIL_SAFETY_SIGNAL`
5. **Review telemetry manifests** to ensure complete coverage
6. **Test with telemetry disabled** to verify functionality

## Troubleshooting

### Telemetry events not appearing

1. Check that `-fdsmil-ot-telemetry` flag is set
2. Verify `DSMIL_OT_TELEMETRY` environment variable is not `0`
3. Check stderr output (events go to stderr by default)
4. Ensure functions are marked with `DSMIL_OT_CRITICAL`

### Manifest not generated

1. Verify `-fdsmil-ot-telemetry` is enabled
2. Check write permissions for manifest directory
3. Look for warnings in compiler output

### Missing annotations

1. Ensure attributes are applied correctly (check with `-S -emit-llvm`)
2. Verify Clang is emitting annotate metadata
3. Check that pass is running (enable debug output)

## See Also

- `dsmil/include/dsmil_attributes.h` - All DSMIL attributes
- `dsmil/include/dsmil_ot_telemetry.h` - OT telemetry API
- `dsmil/examples/ot_telemetry_example.c` - Complete example
- `dsmil/docs/TELEMETRY-ENFORCEMENT.md` - General telemetry enforcement

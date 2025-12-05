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

### Generic Telemetry Annotations (v1.9)

#### `DSMIL_NET_IO`

Marks functions for network I/O telemetry (connect, send, recv, etc.).

```c
DSMIL_NET_IO
DSMIL_LAYER(4)
int connect_to_server(const char *host, int port) {
    // Automatically instrumented with network I/O telemetry
    return socket_connect(host, port);
}
```

#### `DSMIL_CRYPTO`

Marks functions for cryptographic operation telemetry (encrypt, decrypt, sign, verify).

```c
DSMIL_CRYPTO
DSMIL_LAYER(3)
int aes_encrypt(const uint8_t *key, const uint8_t *plaintext, uint8_t *ciphertext) {
    // Automatically instrumented with crypto telemetry
    return do_aes_encrypt(key, plaintext, ciphertext);
}
```

#### `DSMIL_PROCESS`

Marks functions for process/system operation telemetry (fork, exec, kill, etc.).

```c
DSMIL_PROCESS
DSMIL_LAYER(5)
int spawn_child_process(const char *cmd) {
    // Automatically instrumented with process telemetry
    return fork_and_exec(cmd);
}
```

#### `DSMIL_FILE`

Marks functions for file I/O telemetry (open, read, write, close, etc.).

```c
DSMIL_FILE
DSMIL_LAYER(4)
FILE* open_config_file(const char *filename) {
    // Automatically instrumented with file I/O telemetry
    return fopen(filename, "r");
}
```

#### `DSMIL_UNTRUSTED`

Marks functions handling untrusted data (network input, user input, etc.).

```c
DSMIL_UNTRUSTED
DSMIL_LAYER(7)
void process_user_input(const char *input) {
    // Automatically instrumented with untrusted data telemetry
    validate_and_process(input);
}
```

#### `DSMIL_ERROR_HANDLER`

Marks functions as error handlers. If function name suggests panic (e.g., `panic`, `fatal`), emits PANIC events instead of ERROR events.

```c
DSMIL_ERROR_HANDLER
DSMIL_LAYER(5)
void handle_error(int code, const char *msg) {
    // Automatically instrumented with error telemetry
    log_error(code, msg);
}

DSMIL_ERROR_HANDLER
DSMIL_LAYER(5)
void panic(const char *msg) {
    // Automatically emits PANIC events (name suggests panic)
    abort();
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

### Telemetry Levels

Control instrumentation verbosity with `-fdsmil-telemetry-level`:

```bash
# Minimal telemetry (safety-critical only)
dsmil-clang -fdsmil-ot-telemetry \
             -fdsmil-telemetry-level=min \
             example.c -o example

# Normal telemetry (entry probes for annotated functions)
dsmil-clang -fdsmil-ot-telemetry \
             -fdsmil-telemetry-level=normal \
             example.c -o example

# Debug telemetry (entry + exit + timing)
dsmil-clang -fdsmil-ot-telemetry \
             -fdsmil-telemetry-level=debug \
             example.c -o example

# Trace telemetry (all + sampling)
dsmil-clang -fdsmil-ot-telemetry \
             -fdsmil-telemetry-level=trace \
             example.c -o example
```

**Levels**:
- `off` - No telemetry
- `min` - Minimal telemetry (OT events, errors, panics only)
- `normal` - Normal telemetry (entry probes for all annotated functions) - **default**
- `debug` - Debug telemetry (entry + exit + elapsed time)
- `trace` - Trace telemetry (all + probabilistic sampling)

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
- `DSMIL_TELEMETRY_LEVEL=<level>` - Override telemetry level at runtime (off, min, normal, debug, trace)
- `DSMIL_MISSION_PROFILE=<profile>` - Mission profile (affects default telemetry level)

```bash
# Disable telemetry for testing
DSMIL_OT_TELEMETRY=0 ./my_program

# Enable telemetry (default)
DSMIL_OT_TELEMETRY=1 ./my_program

# Override telemetry level at runtime
DSMIL_TELEMETRY_LEVEL=debug ./my_program

# Mission profile affects default level (ics_prod forces min level minimum)
DSMIL_MISSION_PROFILE=ics_prod ./my_program
```

**Level Override Policy**:
- Runtime level override combines with compile-time level
- Lattice enforcement: off < min < normal < debug < trace
- Mission profiles (ics_prod, border_ops) force minimum levels unless CLI demanded stricter

### Runtime API

```c
#include "dsmil/include/dsmil_ot_telemetry.h"

// Initialize telemetry (called automatically, but can be called manually)
dsmil_ot_telemetry_init();

// Check if telemetry is enabled
if (dsmil_ot_telemetry_is_enabled()) {
    // Telemetry is active
}

// Get current telemetry level
dsmil_telemetry_level_t level = dsmil_telemetry_get_level();
// Returns: DSMIL_TELEMETRY_LEVEL_OFF, MIN, NORMAL, DEBUG, or TRACE

// Check if level allows event category
if (dsmil_telemetry_level_allows(DSMIL_TELEMETRY_NET_IO, "net")) {
    // Event will be logged
}

// Shutdown telemetry (flushes pending events)
dsmil_ot_telemetry_shutdown();
```

## Telemetry Events

### Event Types

**OT/Safety Events**:
1. `DSMIL_TELEMETRY_OT_PATH_ENTRY` - OT-critical function entry
2. `DSMIL_TELEMETRY_OT_PATH_EXIT` - OT-critical function exit
3. `DSMIL_TELEMETRY_SES_INTENT` - SES intent sent
4. `DSMIL_TELEMETRY_SES_ACCEPT` - SES intent accepted
5. `DSMIL_TELEMETRY_SES_REJECT` - SES intent rejected
6. `DSMIL_TELEMETRY_INVARIANT_HIT` - Safety invariant checked (passed)
7. `DSMIL_TELEMETRY_INVARIANT_FAIL` - Safety invariant violation

**Telecom Events** (20-24):
- `DSMIL_TELEMETRY_SS7_MSG_RX` - SS7 message received
- `DSMIL_TELEMETRY_SS7_MSG_TX` - SS7 message transmitted
- `DSMIL_TELEMETRY_SIGTRAN_MSG_RX` - SIGTRAN message received
- `DSMIL_TELEMETRY_SIGTRAN_MSG_TX` - SIGTRAN message transmitted
- `DSMIL_TELEMETRY_SIG_ANOMALY` - Signaling anomaly detected

**Generic Annotation Events** (30-36):
- `DSMIL_TELEMETRY_NET_IO` (30) - Network I/O operation
- `DSMIL_TELEMETRY_CRYPTO` (31) - Cryptographic operation
- `DSMIL_TELEMETRY_PROCESS` (32) - Process/system operation
- `DSMIL_TELEMETRY_FILE` (33) - File I/O operation
- `DSMIL_TELEMETRY_UNTRUSTED` (34) - Untrusted data handling
- `DSMIL_TELEMETRY_ERROR` (35) - Error handler invocation
- `DSMIL_TELEMETRY_PANIC` (36) - Panic/fatal error

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

Generic annotation events include new fields (backward compatible):

```json
{
  "type": "net_io",
  "ts": 1234567890123456789,
  "module": "network_daemon",
  "func": "connect",
  "file": "network.c",
  "line": 42,
  "category": "net",
  "op": "connect",
  "status_code": 0,
  "resource": "tcp://example.com:80",
  "elapsed_ns": 1234567
}
```

Error events include error message:

```json
{
  "type": "error",
  "category": "error",
  "op": "error",
  "status_code": -1,
  "error_msg": "Connection failed",
  "func": "handle_error",
  "file": "error.c",
  "line": 100
}
```

## Telemetry Manifests

### Telemetry Manifest

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

### Metrics Manifest

The metrics pass generates `<module>.dsmil.metrics.json` with comprehensive statistics:

```json
{
  "module_id": "network_daemon",
  "mission_profile": "default",
  "metrics": {
    "total_functions": 150,
    "instrumented_functions": 45,
    "instrumentation_coverage": 30.0,
    "ot_critical_count": 10,
    "net_io_count": 15,
    "crypto_count": 8,
    "process_count": 5,
    "file_count": 7,
    "authority_tiers": {
      "tier_0": 2,
      "tier_1": 8,
      "tier_2": 20,
      "tier_3": 15
    },
    "categories": {
      "net": 15,
      "crypto": 8,
      "process": 5,
      "file": 7
    },
    "telecom": {
      "total": 0
    },
    "safety_signals": 3
  }
}
```

### Global Metrics Summary

Use `dsmil-telemetry-summary` to aggregate metrics from all modules:

```bash
# Aggregate all metrics files
dsmil-telemetry-summary \
    --input-glob "*.dsmil.metrics.json" \
    --output dsmil.global.metrics.json

# With telemetry JSON files
dsmil-telemetry-summary \
    --input-glob "*.dsmil.metrics.json" \
    --telemetry-json "*.telemetry.json" \
    --output global_summary.json
```

Outputs `dsmil.global.metrics.json` with aggregated statistics across all modules.

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

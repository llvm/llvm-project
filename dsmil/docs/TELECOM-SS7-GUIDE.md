# DSLLVM SS7/SIGTRAN Telemetry & Flagging Guide

## Overview

The DSLLVM Telecom/SS7/SIGTRAN subsystem provides compile-time annotation discovery and runtime telemetry for telecom signaling code. It enables:

1. **Compile-time manifest generation** - Identifies which modules handle SS7/SIGTRAN signaling
2. **Role identification** - Marks SS7 node roles (STP, MSC, HLR, etc.) and SIGTRAN roles (SG, AS, ASP)
3. **Environment awareness** - Distinguishes production, lab, honeypot, fuzzing environments
4. **Runtime telemetry** - Optional high-level signaling event logging
5. **Security policy enforcement** - Prevents honeypot code from running in production

## Features

- Telecom stack identification (SS7, SIGTRAN, SIP, Diameter)
- SS7 role marking (STP, MSC, HLR, VLR, SMSC, etc.)
- SIGTRAN role marking (SG, AS, ASP, IPSP)
- Environment classification (prod, lab, honeypot, fuzz, sim)
- Security level marking (high_assurance, defense_lab, redteam_sim, low)
- Interface type identification (E1, T1, SCTP, M2PA, M2UA, M3UA, SUA)
- Logical endpoint marking
- Compile-time manifest generation
- Optional runtime telemetry helpers

## Attributes

### Stack and Role Attributes

#### `DSMIL_TELECOM_STACK(name)`

Marks code that implements or interacts with a telecom stack.

```c
DSMIL_TELECOM_STACK("ss7")
DSMIL_LAYER(3)
DSMIL_DEVICE(31)
void ss7_mtp3_process(const uint8_t *msg, size_t len) {
    // SS7 processing
}
```

Supported stacks: `"ss7"`, `"sigtran"`, `"sip"`, `"diameter"`

#### `DSMIL_SS7_ROLE(role)`

Marks SS7 node role in classical SS7 network.

```c
DSMIL_SS7_ROLE("STP")
DSMIL_TELECOM_STACK("ss7")
void stp_routing_function(const uint8_t *msg) {
    // STP routing logic
}
```

Supported roles: `"STP"`, `"MSC"`, `"HLR"`, `"VLR"`, `"SMSC"`, `"GWMSC"`, `"IN"`, `"GMSC"`

#### `DSMIL_SIGTRAN_ROLE(role)`

Marks SIGTRAN role for SS7-over-IP signaling.

```c
DSMIL_SIGTRAN_ROLE("SG")
DSMIL_TELECOM_STACK("sigtran")
void sigtran_sg_function(const uint8_t *msg) {
    // Signaling Gateway processing
}
```

Supported roles: `"SG"`, `"AS"`, `"ASP"`, `"IPSP"`

### Environment and Security Attributes

#### `DSMIL_TELECOM_ENV(env)`

Marks operational environment for signaling code.

```c
DSMIL_TELECOM_ENV("honeypot")
DSMIL_TELECOM_STACK("ss7")
void honeypot_ss7_handler(const uint8_t *msg) {
    // Honeypot handler (must not run in production)
}
```

Supported environments: `"prod"`, `"lab"`, `"honeypot"`, `"fuzz"`, `"sim"`

#### `DSMIL_SIG_SECURITY(level)`

Marks security posture and sensitivity level.

```c
DSMIL_SIG_SECURITY("defense_lab")
DSMIL_TELECOM_ENV("lab")
void defense_lab_analyzer(const uint8_t *msg) {
    // Defense lab analysis
}
```

Supported levels: `"high_assurance"`, `"defense_lab"`, `"redteam_sim"`, `"low"`

### Interface and Endpoint Attributes

#### `DSMIL_TELECOM_INTERFACE(name)`

Marks physical or protocol interface type.

```c
DSMIL_TELECOM_INTERFACE("m3ua")
DSMIL_TELECOM_STACK("sigtran")
void m3ua_message_handler(const uint8_t *msg) {
    // M3UA message processing
}
```

Supported interfaces: `"e1"`, `"t1"`, `"sctp"`, `"m2pa"`, `"m2ua"`, `"m3ua"`, `"sua"`

#### `DSMIL_TELECOM_ENDPOINT(name)`

Marks logical endpoint in telecom network topology.

```c
DSMIL_TELECOM_ENDPOINT("upstream_stp")
DSMIL_TELECOM_STACK("ss7")
void upstream_stp_handler(const uint8_t *msg) {
    // Upstream STP message handling
}
```

## Compilation

### Basic Usage

Enable telecom flagging with `-fdsmil-telecom-flags`:

```bash
dsmil-clang -fdsmil-telecom-flags \
             -fdsmil-mission-profile=ss7_lab \
             -c example.c -o example.o
```

### Auto-Enable for Telecom Profiles

Telecom flagging is automatically enabled when mission profile contains:
- `"ss7"`
- `"telco"`
- `"sigtran"`
- `"telecom"`

```bash
# Automatically enables telecom flagging
dsmil-clang -fdsmil-mission-profile=ss7_honeypot example.c
```

### Manifest Generation

Telecom manifests are automatically generated as `<module>.dsmil.telecom.json`:

```bash
dsmil-clang -fdsmil-telecom-flags example.c
# Generates: example.dsmil.telecom.json
```

Custom manifest path:

```bash
dsmil-clang -fdsmil-telecom-flags \
             -mllvm -dsmil-telecom-manifest-path=telecom/manifest.json \
             example.c
```

## Telecom Manifest

The telecom manifest provides compile-time analysis of telecom-annotated code:

```json
{
  "module_id": "ss7_stp",
  "build_id": "0x12345678",
  "provenance_id": "0xabcdef00",
  "mission_profile": "ss7_lab",
  "telecom": {
    "stacks": ["ss7", "sigtran"],
    "default_env": "lab",
    "default_sig_security": "defense_lab"
  },
  "functions": [
    {
      "name": "ss7_mtp3_process",
      "layer": 3,
      "device": 31,
      "stage": "signaling",
      "telecom_stack": "ss7",
      "ss7_role": "STP",
      "telecom_env": "lab",
      "sig_security": "defense_lab"
    },
    {
      "name": "sigtran_m3ua_rx",
      "layer": 3,
      "device": 32,
      "stage": "signaling",
      "telecom_stack": "sigtran",
      "sigtran_role": "SG",
      "telecom_if": "m3ua"
    }
  ]
}
```

## Runtime Telemetry

### Helper Macros

Use helper macros from `dsmil/include/dsmil_telecom_log.h` for easy telemetry:

```c
#include "dsmil/include/dsmil_telecom_log.h"

void ss7_mtp3_rx(uint32_t opc, uint32_t dpc, uint8_t sio, 
                 uint8_t msg_class, uint8_t msg_type) {
    // Log SS7 message received
    DSMIL_LOG_SS7_RX(opc, dpc, sio, msg_class, msg_type);
    
    // Process message...
}

void sigtran_m3ua_rx(uint32_t rctx) {
    // Log SIGTRAN message received
    DSMIL_LOG_SIGTRAN_RX(rctx);
    
    // Process message...
}

// Log anomaly
if (suspicious_pattern) {
    DSMIL_LOG_SIG_ANOMALY("ss7", "Unexpected message sequence");
}
```

### Available Macros

- `DSMIL_LOG_SS7_RX(opc, dpc, sio, msg_class, msg_type)` - Log SS7 message received
- `DSMIL_LOG_SS7_TX(opc, dpc, sio, msg_class, msg_type)` - Log SS7 message transmitted
- `DSMIL_LOG_SIGTRAN_RX(rctx)` - Log SIGTRAN message received
- `DSMIL_LOG_SIGTRAN_TX(rctx)` - Log SIGTRAN message transmitted
- `DSMIL_LOG_SIG_ANOMALY(stack, description)` - Log signaling anomaly
- `DSMIL_LOG_SS7_FULL(opc, dpc, sio, msg_class, msg_type, role, env)` - Full SS7 logging with context

### Telemetry Event Structure

Telecom fields are embedded in `dsmil_telemetry_event_t`:

```c
typedef struct {
    // ... existing fields ...
    
    // Telecom fields
    const char *telecom_stack;
    const char *ss7_role;
    const char *sigtran_role;
    const char *telecom_env;
    const char *telecom_if;
    const char *telecom_ep;
    
    // SS7 context
    uint32_t ss7_opc;        // Originating Point Code
    uint32_t ss7_dpc;        // Destination Point Code
    uint8_t ss7_sio;         // Service Information Octet
    uint32_t sigtran_rctx;   // Routing Context
    uint8_t ss7_msg_class;  // Message class
    uint8_t ss7_msg_type;    // Message type
} dsmil_telemetry_event_t;
```

## Security Policy Enforcement

The compiler enforces security policies:

### Production vs Honeypot

- **Error**: Honeypot code detected in production mission profile
- **Warning**: Module contains both production and honeypot code

```c
// This will error if compiled with -fdsmil-mission-profile=prod
DSMIL_TELECOM_ENV("honeypot")
void honeypot_handler(void) {
    // Must not run in production
}
```

### Environment Validation

The pass validates:
- Mission profile consistency with code environment
- Mixed production/honeypot code detection
- Security level appropriateness

## Example

See `dsmil/examples/telecom_ss7_example.c` for a complete example:

```c
#include "dsmil/include/dsmil_attributes.h"
#include "dsmil/include/dsmil_telecom_log.h"

// SS7 STP handler
DSMIL_TELECOM_STACK("ss7")
DSMIL_SS7_ROLE("STP")
DSMIL_TELECOM_ENV("lab")
DSMIL_SIG_SECURITY("defense_lab")
DSMIL_LAYER(3)
DSMIL_DEVICE(31)
void ss7_mtp3_process(const uint8_t *msg, size_t len) {
    uint32_t opc = extract_opc(msg);
    uint32_t dpc = extract_dpc(msg);
    DSMIL_LOG_SS7_RX(opc, dpc, 0x08, 1, 2);
    // Process...
}
```

Compile and run:

```bash
dsmil-clang -fdsmil-telecom-flags \
             -fdsmil-mission-profile=ss7_lab \
             telecom_ss7_example.c -o telecom_ss7_example
./telecom_ss7_example
```

## Integration with miltop_ss7

For miltop_ss7 and OSMOCOM-based code:

1. **Annotate functions** with telecom attributes
2. **Use helper macros** for telemetry logging
3. **Compile with telecom flags** to generate manifests
4. **Layer 8/9** will ingest manifests for network awareness

Example integration:

```c
// In miltop_ss7 code
DSMIL_TELECOM_STACK("ss7")
DSMIL_SS7_ROLE("STP")
DSMIL_TELECOM_ENV("honeypot")
void miltop_ss7_handler(const uint8_t *msg, size_t len) {
    // Extract SS7 fields
    uint32_t opc = ...;
    uint32_t dpc = ...;
    
    // Log with helper macro
    DSMIL_LOG_SS7_RX(opc, dpc, sio, msg_class, msg_type);
    
    // Process...
}
```

## Best Practices

1. **Mark all telecom functions** with appropriate stack and role
2. **Set environment correctly** (prod vs lab vs honeypot)
3. **Use security levels** to indicate sensitivity
4. **Mark interfaces** for interface-specific telemetry
5. **Use logical endpoints** for network topology awareness
6. **Review manifests** to ensure complete coverage
7. **Test environment enforcement** (verify honeypot code blocked in prod)

## Troubleshooting

### Manifest not generated

1. Check that `-fdsmil-telecom-flags` is set or mission profile is telecom-related
2. Verify functions have telecom annotations
3. Check write permissions for manifest directory

### Security policy violations

1. Review error messages for environment mismatches
2. Check mission profile matches code environment
3. Separate production and honeypot code into different modules

### Telemetry not appearing

1. Ensure `DSMIL_OT_TELEMETRY=1` environment variable is set
2. Check helper macros are called correctly
3. Verify telemetry initialization (`dsmil_ot_telemetry_init()`)

## See Also

- `dsmil/include/dsmil_attributes.h` - All DSMIL attributes
- `dsmil/include/dsmil_telecom_log.h` - Telecom telemetry helpers
- `dsmil/include/dsmil_ot_telemetry.h` - Telemetry API
- `dsmil/examples/telecom_ss7_example.c` - Complete example
- `dsmil/docs/OT-TELEMETRY-GUIDE.md` - General telemetry guide

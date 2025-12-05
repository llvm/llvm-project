# DSLLVM Build Flags & Feature Guide

## Overview

This guide provides a comprehensive reference for all DSLLVM compiler flags, build options, and feature usage. DSLLVM extends LLVM/Clang with DSMIL-specific capabilities for OT telemetry, SS7/SIGTRAN telecom flagging, mission profiles, and more.

**Version**: 1.8  
**Last Updated**: 2024

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Mission Profiles](#mission-profiles)
3. [OT Telemetry](#ot-telemetry)
4. [SS7/SIGTRAN Telecom](#ss7sigtran-telecom)
5. [Layer & Device Attributes](#layer--device-attributes)
6. [Security & Policy](#security--policy)
7. [Build System Integration](#build-system-integration)
8. [Complete Examples](#complete-examples)

---

## Quick Start

### Basic Compilation

```bash
# Standard DSLLVM compilation
dsmil-clang -c example.c -o example.o

# With mission profile
dsmil-clang -fdsmil-mission-profile=ics_ops example.c -o example

# Enable OT telemetry
dsmil-clang -fdsmil-ot-telemetry \
             -fdsmil-mission-profile=ics_ops \
             example.c -o example

# Enable telecom flagging
dsmil-clang -fdsmil-telecom-flags \
             -fdsmil-mission-profile=ss7_lab \
             example.c -o example
```

### Environment Variables

```bash
# Control OT telemetry at runtime
export DSMIL_OT_TELEMETRY=1  # Enable (default in production)
export DSMIL_OT_TELEMETRY=0  # Disable

# Control telecom flags
export DSMIL_TELECOM_FLAGS=1  # Enable
export DSMIL_TELECOM_FLAGS=0  # Disable
```

---

## Mission Profiles

### Overview

Mission profiles define operational context and enforce compile-time constraints. They control pipeline selection, AI modes, sandbox defaults, telemetry requirements, and security policies.

### Flag

```bash
-fdsmil-mission-profile=<profile_id>
```

### Available Profiles

| Profile | Description | Use Case |
|---------|-------------|----------|
| `ics_ops` | Industrial Control Systems operations | OT/ICS production |
| `grid_ops` | Power grid operations | Smart grid systems |
| `ss7_lab` | SS7 laboratory environment | SS7 testing/development |
| `ss7_honeypot` | SS7 honeypot deployment | SS7 deception systems |
| `cyber_defence` | Cyber defense operations | Security monitoring |
| `border_ops` | Border operations | High-security deployments |
| `exercise_only` | Training exercises | Non-production testing |
| `lab_research` | Laboratory research | Experimental features |

### Usage

```bash
# Production ICS operations
dsmil-clang -fdsmil-mission-profile=ics_ops \
             -O2 \
             control_system.c -o control_system

# SS7 honeypot deployment
dsmil-clang -fdsmil-mission-profile=ss7_honeypot \
             -fdsmil-telecom-flags \
             honeypot_ss7.c -o honeypot_ss7

# Cyber defense with full telemetry
dsmil-clang -fdsmil-mission-profile=cyber_defence \
             -fdsmil-ot-telemetry \
             security_monitor.c -o security_monitor
```

### Profile Effects

Mission profiles automatically:
- Enable relevant telemetry subsystems
- Set security policy levels
- Configure AI modes (local/hybrid/cloud)
- Enforce stage whitelist/blacklist
- Control constant-time enforcement
- Set provenance requirements

---

## OT Telemetry

### Overview

OT (Operational Technology) Telemetry provides safety and OT visibility with minimal runtime overhead. It tracks OT-critical functions, safety signals, and SES (Safety Envelope Supervisor) interactions.

### Flags

```bash
# Enable OT telemetry instrumentation
-fdsmil-ot-telemetry

# Specify manifest output path (optional)
-mllvm -dsmil-telemetry-manifest-path=<path>
```

### Auto-Enable Conditions

OT telemetry is automatically enabled when:
- `-fdsmil-ot-telemetry` is explicitly set, OR
- Mission profile implies OT/ICS usage (`ics_ops`, `grid_ops`)

### Attributes

#### Function-Level Attributes

```c
// Mark OT-critical function
DSMIL_OT_CRITICAL
DSMIL_LAYER(3)
DSMIL_DEVICE(12)
void pump_control_update(double setpoint) {
    // Automatically instrumented with entry/exit telemetry
}

// Set authority tier (0-3)
DSMIL_OT_TIER(1)  // High-impact control
DSMIL_OT_CRITICAL
void critical_valve_control(int valve_id, double position) {
    // Tier 1: Direct control
}

// Mark SES gate function
DSMIL_SES_GATE
DSMIL_OT_CRITICAL
int request_pump_start(int pump_id) {
    // Sends intent to SES (automatically logged)
    return ses_send_intent("pump_start", pump_id);
}
```

#### Variable-Level Attributes

```c
// Mark safety signal
DSMIL_SAFETY_SIGNAL("line7_pressure_setpoint")
static double pressure_setpoint = 100.0;  // PSI

void update_pressure(double new_value) {
    pressure_setpoint = new_value;  // Automatically logged
}
```

### Compilation

```bash
# Basic OT telemetry
dsmil-clang -fdsmil-ot-telemetry \
             -c pump_control.c -o pump_control.o

# With mission profile (auto-enables telemetry)
dsmil-clang -fdsmil-mission-profile=ics_ops \
             pump_control.c -o pump_control

# Custom manifest path
dsmil-clang -fdsmil-ot-telemetry \
             -mllvm -dsmil-telemetry-manifest-path=telemetry/manifest.json \
             pump_control.c -o pump_control
```

### Runtime Control

```bash
# Enable telemetry (default in production)
export DSMIL_OT_TELEMETRY=1
./pump_control

# Disable telemetry (for testing)
export DSMIL_OT_TELEMETRY=0
./pump_control
```

### Output

**Telemetry Events** (stderr, JSON lines):
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

**Telemetry Manifest** (`<module>.dsmil.telemetry.json`):
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
      "ses_gate": false
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

### Complete Example

```c
#include "dsmil/include/dsmil_attributes.h"
#include "dsmil/include/dsmil_ot_telemetry.h"

DSMIL_SAFETY_SIGNAL("line7_pressure_setpoint")
static double pressure_setpoint = 100.0;

DSMIL_OT_CRITICAL
DSMIL_OT_TIER(1)
DSMIL_SES_GATE
DSMIL_LAYER(3)
DSMIL_DEVICE(12)
DSMIL_STAGE("control")
int pump_control_update(int pump_id, double new_pressure) {
    pressure_setpoint = new_pressure;  // Automatically logged
    return ses_send_intent("pump_start", pump_id);
}

int main(void) {
    dsmil_ot_telemetry_init();
    pump_control_update(1, 125.5);
    dsmil_ot_telemetry_shutdown();
    return 0;
}
```

Compile:
```bash
dsmil-clang -fdsmil-ot-telemetry \
             -fdsmil-mission-profile=ics_ops \
             pump_control.c -o pump_control
```

---

## SS7/SIGTRAN Telecom

### Overview

SS7/SIGTRAN Telemetry provides compile-time annotation discovery and runtime telemetry for telecom signaling code. It enables network topology awareness, role identification, and environment classification.

### Flags

```bash
# Enable telecom flagging
-fdsmil-telecom-flags

# Specify manifest output path (optional)
-mllvm -dsmil-telecom-manifest-path=<path>
```

### Auto-Enable Conditions

Telecom flagging is automatically enabled when:
- `-fdsmil-telecom-flags` is explicitly set, OR
- Mission profile contains: `"ss7"`, `"telco"`, `"sigtran"`, or `"telecom"`

### Attributes

#### Stack and Role Attributes

```c
// Mark telecom stack
DSMIL_TELECOM_STACK("ss7")
DSMIL_LAYER(3)
DSMIL_DEVICE(31)
void ss7_mtp3_process(const uint8_t *msg, size_t len) {
    // SS7 processing
}

// Mark SS7 role
DSMIL_SS7_ROLE("STP")  // Signal Transfer Point
DSMIL_TELECOM_STACK("ss7")
void stp_routing_function(const uint8_t *msg) {
    // STP routing logic
}

// Mark SIGTRAN role
DSMIL_SIGTRAN_ROLE("SG")  // Signaling Gateway
DSMIL_TELECOM_STACK("sigtran")
void sigtran_sg_function(const uint8_t *msg) {
    // SG processing
}
```

#### Environment and Security Attributes

```c
// Mark environment
DSMIL_TELECOM_ENV("honeypot")  // prod, lab, honeypot, fuzz, sim
DSMIL_TELECOM_STACK("ss7")
void honeypot_ss7_handler(const uint8_t *msg) {
    // Honeypot handler (must not run in production)
}

// Mark security level
DSMIL_SIG_SECURITY("defense_lab")  // high_assurance, defense_lab, redteam_sim, low
DSMIL_TELECOM_ENV("lab")
void defense_lab_analyzer(const uint8_t *msg) {
    // Defense lab analysis
}
```

#### Interface and Endpoint Attributes

```c
// Mark interface type
DSMIL_TELECOM_INTERFACE("m3ua")  // e1, t1, sctp, m2pa, m2ua, m3ua, sua
DSMIL_TELECOM_STACK("sigtran")
void m3ua_message_handler(const uint8_t *msg) {
    // M3UA message processing
}

// Mark logical endpoint
DSMIL_TELECOM_ENDPOINT("upstream_stp")
DSMIL_TELECOM_STACK("ss7")
void upstream_stp_handler(const uint8_t *msg) {
    // Upstream STP message handling
}
```

### Compilation

```bash
# Basic telecom flagging
dsmil-clang -fdsmil-telecom-flags \
             -c ss7_handler.c -o ss7_handler.o

# With mission profile (auto-enables flagging)
dsmil-clang -fdsmil-mission-profile=ss7_lab \
             ss7_handler.c -o ss7_handler

# Honeypot deployment
dsmil-clang -fdsmil-telecom-flags \
             -fdsmil-mission-profile=ss7_honeypot \
             honeypot_ss7.c -o honeypot_ss7

# Custom manifest path
dsmil-clang -fdsmil-telecom-flags \
             -mllvm -dsmil-telecom-manifest-path=telecom/manifest.json \
             ss7_handler.c -o ss7_handler
```

### Runtime Telemetry

Use helper macros from `dsmil/include/dsmil_telecom_log.h`:

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

### Output

**Telecom Manifest** (`<module>.dsmil.telecom.json`):
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
    }
  ]
}
```

### Security Policy Enforcement

The compiler enforces security policies:

```bash
# Error: Honeypot code with production profile
dsmil-clang -fdsmil-telecom-flags \
             -fdsmil-mission-profile=prod \
             honeypot_code.c
# Error: Honeypot mission profile but production code detected!

# Warning: Mixed production and honeypot code
dsmil-clang -fdsmil-telecom-flags \
             mixed_code.c
# Warning: Module contains both production and honeypot code!
```

### Complete Example

```c
#include "dsmil/include/dsmil_attributes.h"
#include "dsmil/include/dsmil_telecom_log.h"
#include "dsmil/include/dsmil_ot_telemetry.h"

DSMIL_TELECOM_STACK("ss7")
DSMIL_SS7_ROLE("STP")
DSMIL_TELECOM_ENV("lab")
DSMIL_SIG_SECURITY("defense_lab")
DSMIL_LAYER(3)
DSMIL_DEVICE(31)
DSMIL_STAGE("signaling")
void ss7_mtp3_process(const uint8_t *msg, size_t len) {
    if (len < 9) return;
    
    uint32_t opc = (msg[0] << 16) | (msg[1] << 8) | msg[2];
    uint32_t dpc = (msg[3] << 16) | (msg[4] << 8) | msg[5];
    uint8_t sio = msg[6];
    uint8_t msg_class = msg[7];
    uint8_t msg_type = msg[8];
    
    DSMIL_LOG_SS7_RX(opc, dpc, sio, msg_class, msg_type);
    
    // Process MTP3 message...
}

int main(void) {
    dsmil_ot_telemetry_init();
    
    uint8_t ss7_msg[] = {0x00, 0x01, 0x02, 0x00, 0x03, 0x04, 0x08, 0x01, 0x02};
    ss7_mtp3_process(ss7_msg, sizeof(ss7_msg));
    
    dsmil_ot_telemetry_shutdown();
    return 0;
}
```

Compile:
```bash
dsmil-clang -fdsmil-telecom-flags \
             -fdsmil-mission-profile=ss7_lab \
             ss7_handler.c -o ss7_handler
```

---

## Layer & Device Attributes

### Overview

DSMIL uses a 9-layer, 104-device architecture. Functions and globals can be assigned to specific layers and devices for placement, routing, and telemetry.

### Attributes

```c
// Assign to layer
DSMIL_LAYER(7)  // Layer 7: Applications (AI/ML)
void llm_inference_worker(void) {
    // Layer 7 operations
}

// Assign to device
DSMIL_DEVICE(47)  // Device 47: Primary NPU
void npu_workload(void) {
    // Runs on Device 47
}

// Combined assignment
DSMIL_PLACEMENT(7, 47)  // Layer 7, Device 47
void ai_workload(void) {
    // Combined layer/device assignment
}

// MLOps stage
DSMIL_STAGE("serve")  // pretrain, finetune, quantized, serve, debug, experimental
void model_inference_int8(const int8_t *input, int8_t *output) {
    // Quantized inference path
}
```

### Well-Known Layers

| Layer | Name | Description |
|-------|------|-------------|
| 0 | Hardware | Hardware/firmware |
| 1 | Kernel | Kernel core |
| 2 | Drivers | Device drivers |
| 3 | Crypto | Cryptographic services |
| 4 | Network | Network stack |
| 5 | System | System services |
| 6 | Middleware | Middleware/frameworks |
| 7 | Application | Applications (AI/ML) |
| 8 | User | User interface |

### Well-Known Devices

| Device | Name | Description |
|--------|------|-------------|
| 0 | Kernel | Kernel device |
| 30 | Crypto Engine | Cryptographic engine |
| 31 | SS7/SIGTRAN | Telecom signaling |
| 32 | SIGTRAN SG | Signaling Gateway |
| 47 | NPU Primary | Primary NPU |
| 50 | Telemetry | Telemetry/observability |

### Usage

```bash
# No special flags needed - attributes are always processed
dsmil-clang -c example.c -o example.o
```

---

## Security & Policy

### Overview

DSLLVM enforces security policies through attributes, mission profiles, and compile-time checks.

### Attributes

```c
// Security clearance
DSMIL_CLEARANCE(0x07070707)  // 32-bit clearance/compartment mask
void sensitive_operation(void) {
    // Requires specific clearance
}

// Rules of Engagement
DSMIL_ROE("ANALYSIS_ONLY")  // ANALYSIS_ONLY, LIVE_CONTROL, NETWORK_EGRESS, CRYPTO_SIGN
void analyze_data(const void *data) {
    // Read-only operations
}

// Gateway function (cross-layer calls)
DSMIL_GATEWAY
DSMIL_LAYER(5)
int validated_syscall_handler(int syscall_num, void *args) {
    // Can safely transition between layers
}

// Sandbox profile
DSMIL_SANDBOX("l7_llm_worker")
int main(int argc, char **argv) {
    // Runs with l7_llm_worker sandbox restrictions
}

// Untrusted input marking
DSMIL_UNTRUSTED_INPUT
void process_network_input(const char *user_data, size_t len) {
    // Must validate user_data before use
}

// Constant-time enforcement
DSMIL_SECRET
void aes_encrypt(const uint8_t *key, const uint8_t *plaintext, uint8_t *ciphertext) {
    // All operations on key are constant-time
}
```

### Mission Profile Security

Mission profiles enforce security policies:

```bash
# Border operations: Max security, minimal telemetry
dsmil-clang -fdsmil-mission-profile=border_ops \
             secure_system.c -o secure_system

# Cyber defense: AI-enhanced, full telemetry
dsmil-clang -fdsmil-mission-profile=cyber_defence \
             security_monitor.c -o security_monitor
```

---

## Build System Integration

### CMake Integration

```cmake
# Enable DSLLVM features
set(CMAKE_C_COMPILER "dsmil-clang")
set(CMAKE_CXX_COMPILER "dsmil-clang++")

# Add DSLLVM flags
add_compile_options(
    -fdsmil-mission-profile=ics_ops
    -fdsmil-ot-telemetry
    -fdsmil-telecom-flags
)

# Link DSMIL runtime libraries
target_link_libraries(your_target
    dsmil_ot_telemetry
    # ... other libraries
)
```

### Makefile Integration

```makefile
CC = dsmil-clang
CFLAGS = -fdsmil-mission-profile=ics_ops \
         -fdsmil-ot-telemetry \
         -fdsmil-telecom-flags \
         -O2

your_target: source.c
	$(CC) $(CFLAGS) source.c -o your_target
```

### Autotools Integration

```bash
# Configure
./configure CC=dsmil-clang \
            CFLAGS="-fdsmil-mission-profile=ics_ops -fdsmil-ot-telemetry"
```

---

## Complete Examples

### Example 1: OT Control System

```c
#include "dsmil/include/dsmil_attributes.h"
#include "dsmil/include/dsmil_ot_telemetry.h"

DSMIL_SAFETY_SIGNAL("line7_pressure_setpoint")
static double pressure_setpoint = 100.0;

DSMIL_OT_CRITICAL
DSMIL_OT_TIER(1)
DSMIL_SES_GATE
DSMIL_LAYER(3)
DSMIL_DEVICE(12)
DSMIL_STAGE("control")
int pump_control_update(int pump_id, double new_pressure) {
    pressure_setpoint = new_pressure;
    return ses_send_intent("pump_start", pump_id);
}

int main(void) {
    dsmil_ot_telemetry_init();
    pump_control_update(1, 125.5);
    dsmil_ot_telemetry_shutdown();
    return 0;
}
```

Build:
```bash
dsmil-clang -fdsmil-ot-telemetry \
             -fdsmil-mission-profile=ics_ops \
             -O2 \
             pump_control.c -o pump_control
```

### Example 2: SS7 Honeypot

```c
#include "dsmil/include/dsmil_attributes.h"
#include "dsmil/include/dsmil_telecom_log.h"
#include "dsmil/include/dsmil_ot_telemetry.h"

DSMIL_TELECOM_STACK("ss7")
DSMIL_SS7_ROLE("STP")
DSMIL_TELECOM_ENV("honeypot")
DSMIL_SIG_SECURITY("defense_lab")
DSMIL_TELECOM_ENDPOINT("honeypot_stp")
DSMIL_LAYER(3)
DSMIL_DEVICE(31)
DSMIL_STAGE("signaling")
void honeypot_ss7_handler(const uint8_t *msg, size_t len) {
    if (len < 9) return;
    
    uint32_t opc = (msg[0] << 16) | (msg[1] << 8) | msg[2];
    uint32_t dpc = (msg[3] << 16) | (msg[4] << 8) | msg[5];
    uint8_t sio = msg[6];
    
    DSMIL_LOG_SS7_RX(opc, dpc, sio, 1, 2);
    
    if (len > 1000) {
        DSMIL_LOG_SIG_ANOMALY("ss7", "Oversized message");
    }
}

int main(void) {
    dsmil_ot_telemetry_init();
    
    uint8_t msg[] = {0x00, 0x01, 0x02, 0x00, 0x03, 0x04, 0x08, 0x01, 0x02};
    honeypot_ss7_handler(msg, sizeof(msg));
    
    dsmil_ot_telemetry_shutdown();
    return 0;
}
```

Build:
```bash
dsmil-clang -fdsmil-telecom-flags \
             -fdsmil-mission-profile=ss7_honeypot \
             -O2 \
             honeypot_ss7.c -o honeypot_ss7
```

### Example 3: Combined OT + Telecom

```c
#include "dsmil/include/dsmil_attributes.h"
#include "dsmil/include/dsmil_telecom_log.h"
#include "dsmil/include/dsmil_ot_telemetry.h"

// OT-critical telecom function
DSMIL_OT_CRITICAL
DSMIL_OT_TIER(1)
DSMIL_TELECOM_STACK("ss7")
DSMIL_SS7_ROLE("STP")
DSMIL_TELECOM_ENV("prod")
DSMIL_SIG_SECURITY("high_assurance")
DSMIL_LAYER(3)
DSMIL_DEVICE(31)
DSMIL_STAGE("signaling")
void critical_ss7_handler(const uint8_t *msg, size_t len) {
    // Critical SS7 handler with OT telemetry
    uint32_t opc = extract_opc(msg);
    uint32_t dpc = extract_dpc(msg);
    
    DSMIL_LOG_SS7_RX(opc, dpc, 0x08, 1, 2);
    
    // Process critical message...
}

int main(void) {
    dsmil_ot_telemetry_init();
    
    uint8_t msg[] = {0x00, 0x01, 0x02, 0x00, 0x03, 0x04, 0x08, 0x01, 0x02};
    critical_ss7_handler(msg, sizeof(msg));
    
    dsmil_ot_telemetry_shutdown();
    return 0;
}
```

Build:
```bash
dsmil-clang -fdsmil-ot-telemetry \
             -fdsmil-telecom-flags \
             -fdsmil-mission-profile=ics_ops \
             -O2 \
             combined.c -o combined
```

---

## Flag Reference Summary

### Compiler Flags

| Flag | Description | Auto-Enable |
|------|-------------|-------------|
| `-fdsmil-mission-profile=<profile>` | Set mission profile | - |
| `-fdsmil-ot-telemetry` | Enable OT telemetry | `ics_ops`, `grid_ops` |
| `-fdsmil-telecom-flags` | Enable telecom flagging | `ss7_*`, `telco_*`, `sigtran_*` |

### LLVM Pass Flags

| Flag | Description |
|------|-------------|
| `-mllvm -dsmil-telemetry-manifest-path=<path>` | OT telemetry manifest path |
| `-mllvm -dsmil-telecom-manifest-path=<path>` | Telecom manifest path |
| `-mllvm -dsmil-mission-profile=<profile>` | Mission profile (LLVM level) |

### Environment Variables

| Variable | Values | Default | Description |
|----------|--------|---------|-------------|
| `DSMIL_OT_TELEMETRY` | `0`, `1` | `1` | Enable/disable OT telemetry |
| `DSMIL_TELECOM_FLAGS` | `0`, `1` | `1` | Enable/disable telecom flags |

---

## Troubleshooting

### Telemetry Not Appearing

1. Check flags are set: `-fdsmil-ot-telemetry` or `-fdsmil-telecom-flags`
2. Verify environment variable: `DSMIL_OT_TELEMETRY=1`
3. Check stderr output (telemetry goes to stderr)
4. Ensure functions are properly annotated

### Manifest Not Generated

1. Verify flags are set
2. Check write permissions for output directory
3. Look for warnings in compiler output
4. Ensure functions have relevant annotations

### Security Policy Violations

1. Review error messages for environment mismatches
2. Check mission profile matches code environment
3. Separate production and honeypot code into different modules
4. Verify `DSMIL_TELECOM_ENV` matches mission profile

### Build Errors

1. Ensure DSLLVM is properly installed
2. Check runtime libraries are linked: `libdsmil_ot_telemetry.a`
3. Verify header paths: `dsmil/include/`
4. Check LLVM version compatibility

---

## See Also

- `dsmil/include/dsmil_attributes.h` - All attribute definitions
- `dsmil/include/dsmil_ot_telemetry.h` - OT telemetry API
- `dsmil/include/dsmil_telecom_log.h` - Telecom telemetry helpers
- `dsmil/docs/OT-TELEMETRY-GUIDE.md` - OT telemetry detailed guide
- `dsmil/docs/TELECOM-SS7-GUIDE.md` - SS7/SIGTRAN detailed guide
- `dsmil/examples/` - Complete examples

---

## Quick Reference Card

```bash
# OT Telemetry
dsmil-clang -fdsmil-ot-telemetry -fdsmil-mission-profile=ics_ops source.c

# SS7/SIGTRAN Telecom
dsmil-clang -fdsmil-telecom-flags -fdsmil-mission-profile=ss7_lab source.c

# Combined
dsmil-clang -fdsmil-ot-telemetry -fdsmil-telecom-flags \
             -fdsmil-mission-profile=ics_ops source.c

# Runtime control
DSMIL_OT_TELEMETRY=1 ./your_program 2>telemetry.log
```

---

**End of Guide**

# DSLLVM Complete Build & Feature Guide

## Overview

This is the **definitive reference** for all DSLLVM compiler flags, feature flags, runtime APIs, and build system integration. Use this guide when building DSLLVM-enabled modules, integrating with IAI building systems, or configuring DSLLVM for any use case.

**Version**: 1.8  
**Last Updated**: 2024

---

## Table of Contents

1. [Quick Reference](#quick-reference)
2. [Compiler Flags](#compiler-flags)
3. [LLVM Pass Flags](#llvm-pass-flags)
4. [Mission Profiles](#mission-profiles)
5. [Attributes Reference](#attributes-reference)
6. [Runtime APIs](#runtime-apis)
7. [Build System Integration](#build-system-integration)
8. [Feature Modules](#feature-modules)
9. [Integration Examples](#integration-examples)
10. [Troubleshooting](#troubleshooting)

---

## Quick Reference

### Essential Commands

```bash
# Basic DSLLVM compilation
dsmil-clang -c source.c -o source.o

# With mission profile
dsmil-clang -fdsmil-mission-profile=ics_ops source.c -o source

# Enable OT telemetry
dsmil-clang -fdsmil-ot-telemetry \
             -fdsmil-mission-profile=ics_ops \
             source.c -o source

# Enable telecom flagging
dsmil-clang -fdsmil-telecom-flags \
             -fdsmil-mission-profile=ss7_lab \
             source.c -o source

# Enable fuzzing
dsmil-clang++ -fsanitize=fuzzer \
               -mllvm -dsmil-fuzz-coverage \
               harness.cpp source.cpp \
               -ldsmil_fuzz_telemetry \
               -o fuzz_target
```

---

## Compiler Flags

### Mission Profile Flag

**Flag**: `-fdsmil-mission-profile=<profile_id>`

**Description**: Sets the mission profile that controls compilation behavior, security policies, and runtime constraints.

**Available Profiles**:
- `ics_ops` - Industrial Control Systems operations
- `grid_ops` - Power grid operations
- `ss7_lab` - SS7 laboratory environment
- `ss7_honeypot` - SS7 honeypot deployment
- `cyber_defence` - Cyber defense operations
- `border_ops` - Border operations (high security)
- `exercise_only` - Training exercises
- `lab_research` - Laboratory research

**Usage**:
```bash
dsmil-clang -fdsmil-mission-profile=ics_ops source.c
```

**Effects**:
- Auto-enables relevant telemetry subsystems
- Sets security policy levels
- Configures AI modes
- Enforces stage whitelist/blacklist
- Controls constant-time enforcement
- Sets provenance requirements

---

### OT Telemetry Flag

**Flag**: `-fdsmil-ot-telemetry`

**Description**: Enables OT (Operational Technology) telemetry instrumentation for safety-critical functions.

**Auto-Enable**: Automatically enabled for `ics_ops` and `grid_ops` mission profiles.

**Usage**:
```bash
dsmil-clang -fdsmil-ot-telemetry source.c
```

**Effects**:
- Instruments functions marked with `DSMIL_OT_CRITICAL`
- Tracks SES gate functions
- Monitors safety signal updates
- Generates telemetry manifest JSON

**Runtime Control**:
```bash
export DSMIL_OT_TELEMETRY=1  # Enable (default)
export DSMIL_OT_TELEMETRY=0  # Disable
```

**Output**:
- Telemetry events to stderr (JSON lines)
- Manifest: `<module>.dsmil.telemetry.json`

---

### Telemetry Level Flag (v1.9)

**Flag**: `-fdsmil-telemetry-level=<level>`

**Description**: Controls telemetry instrumentation verbosity level.

**Levels**:
- `off` - No telemetry instrumentation
- `min` - Minimal telemetry (safety-critical only: OT events, errors, panics)
- `normal` - Normal telemetry (entry probes for all annotated functions) - **default**
- `debug` - Debug telemetry (entry + exit + elapsed time)
- `trace` - Trace telemetry (all + probabilistic sampling)

**Usage**:
```bash
# Minimal telemetry (production)
dsmil-clang -fdsmil-ot-telemetry \
             -fdsmil-telemetry-level=min \
             source.c -o source

# Normal telemetry (default)
dsmil-clang -fdsmil-ot-telemetry \
             -fdsmil-telemetry-level=normal \
             source.c -o source

# Debug telemetry (development)
dsmil-clang -fdsmil-ot-telemetry \
             -fdsmil-telemetry-level=debug \
             source.c -o source

# Trace telemetry (detailed analysis)
dsmil-clang -fdsmil-ot-telemetry \
             -fdsmil-telemetry-level=trace \
             source.c -o source
```

**Effects**:
- Controls which events are instrumented
- Affects entry/exit/timing instrumentation
- Level gating at runtime

**Runtime Override**:
```bash
# Override level at runtime
export DSMIL_TELEMETRY_LEVEL=debug
./my_program

# Mission profile affects default level
export DSMIL_MISSION_PROFILE=ics_prod  # Forces min level minimum
./my_program
```

**Level Lattice**: `off < min < normal < debug < trace`

---

### Telecom Flags Flag

**Flag**: `-fdsmil-telecom-flags`

**Description**: Enables SS7/SIGTRAN telecom annotation discovery and manifest generation.

**Auto-Enable**: Automatically enabled for mission profiles containing `"ss7"`, `"telco"`, `"sigtran"`, or `"telecom"`.

**Usage**:
```bash
dsmil-clang -fdsmil-telecom-flags source.c
```

**Effects**:
- Discovers telecom annotations
- Generates telecom manifest JSON
- Validates security policies (prod vs honeypot)
- Enforces environment consistency

**Runtime Control**:
```bash
export DSMIL_TELECOM_FLAGS=1  # Enable
export DSMIL_TELECOM_FLAGS=0  # Disable
```

**Output**:
- Manifest: `<module>.dsmil.telecom.json`

---

### Fuzzing Flags

**Flags**:
- `-fsanitize=fuzzer` (libFuzzer integration)
- `-mllvm -dsmil-fuzz-coverage` (Coverage instrumentation)
- `-mllvm -dsmil-fuzz-state-machine` (State machine tracking)
- `-mllvm -dsmil-fuzz-metrics` (Operation metrics)
- `-mllvm -dsmil-fuzz-api-misuse` (API misuse detection)

**Description**: Enable fuzzing instrumentation for any target.

**Usage**:
```bash
dsmil-clang++ -fsanitize=fuzzer \
               -mllvm -dsmil-fuzz-coverage \
               -mllvm -dsmil-fuzz-state-machine \
               harness.cpp source.cpp \
               -ldsmil_fuzz_telemetry \
               -o fuzz_target
```

**Effects**:
- Instruments coverage sites
- Tracks state machine transitions
- Collects operation metrics
- Detects API misuse

---

## LLVM Pass Flags

### OT Telemetry Pass

**Flag**: `-mllvm -dsmil-ot-telemetry`

**Description**: Enable OT telemetry instrumentation pass. Instruments OT-critical functions, generic annotations (NET_IO, CRYPTO, PROCESS, FILE, UNTRUSTED, ERROR_HANDLER), and safety signals.

**Additional Flags**:
- `-mllvm -dsmil-telemetry-level=<level>` - Set telemetry level (off, min, normal, debug, trace)
- `-mllvm -dsmil-telemetry-manifest-path=<path>` - Custom manifest path
- `-mllvm -dsmil-mission-profile=<profile>` - Mission profile name

**Usage**:
```bash
dsmil-clang -mllvm -dsmil-ot-telemetry source.c
```

**Features**:
- Entry/exit instrumentation (based on level)
- Timing measurements (debug/trace levels)
- Generic annotation support
- Error handler detection with panic detection
- Libc symbol heuristics

**Related Flags**:
- `-mllvm -dsmil-telemetry-manifest-path=<path>` - Custom manifest path

---

### Telemetry Metrics Pass (v1.9)

**Flag**: `-mllvm -dsmil-metrics`

**Description**: Collects telemetry instrumentation metrics and generates JSON manifest with statistics.

**Additional Flags**:
- `-mllvm -dsmil-metrics-output-dir=<dir>` - Output directory for metrics JSON files
- `-mllvm -dsmil-mission-profile=<profile>` - Mission profile name

**Usage**:
```bash
dsmil-clang -mllvm -dsmil-metrics source.c
```

**Output**:
- Metrics manifest: `<module>.dsmil.metrics.json`
- Contains: function counts, instrumentation coverage, category distribution, OT tier distribution, telecom statistics

**Example Output**:
```json
{
  "module_id": "network_daemon",
  "metrics": {
    "total_functions": 150,
    "instrumented_functions": 45,
    "instrumentation_coverage": 30.0,
    "net_io_count": 15,
    "crypto_count": 8,
    "authority_tiers": {
      "tier_0": 2,
      "tier_1": 8
    }
  }
}
```

---

### Telecom Pass

**Flag**: `-mllvm -dsmil-telecom-flags`

**Description**: Enable telecom annotation discovery pass.

**Usage**:
```bash
dsmil-clang -mllvm -dsmil-telecom-flags source.c
```

**Related Flags**:
- `-mllvm -dsmil-telecom-manifest-path=<path>` - Custom manifest path
- `-mllvm -dsmil-mission-profile=<profile>` - Mission profile (LLVM level)

---

### Fuzzing Passes

**Flags**:
- `-mllvm -dsmil-fuzz-coverage` - Coverage instrumentation
- `-mllvm -dsmil-fuzz-state-machine` - State machine tracking
- `-mllvm -dsmil-fuzz-metrics` - Operation metrics
- `-mllvm -dsmil-fuzz-api-misuse` - API misuse detection
- `-mllvm -dsmil-fuzz-crypto-timing` - Enable timing measurements

**Usage**:
```bash
dsmil-clang++ -mllvm -dsmil-fuzz-coverage \
               -mllvm -dsmil-fuzz-state-machine \
               source.cpp
```

---

### Other Pass Flags

**Constant-Time Enforcement**:
- `-mllvm -dsmil-ct-check` - Enable constant-time checks
- `-mllvm -dsmil-ct-check-strict` - Strict mode (warnings as errors)
- `-mllvm -dsmil-ct-check-output=<path>` - Violations report path

**Telemetry Enforcement**:
- `-mllvm -dsmil-telemetry-check-mode=<enforce|warn|disabled>` - Enforcement mode
- `-mllvm -dsmil-telemetry-check-callgraph` - Check call graph

**Mission Policy**:
- `-mllvm -dsmil-mission-policy-mode=<enforce|warn|disabled>` - Policy mode

---

## Mission Profiles

### Profile Configuration

Mission profiles are defined in `/etc/dsmil/mission-profiles.json` or specified via `-fdsmil-mission-profile=<id>`.

### Profile Effects

Each profile controls:

| Setting | Description |
|---------|-------------|
| Pipeline | Hardened/enhanced/standard/permissive |
| AI Mode | Local/hybrid/cloud |
| Sandbox Defaults | Sandbox profile selection |
| Stage Whitelist | Allowed MLOps stages |
| Telemetry Level | Minimal/standard/full/verbose |
| Constant-Time | Enforcement level |
| Provenance | Requirements |
| Device Access | Layer/device policies |

### Profile Reference

#### `ics_ops`
- **Use Case**: Industrial Control Systems production
- **Telemetry**: Auto-enables OT telemetry
- **Security**: High assurance
- **Stages**: `serve`, `control` only

#### `grid_ops`
- **Use Case**: Power grid operations
- **Telemetry**: Auto-enables OT telemetry
- **Security**: High assurance
- **Stages**: `serve`, `control` only

#### `ss7_lab`
- **Use Case**: SS7 laboratory/testing
- **Telemetry**: Auto-enables telecom flagging
- **Security**: Defense lab
- **Environment**: `lab`

#### `ss7_honeypot`
- **Use Case**: SS7 honeypot deployment
- **Telemetry**: Auto-enables telecom flagging
- **Security**: Defense lab
- **Environment**: `honeypot`
- **Warning**: Must not run in production

#### `cyber_defence`
- **Use Case**: Cyber defense operations
- **Telemetry**: Full telemetry
- **Security**: High assurance
- **AI Mode**: Enhanced

#### `border_ops`
- **Use Case**: Border operations
- **Telemetry**: Minimal (stealth)
- **Security**: Maximum
- **Stages**: `serve` only

#### `exercise_only`
- **Use Case**: Training exercises
- **Telemetry**: Verbose
- **Security**: Relaxed
- **Stages**: All allowed

#### `lab_research`
- **Use Case**: Laboratory research
- **Telemetry**: Verbose
- **Security**: Relaxed
- **Stages**: All allowed (including `experimental`)

---

## Attributes Reference

### Layer & Device Attributes

#### `DSMIL_LAYER(layer)`
**Purpose**: Assign function/global to DSMIL layer (0-8)

**Example**:
```c
DSMIL_LAYER(7)
void ai_function(void) { }
```

#### `DSMIL_DEVICE(device_id)`
**Purpose**: Assign function/global to DSMIL device (0-103)

**Example**:
```c
DSMIL_DEVICE(47)
void npu_function(void) { }
```

#### `DSMIL_PLACEMENT(layer, device_id)`
**Purpose**: Combined layer and device assignment

**Example**:
```c
DSMIL_PLACEMENT(7, 47)
void ai_npu_function(void) { }
```

#### `DSMIL_STAGE(stage_name)`
**Purpose**: MLOps lifecycle stage

**Values**: `"pretrain"`, `"finetune"`, `"quantized"`, `"distilled"`, `"serve"`, `"debug"`, `"experimental"`

**Example**:
```c
DSMIL_STAGE("serve")
void production_function(void) { }
```

---

### Security & Policy Attributes

#### `DSMIL_CLEARANCE(clearance_mask)`
**Purpose**: Security clearance level (32-bit mask)

**Example**:
```c
DSMIL_CLEARANCE(0x07070707)
void sensitive_function(void) { }
```

#### `DSMIL_ROE(rules)`
**Purpose**: Rules of Engagement

**Values**: `"ANALYSIS_ONLY"`, `"LIVE_CONTROL"`, `"NETWORK_EGRESS"`, `"CRYPTO_SIGN"`, `"ADMIN_OVERRIDE"`

**Example**:
```c
DSMIL_ROE("ANALYSIS_ONLY")
void read_only_function(void) { }
```

#### `DSMIL_GATEWAY`
**Purpose**: Mark function as authorized boundary crossing point

**Example**:
```c
DSMIL_GATEWAY
DSMIL_LAYER(5)
int syscall_handler(int num, void *args) { }
```

#### `DSMIL_SANDBOX(profile_name)`
**Purpose**: Sandbox profile for program entry point

**Example**:
```c
DSMIL_SANDBOX("l7_llm_worker")
int main(int argc, char **argv) { }
```

#### `DSMIL_UNTRUSTED_INPUT`
**Purpose**: Mark untrusted data inputs

**Example**:
```c
DSMIL_UNTRUSTED_INPUT
void process_user_input(const char *data, size_t len) { }
```

#### `DSMIL_SECRET`
**Purpose**: Mark cryptographic secrets (constant-time enforcement)

**Example**:
```c
DSMIL_SECRET
void aes_encrypt(const uint8_t *key, const uint8_t *data) { }
```

---

### OT Telemetry Attributes

#### `DSMIL_OT_CRITICAL`
**Purpose**: Mark OT-critical function

**Example**:
```c
DSMIL_OT_CRITICAL
DSMIL_LAYER(3)
DSMIL_DEVICE(12)
void pump_control(double setpoint) { }
```

#### `DSMIL_OT_TIER(level)`
**Purpose**: Authority tier (0-3)

**Values**:
- `0` - Safety kernel / SIS
- `1` - High-impact control
- `2` - Optimization/scheduling
- `3` - Analytics/advisory

**Example**:
```c
DSMIL_OT_TIER(1)
DSMIL_OT_CRITICAL
void critical_control(void) { }
```

#### `DSMIL_SES_GATE`
**Purpose**: Mark SES (Safety Envelope Supervisor) gate function

**Example**:
```c
DSMIL_SES_GATE
DSMIL_OT_CRITICAL
int request_pump_start(int pump_id) { }
```

#### `DSMIL_SAFETY_SIGNAL(name)`
**Purpose**: Mark safety-relevant signal variable

**Example**:
```c
DSMIL_SAFETY_SIGNAL("line7_pressure_setpoint")
static double pressure_setpoint = 100.0;
```

---

### Telecom Attributes

#### `DSMIL_TELECOM_STACK(name)`
**Purpose**: Mark telecom stack

**Values**: `"ss7"`, `"sigtran"`, `"sip"`, `"diameter"`

**Example**:
```c
DSMIL_TELECOM_STACK("ss7")
void ss7_handler(const uint8_t *msg) { }
```

#### `DSMIL_SS7_ROLE(role)`
**Purpose**: SS7 node role

**Values**: `"STP"`, `"MSC"`, `"HLR"`, `"VLR"`, `"SMSC"`, `"GWMSC"`, `"IN"`, `"GMSC"`

**Example**:
```c
DSMIL_SS7_ROLE("STP")
DSMIL_TELECOM_STACK("ss7")
void stp_routing(void) { }
```

#### `DSMIL_SIGTRAN_ROLE(role)`
**Purpose**: SIGTRAN role

**Values**: `"SG"`, `"AS"`, `"ASP"`, `"IPSP"`

**Example**:
```c
DSMIL_SIGTRAN_ROLE("SG")
DSMIL_TELECOM_STACK("sigtran")
void sigtran_gateway(void) { }
```

#### `DSMIL_TELECOM_ENV(env)`
**Purpose**: Environment classification

**Values**: `"prod"`, `"lab"`, `"honeypot"`, `"fuzz"`, `"sim"`

**Example**:
```c
DSMIL_TELECOM_ENV("honeypot")
DSMIL_TELECOM_STACK("ss7")
void honeypot_handler(void) { }
```

#### `DSMIL_SIG_SECURITY(level)`
**Purpose**: Security posture

**Values**: `"high_assurance"`, `"defense_lab"`, `"redteam_sim"`, `"low"`

**Example**:
```c
DSMIL_SIG_SECURITY("defense_lab")
DSMIL_TELECOM_ENV("lab")
void defense_analyzer(void) { }
```

#### `DSMIL_TELECOM_INTERFACE(name)`
**Purpose**: Interface type

**Values**: `"e1"`, `"t1"`, `"sctp"`, `"m2pa"`, `"m2ua"`, `"m3ua"`, `"sua"`

**Example**:
```c
DSMIL_TELECOM_INTERFACE("m3ua")
DSMIL_TELECOM_STACK("sigtran")
void m3ua_handler(void) { }
```

#### `DSMIL_TELECOM_ENDPOINT(name)`
**Purpose**: Logical endpoint identifier

**Example**:
```c
DSMIL_TELECOM_ENDPOINT("upstream_stp")
DSMIL_TELECOM_STACK("ss7")
void upstream_handler(void) { }
```

---

### Fuzzing Attributes

#### `DSMIL_FUZZ_COVERAGE`
**Purpose**: Enable coverage instrumentation

**Example**:
```c
DSMIL_FUZZ_COVERAGE
void parse_input(const uint8_t *data, size_t len) { }
```

#### `DSMIL_FUZZ_ENTRY_POINT`
**Purpose**: Mark primary fuzzing target

**Example**:
```c
DSMIL_FUZZ_ENTRY_POINT
DSMIL_FUZZ_COVERAGE
int main_fuzz_target(const uint8_t *data, size_t len) { }
```

#### `DSMIL_FUZZ_STATE_MACHINE(sm_name)`
**Purpose**: Mark state machine function

**Example**:
```c
DSMIL_FUZZ_STATE_MACHINE("http_parser")
int http_parse(const uint8_t *data, size_t len) { }
```

#### `DSMIL_FUZZ_CRITICAL_OP(op_name)`
**Purpose**: Mark critical operation for metrics

**Example**:
```c
DSMIL_FUZZ_CRITICAL_OP("json_parse")
int json_parse(const char *json) { }
```

#### `DSMIL_FUZZ_API_MISUSE_CHECK(api_name)`
**Purpose**: Enable API misuse detection

**Example**:
```c
DSMIL_FUZZ_API_MISUSE_CHECK("buffer_write")
int buffer_write(void *buf, const void *data, size_t len) { }
```

#### `DSMIL_FUZZ_CONSTANT_TIME_LOOP`
**Purpose**: Mark constant-time critical loop

**Example**:
```c
DSMIL_FUZZ_CONSTANT_TIME_LOOP
for (size_t i = 0; i < len; i++) {
    // Constant-time operations
}
```

---

## Runtime APIs

### OT Telemetry API

**Header**: `dsmil/include/dsmil_ot_telemetry.h`

#### Initialization

```c
int dsmil_ot_telemetry_init(void);
void dsmil_ot_telemetry_shutdown(void);
int dsmil_ot_telemetry_is_enabled(void);
```

#### Telemetry Level Management (v1.9)

```c
dsmil_telemetry_level_t dsmil_telemetry_get_level(void);
int dsmil_telemetry_level_allows(dsmil_telemetry_event_type_t event_type, const char *category);
```

**Telemetry Levels**:
- `DSMIL_TELEMETRY_LEVEL_OFF` (0) - No telemetry
- `DSMIL_TELEMETRY_LEVEL_MIN` (1) - Minimal (safety-critical only)
- `DSMIL_TELEMETRY_LEVEL_NORMAL` (2) - Normal (entry probes) - default
- `DSMIL_TELEMETRY_LEVEL_DEBUG` (3) - Debug (entry + exit + timing)
- `DSMIL_TELEMETRY_LEVEL_TRACE` (4) - Trace (all + sampling)

#### Event Logging

```c
void dsmil_telemetry_event(const dsmil_telemetry_event_t *ev);
void dsmil_telemetry_safety_signal_update(const dsmil_telemetry_event_t *ev);
```

**Event Types**:
- `DSMIL_TELEMETRY_OT_PATH_ENTRY` (1)
- `DSMIL_TELEMETRY_OT_PATH_EXIT` (2)
- `DSMIL_TELEMETRY_SES_INTENT` (3)
- `DSMIL_TELEMETRY_SES_ACCEPT` (4)
- `DSMIL_TELEMETRY_SES_REJECT` (5)
- `DSMIL_TELEMETRY_INVARIANT_HIT` (6)
- `DSMIL_TELEMETRY_INVARIANT_FAIL` (7)
- `DSMIL_TELEMETRY_SS7_MSG_RX` (20)
- `DSMIL_TELEMETRY_SS7_MSG_TX` (21)
- `DSMIL_TELEMETRY_SIGTRAN_MSG_RX` (22)
- `DSMIL_TELEMETRY_SIGTRAN_MSG_TX` (23)
- `DSMIL_TELEMETRY_SIG_ANOMALY` (24)
- `DSMIL_TELEMETRY_NET_IO` (30) - Network I/O
- `DSMIL_TELEMETRY_CRYPTO` (31) - Cryptographic operation
- `DSMIL_TELEMETRY_PROCESS` (32) - Process/system operation
- `DSMIL_TELEMETRY_FILE` (33) - File I/O
- `DSMIL_TELEMETRY_UNTRUSTED` (34) - Untrusted data
- `DSMIL_TELEMETRY_ERROR` (35) - Error handler
- `DSMIL_TELEMETRY_PANIC` (36) - Panic/fatal error

**Event Structure** (v1.9 extended):
```c
typedef struct {
    dsmil_telemetry_event_type_t event_type;
    const char *module_id;
    const char *func_id;
    const char *file;
    uint32_t line;
    uint8_t layer;
    uint8_t device;
    const char *stage;
    const char *mission_profile;
    uint8_t authority_tier;
    uint64_t build_id;
    uint64_t provenance_id;
    // ... safety signal fields ...
    // ... telecom fields ...
    // New fields (v1.9):
    const char *category;      // Event category
    const char *op;            // Operation name
    int32_t status_code;       // Status/return code
    const char *resource;       // Resource identifier
    const char *error_msg;     // Error message
    uint64_t elapsed_ns;       // Elapsed time (debug/trace)
} dsmil_telemetry_event_t;
```

---

### Telecom Telemetry API

**Header**: `dsmil/include/dsmil_telecom_log.h`

#### Helper Macros

```c
DSMIL_LOG_SS7_RX(opc, dpc, sio, msg_class, msg_type)
DSMIL_LOG_SS7_TX(opc, dpc, sio, msg_class, msg_type)
DSMIL_LOG_SIGTRAN_RX(rctx)
DSMIL_LOG_SIGTRAN_TX(rctx)
DSMIL_LOG_SIG_ANOMALY(stack, description)
DSMIL_LOG_SS7_FULL(opc, dpc, sio, msg_class, msg_type, role, env)
```

**Usage**:
```c
#include "dsmil/include/dsmil_telecom_log.h"

void ss7_handler(uint32_t opc, uint32_t dpc, uint8_t sio) {
    DSMIL_LOG_SS7_RX(opc, dpc, sio, 1, 2);
}
```

---

### General Fuzzing API

**Header**: `dsmil/include/dsmil_fuzz_telemetry.h`

#### Initialization

```c
int dsmil_fuzz_telemetry_init(const char *config_path, size_t ring_buffer_size);
void dsmil_fuzz_telemetry_shutdown(void);
```

#### Context Management

```c
void dsmil_fuzz_set_context(uint64_t context_id);
uint64_t dsmil_fuzz_get_context(void);
```

#### Coverage

```c
void dsmil_fuzz_cov_hit(uint32_t site_id);
```

#### State Machine

```c
void dsmil_fuzz_state_transition(uint16_t sm_id, uint16_t state_from, uint16_t state_to);
```

#### Metrics

```c
void dsmil_fuzz_metric_begin(const char *op_name);
void dsmil_fuzz_metric_end(const char *op_name);
void dsmil_fuzz_metric_record(const char *op_name, uint32_t branches,
                              uint32_t loads, uint32_t stores, uint64_t cycles);
```

#### API Misuse

```c
void dsmil_fuzz_api_misuse_report(const char *api, const char *reason, uint64_t context_id);
```

#### State Events

```c
void dsmil_fuzz_state_event(dsmil_state_event_t subtype, uint64_t state_id);
```

#### Export

```c
size_t dsmil_fuzz_get_events(dsmil_fuzz_telemetry_event_t *events, size_t max_events);
int dsmil_fuzz_flush_events(const char *filepath);
void dsmil_fuzz_clear_events(void);
```

#### Budgets

```c
int dsmil_fuzz_check_budget(const char *op_name, uint32_t branches,
                           uint32_t loads, uint32_t stores, uint64_t cycles);
```

---

### Advanced Fuzzing API

**Header**: `dsmil/include/dsmil_fuzz_telemetry_advanced.h`

#### Advanced Initialization

```c
int dsmil_fuzz_telemetry_advanced_init(const char *config_path,
                                       size_t ring_buffer_size,
                                       int enable_perf_counters,
                                       int enable_ml);
```

#### Coverage Maps

```c
int dsmil_fuzz_update_coverage_map(uint64_t input_hash,
                                    const uint32_t *new_edges, size_t new_edges_count,
                                    const uint32_t *new_states, size_t new_states_count);
void dsmil_fuzz_get_coverage_stats(uint32_t *total_edges,
                                   uint32_t *total_states,
                                   uint64_t *unique_inputs);
```

#### ML Integration

```c
double dsmil_fuzz_compute_interestingness(uint64_t input_hash,
                                          const dsmil_coverage_feedback_t *coverage_feedback);
size_t dsmil_fuzz_get_mutation_suggestions(uint32_t seed_input_id,
                                           dsmil_mutation_metadata_t *suggestions,
                                           size_t max_suggestions);
int dsmil_fuzz_export_for_ml(const char *filepath, const char *format);
```

#### Performance

```c
void dsmil_fuzz_record_perf_counters(uint64_t cpu_cycles,
                                     uint64_t cache_misses,
                                     uint64_t branch_mispredicts);
```

#### Statistics

```c
void dsmil_fuzz_get_telemetry_stats(uint64_t *total_events,
                                     double *events_per_sec,
                                     double *ring_buffer_utilization);
```

#### Advanced Export

```c
int dsmil_fuzz_flush_advanced_events(const char *filepath, int compress);
```

---

## Build System Integration

### CMake Integration

#### Basic Setup

```cmake
# Set DSLLVM compilers
set(CMAKE_C_COMPILER "dsmil-clang")
set(CMAKE_CXX_COMPILER "dsmil-clang++")

# Add DSLLVM flags
add_compile_options(
    -fdsmil-mission-profile=ics_ops
    -fdsmil-ot-telemetry
    -fdsmil-telecom-flags
)

# Link DSLLVM runtime libraries
target_link_libraries(your_target
    dsmil_ot_telemetry
    dsmil_fuzz_telemetry
    dsmil_fuzz_telemetry_advanced
)
```

#### Fuzzing Build Type

```cmake
if(DSLLVM_FUZZING)
    add_compile_definitions(DSLLVM_FUZZING=1)
    add_compile_options(
        -fsanitize=fuzzer
        -mllvm -dsmil-fuzz-coverage
        -mllvm -dsmil-fuzz-state-machine
    )
    target_link_libraries(your_target
        dsmil_fuzz_telemetry
        dsmil_fuzz_telemetry_advanced
    )
endif()
```

#### Advanced Features

```cmake
if(DSLLVM_ADVANCED_FUZZING)
    add_compile_definitions(
        DSMIL_ADVANCED_FUZZING=1
        DSMIL_ENABLE_PERF_COUNTERS=1
        DSMIL_ENABLE_ML=1
    )
    add_compile_options(
        -mllvm -dsmil-fuzz-coverage
        -mllvm -dsmil-fuzz-metrics
    )
endif()
```

#### Complete CMakeLists.txt Example

```cmake
cmake_minimum_required(VERSION 3.15)
project(MyProject)

# DSLLVM Configuration
set(CMAKE_C_COMPILER "dsmil-clang")
set(CMAKE_CXX_COMPILER "dsmil-clang++")

# Mission Profile
set(DSMIL_MISSION_PROFILE "ics_ops" CACHE STRING "Mission profile")
add_compile_options(-fdsmil-mission-profile=${DSMIL_MISSION_PROFILE})

# OT Telemetry
option(DSLLVM_OT_TELEMETRY "Enable OT telemetry" ON)
if(DSLLVM_OT_TELEMETRY)
    add_compile_options(-fdsmil-ot-telemetry)
    target_link_libraries(${PROJECT_NAME} dsmil_ot_telemetry)
endif()

# Telecom Flagging
option(DSLLVM_TELECOM_FLAGS "Enable telecom flagging" OFF)
if(DSLLVM_TELECOM_FLAGS)
    add_compile_options(-fdsmil-telecom-flags)
endif()

# Fuzzing
option(DSLLVM_FUZZING "Enable fuzzing" OFF)
if(DSLLVM_FUZZING)
    add_compile_definitions(DSLLVM_FUZZING=1)
    add_compile_options(
        -fsanitize=fuzzer
        -mllvm -dsmil-fuzz-coverage
        -mllvm -dsmil-fuzz-state-machine
    )
    target_link_libraries(${PROJECT_NAME}
        dsmil_fuzz_telemetry
        dsmil_fuzz_telemetry_advanced
    )
endif()

# Advanced Fuzzing
option(DSLLVM_ADVANCED_FUZZING "Enable advanced fuzzing" OFF)
if(DSLLVM_ADVANCED_FUZZING)
    add_compile_definitions(DSMIL_ADVANCED_FUZZING=1)
    add_compile_options(
        -mllvm -dsmil-fuzz-coverage
        -mllvm -dsmil-fuzz-metrics
        -mllvm -dsmil-fuzz-api-misuse
    )
endif()

# Include directories
target_include_directories(${PROJECT_NAME} PRIVATE
    ${DSMIL_INCLUDE_DIR}
)

# Source files
add_executable(${PROJECT_NAME} source.c)
```

---

### Makefile Integration

```makefile
CC = dsmil-clang
CXX = dsmil-clang++
CFLAGS = -fdsmil-mission-profile=ics_ops \
         -fdsmil-ot-telemetry \
         -fdsmil-telemetry-level=normal \
         -O2
CXXFLAGS = $(CFLAGS)

# Fuzzing flags
FUZZ_FLAGS = -fsanitize=fuzzer \
             -mllvm -dsmil-fuzz-coverage \
             -mllvm -dsmil-fuzz-state-machine

# Libraries
LIBS = -ldsmil_ot_telemetry \
       -ldsmil_fuzz_telemetry

# Targets
your_target: source.c
	$(CC) $(CFLAGS) source.c -o your_target $(LIBS)

fuzz_target: harness.cpp source.cpp
	$(CXX) $(FUZZ_FLAGS) harness.cpp source.cpp -o fuzz_target $(LIBS)
```

---

### Autotools Integration

```bash
# Configure
./configure CC=dsmil-clang \
            CXX=dsmil-clang++ \
            CFLAGS="-fdsmil-mission-profile=ics_ops -fdsmil-ot-telemetry" \
            LIBS="-ldsmil_ot_telemetry"
```

---

### Bazel Integration

```python
# BUILD file
cc_binary(
    name = "your_target",
    srcs = ["source.c"],
    copts = [
        "-fdsmil-mission-profile=ics_ops",
        "-fdsmil-ot-telemetry",
    ],
    deps = [
        "@dsmil//:ot_telemetry",
        "@dsmil//:fuzz_telemetry",
    ],
)
```

---

## Feature Modules

### Module 1: OT Telemetry

**Purpose**: Safety and OT visibility for industrial control systems

**Components**:
- Attributes: `DSMIL_OT_CRITICAL`, `DSMIL_OT_TIER`, `DSMIL_SES_GATE`, `DSMIL_SAFETY_SIGNAL`
- Runtime: `libdsmil_ot_telemetry.a`
- Pass: `DsmilTelemetryPass`
- Header: `dsmil/include/dsmil_ot_telemetry.h`

**Build**:
```bash
dsmil-clang -fdsmil-ot-telemetry source.c -ldsmil_ot_telemetry
```

**Use Cases**:
- Industrial control systems
- Power grid operations
- Safety-critical systems

---

### Module 2: Telecom Flagging

**Purpose**: SS7/SIGTRAN annotation discovery and telemetry

**Components**:
- Attributes: `DSMIL_TELECOM_STACK`, `DSMIL_SS7_ROLE`, `DSMIL_SIGTRAN_ROLE`, etc.
- Pass: `DsmilTelecomPass`
- Helper Macros: `dsmil/include/dsmil_telecom_log.h`

**Build**:
```bash
dsmil-clang -fdsmil-telecom-flags source.c
```

**Use Cases**:
- SS7/SIGTRAN stacks
- Telecom honeypots
- Network protocol testing

---

### Module 3: General Fuzzing Foundation

**Purpose**: Target-agnostic fuzzing infrastructure

**Components**:
- Attributes: `DSMIL_FUZZ_*`
- Runtime: `libdsmil_fuzz_telemetry.a`, `libdsmil_fuzz_telemetry_advanced.a`
- Passes: `DsmilFuzzCoveragePass`, `DsmilFuzzMetricsPass`, `DsmilFuzzApiMisusePass`
- Generator: `dsmil-gen-fuzz-harness`
- Headers: `dsmil_fuzz_telemetry.h`, `dsmil_fuzz_telemetry_advanced.h`

**Build**:
```bash
dsmil-clang++ -fsanitize=fuzzer \
               -mllvm -dsmil-fuzz-coverage \
               harness.cpp source.cpp \
               -ldsmil_fuzz_telemetry
```

**Use Cases**:
- Any fuzzing target
- Protocol fuzzing
- Parser fuzzing
- API fuzzing
- Kernel fuzzing

---

## Integration Examples

### Example 1: IAI Building Module with OT Telemetry

```cmake
# CMakeLists.txt for IAI module
cmake_minimum_required(VERSION 3.15)
project(IAI_ControlModule)

# DSLLVM Configuration
set(CMAKE_C_COMPILER "dsmil-clang")
set(CMAKE_CXX_COMPILER "dsmil-clang++")

# Mission Profile
set(DSMIL_MISSION_PROFILE "ics_ops" CACHE STRING "Mission profile")
add_compile_options(-fdsmil-mission-profile=${DSMIL_MISSION_PROFILE})

# OT Telemetry (required for IAI control modules)
add_compile_options(-fdsmil-ot-telemetry)
target_link_libraries(${PROJECT_NAME} dsmil_ot_telemetry)

# Include DSLLVM headers
target_include_directories(${PROJECT_NAME} PRIVATE
    ${DSMIL_INCLUDE_DIR}
)

# Source files
add_executable(${PROJECT_NAME}
    control_module.c
    pump_control.c
    valve_control.c
)
```

**Source Code** (`control_module.c`):
```c
#include "dsmil/include/dsmil_attributes.h"
#include "dsmil/include/dsmil_ot_telemetry.h"

DSMIL_OT_CRITICAL
DSMIL_OT_TIER(1)
DSMIL_SES_GATE
DSMIL_LAYER(3)
DSMIL_DEVICE(12)
DSMIL_STAGE("control")
int control_module_update(double setpoint) {
    dsmil_ot_telemetry_init();
    
    // Control logic
    
    dsmil_ot_telemetry_shutdown();
    return 0;
}
```

**Build**:
```bash
mkdir build && cd build
cmake -DDSMIL_MISSION_PROFILE=ics_ops ..
make
```

---

### Example 2: Network Protocol Module with Telecom Flagging

```cmake
# CMakeLists.txt for protocol module
project(NetworkProtocol)

set(CMAKE_C_COMPILER "dsmil-clang")
add_compile_options(
    -fdsmil-mission-profile=ss7_lab
    -fdsmil-telecom-flags
)

target_include_directories(${PROJECT_NAME} PRIVATE
    ${DSMIL_INCLUDE_DIR}
)

add_executable(${PROJECT_NAME} protocol.c)
```

**Source Code** (`protocol.c`):
```c
#include "dsmil/include/dsmil_attributes.h"
#include "dsmil/include/dsmil_telecom_log.h"

DSMIL_TELECOM_STACK("ss7")
DSMIL_SS7_ROLE("STP")
DSMIL_TELECOM_ENV("lab")
DSMIL_SIG_SECURITY("defense_lab")
DSMIL_LAYER(3)
DSMIL_DEVICE(31)
void ss7_process_message(const uint8_t *msg, size_t len) {
    uint32_t opc = extract_opc(msg);
    uint32_t dpc = extract_dpc(msg);
    DSMIL_LOG_SS7_RX(opc, dpc, 0x08, 1, 2);
    // Process message
}
```

---

### Example 3: Parser Module with Fuzzing Support

```cmake
# CMakeLists.txt for parser module
project(JSONParser)

set(CMAKE_C_COMPILER "dsmil-clang")
set(CMAKE_CXX_COMPILER "dsmil-clang++")

# Fuzzing support
option(BUILD_FUZZING "Build fuzzing harness" OFF)
if(BUILD_FUZZING)
    add_compile_options(
        -fsanitize=fuzzer
        -mllvm -dsmil-fuzz-coverage
        -mllvm -dsmil-fuzz-state-machine
    )
    target_link_libraries(${PROJECT_NAME}
        dsmil_fuzz_telemetry
    )
endif()

add_executable(${PROJECT_NAME} json_parser.c)
```

**Source Code** (`json_parser.c`):
```c
#include "dsmil/include/dsmil_fuzz_attributes.h"
#include "dsmil/include/dsmil_fuzz_telemetry.h"

DSMIL_FUZZ_ENTRY_POINT
DSMIL_FUZZ_COVERAGE
DSMIL_FUZZ_CRITICAL_OP("json_parse")
DSMIL_FUZZ_STATE_MACHINE("json_parser")
int json_parse(const char *json, size_t len) {
    dsmil_fuzz_telemetry_init(NULL, 65536);
    dsmil_fuzz_metric_begin("json_parse");
    
    // Parse JSON
    dsmil_fuzz_state_transition(1, 0, 1);
    
    dsmil_fuzz_metric_end("json_parse");
    return 0;
}
```

**Generate Harness**:
```bash
dsmil-gen-fuzz-harness config/json_parser.yaml harness.cpp
```

**Build Fuzzing Target**:
```bash
cmake -DBUILD_FUZZING=ON ..
make
```

---

### Example 4: Multi-Module IAI System

```cmake
# Top-level CMakeLists.txt
cmake_minimum_required(VERSION 3.15)
project(IAI_System)

# DSLLVM Configuration
set(CMAKE_C_COMPILER "dsmil-clang")
set(CMAKE_CXX_COMPILER "dsmil-clang++")

# Common flags
set(DSMIL_COMMON_FLAGS
    -fdsmil-mission-profile=ics_ops
    -fdsmil-ot-telemetry
)

# Control Module (OT-critical)
add_subdirectory(control_module)
target_compile_options(control_module PRIVATE ${DSMIL_COMMON_FLAGS})

# Network Module (Telecom)
add_subdirectory(network_module)
target_compile_options(network_module PRIVATE
    -fdsmil-telecom-flags
    -fdsmil-mission-profile=ss7_lab
)

# Parser Module (Fuzzing)
add_subdirectory(parser_module)
option(BUILD_PARSER_FUZZING OFF)
if(BUILD_PARSER_FUZZING)
    target_compile_options(parser_module PRIVATE
        -fsanitize=fuzzer
        -mllvm -dsmil-fuzz-coverage
    )
    target_link_libraries(parser_module dsmil_fuzz_telemetry)
endif()

# Main executable
add_executable(${PROJECT_NAME} main.c)
target_link_libraries(${PROJECT_NAME}
    control_module
    network_module
    parser_module
)
```

---

## Environment Variables

### OT Telemetry

```bash
export DSMIL_OT_TELEMETRY=1      # Enable (default in production)
export DSMIL_OT_TELEMETRY=0      # Disable
```

### Telecom Flags

```bash
export DSMIL_TELECOM_FLAGS=1     # Enable
export DSMIL_TELECOM_FLAGS=0     # Disable
```

### Fuzzing

```bash
export DSMIL_FUZZ_CONFIG=/path/to/config.yaml
export DSMIL_ML_MODEL_PATH=/path/to/model.onnx
export DSMIL_WORKER_ID=0
export DSMIL_NUM_WORKERS=16
```

### Advanced Fuzzing

```bash
export DSMIL_ADVANCED_FUZZING=1
export DSMIL_ENABLE_PERF_COUNTERS=1
export DSMIL_ENABLE_ML=1
```

---

## Complete Flag Reference

### Compiler Flags (Clang Level)

| Flag | Description | Auto-Enable |
|------|-------------|-------------|
| `-fdsmil-mission-profile=<profile>` | Set mission profile | - |
| `-fdsmil-ot-telemetry` | Enable OT telemetry | `ics_ops`, `grid_ops` |
| `-fdsmil-telecom-flags` | Enable telecom flagging | `ss7_*`, `telco_*` |
| `-fsanitize=fuzzer` | Enable libFuzzer | - |

### LLVM Pass Flags

| Flag | Description |
|------|-------------|
| `-mllvm -dsmil-ot-telemetry` | OT telemetry pass |
| `-mllvm -dsmil-telemetry-manifest-path=<path>` | OT manifest path |
| `-mllvm -dsmil-telecom-flags` | Telecom pass |
| `-mllvm -dsmil-telecom-manifest-path=<path>` | Telecom manifest path |
| `-mllvm -dsmil-fuzz-coverage` | Coverage instrumentation |
| `-mllvm -dsmil-fuzz-state-machine` | State machine tracking |
| `-mllvm -dsmil-fuzz-metrics` | Operation metrics |
| `-mllvm -dsmil-fuzz-api-misuse` | API misuse detection |
| `-mllvm -dsmil-fuzz-crypto-timing` | Enable timing measurements |
| `-mllvm -dsmil-ct-check` | Constant-time checks |
| `-mllvm -dsmil-telemetry-check-mode=<mode>` | Telemetry enforcement |

### Build System Variables

| Variable | Values | Description |
|----------|--------|-------------|
| `DSLLVM_FUZZING` | `ON`, `OFF` | Enable fuzzing mode |
| `DSLLVM_TELEMETRY` | `ON`, `OFF` | Enable telemetry |
| `DSLLVM_ADVANCED_FUZZING` | `ON`, `OFF` | Enable advanced fuzzing |
| `DSLLVM_ENABLE_PERF_COUNTERS` | `ON`, `OFF` | Enable performance counters |
| `DSLLVM_ENABLE_ML` | `ON`, `OFF` | Enable ML integration |
| `DSLLVM_CRYPTO_BUDGETS_CONFIG` | `<path>` | Budget config path |
| `DSMIL_MISSION_PROFILE` | `<profile>` | Mission profile (CMake) |

---

## Function Reference

### OT Telemetry Functions

| Function | Purpose |
|----------|---------|
| `dsmil_ot_telemetry_init()` | Initialize OT telemetry |
| `dsmil_ot_telemetry_shutdown()` | Shutdown OT telemetry |
| `dsmil_ot_telemetry_is_enabled()` | Check if enabled |
| `dsmil_telemetry_event()` | Log telemetry event |
| `dsmil_telemetry_safety_signal_update()` | Log safety signal |

### Telecom Functions

| Function | Purpose |
|----------|---------|
| `DSMIL_LOG_SS7_RX()` | Log SS7 message received |
| `DSMIL_LOG_SS7_TX()` | Log SS7 message transmitted |
| `DSMIL_LOG_SIGTRAN_RX()` | Log SIGTRAN message received |
| `DSMIL_LOG_SIGTRAN_TX()` | Log SIGTRAN message transmitted |
| `DSMIL_LOG_SIG_ANOMALY()` | Log signaling anomaly |

### Fuzzing Functions

| Function | Purpose |
|----------|---------|
| `dsmil_fuzz_telemetry_init()` | Initialize fuzzing telemetry |
| `dsmil_fuzz_set_context()` | Set context ID |
| `dsmil_fuzz_cov_hit()` | Record coverage hit |
| `dsmil_fuzz_state_transition()` | Record state transition |
| `dsmil_fuzz_metric_begin()` | Begin metric collection |
| `dsmil_fuzz_metric_end()` | End metric collection |
| `dsmil_fuzz_api_misuse_report()` | Report API misuse |
| `dsmil_fuzz_flush_events()` | Flush events to file |

### Advanced Fuzzing Functions

| Function | Purpose |
|----------|---------|
| `dsmil_fuzz_telemetry_advanced_init()` | Initialize advanced telemetry |
| `dsmil_fuzz_update_coverage_map()` | Update coverage bitmap |
| `dsmil_fuzz_get_coverage_stats()` | Get coverage statistics |
| `dsmil_fuzz_compute_interestingness()` | Compute ML interestingness score |
| `dsmil_fuzz_get_mutation_suggestions()` | Get ML mutation suggestions |
| `dsmil_fuzz_record_perf_counters()` | Record performance counters |
| `dsmil_fuzz_export_for_ml()` | Export for ML training |
| `dsmil_fuzz_get_telemetry_stats()` | Get telemetry statistics |

---

## Complete Build Examples

### Example 1: Production OT System

```bash
# Build
dsmil-clang -fdsmil-mission-profile=ics_ops \
             -fdsmil-ot-telemetry \
             -O2 \
             -c control.c -o control.o

# Link
dsmil-clang control.o -ldsmil_ot_telemetry -o control

# Run
DSMIL_OT_TELEMETRY=1 ./control
```

### Example 2: SS7 Honeypot

```bash
# Build
dsmil-clang -fdsmil-mission-profile=ss7_honeypot \
             -fdsmil-telecom-flags \
             -O2 \
             honeypot.c -o honeypot

# Run
./honeypot
# Manifest: honeypot.dsmil.telecom.json
```

### Example 3: Fuzzing Target

```bash
# Generate harness
dsmil-gen-fuzz-harness config.yaml harness.cpp

# Build
dsmil-clang++ -fsanitize=fuzzer \
               -mllvm -dsmil-fuzz-coverage \
               -mllvm -dsmil-fuzz-state-machine \
               harness.cpp target.cpp \
               -ldsmil_fuzz_telemetry \
               -o fuzz_target

# Run
./fuzz_target -runs=1000000 corpus/
```

### Example 4: Advanced Fuzzing

```bash
# Build with advanced features
dsmil-clang++ -fsanitize=fuzzer \
               -mllvm -dsmil-fuzz-coverage \
               -DDSMIL_ADVANCED_FUZZING=1 \
               -DDSMIL_ENABLE_PERF_COUNTERS=1 \
               harness.cpp target.cpp \
               -ldsmil_fuzz_telemetry \
               -ldsmil_fuzz_telemetry_advanced \
               -o fuzz_advanced

# Run with perf counters (requires root)
sudo ./fuzz_advanced corpus/
```

---

## Troubleshooting

### Common Issues

#### Telemetry Not Appearing

1. Check flags: `-fdsmil-ot-telemetry` or `-fdsmil-telecom-flags` set?
2. Check environment: `DSMIL_OT_TELEMETRY=1`?
3. Check stderr: Telemetry goes to stderr
4. Check annotations: Functions properly annotated?

#### Manifest Not Generated

1. Verify flags are set
2. Check write permissions
3. Look for warnings in compiler output
4. Ensure functions have relevant annotations

#### Build Errors

1. Verify DSLLVM installation
2. Check runtime libraries linked: `-ldsmil_ot_telemetry`, etc.
3. Verify header paths: `dsmil/include/`
4. Check LLVM version compatibility

#### Performance Issues

1. Reduce ring buffer size
2. Disable telemetry: `DSMIL_OT_TELEMETRY=0`
3. Use batch processing
4. Flush telemetry more frequently

---

## Integration with IAI Building Systems

### IAI Module Structure

```
iai_module/
├── CMakeLists.txt          # DSLLVM-enabled build
├── src/
│   ├── module.c           # Annotated source
│   └── ...
├── config/
│   └── dsmil_config.yaml   # DSLLVM config
└── harness/                # Optional fuzzing harnesses
    └── fuzz_harness.cpp
```

### IAI CMake Template

```cmake
# IAI Module CMakeLists.txt Template
cmake_minimum_required(VERSION 3.15)
project(IAI_ModuleName)

# DSLLVM Configuration
if(DSLLVM_FOUND)
    set(CMAKE_C_COMPILER ${DSLLVM_CLANG})
    set(CMAKE_CXX_COMPILER ${DSLLVM_CLANGXX})
    
    # Mission Profile (from IAI config or default)
    if(NOT DEFINED DSMIL_MISSION_PROFILE)
        set(DSMIL_MISSION_PROFILE "ics_ops")
    endif()
    add_compile_options(-fdsmil-mission-profile=${DSMIL_MISSION_PROFILE})
    
    # OT Telemetry (if control module)
    if(IAI_MODULE_TYPE STREQUAL "control")
        add_compile_options(-fdsmil-ot-telemetry)
        target_link_libraries(${PROJECT_NAME} dsmil_ot_telemetry)
    endif()
    
    # Telecom Flagging (if network module)
    if(IAI_MODULE_TYPE STREQUAL "network")
        add_compile_options(-fdsmil-telecom-flags)
    endif()
    
    # Fuzzing (if enabled)
    if(IAI_ENABLE_FUZZING)
        add_compile_options(
            -fsanitize=fuzzer
            -mllvm -dsmil-fuzz-coverage
        )
        target_link_libraries(${PROJECT_NAME} dsmil_fuzz_telemetry)
    endif()
    
    # Include directories
    target_include_directories(${PROJECT_NAME} PRIVATE
        ${DSLLVM_INCLUDE_DIR}
    )
endif()

# Source files
add_executable(${PROJECT_NAME} src/module.c)
```

### IAI Build Command

```bash
# Configure
cmake -DDSLLVM_DIR=/path/to/dsllvm \
      -DIAI_MODULE_TYPE=control \
      -DDSMIL_MISSION_PROFILE=ics_ops \
      ..

# Build
make

# With fuzzing
cmake -DIAI_ENABLE_FUZZING=ON ..
make
```

---

## Feature Compatibility Matrix

| Feature | Requires | Conflicts With | Compatible With |
|---------|----------|----------------|-----------------|
| OT Telemetry | `-fdsmil-ot-telemetry` | - | All |
| Telecom Flags | `-fdsmil-telecom-flags` | - | All |
| Fuzzing | `-fsanitize=fuzzer` | - | All |
| Advanced Fuzzing | `-DDSMIL_ADVANCED_FUZZING=1` | - | Fuzzing |
| Perf Counters | Root/perf permissions | - | Advanced Fuzzing |
| ML Integration | ONNX Runtime | - | Advanced Fuzzing |

---

## Performance Tuning

### For High-Throughput Systems (1+ Petaops)

```yaml
# config.yaml
telemetry:
  ring_buffer_size: 1048576  # 1MB
  enable_timing: true
  enable_perf_counters: true

performance:
  enable_parallel: true
  num_threads: 64
  enable_batch: true
  batch_size: 100000
  preallocate_buffers: true
  buffer_size: 16777216  # 16MB
```

### Compilation Flags

```bash
dsmil-clang -O3 \
             -fdsmil-ot-telemetry \
             -mllvm -dsmil-telemetry-manifest-path=/fast/ssd/manifests \
             source.c
```

---

## Complete API Summary

### Initialization Functions

```c
// OT Telemetry
int dsmil_ot_telemetry_init(void);
void dsmil_ot_telemetry_shutdown(void);

// General Fuzzing
int dsmil_fuzz_telemetry_init(const char *config_path, size_t ring_buffer_size);
void dsmil_fuzz_telemetry_shutdown(void);

// Advanced Fuzzing
int dsmil_fuzz_telemetry_advanced_init(const char *config_path,
                                       size_t ring_buffer_size,
                                       int enable_perf_counters,
                                       int enable_ml);
```

### Event Logging Functions

```c
// OT Telemetry
void dsmil_telemetry_event(const dsmil_telemetry_event_t *ev);
void dsmil_telemetry_safety_signal_update(const dsmil_telemetry_event_t *ev);

// Fuzzing
void dsmil_fuzz_cov_hit(uint32_t site_id);
void dsmil_fuzz_state_transition(uint16_t sm_id, uint16_t from, uint16_t to);
void dsmil_fuzz_metric_record(const char *op_name, uint32_t branches,
                             uint32_t loads, uint32_t stores, uint64_t cycles);
void dsmil_fuzz_api_misuse_report(const char *api, const char *reason, uint64_t context_id);

// Advanced
void dsmil_fuzz_record_advanced_event(const dsmil_advanced_fuzz_event_t *event);
```

### Coverage Functions

```c
// Basic
void dsmil_fuzz_cov_hit(uint32_t site_id);

// Advanced
int dsmil_fuzz_update_coverage_map(uint64_t input_hash,
                                    const uint32_t *new_edges, size_t new_edges_count,
                                    const uint32_t *new_states, size_t new_states_count);
void dsmil_fuzz_get_coverage_stats(uint32_t *total_edges,
                                   uint32_t *total_states,
                                   uint64_t *unique_inputs);
```

### ML Functions

```c
double dsmil_fuzz_compute_interestingness(uint64_t input_hash,
                                          const dsmil_coverage_feedback_t *coverage_feedback);
size_t dsmil_fuzz_get_mutation_suggestions(uint32_t seed_input_id,
                                           dsmil_mutation_metadata_t *suggestions,
                                           size_t max_suggestions);
int dsmil_fuzz_export_for_ml(const char *filepath, const char *format);
```

### Statistics Functions

```c
void dsmil_fuzz_get_telemetry_stats(uint64_t *total_events,
                                     double *events_per_sec,
                                     double *ring_buffer_utilization);
```

### Export Functions

```c
size_t dsmil_fuzz_get_events(dsmil_fuzz_telemetry_event_t *events, size_t max_events);
int dsmil_fuzz_flush_events(const char *filepath);
int dsmil_fuzz_flush_advanced_events(const char *filepath, int compress);
void dsmil_fuzz_clear_events(void);
```

---

## Attribute Quick Reference

### Coverage & Entry Points
- `DSMIL_FUZZ_COVERAGE` - Enable coverage tracking
- `DSMIL_FUZZ_ENTRY_POINT` - Mark primary fuzzing target

### State Machines
- `DSMIL_FUZZ_STATE_MACHINE(name)` - Mark state machine
- `DSMIL_STATE_MACHINE(name)` - (OT telemetry) State machine

### Operations
- `DSMIL_FUZZ_CRITICAL_OP(name)` - Track operation metrics
- `DSMIL_CRYPTO(name)` - (Legacy) Crypto operation
- `DSMIL_FUZZ_CONSTANT_TIME_LOOP` - Constant-time loop

### API Misuse
- `DSMIL_FUZZ_API_MISUSE_CHECK(name)` - Enable misuse detection
- `DSMIL_API_MISUSE_CHECK(name)` - (Legacy) API misuse

### OT/Safety
- `DSMIL_OT_CRITICAL` - OT-critical function
- `DSMIL_OT_TIER(level)` - Authority tier
- `DSMIL_SES_GATE` - SES gate function
- `DSMIL_SAFETY_SIGNAL(name)` - Safety signal variable

### Generic Telemetry Annotations (v1.9)
- `DSMIL_NET_IO` - Network I/O operation
- `DSMIL_CRYPTO` - Cryptographic operation
- `DSMIL_PROCESS` - Process/system operation
- `DSMIL_FILE` - File I/O operation
- `DSMIL_UNTRUSTED` - Untrusted data handling
- `DSMIL_ERROR_HANDLER` - Error handler function

### Telecom
- `DSMIL_TELECOM_STACK(name)` - Telecom stack
- `DSMIL_SS7_ROLE(role)` - SS7 role
- `DSMIL_SIGTRAN_ROLE(role)` - SIGTRAN role
- `DSMIL_TELECOM_ENV(env)` - Environment
- `DSMIL_SIG_SECURITY(level)` - Security level
- `DSMIL_TELECOM_INTERFACE(name)` - Interface type
- `DSMIL_TELECOM_ENDPOINT(name)` - Logical endpoint

### Layer & Device
- `DSMIL_LAYER(layer)` - Assign to layer
- `DSMIL_DEVICE(device_id)` - Assign to device
- `DSMIL_PLACEMENT(layer, device)` - Combined assignment
- `DSMIL_STAGE(stage)` - MLOps stage

### Security
- `DSMIL_CLEARANCE(mask)` - Security clearance
- `DSMIL_ROE(rules)` - Rules of Engagement
- `DSMIL_GATEWAY` - Cross-layer gateway
- `DSMIL_SANDBOX(profile)` - Sandbox profile
- `DSMIL_UNTRUSTED_INPUT` - Untrusted input
- `DSMIL_SECRET` - Cryptographic secret

---

## File Locations Reference

### Headers

```
dsmil/include/
├── dsmil_attributes.h                    # All DSMIL attributes
├── dsmil_ot_telemetry.h                  # OT telemetry API
├── dsmil_telecom_log.h                   # Telecom helper macros
├── dsmil_fuzz_telemetry.h                # General fuzzing API
├── dsmil_fuzz_telemetry_advanced.h       # Advanced fuzzing API
└── dsmil_fuzz_attributes.h              # Fuzzing attributes
```

### Runtime Libraries

```
dsmil/runtime/
├── dsmil_ot_telemetry.c                  # OT telemetry runtime
├── dsmil_fuzz_telemetry.c                # Basic fuzzing runtime
└── dsmil_fuzz_telemetry_advanced.c      # Advanced fuzzing runtime
```

### LLVM Passes

```
dsmil/lib/Passes/
├── DsmilTelemetryPass.cpp                # OT telemetry pass (v1.9: expanded)
├── DsmilMetricsPass.cpp                  # Telemetry metrics pass (v1.9: NEW)
├── DsmilTelecomPass.cpp                  # Telecom pass
├── DsmilFuzzCoveragePass.cpp            # Coverage pass
├── DsmilFuzzMetricsPass.cpp             # Metrics pass
└── DsmilFuzzApiMisusePass.cpp           # API misuse pass
```

### Tools

```
dsmil/tools/
├── dsmil-telemetry-summary/              # Telemetry summary tool (v1.9: NEW)
│   └── dsmil-telemetry-summary.cpp      # Aggregates metrics from all modules
├── dsmil-gen-fuzz-harness/
│   └── dsmil-gen-fuzz-harness.cpp       # Harness generator
└── ...
```

### Configurations

```
dsmil/config/
├── fuzz_telemetry_generic.yaml          # Generic fuzzing config
├── dsssl_fuzz_telemetry_advanced.yaml   # Advanced fuzzing config
├── fuzz_target_http_parser.yaml         # HTTP parser example
└── fuzz_target_json_parser.yaml        # JSON parser example
```

---

## Complete Build Command Reference

### Basic Compilation

```bash
dsmil-clang -c source.c -o source.o
dsmil-clang source.c -o source
```

### With Mission Profile

```bash
dsmil-clang -fdsmil-mission-profile=ics_ops source.c -o source
```

### With OT Telemetry

```bash
dsmil-clang -fdsmil-ot-telemetry \
             -fdsmil-mission-profile=ics_ops \
             source.c -o source
```

### With Telemetry Level Control

```bash
# Production: minimal telemetry
dsmil-clang -fdsmil-ot-telemetry \
             -fdsmil-telemetry-level=min \
             -fdsmil-mission-profile=ics_prod \
             source.c -o source

# Development: debug telemetry with timing
dsmil-clang -fdsmil-ot-telemetry \
             -fdsmil-telemetry-level=debug \
             source.c -o source

# Analysis: trace telemetry with sampling
dsmil-clang -fdsmil-ot-telemetry \
             -fdsmil-telemetry-level=trace \
             source.c -o source
```

### With Metrics Collection

```bash
# Generate telemetry manifest and metrics
dsmil-clang -fdsmil-ot-telemetry \
             -mllvm -dsmil-metrics \
             -mllvm -dsmil-metrics-output-dir=./metrics \
             source.c -o source

# Aggregate metrics from all modules
dsmil-telemetry-summary \
    --input-glob "*.dsmil.metrics.json" \
    --output dsmil.global.metrics.json
```

### With Telecom Flagging

```bash
dsmil-clang -fdsmil-telecom-flags \
             -fdsmil-mission-profile=ss7_lab \
             source.c -o source
```

### With Fuzzing

```bash
dsmil-clang++ -fsanitize=fuzzer \
               -mllvm -dsmil-fuzz-coverage \
               -mllvm -dsmil-fuzz-state-machine \
               harness.cpp source.cpp \
               -ldsmil_fuzz_telemetry \
               -o fuzz_target
```

### With Advanced Fuzzing

```bash
dsmil-clang++ -fsanitize=fuzzer \
               -mllvm -dsmil-fuzz-coverage \
               -DDSMIL_ADVANCED_FUZZING=1 \
               harness.cpp source.cpp \
               -ldsmil_fuzz_telemetry \
               -ldsmil_fuzz_telemetry_advanced \
               -o fuzz_advanced
```

### Combined Features

```bash
dsmil-clang++ -fdsmil-mission-profile=ics_ops \
               -fdsmil-ot-telemetry \
               -fdsmil-telecom-flags \
               -fsanitize=fuzzer \
               -mllvm -dsmil-fuzz-coverage \
               source.cpp \
               -ldsmil_ot_telemetry \
               -ldsmil_fuzz_telemetry \
               -o combined_target
```

---

## Integration Checklist

### For IAI Building Modules

- [ ] Set `CMAKE_C_COMPILER` to `dsmil-clang`
- [ ] Set `CMAKE_CXX_COMPILER` to `dsmil-clang++`
- [ ] Add mission profile flag: `-fdsmil-mission-profile=<profile>`
- [ ] Add OT telemetry if control module: `-fdsmil-ot-telemetry`
- [ ] Add telecom flags if network module: `-fdsmil-telecom-flags`
- [ ] Link appropriate runtime libraries
- [ ] Include DSLLVM headers: `dsmil/include/`
- [ ] Annotate source code with appropriate attributes
- [ ] Initialize telemetry in code: `dsmil_ot_telemetry_init()` or `dsmil_fuzz_telemetry_init()`
- [ ] Test compilation and runtime behavior

### For Fuzzing Targets

- [ ] Create YAML config file
- [ ] Generate harness: `dsmil-gen-fuzz-harness config.yaml harness.cpp`
- [ ] Add fuzzing flags: `-fsanitize=fuzzer`, `-mllvm -dsmil-fuzz-coverage`
- [ ] Link fuzzing libraries: `-ldsmil_fuzz_telemetry`
- [ ] Annotate target code with `DSMIL_FUZZ_*` attributes
- [ ] Compile harness and target together
- [ ] Run fuzzer: `./fuzz_target corpus/`

---

## See Also

- `dsmil/docs/DSLLVM-BUILD-FLAGS-GUIDE.md` - Feature-specific guide
- `dsmil/docs/OT-TELEMETRY-GUIDE.md` - OT telemetry details
- `dsmil/docs/TELECOM-SS7-GUIDE.md` - Telecom flagging details
- `dsmil/docs/DSMIL-GENERAL-FUZZING-GUIDE.md` - Fuzzing foundation guide
- `dsmil/include/dsmil_attributes.h` - Complete attribute reference

---

**End of Complete Build Guide**

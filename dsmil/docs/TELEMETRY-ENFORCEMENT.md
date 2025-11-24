# DSLLVM Telemetry Enforcement Guide

**Version:** 1.3.0
**Feature:** Minimum Telemetry Enforcement (Phase 1, Feature 1.3)
**SPDX-License-Identifier:** Apache-2.0 WITH LLVM-exception

## Overview

Telemetry enforcement prevents "dark functions" - critical code paths with zero forensic trail. DSLLVM enforces compile-time telemetry requirements for safety-critical and mission-critical functions, ensuring observability for:

- **Layer 5 Performance AI**: Optimization feedback
- **Layer 62 Forensics**: Post-incident analysis
- **Mission compliance**: Telemetry level enforcement

## Enforcement Levels

### Safety-Critical (`DSMIL_SAFETY_CRITICAL`)

**Requirement**: At least ONE telemetry call
**Use Case**: Important functions requiring basic observability

```c
DSMIL_SAFETY_CRITICAL("crypto")
DSMIL_LAYER(3)
void ml_kem_encapsulate(const uint8_t *pk, uint8_t *ct) {
    dsmil_counter_inc("ml_kem_calls");  // ✓ Satisfies requirement
    // ... crypto operations ...
}
```

### Mission-Critical (`DSMIL_MISSION_CRITICAL`)

**Requirement**: BOTH counter AND event telemetry + error path coverage
**Use Case**: Critical functions requiring comprehensive observability

```c
DSMIL_MISSION_CRITICAL
DSMIL_LAYER(8)
int detect_threat(const uint8_t *pkt, size_t len, float *score) {
    dsmil_counter_inc("threat_detection_calls");  // Counter required
    dsmil_event_log("threat_detection_start");    // Event required

    int result = analyze(pkt, len, score);

    if (result < 0) {
        dsmil_event_log("threat_detection_error");  // Error path logged
        return result;
    }

    dsmil_event_log("threat_detection_complete");
    return 0;
}
```

## Telemetry API

### Counter Telemetry

```c
// Increment counter (atomic, thread-safe)
void dsmil_counter_inc(const char *counter_name);

// Add value to counter
void dsmil_counter_add(const char *counter_name, uint64_t value);
```

**Use for**: Call frequency, item counts, resource usage

### Event Telemetry

```c
// Simple event (INFO severity)
void dsmil_event_log(const char *event_name);

// Event with severity
void dsmil_event_log_severity(const char *event_name,
                              dsmil_event_severity_t severity);

// Event with message
void dsmil_event_log_msg(const char *event_name,
                         dsmil_event_severity_t severity,
                         const char *message);
```

**Use for**: State transitions, errors, security events

### Performance Metrics

```c
void *timer = dsmil_perf_start("operation_name");
// ... operation ...
dsmil_perf_end(timer);
```

**Use for**: Latency measurement, performance optimization

## Compilation

```bash
# Enforce telemetry requirements (default)
dsmil-clang -fdsmil-telemetry-check src.c -o app

# Warn only
dsmil-clang -mllvm -dsmil-telemetry-check-mode=warn src.c

# Disable
dsmil-clang -mllvm -dsmil-telemetry-check-mode=disabled src.c
```

## Mission Profile Integration

Mission profiles enforce telemetry levels:

- `border_ops`: minimal (counter-only acceptable)
- `cyber_defence`: full (comprehensive required)
- `exercise_only`: verbose (all telemetry enabled)

```bash
dsmil-clang -fdsmil-mission-profile=cyber_defence \
            -fdsmil-telemetry-check src.c
```

## Common Violations

### Missing Telemetry

```c
// ✗ VIOLATION
DSMIL_SAFETY_CRITICAL
void critical_op() {
    // No telemetry calls!
}
```

**Error:**
```
ERROR: Function 'critical_op' is marked dsmil_safety_critical
       but has no telemetry calls
```

### Missing Counter (Mission-Critical)

```c
// ✗ VIOLATION
DSMIL_MISSION_CRITICAL
int mission_op() {
    dsmil_event_log("start");  // Event only, no counter!
    return do_work();
}
```

**Error:**
```
ERROR: Function 'mission_op' is marked dsmil_mission_critical
       but has no counter telemetry (dsmil_counter_inc/add required)
```

## Best Practices

1. **Add telemetry early**: At function entry
2. **Log errors**: All error paths need telemetry
3. **Use descriptive names**: `"ml_kem_calls"` not `"calls"`
4. **Component prefix**: `"crypto.ml_kem_calls"` for routing
5. **Avoid PII**: Don't log sensitive data

## References

- **API Header**: `dsmil/include/dsmil_telemetry.h`
- **Attributes**: `dsmil/include/dsmil_attributes.h`
- **Check Pass**: `dsmil/lib/Passes/DsmilTelemetryCheckPass.cpp`
- **Roadmap**: `dsmil/docs/DSLLVM-ROADMAP.md`

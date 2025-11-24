# DSLLVM Stealth Mode Guide (Feature 2.1)

**Version**: 1.4
**Feature**: Operational Stealth Modes for AI-Laden Binaries
**Status**: Implemented
**Date**: 2025-11-24

---

## Table of Contents

1. [Overview](#overview)
2. [Motivation](#motivation)
3. [Stealth Levels](#stealth-levels)
4. [Attributes](#attributes)
5. [Transformations](#transformations)
6. [Mission Profile Integration](#mission-profile-integration)
7. [Usage Examples](#usage-examples)
8. [Trade-offs and Guardrails](#trade-offs-and-guardrails)
9. [Layer 5/8 AI Integration](#layer-58-ai-integration)
10. [Best Practices](#best-practices)

---

## Overview

Stealth mode provides compiler-level transformations to reduce the detectability of binaries deployed in hostile network environments. DSLLVM implements "operational stealth" through:

- **Telemetry reduction**: Strip non-critical logging and metrics
- **Constant-rate execution**: Normalize timing patterns to prevent analysis
- **Jitter suppression**: Minimize timing variance
- **Network fingerprint reduction**: Batch and delay network I/O

These transformations are controlled by source-level attributes and mission profiles, allowing a single codebase to support both high-observability (debugging) and low-signature (covert ops) deployments.

---

## Motivation

Binaries with embedded AI/ML capabilities have distinct runtime signatures:

- **Telemetry patterns**: Frequent logging exposes activity patterns
- **Timing patterns**: Bursty computation reveals model inference
- **Network patterns**: Periodic updates create fingerprints
- **CPU patterns**: Predictable AI workloads are detectable

In hostile environments (border operations, covert surveillance), these signatures enable:
- **Detection**: Adversaries identify presence via timing/network analysis
- **Classification**: Workload patterns reveal system purpose
- **Targeting**: Known signatures enable focused attacks

Stealth mode addresses these risks by making binaries **harder to detect, classify, and target** while maintaining operational capability.

---

## Stealth Levels

DSLLVM provides three stealth levels with increasing detectability reduction:

### `STEALTH_MINIMAL` (Level 1)

**Philosophy**: Basic telemetry cleanup, preserve debugging capability

**Transformations**:
- Strip verbose/debug telemetry only
- Keep all critical and standard telemetry
- No timing transformations
- Minimal impact on observability

**Use Case**: Border operations with moderate threat

**Example**:
```c
DSMIL_LOW_SIGNATURE("minimal")
void sensor_data_processing(const uint8_t *data, size_t len) {
    // Critical telemetry preserved
    dsmil_counter_inc("sensor_readings");

    // Verbose logging stripped
    dsmil_event_log("debug_info"); // Removed by compiler
}
```

### `STEALTH_STANDARD` (Level 2)

**Philosophy**: Balance detectability and debugging

**Transformations**:
- Strip verbose and performance telemetry
- Keep critical telemetry only
- Jitter suppression enabled
- Network fingerprint reduction
- Moderate timing normalization

**Use Case**: Border operations in hostile areas, forward observers

**Example**:
```c
DSMIL_LOW_SIGNATURE("standard")
DSMIL_JITTER_SUPPRESS
void network_status_update(void) {
    // Critical counter preserved
    dsmil_counter_inc("status_updates");

    // Performance metrics stripped
    dsmil_perf_latency("update_time", 50); // Removed

    // Network I/O batched to reduce patterns
    send_status_packet();
}
```

### `STEALTH_AGGRESSIVE` (Level 3)

**Philosophy**: Maximum stealth, minimal forensic trail

**Transformations**:
- Strip all non-critical telemetry
- Constant-rate execution (timing padding)
- Maximum jitter suppression
- Aggressive network batching/delays
- Minimal fingerprints

**Use Case**: Covert operations, deep cover surveillance

**Example**:
```c
DSMIL_LOW_SIGNATURE("aggressive")
DSMIL_CONSTANT_RATE
DSMIL_NETWORK_STEALTH
void covert_data_exfiltration(const uint8_t *data, size_t len) {
    // Only critical forensic telemetry
    dsmil_forensic_checkpoint("exfil_start");

    // All other telemetry stripped
    // Function takes constant time (100ms default)
    // Network I/O heavily batched and delayed

    encrypt_and_send(data, len);
}
```

---

## Attributes

### Core Stealth Attributes

#### `DSMIL_LOW_SIGNATURE(level)`

Mark function for low-signature execution with specified stealth level.

**Parameters**:
- `level`: `"minimal"`, `"standard"`, or `"aggressive"`

**Example**:
```c
DSMIL_LOW_SIGNATURE("aggressive")
void covert_operation(void) {
    // Maximum stealth transformations applied
}
```

#### `DSMIL_STEALTH`

Alias for `DSMIL_LOW_SIGNATURE("standard")`.

**Example**:
```c
DSMIL_STEALTH
void stealth_function(void) {
    // Standard stealth level
}
```

### Timing Attributes

#### `DSMIL_CONSTANT_RATE`

Enforce constant-rate execution to prevent timing pattern analysis.

**Behavior**:
- Measures function execution time
- Pads with delay to reach target duration (default: 100ms)
- All code paths take same time

**Example**:
```c
DSMIL_CONSTANT_RATE
void heartbeat(void) {
    // Always takes exactly 100ms
    check_status();
    // Compiler adds padding delay
}
```

#### `DSMIL_JITTER_SUPPRESS`

Minimize timing variance for predictable execution.

**Optimizations**:
- Disable dynamic frequency scaling hints
- Avoid jump tables (no timing variance)
- Consistent vector width
- Predictable cache behavior

**Example**:
```c
DSMIL_JITTER_SUPPRESS
void predictable_crypto(void) {
    // Minimal timing variance
}
```

### Network Attributes

#### `DSMIL_NETWORK_STEALTH`

Transform network I/O to reduce fingerprints.

**Transformations**:
- Batch operations to avoid patterns
- Add controlled delays between sends
- Normalize packet sizes/timing

**Example**:
```c
DSMIL_NETWORK_STEALTH
void send_covert_message(const char *msg) {
    // Network I/O batched and delayed
    network_send(msg);
}
```

---

## Transformations

### 1. Telemetry Stripping

**Goal**: Remove observability hooks that expose activity patterns

**Classification**:
```c
// Critical telemetry - ALWAYS preserved
dsmil_forensic_security_event(...);
dsmil_forensic_checkpoint(...);

// Standard telemetry - stripped in STANDARD/AGGRESSIVE
dsmil_counter_inc("calls");
dsmil_event_log("event");

// Performance metrics - stripped in STANDARD/AGGRESSIVE
dsmil_perf_latency(...);
dsmil_perf_throughput(...);

// Verbose/debug - stripped in ALL stealth modes
dsmil_event_log_severity("debug", DSMIL_EVENT_DEBUG);
```

**Safety-Critical Override**:
Functions marked `DSMIL_SAFETY_CRITICAL` or `DSMIL_MISSION_CRITICAL` retain minimum telemetry even in aggressive mode:

```c
DSMIL_SAFETY_CRITICAL("crypto")
DSMIL_LOW_SIGNATURE("aggressive")
void ml_kem_decapsulate(const uint8_t *ct, uint8_t *ss) {
    // This counter is ALWAYS preserved
    dsmil_counter_inc("ml_kem_decapsulate_calls");

    // Crypto operations...
}
```

### 2. Constant-Rate Execution

**Goal**: Prevent timing pattern analysis

**Implementation**:
```c
// Compiler transformation:
void my_function() {
    uint64_t start = dsmil_get_timestamp_ns();

    // Original function body
    do_work();

    uint64_t elapsed = dsmil_get_timestamp_ns() - start;
    uint64_t target_ns = 100 * 1000000; // 100ms
    if (elapsed < target_ns) {
        dsmil_nanosleep(target_ns - elapsed);
    }
}
```

**Configuration**:
```bash
# Set target execution time
dsmil-clang -dsmil-stealth-constant-rate \
            -dsmil-stealth-rate-target-ms=200 \
            -o output input.c
```

### 3. Jitter Suppression

**Goal**: Minimize timing variance across invocations

**Compiler Attributes Added**:
```llvm
attributes #0 = {
  "no-jump-tables"           ; Avoid timing variance
  "prefer-vector-width"="256" ; Consistent SIMD width
  optsize                     ; More predictable code size (aggressive)
}
```

**Runtime Effects**:
- Consistent cache behavior
- Predictable branch patterns
- Minimal frequency scaling impact

### 4. Network Fingerprint Reduction

**Goal**: Reduce detectability via network timing/size patterns

**Batching Example**:
```c
// Normal mode: send immediately
void normal_send(const char *msg) {
    network_send(msg, strlen(msg));
}

// Stealth mode: batched and delayed
DSMIL_NETWORK_STEALTH
void stealth_send(const char *msg) {
    // Transformed by compiler to:
    dsmil_network_stealth_wrapper(msg, strlen(msg));
}

// Runtime batches operations and adds delay
void dsmil_network_stealth_wrapper(const void *data, uint64_t len) {
    static uint64_t last_send = 0;
    uint64_t now = dsmil_get_timestamp_ns();

    // Minimum 10ms between sends
    if (now - last_send < 10000000ULL) {
        dsmil_nanosleep(10000000ULL - (now - last_send));
    }

    // Add to batch queue or send immediately
    actual_network_send(data, len);
    last_send = dsmil_get_timestamp_ns();
}
```

---

## Mission Profile Integration

Stealth mode is integrated with mission profiles for deployment-wide control.

### Covert Operations Profile

**File**: `/etc/dsmil/mission-profiles.json`

```json
{
  "covert_ops": {
    "description": "Covert operations: minimal signature",
    "telemetry_level": "stealth",
    "behavioral_constraints": {
      "constant_rate_ops": true,
      "jitter_suppression": true,
      "network_fingerprint": "minimal"
    },
    "stealth_config": {
      "mode": "aggressive",
      "strip_telemetry": true,
      "preserve_safety_critical": true,
      "constant_rate_execution": true,
      "constant_rate_target_ms": 100,
      "jitter_suppression": true,
      "network_fingerprint_reduction": true
    }
  }
}
```

### Border Operations (Stealth Variant)

```json
{
  "border_ops_stealth": {
    "description": "Border operations with enhanced stealth",
    "telemetry_level": "stealth",
    "stealth_config": {
      "mode": "standard",
      "constant_rate_target_ms": 200
    }
  }
}
```

### Compilation

```bash
# Use mission profile
dsmil-clang -fdsmil-mission-profile=covert_ops \
            -O3 -o covert_bin input.c

# Or explicit stealth flags
dsmil-clang -dsmil-stealth-mode=aggressive \
            -dsmil-stealth-strip-telemetry \
            -dsmil-stealth-constant-rate \
            -dsmil-stealth-jitter-suppress \
            -dsmil-stealth-network-reduce \
            -O3 -o stealth_bin input.c
```

---

## Usage Examples

### Example 1: Covert Sensor Node

```c
#include <dsmil_attributes.h>
#include <dsmil_telemetry.h>

DSMIL_MISSION_PROFILE("covert_ops")
DSMIL_LOW_SIGNATURE("aggressive")
DSMIL_LAYER(7)
DSMIL_DEVICE(47)
int main(int argc, char **argv) {
    // Initialize (minimal setup, no verbose logging)
    dsmil_stealth_init();

    // Main loop
    while (running) {
        // Collect sensor data
        collect_environmental_data();

        // Process with AI (Layer 7, Device 47)
        analyze_patterns();

        // Covert exfiltration (batched, delayed)
        exfiltrate_findings();

        // Constant-rate heartbeat (100ms)
        heartbeat();
    }

    dsmil_stealth_shutdown();
    return 0;
}

DSMIL_CONSTANT_RATE
DSMIL_NETWORK_STEALTH
void heartbeat(void) {
    // Always takes 100ms
    // Network send batched and delayed
    send_status_update("alive");
}
```

### Example 2: Border Operations with Fallback

```c
DSMIL_MISSION_PROFILE("border_ops_stealth")
DSMIL_LOW_SIGNATURE("standard")
DSMIL_SAFETY_CRITICAL("border")
void border_surveillance(void) {
    // Standard stealth: reduced telemetry but debuggable
    dsmil_counter_inc("surveillance_cycles"); // Preserved (safety-critical)

    // Process data
    detect_intrusions();

    // Verbose logging stripped
    // dsmil_event_log("scan_complete"); // Removed by compiler

    // Critical events preserved
    if (threat_detected) {
        dsmil_forensic_security_event("threat_detected",
                                      DSMIL_EVENT_CRITICAL,
                                      threat_details);
    }
}
```

### Example 3: Crypto Worker (Constant-Time + Stealth)

```c
DSMIL_SECRET
DSMIL_SAFETY_CRITICAL("crypto")
DSMIL_LOW_SIGNATURE("aggressive")
DSMIL_LAYER(8)
void secure_key_derivation(const uint8_t *ikm, uint8_t *okm) {
    // Constant-time enforcement (DSMIL_SECRET)
    // + Stealth mode (low signature)
    // + Safety-critical telemetry preserved

    dsmil_counter_inc("key_derivations"); // Preserved

    // Constant-time HKDF
    hkdf_extract(ikm, prk);
    hkdf_expand(prk, okm);

    // Forensic checkpoint (preserved)
    dsmil_forensic_checkpoint("key_derived");
}
```

---

## Trade-offs and Guardrails

### Benefits

✅ **Reduced Detectability**
- Lower network fingerprint (batched I/O)
- Harder to analyze via timing (constant-rate)
- Minimal observability signature (stripped telemetry)

✅ **Mission Flexibility**
- Single codebase for covert/observable modes
- Flip via mission profile
- No code changes required

✅ **AI-Optimized**
- Layer 5/8 AI models detectability
- Trade-off analysis (stealth vs debugging)

### Costs

⚠️ **Lower Observability**
- Harder to debug issues in production
- Limited forensic trail
- Reduced performance insights

⚠️ **Performance Impact**
- Constant-rate execution adds delays
- Network batching increases latency
- Timing normalization may degrade throughput

⚠️ **Operational Complexity**
- Must maintain companion high-fidelity test builds
- Requires post-mission data exfiltration
- Stealth builds should not be default

### Guardrails

🛡️ **Safety-Critical Functions**
Always retain minimum telemetry even in aggressive mode:
```c
DSMIL_SAFETY_CRITICAL("component")
DSMIL_LOW_SIGNATURE("aggressive")
void critical_operation(void) {
    // This telemetry is NEVER stripped
    dsmil_counter_inc("critical_calls");
}
```

🛡️ **Companion Test Builds**
Require high-fidelity build for testing:
```bash
# Stealth build for deployment
dsmil-clang -fdsmil-mission-profile=covert_ops -o deploy.bin src.c

# High-fidelity build for testing
dsmil-clang -fdsmil-mission-profile=cyber_defence -o test.bin src.c
```

🛡️ **Deployment Restrictions**
Stealth builds should only deploy to hostile environments:
```json
{
  "covert_ops": {
    "deployment_restrictions": {
      "approved_networks": ["FIELD_OPS_NET"],
      "expiry_date": "2026-01-01",
      "max_deployment_days": null
    }
  }
}
```

🛡️ **Forensic Fallback**
Always preserve critical security events:
```c
// Even in aggressive stealth, this is logged
dsmil_forensic_security_event("intrusion_detected",
                              DSMIL_EVENT_CRITICAL,
                              details);
```

---

## Layer 5/8 AI Integration

### Layer 5: Detectability Modeling

L5 Performance AI models **detectability** based on:

```json
{
  "detectability_features": {
    "timing_patterns": {
      "burst_ratio": 0.23,
      "periodicity": 0.87,
      "variance_coefficient": 0.05
    },
    "network_patterns": {
      "packet_size_entropy": 2.1,
      "inter_packet_delay_variance": 12.3,
      "protocol_fingerprint_uniqueness": 0.91
    },
    "cpu_patterns": {
      "load_predictability": 0.78,
      "frequency_scaling_events": 23
    }
  },
  "detectability_score": 0.82,
  "recommendation": "Use STEALTH_STANDARD or higher for this deployment"
}
```

### Layer 8: Security AI Validation

L8 Security AI validates stealth transformations:

```json
{
  "stealth_validation": {
    "telemetry_stripped": 127,
    "safety_critical_preserved": 8,
    "constant_rate_functions": 3,
    "network_calls_modified": 5,
    "detectability_reduction": "67%",
    "forensic_capability": "minimal",
    "risk_assessment": {
      "lower_observability_risk": "high",
      "mitigation": "Require companion test build + post-mission exfil"
    }
  }
}
```

### Feedback Loop

```
┌──────────────────────────────────────────┐
│ DSLLVM Stealth Pass                      │
│ ├─ Strip telemetry                       │
│ ├─ Add constant-rate padding             │
│ └─ Transform network calls               │
└──────────────────┬───────────────────────┘
                   │ Binary + metadata
                   ▼
┌──────────────────────────────────────────┐
│ Layer 5 Performance AI (Devices 43-58)   │
│ ├─ Model detectability                   │
│ ├─ Estimate timing patterns              │
│ └─ Suggest stealth level                 │
└──────────────────┬───────────────────────┘
                   │ Detectability score
                   ▼
┌──────────────────────────────────────────┐
│ Layer 8 Security AI (Devices 80-87)      │
│ ├─ Validate stealth transformations      │
│ ├─ Check safety-critical preservation    │
│ └─ Balance stealth vs forensics          │
└──────────────────────────────────────────┘
```

---

## Best Practices

### 1. Choose Appropriate Stealth Level

```c
// Low-threat: minimal stealth
DSMIL_LOW_SIGNATURE("minimal")
void border_scan(void) { /* ... */ }

// Moderate threat: standard stealth
DSMIL_LOW_SIGNATURE("standard")
void forward_observer(void) { /* ... */ }

// High-threat: aggressive stealth
DSMIL_LOW_SIGNATURE("aggressive")
void deep_cover_ops(void) { /* ... */ }
```

### 2. Always Mark Safety-Critical Functions

```c
// Ensures minimum telemetry even in aggressive mode
DSMIL_SAFETY_CRITICAL("crypto")
DSMIL_LOW_SIGNATURE("aggressive")
void crypto_operation(void) {
    // Critical telemetry preserved
    dsmil_counter_inc("crypto_ops");
}
```

### 3. Maintain Test Builds

```bash
# Production stealth build
dsmil-clang -fdsmil-mission-profile=covert_ops -o prod.bin src.c

# Test build with full telemetry
dsmil-clang -fdsmil-mission-profile=cyber_defence -o test.bin src.c

# Verify both before deployment
dsmil-verify --check-mission-profile=covert_ops prod.bin
dsmil-verify --check-mission-profile=cyber_defence test.bin
```

### 4. Use Mission Profiles

```c
// Preferred: Use mission profile
DSMIL_MISSION_PROFILE("covert_ops")
int main() { /* ... */ }

// Avoid: Manual stealth flags (harder to maintain)
```

### 5. Plan for Post-Mission Data Collection

```c
DSMIL_LOW_SIGNATURE("aggressive")
void mission_loop(void) {
    // Minimal real-time telemetry
    while (running) {
        do_covert_work();
    }

    // Post-mission: exfiltrate full logs
    if (mission_complete) {
        exfiltrate_mission_logs();
    }
}
```

### 6. Combine with Constant-Time Crypto

```c
// Stealth + constant-time = defense in depth
DSMIL_SECRET
DSMIL_LOW_SIGNATURE("aggressive")
void secure_operation(const uint8_t *key) {
    // DSMIL_SECRET: constant-time enforcement (no timing leaks)
    // DSMIL_LOW_SIGNATURE: reduced detectability (no pattern leaks)
    crypto_constant_time(key);
}
```

### 7. Let AI Guide Stealth Level

```bash
# Compile with AI advisor
dsmil-clang -fdsmil-ai-mode=advisor \
            -fdsmil-mission-profile=border_ops_stealth \
            -o output input.c

# AI suggests: "Detectability: 0.67, recommend STEALTH_STANDARD"
```

---

## CLI Reference

### Compilation Flags

```bash
# Stealth mode
-dsmil-stealth-mode=<off|minimal|standard|aggressive>

# Telemetry stripping
-dsmil-stealth-strip-telemetry

# Preserve safety-critical telemetry
-dsmil-stealth-preserve-safety

# Constant-rate execution
-dsmil-stealth-constant-rate
-dsmil-stealth-rate-target-ms=<milliseconds>

# Jitter suppression
-dsmil-stealth-jitter-suppress

# Network fingerprint reduction
-dsmil-stealth-network-reduce
```

### Example Commands

```bash
# Minimal stealth
dsmil-clang -dsmil-stealth-mode=minimal -O3 -o output input.c

# Standard stealth
dsmil-clang -dsmil-stealth-mode=standard \
            -dsmil-stealth-jitter-suppress \
            -O3 -o output input.c

# Aggressive stealth
dsmil-clang -dsmil-stealth-mode=aggressive \
            -dsmil-stealth-strip-telemetry \
            -dsmil-stealth-constant-rate \
            -dsmil-stealth-rate-target-ms=150 \
            -dsmil-stealth-jitter-suppress \
            -dsmil-stealth-network-reduce \
            -O3 -o output input.c

# Use mission profile (recommended)
dsmil-clang -fdsmil-mission-profile=covert_ops \
            -O3 -o output input.c
```

---

## Provenance Integration

Stealth mode is recorded in binary provenance:

```json
{
  "compiler_version": "dsmil-clang 19.0.0-v1.4",
  "mission_profile": "covert_ops",
  "stealth_mode": {
    "level": "aggressive",
    "telemetry_stripped": 127,
    "constant_rate_functions": 3,
    "network_calls_modified": 5,
    "safety_critical_preserved": 8
  },
  "detectability_estimate": 0.23,
  "forensic_capability": "minimal",
  "deployment_restrictions": {
    "approved_networks": ["FIELD_OPS_NET"],
    "requires_companion_test_build": true
  }
}
```

---

## Summary

**Stealth Mode** (Feature 2.1) provides compiler-level transformations for low-signature execution in hostile environments:

- **Three levels**: minimal, standard, aggressive
- **Four transformations**: telemetry stripping, constant-rate execution, jitter suppression, network fingerprint reduction
- **Mission profile integration**: covert_ops, border_ops_stealth
- **AI-optimized**: Layer 5/8 model detectability and validate safety
- **Guardrails**: Safety-critical preservation, companion test builds, deployment restrictions

Use stealth mode for **covert operations**, **border surveillance**, and **forward observers** where **detectability is a primary threat**.

---

**Document Version**: 1.0
**Date**: 2025-11-24
**Next Review**: After v1.4 deployment feedback

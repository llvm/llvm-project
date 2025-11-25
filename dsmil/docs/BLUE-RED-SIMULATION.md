# DSLLVM Blue vs Red Scenario Simulation Guide (Feature 2.3)

**Version**: 1.4
**Feature**: Compiler-Level "Blue vs Red" Scenario Simulation
**Status**: Implemented
**Date**: 2025-11-25

---

## Table of Contents

1. [Overview](#overview)
2. [Motivation](#motivation)
3. [Architecture](#architecture)
4. [Build Roles](#build-roles)
5. [Attributes](#attributes)
6. [Usage Examples](#usage-examples)
7. [Mission Profiles](#mission-profiles)
8. [Runtime Control](#runtime-control)
9. [Analysis & Reporting](#analysis--reporting)
10. [Guardrails & Safety](#guardrails--safety)
11. [Integration with CI/CD](#integration-with-cicd)
12. [Best Practices](#best-practices)

---

## Overview

Blue vs Red Scenario Simulation enables **dual-build adversarial testing** from a single codebase:

- **Blue Build (Defender)**: Production configuration with full security
- **Red Build (Attacker)**: Testing configuration with adversarial instrumentation

Red builds simulate attack scenarios, map attack surfaces, and model blast radius - all without deploying vulnerable code to production.

---

## Motivation

Modern AI-laden systems need structured adversarial testing:

**Problems**:
- Separate red team tools are disconnected from production code
- Manual penetration testing misses compiler-level insights
- No systematic way to model "what if validation is bypassed?"
- Blast radius analysis requires manual threat modeling

**Solution**:
- Same codebase compiles to both blue (production) and red (testing)
- Compiler instruments red builds with attack scenarios
- Layer 5/8 AI models campaign-level effects
- Automated attack surface mapping and blast radius tracking

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│ Single Codebase (source.c)                              │
│ ├─ Normal logic (shared)                                │
│ ├─ #ifdef DSMIL_RED_BUILD                               │
│ │   └─ Red team instrumentation                         │
│ └─ Attributes (DSMIL_RED_TEAM_HOOK, etc.)               │
└───────────────┬─────────────────────────────────────────┘
                │
        ┌───────┴───────┐
        │               │
        ▼               ▼
┌──────────────┐ ┌──────────────┐
│ Blue Build   │ │ Red Build    │
├──────────────┤ ├──────────────┤
│ -fdsmil-role=│ │ -fdsmil-role=│
│  blue        │ │  red         │
├──────────────┤ ├──────────────┤
│ PRODUCTION   │ │ TESTING ONLY │
│ Full security│ │ Extra hooks  │
│ CNSA 2.0     │ │ Attack sims  │
│ Strict       │ │ Vuln inject  │
│ Deploy: YES  │ │ Deploy: NEVER│
└──────────────┘ └──────────────┘
        │               │
        │               ▼
        │        ┌─────────────────┐
        │        │ Analysis Report │
        │        │ - Attack surface│
        │        │ - Blast radius  │
        │        │ - Vuln points   │
        │        └─────────────────┘
        ▼
   Production
   Deployment
```

---

## Build Roles

### Blue Build (Defender/Production)

**Configuration**:
```bash
dsmil-clang -fdsmil-role=blue -O3 -o blue.bin source.c
```

**Characteristics**:
- ✅ Production-ready
- ✅ CNSA 2.0 provenance
- ✅ Strict sandboxing
- ✅ Full telemetry
- ✅ Constant-time enforcement
- ✅ Deploy to production: YES

**Use Cases**:
- Production deployments
- Cyber defense operations
- Border operations
- Any operational mission

### Red Build (Attacker/Testing)

**Configuration**:
```bash
dsmil-clang -fdsmil-role=red -O3 -o red.bin source.c
```

**Characteristics**:
- ⚠️ TESTING ONLY - NEVER PRODUCTION
- 📊 Extra instrumentation
- 🎯 Attack surface mapping
- 💥 Vulnerability injection points
- 📈 Blast radius tracking
- 🔒 Aggressively isolated
- ⏰ 7-day max deployment
- 🔑 Separate signing key

**Use Cases**:
- Adversarial stress-testing
- Vulnerability discovery
- Blast radius analysis
- Campaign-level modeling
- Security training exercises

---

## Attributes

### Core Attributes

#### `DSMIL_RED_TEAM_HOOK(hook_name)`

Mark function for red team instrumentation.

**Example**:
```c
DSMIL_RED_TEAM_HOOK("user_input_injection")
void process_user_input(const char *input) {
    #ifdef DSMIL_RED_BUILD
        dsmil_red_log("input_processing", __func__);

        // Simulate bypassing validation
        if (dsmil_red_scenario("bypass_validation")) {
            raw_process(input);  // Vulnerable path
            return;
        }
    #endif

    // Normal path (both builds)
    validate_and_process(input);
}
```

#### `DSMIL_ATTACK_SURFACE`

Mark functions exposed to untrusted input.

**Example**:
```c
DSMIL_ATTACK_SURFACE
void handle_network_packet(const uint8_t *pkt, size_t len) {
    // Red build: logged as attack surface
    // Layer 8 AI analyzes vulnerability potential
    parse_packet(pkt, len);
}
```

#### `DSMIL_VULN_INJECT(vuln_type)`

Mark vulnerability injection points for testing defenses.

**Vulnerability Types**:
- `"buffer_overflow"`: Buffer overflow simulation
- `"use_after_free"`: UAF simulation
- `"race_condition"`: Race condition injection
- `"injection"`: SQL/command injection
- `"auth_bypass"`: Authentication bypass

**Example**:
```c
DSMIL_VULN_INJECT("buffer_overflow")
void copy_data(char *dest, const char *src, size_t len) {
    #ifdef DSMIL_RED_BUILD
        if (dsmil_red_scenario("trigger_overflow")) {
            memcpy(dest, src, len + 100);  // Overflow
            return;
        }
    #endif

    memcpy(dest, src, len);  // Safe
}
```

#### `DSMIL_BLAST_RADIUS`

Track blast radius for compromise analysis.

**Example**:
```c
DSMIL_BLAST_RADIUS
DSMIL_LAYER(8)
void critical_security_function(void) {
    // If compromised, what cascades?
    // L5/L9 AI models campaign effects
}
```

#### `DSMIL_BUILD_ROLE(role)`

Specify build role at translation unit level.

**Example**:
```c
DSMIL_BUILD_ROLE("blue")
int main(int argc, char **argv) {
    return run_production();
}
```

---

## Usage Examples

### Example 1: Input Validation Bypass

```c
#include <dsmil_attributes.h>

DSMIL_RED_TEAM_HOOK("sql_injection")
DSMIL_ATTACK_SURFACE
void execute_query(const char *user_input) {
    #ifdef DSMIL_RED_BUILD
        dsmil_red_log("sql_query", __func__);

        // Red: simulate SQL injection
        if (dsmil_red_scenario("sql_injection")) {
            printf("[RED] Injecting: %s\n", user_input);
            execute_raw_sql(user_input);
            return;
        }
    #endif

    // Blue: normal validation
    if (!is_safe_sql(user_input)) {
        reject_query();
        return;
    }
    execute_safe_sql(sanitize(user_input));
}
```

**Blue Build**: Validates input, executes safely
**Red Build**: Can bypass validation via scenario

### Example 2: Authentication Bypass

```c
DSMIL_VULN_INJECT("auth_bypass")
DSMIL_BLAST_RADIUS
int authenticate_user(const char *username, const char *password) {
    #ifdef DSMIL_RED_BUILD
        if (dsmil_red_scenario("bypass_auth")) {
            dsmil_red_blast_radius_event(__func__, "auth_bypassed",
                                         "Testing authentication bypass");
            return 1;  // Bypass successful
        }
    #endif

    return check_credentials(username, password);
}
```

### Example 3: Campaign-Level Analysis

```c
DSMIL_BLAST_RADIUS
DSMIL_LAYER(8)
void compromise_detection_system(void) {
    // If this is compromised, attacker can:
    // 1. Disable intrusion detection
    // 2. Tamper with logs
    // 3. Pivot to Layer 7 AI systems

    // L5/L9 models: "If 3 Layer 8 systems compromised,
    //                what's the blast radius?"
}
```

---

## Mission Profiles

### Blue Production Profile

**File**: `mission-profiles-blue-red.json`

```json
{
  "blue_production": {
    "build_role": "blue",
    "pipeline": "dsmil-hardened",
    "ai_mode": "advisor",
    "ct_enforcement": "strict",
    "telemetry_level": "full",
    "deployment_restrictions": {
      "approved_networks": ["SIPRNET", "JWICS"],
      "max_deployment_days": null
    }
  }
}
```

**Usage**:
```bash
dsmil-clang -fdsmil-mission-profile=blue_production \
            -O3 -o production.bin source.c
```

### Red Stress Test Profile

```json
{
  "red_stress_test": {
    "build_role": "red",
    "pipeline": "dsmil-lab",
    "red_build_config": {
      "instrument": true,
      "attack_surface_mapping": true,
      "vuln_injection": true,
      "blast_radius_tracking": true
    },
    "deployment_restrictions": {
      "approved_networks": ["TEST_NET_ONLY"],
      "never_production": true,
      "max_deployment_days": 7
    },
    "warnings": [
      "RED BUILD - FOR TESTING ONLY",
      "NEVER DEPLOY TO PRODUCTION"
    ]
  }
}
```

**Usage**:
```bash
dsmil-clang -fdsmil-mission-profile=red_stress_test \
            -O3 -o red_test.bin source.c
```

---

## Runtime Control

### Scenario Activation

Control which attack scenarios execute via environment variable:

```bash
# No scenarios (normal execution)
./red.bin

# Single scenario
DSMIL_RED_SCENARIOS="bypass_validation" ./red.bin

# Multiple scenarios
DSMIL_RED_SCENARIOS="bypass_validation,trigger_overflow" ./red.bin

# All scenarios
DSMIL_RED_SCENARIOS="all" ./red.bin
```

### Red Team Logging

Red builds log to file:

```bash
# Default log location
/tmp/dsmil-red.log

# Custom log location
DSMIL_RED_LOG=/var/log/red-test.log ./red.bin
```

### Runtime API

```c
// Initialize red runtime
dsmil_blue_red_init(1);  // 1 = red build

// Check if scenario is active
if (dsmil_red_scenario("bypass_auth")) {
    // Simulate attack
}

// Log red event
dsmil_red_log("hook_name", __func__);

// Log with details
dsmil_red_log_detailed("hook", __func__, "details: %s", info);

// Shutdown
dsmil_blue_red_shutdown();
```

---

## Analysis & Reporting

Red builds generate JSON analysis reports:

### Attack Surface Report

```json
{
  "schema": "dsmil-red-analysis-v1",
  "module": "sensor_daemon",
  "build_role": "red",
  "statistics": {
    "red_hooks_inserted": 12,
    "attack_surfaces_mapped": 5,
    "vuln_injections_added": 3,
    "blast_radius_tracked": 8
  },
  "attack_surfaces": [
    {
      "function": "process_network_packet",
      "layer": 7,
      "device": 47,
      "has_untrusted_input": true,
      "blast_radius_score": 87
    }
  ],
  "red_hooks": [
    {
      "hook_name": "user_input_injection",
      "function": "process_user_input",
      "type": "instrumentation"
    }
  ]
}
```

**Generated via**:
```bash
dsmil-clang -fdsmil-role=red \
            -dsmil-red-output=analysis.json \
            -O3 -o red.bin source.c
```

---

## Guardrails & Safety

### Runtime Verification

Red builds are rejected at runtime if deployed incorrectly:

```c
// Loader checks build role
if (!dsmil_verify_build_role("blue")) {
    fprintf(stderr, "ERROR: Red build in production!\n");
    exit(1);
}
```

### Separate Signing Key

Red builds use different provenance key:

```bash
# Blue: signed with TSK (Trusted Signing Key)
# Red: signed with RTSK (Red Team Signing Key)
```

### Time Limits

Red builds expire after 7 days:

```json
{
  "provenance": {
    "build_role": "red",
    "build_date": "2025-11-25",
    "expiry_date": "2025-12-02"
  }
}
```

### Network Isolation

Red builds restricted to test networks:

```json
{
  "deployment_restrictions": {
    "approved_networks": ["TEST_NET_ONLY"],
    "never_production": true
  }
}
```

---

## Integration with CI/CD

### Parallel Blue/Red Testing

```yaml
# .github/workflows/blue-red-test.yml
jobs:
  blue-build:
    runs-on: meteor-lake
    steps:
      - name: Build Blue (Production)
        run: |
          dsmil-clang -fdsmil-role=blue -O3 \
            -o blue.bin src/*.c

      - name: Test Blue
        run: |
          ./blue.bin --test-mode

      - name: Deploy Blue
        run: |
          deploy-to-production blue.bin

  red-build:
    runs-on: test-cluster
    steps:
      - name: Build Red (Stress Test)
        run: |
          dsmil-clang -fdsmil-role=red -O3 \
            -dsmil-red-output=red-analysis.json \
            -o red.bin src/*.c

      - name: Run Red Scenarios
        run: |
          DSMIL_RED_SCENARIOS="all" ./red.bin

      - name: Analyze Results
        run: |
          cat red-analysis.json
          check-for-vulnerabilities red-analysis.json

      - name: NEVER Deploy Red
        run: |
          echo "Red builds never deployed"
```

---

## Best Practices

### 1. Always Build Both Flavors

```bash
# Blue for production
dsmil-clang -fdsmil-role=blue -O3 -o prod.bin src.c

# Red for testing
dsmil-clang -fdsmil-role=red -O3 -o test.bin src.c
```

### 2. Use Scenarios Selectively

```bash
# Start with no scenarios (baseline)
./red.bin

# Enable specific scenarios
DSMIL_RED_SCENARIOS="bypass_validation" ./red.bin

# Gradually increase
DSMIL_RED_SCENARIOS="bypass_validation,trigger_overflow" ./red.bin
```

### 3. Mark Critical Functions

```c
// High-value targets for red team analysis
DSMIL_BLAST_RADIUS
DSMIL_ATTACK_SURFACE
void critical_function(void) {
    // Analyze compromise impact
}
```

### 4. Review Red Analysis Reports

```bash
# Generate report
dsmil-clang -fdsmil-role=red -dsmil-red-output=report.json ...

# Review with team
cat report.json | jq '.attack_surfaces[] | select(.blast_radius_score > 70)'
```

### 5. Isolate Red Builds

```bash
# Run in isolated container
docker run --network=test-net red-container ./red.bin

# Never allow production network access
iptables -A OUTPUT -m owner --uid-owner red-user -j DROP
```

### 6. Time-Box Red Testing

```bash
# Red builds expire after 7 days
# Plan testing accordingly:
# - Day 1-2: Setup and baseline
# - Day 3-5: Scenario execution
# - Day 6-7: Analysis and reporting
```

---

## CLI Reference

### Compilation Flags

```bash
# Build role
-fdsmil-role=<blue|red>

# Red instrumentation
-dsmil-red-instrument          # Enable red team hooks
-dsmil-red-attack-surface      # Map attack surfaces
-dsmil-red-vuln-inject         # Enable vulnerability injection
-dsmil-red-output=<path>       # Analysis report output

# Mission profile
-fdsmil-mission-profile=<profile_id>
```

### Example Commands

```bash
# Blue production build
dsmil-clang -fdsmil-role=blue \
            -fdsmil-mission-profile=blue_production \
            -O3 -o blue.bin source.c

# Red stress test build
dsmil-clang -fdsmil-role=red \
            -fdsmil-mission-profile=red_stress_test \
            -dsmil-red-instrument \
            -dsmil-red-attack-surface \
            -dsmil-red-vuln-inject \
            -dsmil-red-output=red-report.json \
            -O3 -o red.bin source.c

# Verify provenance
dsmil-verify --check-build-role=blue blue.bin
dsmil-verify --check-build-role=red red.bin  # Should be rejected in prod
```

---

## Summary

**Blue vs Red Scenario Simulation** enables structured adversarial testing from a single codebase:

- **Blue Builds**: Production-ready, fully secured, deployable
- **Red Builds**: Testing-only, instrumented, never production
- **Same Code**: 95% shared, only instrumentation differs
- **AI-Enhanced**: Layer 5/8/9 campaign-level modeling
- **Guardrails**: Separate keys, time limits, network isolation

Use blue builds for operations, red builds for continuous adversarial testing.

---

**Document Version**: 1.0
**Date**: 2025-11-25
**Next Review**: After first red team exercise

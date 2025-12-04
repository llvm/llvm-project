# DSLLVM Mission Profiles - User Guide

**Version:** 1.3.0
**Feature:** Mission Profiles as First-Class Compile Targets
**SPDX-License-Identifier:** Apache-2.0 WITH LLVM-exception

## Table of Contents

1. [Introduction](#introduction)
2. [Mission Profile Overview](#mission-profile-overview)
3. [Installation and Setup](#installation-and-setup)
4. [Using Mission Profiles](#using-mission-profiles)
5. [Source Code Annotations](#source-code-annotations)
6. [Compilation Examples](#compilation-examples)
7. [Common Workflows](#common-workflows)
8. [Troubleshooting](#troubleshooting)
9. [Best Practices](#best-practices)

## Introduction

Mission profiles are first-class compile targets in DSLLVM that replace traditional `debug` and `release` configurations with operational context awareness. A mission profile defines:

- **Operational Context:** Where and how the binary will be deployed (hostile environment, training, lab, etc.)
- **Security Constraints:** Clearance levels, device access, layer policies
- **Compilation Behavior:** Optimization levels, constant-time enforcement, AI assistance
- **Runtime Requirements:** Memory limits, network access, telemetry levels
- **Compliance Requirements:** Provenance, attestation, expiration

By compiling with a specific mission profile, you ensure the resulting binary is purpose-built for its deployment environment and complies with all operational constraints.

## Mission Profile Overview

### Standard Profiles

DSLLVM 1.3 includes four standard mission profiles:

#### 1. `border_ops` - Border Operations

**Use Case:** Maximum security deployments in hostile or contested environments

**Characteristics:**
- **Classification:** RESTRICTED
- **Operational Context:** Hostile environment
- **Security:** Maximum (strict constant-time, minimal telemetry, no quantum export)
- **Optimization:** Aggressive (-O3)
- **AI Mode:** Local only (no cloud dependencies)
- **Stages Allowed:** quantized, serve (production only)
- **Device Access:** Strict whitelist (critical devices only)
- **Provenance:** Mandatory with TPM-backed ML-DSA-87 signature
- **Expiration:** None (indefinite deployment)
- **Network Egress:** Forbidden
- **Filesystem Write:** Forbidden

**When to Use:**
- Border security operations
- Air-gapped deployments
- Classified operations
- Zero-trust environments

#### 2. `cyber_defence` - Cyber Defence Operations

**Use Case:** AI-enhanced cyber defense with full observability

**Characteristics:**
- **Classification:** CONFIDENTIAL
- **Operational Context:** Defensive operations
- **Security:** High (strict constant-time, full telemetry)
- **Optimization:** Aggressive (-O3)
- **AI Mode:** Hybrid (local + cloud for updates)
- **Stages Allowed:** quantized, serve, finetune
- **AI Features:** Layer 5/7/8 AI advisors enabled
- **Provenance:** Mandatory with TPM-backed ML-DSA-87 signature
- **Expiration:** 90 days (enforced recompilation)
- **Network Egress:** Allowed (for telemetry and AI updates)
- **Filesystem Write:** Allowed

**When to Use:**
- Cyber defense operations
- Threat intelligence systems
- Adaptive security systems
- AI-powered defense platforms

#### 3. `exercise_only` - Training and Exercises

**Use Case:** Realistic training environments with relaxed constraints

**Characteristics:**
- **Classification:** UNCLASSIFIED
- **Operational Context:** Training simulation
- **Security:** Medium (relaxed constant-time, verbose telemetry)
- **Optimization:** Moderate (-O2)
- **AI Mode:** Cloud (full AI assistance)
- **Stages Allowed:** quantized, serve, finetune, debug
- **Provenance:** Basic with software ML-DSA-65 signature
- **Expiration:** 30 days (prevents accidental production use)
- **Simulation Features:** Blue/Red team modes, fault injection
- **Network Egress:** Allowed
- **Filesystem Write:** Allowed

**When to Use:**
- Training exercises
- Red team operations
- Blue team defense simulations
- Operator training

#### 4. `lab_research` - Laboratory Research

**Use Case:** Unrestricted research and development

**Characteristics:**
- **Classification:** UNCLASSIFIED
- **Operational Context:** Research and development
- **Security:** Minimal (constant-time disabled, verbose telemetry)
- **Optimization:** None (-O0 with debug symbols)
- **AI Mode:** Cloud (full experimental features)
- **Stages Allowed:** All (including experimental)
- **Provenance:** Optional
- **Expiration:** None
- **Experimental Features:** RL loop, quantum offload, custom passes
- **Network Egress:** Allowed
- **Filesystem Write:** Allowed

**When to Use:**
- Algorithm development
- Performance research
- ML model experimentation
- Prototyping new features

### Profile Comparison Matrix

| Feature | border_ops | cyber_defence | exercise_only | lab_research |
|---------|-----------|---------------|---------------|--------------|
| Classification | RESTRICTED | CONFIDENTIAL | UNCLASSIFIED | UNCLASSIFIED |
| Optimization | -O3 | -O3 | -O2 | -O0 |
| CT Enforcement | Strict | Strict | Relaxed | Disabled |
| Telemetry | Minimal | Full | Verbose | Verbose |
| AI Mode | Local | Hybrid | Cloud | Cloud |
| Provenance | ML-DSA-87 (TPM) | ML-DSA-87 (TPM) | ML-DSA-65 (SW) | Optional |
| Expiration | None | 90 days | 30 days | None |
| Production Ready | ✓ | ✓ | ✗ | ✗ |

## Installation and Setup

### 1. Install Mission Profile Configuration

The mission profile configuration file can be installed in multiple locations (checked in order):

1. `${DSMIL_CONFIG_DIR}` (default: `${DSMIL_PREFIX}/etc` or `/etc/dsmil`)
2. `${XDG_CONFIG_HOME}/dsmil` or `$HOME/.config/dsmil` (user-specific)
3. System default: `/etc/dsmil`

```bash
# System-wide installation (requires root)
# Uses dynamic path resolution
export DSMIL_CONFIG_DIR=/etc/dsmil  # Optional: override default
sudo mkdir -p ${DSMIL_CONFIG_DIR:-/etc/dsmil}
sudo cp dsmil/config/mission-profiles.json ${DSMIL_CONFIG_DIR:-/etc/dsmil}/
sudo chmod 644 ${DSMIL_CONFIG_DIR:-/etc/dsmil}/mission-profiles.json

# Or use runtime API in C code:
# #include <dsmil_paths.h>
# char config_path[PATH_MAX];
# dsmil_resolve_config("mission-profiles.json", config_path, sizeof(config_path));

# Verify installation
dsmil-clang --version
cat ${DSMIL_CONFIG_DIR:-/etc/dsmil}/mission-profiles.json | jq '.profiles | keys'
# Output: ["border_ops", "cyber_defence", "exercise_only", "lab_research"]
```

### 2. Custom Configuration Path (Optional)

For non-standard installations or custom profiles:

```bash
# Use custom config path
export DSMIL_MISSION_PROFILE_CONFIG=/path/to/custom-profiles.json

# Or specify at compile time
dsmil-clang -fdsmil-mission-profile-config=/path/to/custom-profiles.json ...
```

### 3. Signing Key Setup

For production profiles (`border_ops`, `cyber_defence`), configure signing keys:

```bash
# TPM-backed signing (recommended for production)
# Requires TPM 2.0 hardware and tpm2-tools
tpm2_createprimary -C o -g sha384 -G ecc -c primary.ctx
tpm2_create -C primary.ctx -g sha384 -G ecc -u dsmil.pub -r dsmil.priv
tpm2_load -C primary.ctx -u dsmil.pub -r dsmil.priv -c dsmil.ctx

# Set DSLLVM to use TPM key
export DSMIL_PROVENANCE_KEY=tpm://dsmil

# Software signing (development/exercise_only)
openssl genpkey -algorithm dilithium5 -out dsmil-dev.pem
export DSMIL_PROVENANCE_KEY=file:///path/to/dsmil-dev.pem
```

## Using Mission Profiles

### Basic Compilation

```bash
# Compile with border_ops profile
dsmil-clang -fdsmil-mission-profile=border_ops src/main.c -o bin/main

# Compile with cyber_defence profile
dsmil-clang -fdsmil-mission-profile=cyber_defence src/server.c -o bin/server

# Multiple source files
dsmil-clang -fdsmil-mission-profile=exercise_only \
  src/trainer.c src/scenario.c -o bin/trainer
```

### Makefile Integration

```makefile
# Makefile with mission profile support

CC = dsmil-clang
MISSION_PROFILE ?= lab_research
CFLAGS = -fdsmil-mission-profile=$(MISSION_PROFILE) -Wall -Wextra

# Production build
.PHONY: prod
prod: MISSION_PROFILE=border_ops
prod: CFLAGS += -O3
prod: clean all

# Development build
.PHONY: dev
dev: MISSION_PROFILE=lab_research
dev: CFLAGS += -O0 -g
dev: clean all

# Exercise build
.PHONY: exercise
exercise: MISSION_PROFILE=exercise_only
exercise: clean all

all: bin/llm_worker

bin/llm_worker: src/main.c src/inference.c
	$(CC) $(CFLAGS) $^ -o $@

clean:
	rm -f bin/*
```

### CMake Integration

```cmake
# CMakeLists.txt with mission profile support

cmake_minimum_required(VERSION 3.20)
project(DSLLVMApp C)

# Mission profile selection
set(DSMIL_MISSION_PROFILE "lab_research" CACHE STRING "DSMIL mission profile")
set_property(CACHE DSMIL_MISSION_PROFILE PROPERTY STRINGS
  "border_ops" "cyber_defence" "exercise_only" "lab_research")

# Apply mission profile flag
add_compile_options(-fdsmil-mission-profile=${DSMIL_MISSION_PROFILE})
add_link_options(-fdsmil-mission-profile=${DSMIL_MISSION_PROFILE})

# Targets
add_executable(llm_worker src/main.c src/inference.c)

# Installation rules
install(TARGETS llm_worker DESTINATION bin)

# Build types
# cmake -B build -DDSMIL_MISSION_PROFILE=border_ops
# cmake -B build -DDSMIL_MISSION_PROFILE=cyber_defence
```

## Source Code Annotations

### Mission Profile Attribute

Use `DSMIL_MISSION_PROFILE()` to explicitly tag functions with their intended profile:

```c
#include <dsmil_attributes.h>

// Border operations worker
DSMIL_MISSION_PROFILE("border_ops")
DSMIL_LAYER(7)
DSMIL_DEVICE(47)
DSMIL_ROE("ANALYSIS_ONLY")
int main(int argc, char **argv) {
    // Compiled with border_ops constraints:
    // - Only quantized or serve stages allowed
    // - Strict constant-time enforcement
    // - Minimal telemetry
    // - Local AI mode only
    return run_llm_inference();
}
```

### Stage Annotations

Ensure stage annotations comply with mission profile:

```c
// ✓ VALID for border_ops (allows "serve" stage)
DSMIL_MISSION_PROFILE("border_ops")
DSMIL_STAGE("serve")
void production_inference(const float *input, float *output) {
    // Production inference code
}

// ✗ INVALID for border_ops (does not allow "debug" stage)
DSMIL_MISSION_PROFILE("border_ops")
DSMIL_STAGE("debug")  // Compile error!
void debug_inference(const float *input, float *output) {
    // Debug code not allowed in border_ops
}

// ✓ VALID for exercise_only (allows "debug" stage)
DSMIL_MISSION_PROFILE("exercise_only")
DSMIL_STAGE("debug")
void exercise_debug(const float *input, float *output) {
    // Debug code allowed in exercises
}
```

### Layer and Device Constraints

```c
// ✓ VALID for border_ops (device 47 is whitelisted)
DSMIL_MISSION_PROFILE("border_ops")
DSMIL_LAYER(7)
DSMIL_DEVICE(47)  // NPU primary (whitelisted)
void npu_inference(void) {
    // NPU inference
}

// ✗ INVALID for border_ops (device 40 not whitelisted)
DSMIL_MISSION_PROFILE("border_ops")
DSMIL_LAYER(7)
DSMIL_DEVICE(40)  // GPU (not whitelisted) - Compile error!
void gpu_inference(void) {
    // GPU inference not allowed
}
```

### Quantum Export Restrictions

```c
// ✗ INVALID for border_ops (quantum_export: false)
DSMIL_MISSION_PROFILE("border_ops")
DSMIL_QUANTUM_CANDIDATE("placement")  // Compile error!
int optimize_placement(void) {
    // Quantum candidates not allowed in border_ops
}

// ✓ VALID for cyber_defence (quantum_export: true)
DSMIL_MISSION_PROFILE("cyber_defence")
DSMIL_QUANTUM_CANDIDATE("placement")
int optimize_placement(void) {
    // Quantum optimization allowed
}
```

## Compilation Examples

### Example 1: Border Operations LLM Worker

**Source: `llm_worker.c`**
```c
#include <dsmil_attributes.h>
#include <stdint.h>

// Main entry point - border operations profile
DSMIL_MISSION_PROFILE("border_ops")
DSMIL_LLM_WORKER_MAIN  // Expands to layer 7, device 47, etc.
int main(int argc, char **argv) {
    return llm_inference_loop();
}

// Production inference function
DSMIL_STAGE("serve")
DSMIL_LAYER(7)
DSMIL_DEVICE(47)
int llm_inference_loop(void) {
    // Inference loop
    return 0;
}

// Crypto key handling - strict constant-time
DSMIL_SECRET
DSMIL_LAYER(3)
DSMIL_DEVICE(30)
void derive_session_key(const uint8_t *master, uint8_t *session) {
    // Constant-time key derivation
}
```

**Compile:**
```bash
dsmil-clang \
  -fdsmil-mission-profile=border_ops \
  -fdsmil-provenance=full \
  -fdsmil-provenance-sign-key=tpm://dsmil \
  llm_worker.c \
  -o bin/llm_worker

# Output:
# [DSMIL Mission Policy] Enforcing mission profile: border_ops (Border Operations)
#   Classification: RESTRICTED
#   CT Enforcement: strict
#   Telemetry Level: minimal
# [DSMIL CT Check] Verifying constant-time enforcement...
# [DSMIL CT Check] ✓ Function 'derive_session_key' is constant-time
# [DSMIL Provenance] Generating provenance record
#   Mission Profile Hash: sha384:a1b2c3...
#   Signing with ML-DSA-87 (TPM key)
# [DSMIL Mission Policy] ✓ All functions comply with mission profile
```

**Verify:**
```bash
# Inspect compiled binary
dsmil-inspect bin/llm_worker
# Output:
#   Mission Profile: border_ops
#   Classification: RESTRICTED
#   Compiled: 2026-01-15T14:30:00Z
#   Signature: VALID (ML-DSA-87, TPM key)
#   Devices: [0, 1, 2, 3, 30, 31, 32, 33, 47, 50, 53]
#   Stages: [quantized, serve]
#   Expiration: None
#   Status: DEPLOYABLE
```

### Example 2: Cyber Defence Threat Analyzer

**Source: `threat_analyzer.c`**
```c
#include <dsmil_attributes.h>

// Cyber defence profile with AI assistance
DSMIL_MISSION_PROFILE("cyber_defence")
DSMIL_LAYER(8)
DSMIL_DEVICE(80)
DSMIL_ROE("ANALYSIS_ONLY")
int main(int argc, char **argv) {
    return analyze_threats();
}

// Threat analysis with Layer 8 Security AI
DSMIL_STAGE("serve")
DSMIL_LAYER(8)
DSMIL_DEVICE(80)
int analyze_threats(void) {
    // L8 Security AI analysis
    return 0;
}

// Network input handling
DSMIL_UNTRUSTED_INPUT
void process_network_packet(const uint8_t *packet, size_t len) {
    // Must validate before use
}
```

**Compile:**
```bash
dsmil-clang \
  -fdsmil-mission-profile=cyber_defence \
  -fdsmil-l8-security-ai=enabled \
  -fdsmil-provenance=full \
  threat_analyzer.c \
  -o bin/threat_analyzer

# Output:
# [DSMIL Mission Policy] Enforcing mission profile: cyber_defence
# [DSMIL L8 Security AI] Analyzing untrusted input flows...
# [DSMIL L8 Security AI] Found 1 untrusted input: 'process_network_packet'
# [DSMIL L8 Security AI] Risk score: 0.87 (HIGH)
# [DSMIL Provenance] Expiration: 2026-04-15T14:30:00Z (90 days)
# [DSMIL Mission Policy] ✓ All functions comply
```

### Example 3: Exercise Scenario

**Source: `exercise.c`**
```c
#include <dsmil_attributes.h>

// Exercise profile with debug support
DSMIL_MISSION_PROFILE("exercise_only")
DSMIL_LAYER(5)
int main(int argc, char **argv) {
    return run_exercise();
}

// Debug instrumentation allowed
DSMIL_STAGE("debug")
void debug_print_state(void) {
    // Debug output
}

// Production-like inference
DSMIL_STAGE("serve")
void exercise_inference(void) {
    debug_print_state();  // OK in exercise mode
}
```

**Compile:**
```bash
dsmil-clang \
  -fdsmil-mission-profile=exercise_only \
  exercise.c \
  -o bin/exercise

# Output:
# [DSMIL Mission Policy] Enforcing mission profile: exercise_only
#   Expiration: 2026-02-14T14:30:00Z (30 days)
# [DSMIL Mission Policy] ✓ All functions comply
```

## Common Workflows

### Workflow 1: Development → Exercise → Production

```bash
# Phase 1: Development (lab_research)
dsmil-clang -fdsmil-mission-profile=lab_research \
  -O0 -g src/*.c -o bin/prototype
./bin/prototype  # Full debugging, no restrictions

# Phase 2: Exercise Testing (exercise_only)
dsmil-clang -fdsmil-mission-profile=exercise_only \
  -O2 src/*.c -o bin/exercise
./bin/exercise   # 30-day expiration enforced

# Phase 3: Production (border_ops or cyber_defence)
dsmil-clang -fdsmil-mission-profile=border_ops \
  -fdsmil-provenance=full -fdsmil-provenance-sign-key=tpm://dsmil \
  -O3 src/*.c -o bin/production
dsmil-verify bin/production  # Signature verification
./bin/production  # Full security enforcement
```

### Workflow 2: CI/CD Pipeline

```yaml
# .gitlab-ci.yml example
stages:
  - build
  - test
  - deploy

build:dev:
  stage: build
  script:
    - dsmil-clang -fdsmil-mission-profile=lab_research src/*.c -o bin/dev
  artifacts:
    paths: [bin/dev]

build:exercise:
  stage: build
  script:
    - dsmil-clang -fdsmil-mission-profile=exercise_only src/*.c -o bin/exercise
  artifacts:
    paths: [bin/exercise]
    expire_in: 30 days

build:production:
  stage: build
  only: [tags]
  script:
    - dsmil-clang -fdsmil-mission-profile=border_ops \
        -fdsmil-provenance=full -fdsmil-provenance-sign-key=tpm://dsmil \
        src/*.c -o bin/production
    - dsmil-verify bin/production
  artifacts:
    paths: [bin/production]

test:exercise:
  stage: test
  script:
    - ./bin/exercise --self-test

deploy:production:
  stage: deploy
  only: [tags]
  script:
    - scp bin/production deploy-server:${DSMIL_BIN_DIR:-/opt/dsmil/bin}/
    - ssh deploy-server "dsmil-inspect \${DSMIL_BIN_DIR:-/opt/dsmil/bin}/production"
```

## Troubleshooting

### Error: Mission Profile Not Found

```
[DSMIL Mission Policy] ERROR: Profile 'cyber_defense' not found.
Available profiles: border_ops cyber_defence exercise_only lab_research
```

**Solution:** Check spelling (note: `cyber_defence` with British spelling)

### Error: Stage Not Allowed

```
ERROR: Function 'debug_func' uses stage 'debug' which is not allowed by
mission profile 'border_ops'
```

**Solution:**
- Remove `DSMIL_STAGE("debug")` or switch to `lab_research` profile
- Use `exercise_only` if debug stages are needed

### Error: Device Not Whitelisted

```
ERROR: Function 'gpu_compute' assigned to device 40 which is not
whitelisted by mission profile 'border_ops'
```

**Solution:**
- Switch to NPU (device 47) or another whitelisted device
- Use `cyber_defence` or `lab_research` profiles for unrestricted device access

### Error: Binary Expired

```
[DSMIL Runtime] ✗ BINARY EXPIRED (6 days overdue)
FATAL: Cannot execute expired cyber_defence binary
```

**Solution:**
- Recompile with current DSLLVM toolchain
- `cyber_defence` binaries expire after 90 days
- `exercise_only` binaries expire after 30 days

### Warning: Mission Profile Mismatch

```
[DSMIL Runtime] WARNING: Binary compiled with mission profile hash
sha384:OLD_HASH but current config is sha384:NEW_HASH
```

**Solution:**
- Mission profile configuration has changed since compilation
- Recompile with updated configuration
- If intentional, use `DSMIL_ALLOW_STALE_PROFILE=1` (NOT recommended for production)

## Best Practices

### 1. Always Specify Mission Profile in Source

```c
// ✓ GOOD: Explicit mission profile annotation
DSMIL_MISSION_PROFILE("border_ops")
int main() { ... }

// ✗ BAD: Relying only on compile-time flag
int main() { ... }  // No annotation
```

### 2. Validate Profile at Compile Time

```bash
# ✓ GOOD: Enforce mode (default)
dsmil-clang -fdsmil-mission-profile=border_ops src.c

# ✗ BAD: Warn mode (ignores violations)
dsmil-clang -fdsmil-mission-profile=border_ops \
  -mllvm -dsmil-mission-policy-mode=warn src.c
```

### 3. Use TPM Signing for Production

```bash
# ✓ GOOD: Hardware-backed signing
dsmil-clang -fdsmil-mission-profile=border_ops \
  -fdsmil-provenance-sign-key=tpm://dsmil src.c

# ✗ BAD: Software signing for production profiles
dsmil-clang -fdsmil-mission-profile=border_ops \
  -fdsmil-provenance-sign-key=file://key.pem src.c
```

### 4. Verify Binaries Before Deployment

```bash
# Always verify signature and provenance
dsmil-verify bin/production
dsmil-inspect bin/production

# Check expiration
dsmil-inspect bin/cyber_defence_tool | grep Expiration
```

### 5. Document Profile Selection

```c
/**
 * LLM Inference Worker
 *
 * Mission Profile: border_ops
 * Rationale: Deployed in hostile environment with no external network access
 * Security: RESTRICTED classification, minimal telemetry
 * Deployment: Air-gapped systems at border stations
 */
DSMIL_MISSION_PROFILE("border_ops")
int main() { ... }
```

### 6. Use Appropriate Profile for Development Phase

```
Development Phase    →  Mission Profile
─────────────────────────────────────────
Prototyping         →  lab_research
Feature Development →  lab_research
Integration Testing →  exercise_only
Security Testing    →  exercise_only
Staging             →  cyber_defence (short expiration)
Production          →  border_ops or cyber_defence
```

### 7. Rotate Cyber Defence Binaries

```bash
# Set up automatic recompilation for cyber_defence
# (90-day expiration enforces this)
0 0 * * 0 ${DSMIL_PREFIX:-/opt/dsmil}/scripts/rebuild-cyber-defence.sh
```

### 8. Archive Provenance Records

```bash
# Extract and archive provenance for forensics
dsmil-extract-provenance bin/production > provenance-$(date +%s).json
# Store in forensics database (Layer 62)
```

## References

- **Mission Profiles Configuration:** `dsmil/config/mission-profiles.json`
- **Attributes Header:** `dsmil/include/dsmil_attributes.h`
- **Mission Policy Pass:** `dsmil/lib/Passes/DsmilMissionPolicyPass.cpp`
- **Provenance Integration:** `dsmil/docs/MISSION-PROFILE-PROVENANCE.md`
- **DSLLVM Roadmap:** `dsmil/docs/DSLLVM-ROADMAP.md`

## Support

For questions or issues:
- Documentation: https://dsmil.org/docs/mission-profiles
- Issues: https://github.com/dsllvm/dsllvm/issues
- Mailing List: dsllvm-users@lists.llvm.org

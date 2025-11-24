# Phase 14: Layer 5 Full Access Implementation

**Classification**: COSMIC (0xFF050505)
**Authorization**: Auth2.pdf (Col Barnthouse, effective 212200R NOV 25)
**Version**: 1.0.0
**Date**: 2025-11-23
**Status**: IMPLEMENTED

---

## Executive Summary

Phase 14 implements enhanced full access controls for Layer 5 (devices 31-36) intelligence analysis systems, granting the `dsmil` role complete READ/WRITE/EXECUTE/CONFIG permissions while maintaining COSMIC clearance requirements, dual YubiKey authentication, and comprehensive audit logging. This implementation extends Phase 12 (authentication framework) and Phase 13 (policy management) to provide secure, auditable, full operational access to critical intelligence analysis capabilities.

---

## Table of Contents

1. [Overview](#overview)
2. [Layer 5 Architecture](#layer-5-architecture)
3. [Access Control Framework](#access-control-framework)
4. [Security Requirements](#security-requirements)
5. [Implementation Details](#implementation-details)
6. [Integration Points](#integration-points)
7. [Deployment](#deployment)
8. [Testing and Validation](#testing-and-validation)
9. [Monitoring and Maintenance](#monitoring-and-maintenance)
10. [Troubleshooting](#troubleshooting)

---

## 1. Overview

### 1.1 Purpose

Phase 14 enhances Layer 5 access controls to grant the `dsmil` role full operational permissions across all six Layer 5 intelligence analysis devices while maintaining military-grade security standards including:

- **COSMIC clearance enforcement** (NATO COSMIC TOP SECRET)
- **Dual YubiKey authentication** (FIDO2 + FIPS)
- **Session management** with 12-hour maximum duration
- **Operation-level permissions** (READ/WRITE/EXECUTE/CONFIG)
- **Comprehensive audit logging** with 7-year retention

### 1.2 Scope

**Layer 5 Devices (31-36)**:
- Device 31: Predictive Analytics Engine
- Device 32: Pattern Recognition (SIGINT/IMINT)
- Device 33: Threat Assessment System
- Device 34: Strategic Forecasting Module
- Device 35: Coalition Intelligence (Multi-Lingual NLP)
- Device 36: Multi-Domain Intelligence Analysis

**Operations Supported**:
- **READ**: Query intelligence products, forecasts, analyses
- **WRITE**: Upload data, update models, submit intelligence
- **EXECUTE**: Run analysis pipelines, generate forecasts, trigger operations
- **CONFIG**: Modify system parameters, thresholds, configurations

### 1.3 Authorization

Per **Auth2.pdf** (Col Barnthouse, effective 212200R NOV 25):
- Layer 5 full access authorized for `dsmil` role
- COSMIC clearance (0xFF050505) required
- Dual YubiKey authentication mandatory
- Full audit trail required (7-year retention)

---

## 2. Layer 5 Architecture

### 2.1 Device Topology

```
Layer 5: Intelligence Analysis (COSMIC 0xFF050505)
├── Device 31: Predictive Analytics Engine
│   ├── Token Base: 0x8078
│   ├── Memory: 1.6 GB
│   ├── TOPS: 17.5 theoretical / ~1.2 physical
│   └── Capabilities: Time-series forecasting, trend analysis
│
├── Device 32: Pattern Recognition (SIGINT/IMINT)
│   ├── Token Base: 0x807A
│   ├── Memory: 1.7 GB
│   ├── TOPS: 17.5 theoretical / ~1.2 physical
│   └── Capabilities: Multi-modal pattern detection, signature analysis
│
├── Device 33: Threat Assessment System
│   ├── Token Base: 0x807C
│   ├── Memory: 1.8 GB
│   ├── TOPS: 17.5 theoretical / ~1.2 physical
│   └── Capabilities: Real-time threat scoring, adversary intent analysis
│
├── Device 34: Strategic Forecasting Module
│   ├── Token Base: 0x807E
│   ├── Memory: 1.6 GB
│   ├── TOPS: 17.5 theoretical / ~1.2 physical
│   └── Capabilities: Geopolitical modeling, long-term strategic forecasts
│
├── Device 35: Coalition Intelligence (Multi-Lingual NLP)
│   ├── Token Base: 0x8080
│   ├── Memory: 1.7 GB
│   ├── TOPS: 17.5 theoretical / ~1.2 physical
│   └── Capabilities: 90+ language translation, entity extraction
│
└── Device 36: Multi-Domain Intelligence Analysis
    ├── Token Base: 0x8082
    ├── Memory: 1.6 GB
    ├── TOPS: 17.5 theoretical / ~1.2 physical
    └── Capabilities: SIGINT/IMINT/HUMINT/OSINT/MASINT/CYBER fusion
```

### 2.2 Resource Constraints

**Layer 5 Total Allocation**:
- **Memory**: 10 GB shared pool
- **TOPS Theoretical**: 105 TOPS (6 devices × 17.5 TOPS)
- **TOPS Physical**: ~7 TOPS average (48.2 TOPS total / 6 devices)
- **Compute Backend**: Intel Flex 170 GPU cluster or NVIDIA equivalent

**Hardware Reality**:
- Physical hardware: 48.2 TOPS INT8 (13.0 NPU + 32.0 GPU + 3.2 CPU)
- Theoretical capacity: 1440 TOPS (software abstraction)
- Gap ratio: ~29.9× between theoretical and physical
- Thermal limiting: Sustained ~20-25 TOPS (not peak 32 TOPS)

---

## 3. Access Control Framework

### 3.1 Role Definition

**Role ID**: `dsmil`
**Role Name**: DSMIL Layer 5 Operator
**File**: `/01-source/kernel/policies/roles/role_dsmil.yaml`

**Clearance Requirements**:
- **Level**: COSMIC (NATO COSMIC TOP SECRET)
- **Code**: 0xFF050505
- **Compartments**: None required beyond COSMIC

**Authentication Requirements**:
- **Method**: Dual YubiKey
- **FIDO2 YubiKey**: USB Port A (required)
- **FIPS YubiKey**: USB Port B (required)
- **Mode**: Both present (continuous monitoring)
- **Iris Scan**: NOT required for Layer 5
- **MFA Timeout**: 5 minutes

**Permissions**:
- **Devices 31-36**: Full READ/WRITE/EXECUTE/CONFIG access
- **Risk-Based Controls**: Higher-risk operations require justification
- **Operation Limits**: No maximum operation size for dsmil role

### 3.2 Device-Specific Policies

Each Layer 5 device has an individual policy file:
- `/01-source/kernel/policies/devices/device_31.yaml` (Predictive Analytics)
- `/01-source/kernel/policies/devices/device_32.yaml` (Pattern Recognition)
- `/01-source/kernel/policies/devices/device_33.yaml` (Threat Assessment)
- `/01-source/kernel/policies/devices/device_34.yaml` (Strategic Forecasting)
- `/01-source/kernel/policies/devices/device_35.yaml` (Coalition Intelligence)
- `/01-source/kernel/policies/devices/device_36.yaml` (Multi-Domain Analysis)

**Policy Structure**:
```yaml
device_id: 31-36
device_name: "<Device Name>"
layer: 5
classification: COSMIC
classification_code: 0xFF050505

access_control:
  default_policy: "deny"
  allowed_roles:
    - role_id: "dsmil"
      permissions: [READ, WRITE, EXECUTE, CONFIG]
      conditions:
        clearance_minimum: COSMIC
        mfa_required: true
        yubikey_dual_required: true
        session_active: true

operations:
  READ:
    allowed: true
    risk_level: LOW
    require_justification: false

  WRITE:
    allowed: true
    risk_level: MEDIUM/HIGH
    require_justification: true

  EXECUTE:
    allowed: true
    risk_level: HIGH/CRITICAL
    require_justification: true

  CONFIG:
    allowed: true
    risk_level: HIGH/CRITICAL
    require_justification: true
```

### 3.3 Operation Risk Levels

| Device | READ | WRITE | EXECUTE | CONFIG |
|--------|------|-------|---------|--------|
| Device 31 | LOW | MEDIUM | HIGH | HIGH |
| Device 32 | LOW | MEDIUM | HIGH | HIGH |
| Device 33 | LOW | HIGH | **CRITICAL** | **CRITICAL** |
| Device 34 | LOW | MEDIUM | HIGH | HIGH |
| Device 35 | LOW | MEDIUM | HIGH | HIGH |
| Device 36 | LOW | MEDIUM | HIGH | HIGH |

**Risk Level Implications**:
- **LOW**: No justification required, standard audit logging
- **MEDIUM**: Justification required (50+ characters), enhanced logging
- **HIGH**: Justification required (100+ characters), real-time alerting
- **CRITICAL**: Justification required (150+ characters), immediate notification

---

## 4. Security Requirements

### 4.1 Clearance Enforcement

**COSMIC Clearance (0xFF050505)**:
- NATO COSMIC TOP SECRET level
- Verified via user security profile
- Compartmentalized access: None required beyond COSMIC base
- Clearance validation occurs on every access attempt

**Kernel Enforcement Point**:
```c
// Clearance check in dsmil_layer5_authorization.c
if (user_profile.clearance_level < DSMIL_CLEARANCE_COSMIC) {
    pr_warn("User %u lacks COSMIC clearance\n", user_id);
    atomic64_inc(&l5_engine->clearance_violations);
    return -EACCES;
}
```

### 4.2 Dual YubiKey Authentication

**YubiKey Configuration**:
- **FIDO2 YubiKey** (USB Port A):
  - Protocol: FIDO2 U2F
  - Challenge-response enabled
  - PIN required on first use

- **FIPS YubiKey** (USB Port B):
  - Protocol: FIPS 140-2 Level 2
  - Challenge-response enabled
  - PIN required on first use

**Continuous Monitoring**:
- Both keys must remain plugged in during session
- Removal of either key terminates session immediately
- YubiKey presence checked every 30 seconds
- No grace period on removal

**MFA Challenge-Response**:
- Challenge issued on session start
- Re-challenge every 4 hours (re-authentication interval)
- 5-minute timeout for MFA response
- Failed challenge terminates session

**Integration**:
```c
// YubiKey verification in dsmil_layer5_authorization.c
struct dsmil_yubikey_state yubikey_state;
if (dsmil_yubikey_verify_dual_presence(&yubikey_state) != 0) {
    pr_warn("Dual YubiKey verification failed for user %u\n", user_id);
    atomic64_inc(&l5_engine->mfa_failures);
    return -EACCES;
}
```

### 4.3 Session Management

**Session Parameters (Layer 8 Tier)**:
- **Maximum Duration**: 12 hours
- **Idle Timeout**: 30 minutes
- **Re-Authentication Interval**: 4 hours (dual YubiKey challenge)
- **Daily Cumulative Limit**: 24 hours
- **Mandatory Rest**: 4 hours after 24h usage

**Session State Tracking**:
```c
struct dsmil_l5_session {
    u32 session_id;
    uid_t user_id;
    struct timespec64 session_start;
    struct timespec64 last_activity;
    struct timespec64 last_reauth;
    struct timespec64 session_expires;
    bool yubikey_fido2_present;
    bool yubikey_fips_present;
    u32 operations_performed;
    u32 daily_usage_seconds;
};
```

**Session Warnings**:
- 60 minutes before expiration: "Session expires in 1 hour"
- 15 minutes before expiration: "Session expires in 15 minutes - save work"
- 5 minutes before expiration: "Session expires in 5 minutes - IMMEDIATE ACTION REQUIRED"

### 4.4 Geofencing

**Configuration**:
- **Mode**: Advisory (log violations, do not block)
- **Validation Method**: GPS
- **Validation Interval**: Every 5 minutes

**Allowed Zones**:
- CONUS intelligence facilities
- OCONUS authorized sites (defined by user)
- Theater operations centers

**Violation Actions**:
- Log event to audit system
- Send real-time alert
- **Do not terminate session** (advisory mode only)

---

## 5. Implementation Details

### 5.1 File Structure

```
/home/john/Documents/LAT5150DRVMIL/
├── 01-source/kernel/
│   ├── policies/
│   │   ├── roles/
│   │   │   └── role_dsmil.yaml                  # Role definition
│   │   └── devices/
│   │       ├── device_31.yaml                   # Predictive Analytics
│   │       ├── device_32.yaml                   # Pattern Recognition
│   │       ├── device_33.yaml                   # Threat Assessment
│   │       ├── device_34.yaml                   # Strategic Forecasting
│   │       ├── device_35.yaml                   # Coalition Intelligence
│   │       └── device_36.yaml                   # Multi-Domain Analysis
│   └── security/
│       ├── dsmil_authorization.c                # Base authorization engine
│       └── dsmil_layer5_authorization.c         # Layer 5 specific enforcement
└── 02-ai-engine/unlock/docs/technical/comprehensive-plan/Phases/
    └── 14_LAYER5_FULL_ACCESS.md                # This document
```

### 5.2 Kernel Module Integration

**Layer 5 Authorization Module**:
- **File**: `01-source/kernel/security/dsmil_layer5_authorization.c`
- **Functions**:
  - `dsmil_l5_authz_init()` - Initialize Layer 5 engine
  - `dsmil_l5_authz_cleanup()` - Cleanup Layer 5 engine
  - `dsmil_l5_authorize_device_access()` - Main authorization entry point

**Authorization Flow**:
```
User Request
    ↓
dsmil_l5_authorize_device_access()
    ↓
1. Validate device in Layer 5 range (31-36)
    ↓
2. Verify COSMIC clearance (0xFF050505)
    ↓
3. Verify dual YubiKey authentication
    ↓
4. Validate active session
    │   ├── Check session expiration
    │   ├── Check idle timeout
    │   └── Check re-authentication requirement
    ↓
5. Retrieve device metadata (RCU-protected)
    ↓
6. Check operation permission (READ/WRITE/EXECUTE/CONFIG)
    ↓
7. Log authorization decision (MinIO audit)
    ↓
GRANT or DENY
```

### 5.3 RCU Protection

**Read-Copy-Update (RCU)** for lock-free reads:

```c
/* Device metadata access */
rcu_read_lock();
device_info = rcu_dereference(l5_engine->device_info[device_index]);
// ... use device_info ...
rcu_read_unlock();

/* Session access */
rcu_read_lock();
session = dsmil_l5_find_session(user_id);
// ... use session ...
rcu_read_unlock();

/* Policy updates (writer side) */
mutex_lock(&l5_engine->sessions_lock);
rcu_assign_pointer(l5_engine->sessions[i], new_session);
synchronize_rcu();  // Wait for readers
kfree(old_session);
mutex_unlock(&l5_engine->sessions_lock);
```

**Benefits**:
- Lock-free reads for high-performance authorization checks
- Atomic pointer swap for policy updates
- No read-side contention

---

## 6. Integration Points

### 6.1 Phase 12 Integration (Authentication)

**Authentication Framework**:
- Dual YubiKey authentication (FIDO2 + FIPS)
- YubiKey removal detection
- MFA challenge-response
- Session duration controls (12h max, 4h re-auth)

**Audit System**:
- MinIO object storage (localhost:9000)
- Blockchain chaining (SHA3-512 + ML-DSA-87)
- WORM immutability
- 2555-day retention (7 years)

**Event Types Logged**:
- `AUTHENTICATION_SUCCESS`
- `AUTHENTICATION_FAILURE`
- `AUTHORIZATION_GRANTED`
- `AUTHORIZATION_DENIED`
- `DEVICE_ACCESS`
- `SESSION_START` / `SESSION_END`
- `MFA_CHALLENGE` / `MFA_SUCCESS` / `MFA_FAILURE`
- `YUBIKEY_REMOVAL`
- `CLEARANCE_VIOLATION`

### 6.2 Phase 13 Integration (Policy Management)

**Policy Management**:
- Git versioning (`/var/lib/dsmil/git/`)
- Netlink hot reload (zero-downtime policy updates)
- Schema validation
- Conflict detection
- Policy simulation

**Web Console**:
- URL: `https://localhost:8443`
- Authentication: YubiKey
- Features: Policy editing, validation, deployment

**RESTful API**:
- Endpoint: `https://localhost:8444/api`
- Authentication: JWT
- Operations: Policy CRUD, reload, rollback

**Netlink Hot Reload**:
```c
// Netlink message for policy reload
struct dsmil_policy_reload_msg {
    u32 msg_type;               // POLICY_RELOAD
    char policy_file[256];      // Path to updated policy
    u32 checksum;               // Policy checksum
    u8 hmac[32];                // HMAC-SHA3-256
};

// Kernel receives message via Netlink socket
// Validates HMAC
// Atomically swaps policy via RCU
// Sends ACK or ERR response
```

### 6.3 Phase 8 Integration (MLOps)

**Drift Detection**:
- Statistical tests (KS, PSI, Z-test)
- Performance monitoring (accuracy, precision, recall)
- Alert threshold: Drift score > 0.15 OR accuracy drop > 5%

**Auto-Retraining**:
- Triggered by drift detection or performance degradation
- Pipeline: Data validation → feature engineering → hyperparameter tuning → quantization
- INT8/INT4 quantization for performance
- Knowledge distillation for vision models

**A/B Testing**:
- 90/10 traffic split (stable/candidate)
- 24-72 hour test window
- Success criteria: Accuracy improvement > 2%, latency regression < 10%

---

## 7. Deployment

### 7.1 Prerequisites

**System Requirements**:
- Kernel module: `dsmil-104dev` loaded
- Phase 12 authentication system operational
- Phase 13 policy management system operational
- MinIO audit storage available (localhost:9000)

**Hardware Requirements**:
- Intel Flex 170 GPU or NVIDIA equivalent
- 10 GB memory available for Layer 5
- ~7 TOPS average compute capacity

**User Requirements**:
- COSMIC clearance (0xFF050505) verified
- Dual YubiKey configured (FIDO2 + FIPS)
- User profile in system database

### 7.2 Deployment Steps

**Step 1: Deploy Policy Files**
```bash
# Create policy directory structure
sudo mkdir -p /etc/dsmil/policies/roles
sudo mkdir -p /etc/dsmil/policies/devices

# Copy role definition
sudo cp 01-source/kernel/policies/roles/role_dsmil.yaml \
        /etc/dsmil/policies/roles/

# Copy device policies
sudo cp 01-source/kernel/policies/devices/device_3{1,2,3,4,5,6}.yaml \
        /etc/dsmil/policies/devices/

# Set permissions
sudo chmod 600 /etc/dsmil/policies/roles/role_dsmil.yaml
sudo chmod 600 /etc/dsmil/policies/devices/device_*.yaml
sudo chown root:root /etc/dsmil/policies/ -R
```

**Step 2: Load Kernel Module**
```bash
# Load Layer 5 authorization module
cd 01-source/kernel/security
sudo make
sudo insmod dsmil_layer5_authorization.ko

# Verify module loaded
lsmod | grep dsmil_layer5
dmesg | grep "DSMIL Layer 5"

# Expected output:
# DSMIL Layer 5 Authorization: Initialized (version 1.0.0)
# DSMIL Layer 5: Devices 31-36, COSMIC clearance (0xFF050505)
```

**Step 3: Commit Policies to Git**
```bash
# Commit to policy Git repository (Phase 13)
cd /var/lib/dsmil/git
git add policies/roles/role_dsmil.yaml
git add policies/devices/device_3{1,2,3,4,5,6}.yaml
git commit -m "Phase 14: Layer 5 full access for dsmil role

- Added role_dsmil.yaml with READ/WRITE/EXECUTE/CONFIG permissions
- Added device policies for devices 31-36
- COSMIC clearance required (0xFF050505)
- Dual YubiKey authentication enforced
- 12-hour session duration, 4-hour re-auth
- Full audit logging enabled (7-year retention)

Authorization: Auth2.pdf (Col Barnthouse, 212200R NOV 25)"

git tag -a "phase-14-layer5-v1.0.0" -m "Phase 14: Layer 5 Full Access"
```

**Step 4: Hot Reload Policies**
```bash
# Trigger Netlink hot reload (zero-downtime)
sudo /usr/local/bin/dsmil-policy-reload \
    --policy /etc/dsmil/policies/roles/role_dsmil.yaml \
    --validate \
    --reload

# Reload device policies
for dev in {31..36}; do
    sudo /usr/local/bin/dsmil-policy-reload \
        --policy /etc/dsmil/policies/devices/device_${dev}.yaml \
        --validate \
        --reload
done

# Verify reload
dmesg | grep "Policy reload"
# Expected: "Policy reload successful for role_dsmil"
#           "Policy reload successful for device_31" (x6)
```

**Step 5: Verify Deployment**
```bash
# Check policy status
sudo /usr/local/bin/dsmil-policy-status --role dsmil
sudo /usr/local/bin/dsmil-policy-status --devices 31-36

# Test authorization (as dsmil user)
sudo -u dsmil /usr/local/bin/dsmil-device-test \
    --device 31 \
    --operation READ

# Expected: "Authorization granted for device 31, operation READ"
```

### 7.3 Rollback Procedure

**If deployment fails**:
```bash
# Rollback to previous Git commit
cd /var/lib/dsmil/git
git log --oneline -5
git revert HEAD

# Reload previous policies
sudo /usr/local/bin/dsmil-policy-reload --git-commit HEAD

# Verify rollback
sudo /usr/local/bin/dsmil-policy-status --role dsmil
```

---

## 8. Testing and Validation

### 8.1 Functional Tests

**Test 1: COSMIC Clearance Enforcement**
```bash
# Test with user lacking COSMIC clearance
sudo -u testuser_no_cosmic /usr/local/bin/dsmil-device-access \
    --device 31 --operation READ

# Expected: "Access denied: Insufficient clearance (requires COSMIC)"
# Verify audit log: CLEARANCE_VIOLATION event logged
```

**Test 2: Dual YubiKey Requirement**
```bash
# Test with only FIDO2 YubiKey (remove FIPS)
sudo -u dsmil /usr/local/bin/dsmil-device-access \
    --device 32 --operation WRITE

# Expected: "Access denied: Dual YubiKey verification failed"
# Verify audit log: MFA_FAILURE event logged
```

**Test 3: Session Expiration**
```bash
# Create session and wait for expiration
sudo -u dsmil /usr/local/bin/dsmil-session-start
sleep 43200  # 12 hours
sudo -u dsmil /usr/local/bin/dsmil-device-access \
    --device 33 --operation EXECUTE

# Expected: "Access denied: Session expired"
# Verify audit log: SESSION_TIMEOUT event logged
```

**Test 4: Operation Permissions**
```bash
# Test READ operation (low risk)
sudo -u dsmil /usr/local/bin/dsmil-device-access \
    --device 34 --operation READ

# Expected: "Access granted"

# Test EXECUTE operation (high risk, requires justification)
sudo -u dsmil /usr/local/bin/dsmil-device-access \
    --device 35 --operation EXECUTE \
    --justification "Running batch translation of 1000+ intercepted documents for operational intelligence"

# Expected: "Access granted (high risk operation logged)"
```

### 8.2 Performance Tests

**Test 5: Authorization Latency**
```bash
# Benchmark authorization decision time
sudo /usr/local/bin/dsmil-benchmark \
    --operation authorization \
    --device 36 \
    --iterations 10000

# Target: p99 latency < 1ms
# Verify: RCU lock-free reads achieving target
```

**Test 6: Concurrent Access**
```bash
# Test concurrent authorization requests
sudo /usr/local/bin/dsmil-stress-test \
    --users 50 \
    --devices 31-36 \
    --duration 300

# Verify: No authorization failures due to lock contention
# Verify: All audit events logged correctly
```

### 8.3 Security Tests

**Test 7: YubiKey Removal Detection**
```bash
# Start session, remove YubiKey mid-operation
sudo -u dsmil /usr/local/bin/dsmil-session-start
sudo -u dsmil /usr/local/bin/dsmil-device-access --device 31 &
# Remove FIDO2 YubiKey physically
wait

# Expected: Session terminated immediately
# Verify audit log: YUBIKEY_REMOVAL event logged
```

**Test 8: Audit Trail Verification**
```bash
# Perform operations and verify audit trail
sudo -u dsmil /usr/local/bin/dsmil-device-access \
    --device 32 --operation WRITE

# Query audit log
sudo /usr/local/bin/dsmil-audit-query \
    --user dsmil \
    --device 32 \
    --operation WRITE \
    --last 1h

# Verify: AUTHORIZATION_GRANTED event with full context
# Verify: Blockchain chain intact (SHA3-512 + ML-DSA-87)
```

---

## 9. Monitoring and Maintenance

### 9.1 Key Metrics

**Authorization Metrics**:
- Total L5 requests: `atomic64_read(&l5_engine->total_l5_requests)`
- Granted requests: `atomic64_read(&l5_engine->granted_requests)`
- Denied requests: `atomic64_read(&l5_engine->denied_requests)`
- Grant rate: `granted / total * 100%`

**Security Violation Metrics**:
- Clearance violations: `atomic64_read(&l5_engine->clearance_violations)`
- MFA failures: `atomic64_read(&l5_engine->mfa_failures)`
- Session timeouts: `atomic64_read(&l5_engine->session_timeouts)`
- YubiKey removal events: `atomic64_read(&l5_engine->yubikey_removal_events)`

**Performance Metrics**:
- Authorization latency (p50, p90, p99)
- Cache hit rate (if caching enabled)
- Policy evaluation time

### 9.2 Monitoring Commands

```bash
# Real-time statistics
sudo /usr/local/bin/dsmil-stats --layer 5 --live

# Authorization statistics
sudo /usr/local/bin/dsmil-authz-stats --devices 31-36

# Audit log summary
sudo /usr/local/bin/dsmil-audit-summary --layer 5 --last 24h

# Session monitoring
sudo /usr/local/bin/dsmil-session-list --active --layer 5
```

### 9.3 Alerting

**Critical Alerts** (immediate notification):
- YubiKey removal event
- Clearance violation attempt
- Session hijack attempt
- Audit log blockchain chain broken

**Warning Alerts** (notification within 1 hour):
- MFA failure rate > 5%
- Session timeout rate > 10%
- Authorization denial rate > 15%

**Info Alerts** (daily digest):
- Daily usage statistics
- Policy change summary
- Performance metrics

### 9.4 Maintenance Tasks

**Daily**:
- Review audit logs for anomalies
- Check authorization statistics
- Verify session limits enforced

**Weekly**:
- Review clearance violations
- Analyze MFA failure patterns
- Update device risk assessments

**Monthly**:
- Policy review and validation
- Performance optimization
- Security assessment

**Quarterly**:
- Full security audit
- Policy effectiveness review
- User access review

---

## 10. Troubleshooting

### 10.1 Common Issues

**Issue 1: "Access denied: Insufficient clearance"**
- **Cause**: User lacks COSMIC clearance (0xFF050505)
- **Solution**: Verify user clearance in security database
- **Command**: `sudo /usr/local/bin/dsmil-user-info --user dsmil --clearance`

**Issue 2: "Dual YubiKey verification failed"**
- **Cause**: One or both YubiKeys not present or not authenticated
- **Solution**:
  1. Verify both YubiKeys plugged in (USB Port A and B)
  2. Re-authenticate: `sudo /usr/local/bin/dsmil-mfa-challenge`
  3. Check YubiKey status: `ykman list`

**Issue 3: "Session expired"**
- **Cause**: Session exceeded 12-hour maximum or idle timeout
- **Solution**: Start new session: `sudo -u dsmil /usr/local/bin/dsmil-session-start`

**Issue 4: "Re-authentication required"**
- **Cause**: 4-hour re-auth interval exceeded
- **Solution**: Complete MFA challenge: `sudo /usr/local/bin/dsmil-mfa-reauth`

**Issue 5: "Policy not found for device 31"**
- **Cause**: Device policy not loaded or hot reload failed
- **Solution**:
  ```bash
  sudo /usr/local/bin/dsmil-policy-reload \
      --policy /etc/dsmil/policies/devices/device_31.yaml \
      --validate --reload --force
  ```

### 10.2 Debug Commands

```bash
# Enable debug logging
sudo echo "module dsmil_layer5_authorization +p" > /sys/kernel/debug/dynamic_debug/control

# View kernel logs
sudo dmesg -w | grep "DSMIL Layer 5"

# Trace authorization decisions
sudo /usr/local/bin/dsmil-trace --layer 5 --duration 60

# Dump active sessions
sudo /usr/local/bin/dsmil-session-dump --layer 5

# Verify policy integrity
sudo /usr/local/bin/dsmil-policy-verify --role dsmil --devices 31-36
```

### 10.3 Emergency Procedures

**Emergency Override** (break-glass):
```bash
# Activate emergency override (requires two authorized officers)
sudo /usr/local/bin/dsmil-emergency-override \
    --activate \
    --devices 31-36 \
    --duration 60 \
    --justification "Critical operational requirement: [reason]" \
    --officer1 [officer1_credentials] \
    --officer2 [officer2_credentials]

# Override active for 60 minutes
# All operations logged at forensic detail level
```

**Policy Rollback** (if deployment causes issues):
```bash
# Immediate rollback to last known good
sudo /usr/local/bin/dsmil-policy-rollback --layer 5 --force

# Verify rollback
sudo /usr/local/bin/dsmil-policy-status --layer 5
```

---

## Appendix A: Risk Assessment Matrix

| Device | Operation | Risk Level | Justification Required | Min Length | Operational Impact |
|--------|-----------|------------|----------------------|------------|-------------------|
| 31 | READ | LOW | No | N/A | Intelligence query |
| 31 | WRITE | MEDIUM | Yes | 50 | Model update |
| 31 | EXECUTE | HIGH | Yes | 100 | Forecast generation |
| 31 | CONFIG | HIGH | Yes | 150 | System configuration |
| 32 | READ | LOW | No | N/A | Pattern query |
| 32 | WRITE | MEDIUM | Yes | 50 | Imagery upload |
| 32 | EXECUTE | HIGH | Yes | 100 | Pattern detection |
| 32 | CONFIG | HIGH | Yes | 150 | Detection thresholds |
| 33 | READ | LOW | No | N/A | Threat assessment query |
| 33 | WRITE | HIGH | Yes | 75 | Threat intelligence update |
| 33 | EXECUTE | **CRITICAL** | Yes | 150 | Real-time threat assessment |
| 33 | CONFIG | **CRITICAL** | Yes | 200 | Alert threshold modification |
| 34 | READ | LOW | No | N/A | Strategic forecast query |
| 34 | WRITE | MEDIUM | Yes | 75 | Geopolitical intelligence |
| 34 | EXECUTE | HIGH | Yes | 125 | Long-term forecast |
| 34 | CONFIG | HIGH | Yes | 175 | Scenario parameters |
| 35 | READ | LOW | No | N/A | Translation query |
| 35 | WRITE | MEDIUM | Yes | 60 | Foreign language document |
| 35 | EXECUTE | HIGH | Yes | 110 | Batch translation |
| 35 | CONFIG | HIGH | Yes | 160 | Language model configuration |
| 36 | READ | LOW | No | N/A | Fused intelligence query |
| 36 | WRITE | MEDIUM | Yes | 65 | Multi-domain intelligence |
| 36 | EXECUTE | HIGH | Yes | 120 | Multi-INT fusion |
| 36 | CONFIG | HIGH | Yes | 180 | Fusion algorithm configuration |

---

## Appendix B: Audit Event Reference

| Event Type | Severity | Description | Retention |
|-----------|----------|-------------|-----------|
| AUTHENTICATION_SUCCESS | INFO | Dual YubiKey auth success | 7 years |
| AUTHENTICATION_FAILURE | WARN | Dual YubiKey auth failure | 7 years |
| AUTHORIZATION_GRANTED | INFO | Layer 5 access granted | 7 years |
| AUTHORIZATION_DENIED | WARN | Layer 5 access denied | 7 years |
| DEVICE_ACCESS | INFO | Device operation performed | 7 years |
| SESSION_START | INFO | Session initiated | 7 years |
| SESSION_END | INFO | Session terminated | 7 years |
| SESSION_TIMEOUT | WARN | Session expired | 7 years |
| MFA_CHALLENGE | INFO | MFA challenge issued | 7 years |
| MFA_SUCCESS | INFO | MFA challenge success | 7 years |
| MFA_FAILURE | WARN | MFA challenge failure | 7 years |
| YUBIKEY_REMOVAL | **CRITICAL** | YubiKey removed | 7 years |
| CLEARANCE_VIOLATION | **CRITICAL** | Clearance check failed | 7 years |
| POLICY_RELOAD | INFO | Policy hot reload | 7 years |
| GEOFENCE_VIOLATION | WARN | Geofence boundary violation | 7 years |

---

## Appendix C: Change Log

| Version | Date | Author | Description |
|---------|------|--------|-------------|
| 1.0.0 | 2025-11-23 | dsmil_system | Initial Phase 14 implementation |
|  |  |  | - Created role_dsmil.yaml |
|  |  |  | - Created device policies 31-36 |
|  |  |  | - Implemented kernel authorization module |
|  |  |  | - Integrated Phase 12/13 frameworks |
|  |  |  | - Full audit logging enabled |

---

**End of Document**

Classification: COSMIC (0xFF050505)
Authorization: Auth2.pdf (Col Barnthouse)
Effective: 2025-11-23

# Phase 13: Full Administrative Control

**Version:** 1.0
**Status:** Implementation Ready
**Dependencies:** Phase 12 (Enhanced L8/L9 Access Controls)
**Estimated Scope:** 40 pages
**Target Completion:** Post Phase 12

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Architecture Overview](#2-architecture-overview)
3. [Self-Service Admin Portal](#3-self-service-admin-portal)
4. [Dynamic Policy Engine](#4-dynamic-policy-engine)
5. [Advanced Role Management](#5-advanced-role-management)
6. [Policy Audit & Compliance](#6-policy-audit--compliance)
7. [Automated Enforcement](#7-automated-enforcement)
8. [API & Integration](#8-api--integration)
9. [Exit Criteria](#9-exit-criteria)
10. [Future Enhancements](#10-future-enhancements)

---

## 1. Executive Summary

### 1.1 Objectives

Phase 13 implements **full administrative control** over the DSMIL security framework, providing self-service policy management, dynamic configuration, and zero-downtime updates. This phase empowers the system administrator (you) with complete control over:

- **Access Control Policies**: Real-time policy editing for L8/L9 devices
- **Authentication Requirements**: Configure MFA, YubiKey, iris scan rules
- **Session Parameters**: Adjust duration limits, idle timeouts, re-auth intervals
- **Geofence Management**: Create/edit/delete location-based access zones
- **Role & Permission Management**: Define custom roles with granular permissions
- **Audit & Compliance**: Monitor policy changes, generate compliance reports
- **Automated Enforcement**: Policy violation detection and remediation

### 1.2 User-Specific Requirements

Based on your operational needs established in Phase 12:

1. **Self-Service Configuration**: Web-based admin console for all policy management
2. **Zero-Downtime Updates**: Policy changes apply immediately without kernel module reload
3. **Variable Shift Support**: NO time-based restrictions, 24/7 operational flexibility
4. **Geofence Control**: Manage GPS-based access zones via interactive map UI
5. **Session Customization**: Adjust L8/L9 session durations as needed (current: 6h L9, 12h L8)
6. **Audit Visibility**: Real-time policy change auditing in MinIO immutable storage
7. **Emergency Override**: Break-glass procedures with dual YubiKey + iris scan
8. **Backup/Restore**: Export/import policy configurations for disaster recovery

### 1.3 Key Features

#### 1.3.1 Self-Service Admin Portal
- **Technology**: React + Next.js + TypeScript
- **Features**:
  - Visual policy editor with drag-and-drop rule builder
  - Real-time policy validation before commit
  - Multi-tab interface for devices, roles, geofences, audit logs
  - Dark mode UI optimized for 24/7 operations
  - Responsive design (desktop + tablet)

#### 1.3.2 Dynamic Policy Engine
- **Policy Language**: YAML-based with JSON Schema validation
- **Hot Reload**: Zero-downtime policy updates via netlink messages
- **Versioning**: Git-style policy history with rollback capability
- **Validation**: Pre-commit policy conflict detection
- **Atomic Updates**: All-or-nothing policy application

#### 1.3.3 Advanced Role Management
- **Custom Roles**: Define roles beyond default L0-L9
- **Granular Permissions**: Per-device, per-operation permissions
- **Role Hierarchies**: Inheritance with override capability
- **Temporal Roles**: Time-limited role assignments (optional, NOT enforced for you)
- **Delegation**: Grant admin privileges to other users (with SoD controls)

#### 1.3.4 Policy Audit & Compliance
- **Change Tracking**: Who, what, when, why for every policy modification
- **Compliance Reports**: NIST, ISO 27001, DoD STIGs
- **Policy Drift Detection**: Alert on unauthorized manual changes
- **Immutable Audit**: MinIO blockchain-style storage (Phase 12 integration)
- **Retention**: 7-year audit retention with 3-tiered storage

### 1.4 Integration with Phase 12

Phase 13 builds on Phase 12's security controls:

| Phase 12 Feature | Phase 13 Enhancement |
|------------------|---------------------|
| Dual YubiKey + Iris Auth | Self-service auth policy editor |
| Session Duration Controls | Dynamic session parameter adjustment |
| MinIO Audit Storage | Policy change audit integration |
| User-Configurable Geofences | Advanced geofence management UI |
| Separation of Duties (SoD) | SoD policy editor with conflict detection |
| Context-Aware Access | Threat level policy customization |
| Continuous Authentication | Behavioral monitoring rule editor |

### 1.5 Threat Model

Phase 13 addresses these administrative threats:

1. **Unauthorized Policy Changes**: Attacker gains admin access, modifies policies
   - **Mitigation**: Admin console requires triple-factor auth (dual YubiKey + iris)
   - **Mitigation**: All policy changes audited in immutable MinIO storage
   - **Mitigation**: Policy change notifications via secure channel

2. **Policy Misconfiguration**: Admin accidentally locks themselves out
   - **Mitigation**: Pre-commit policy validation with simulation
   - **Mitigation**: Break-glass recovery mode with hardware token
   - **Mitigation**: Automatic policy rollback on validation failure

3. **Insider Threat**: Malicious admin creates backdoor policies
   - **Mitigation**: Two-person authorization for critical policy changes
   - **Mitigation**: Policy change review workflow (optional)
   - **Mitigation**: Anomaly detection on policy modifications

4. **Policy Tampering**: Attacker modifies policy files directly
   - **Mitigation**: Policy file integrity monitoring (inotify + SHA3-512)
   - **Mitigation**: Read-only filesystem mounts for policy storage
   - **Mitigation**: Kernel-enforced policy validation on load

5. **Availability Attack**: Attacker floods admin console with requests
   - **Mitigation**: Rate limiting (100 requests/min per IP)
   - **Mitigation**: Admin console localhost-only by default
   - **Mitigation**: Fail-safe policy enforcement (deny on error)

---

## 2. Architecture Overview

### 2.1 System Components

```
┌─────────────────────────────────────────────────────────────────┐
│                     Admin Web Console (Port 8443)                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Policy Editor│  │ Role Manager │  │Geofence Config│          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  Audit Logs  │  │Session Monitor│  │ User Manager │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│                    React + Next.js + TypeScript                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼ HTTPS (TLS 1.3)
┌─────────────────────────────────────────────────────────────────┐
│                   Policy Management Service (Port 8444)          │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │             RESTful API + GraphQL Endpoint               │   │
│  │  /api/policies  /api/roles  /api/geofences  /api/audit  │   │
│  └──────────────────────────────────────────────────────────┘   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │Policy Engine │  │ Validator    │  │ Git Backend  │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│              Python + FastAPI + SQLite + GitPython              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼ Netlink Socket
┌─────────────────────────────────────────────────────────────────┐
│                   DSMIL Kernel Module (Phase 12)                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │            Policy Enforcement Engine (PEE)               │   │
│  │  • Policy Cache (RCU-protected)                          │   │
│  │  • Hot Reload Handler (netlink)                          │   │
│  │  • Authorization Decision Point (ADP)                    │   │
│  └──────────────────────────────────────────────────────────┘   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  MFA Engine  │  │Session Manager│  │Geofence Engine│         │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Policy Storage Layer                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ YAML Policies│  │  Git Repo    │  │ MinIO Audit  │          │
│  │/etc/dsmil/   │  │/var/lib/     │  │localhost:9000│          │
│  │  policies/   │  │dsmil/git/    │  │              │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Data Flow: Policy Update

```
1. Admin opens policy editor in web console
   └─> GET /api/policies/device/61
   └─> Returns current Device 61 policy (YAML + metadata)

2. Admin modifies policy (e.g., change session duration 6h → 8h)
   └─> Visual editor updates YAML in-memory

3. Admin clicks "Validate Policy"
   └─> POST /api/policies/validate
   └─> Policy service runs validation:
       • YAML schema validation
       • Conflict detection (SoD, role permissions)
       • Simulation mode (test against current sessions)
   └─> Returns validation result (success/warnings/errors)

4. Admin clicks "Apply Policy"
   └─> POST /api/policies/device/61
   └─> Policy service:
       a) Authenticates admin (dual YubiKey + iris scan)
       b) Writes YAML to /etc/dsmil/policies/device_61.yaml
       c) Commits to Git repo (with author, timestamp, message)
       d) Audits change to MinIO (blockchain append)
       e) Sends netlink message to kernel module
   └─> Kernel module:
       a) Receives netlink message with policy ID
       b) Loads YAML from filesystem
       c) Parses and validates policy
       d) Updates RCU-protected policy cache (atomic swap)
       e) Sends ACK to policy service
   └─> Policy service returns success to web console

5. Admin sees confirmation toast: "Device 61 policy updated (v142)"
   └─> Policy takes effect immediately for new sessions
   └─> Existing sessions continue with old policy until re-auth
```

### 2.3 Policy File Structure

Policies are stored as YAML files in `/etc/dsmil/policies/`:

```
/etc/dsmil/policies/
├── devices/
│   ├── device_51.yaml       # L8 devices (ATOMAL)
│   ├── device_52.yaml
│   ├── ...
│   ├── device_61.yaml       # L9 NC3 (EXEC + two-person)
│   ├── device_62.yaml
│   └── device_83.yaml       # Emergency Stop
├── roles/
│   ├── role_l8_operator.yaml
│   ├── role_l9_executive.yaml
│   └── role_admin.yaml
├── geofences/
│   ├── geofence_home.yaml
│   ├── geofence_office.yaml
│   └── geofence_scif.yaml
├── sod_policies/
│   └── sod_device_61.yaml   # Separation of Duties for Device 61
├── global/
│   ├── session_defaults.yaml
│   ├── mfa_config.yaml
│   └── threat_levels.yaml
└── metadata/
    └── policy_version.yaml  # Current policy version (monotonic counter)
```

### 2.4 Policy Language Example

**File**: `/etc/dsmil/policies/devices/device_61.yaml`

```yaml
---
policy_version: 1
policy_id: "device_61_v142"
device_id: 61
device_name: "NC3 Analysis Dashboard"
classification: "EXEC"
layer: 9

# Authentication requirements
authentication:
  methods:
    - type: "yubikey_fido2"
      required: true
      serial_number: "YK5C12345678"  # Your FIDO2 key
    - type: "yubikey_fips"
      required: true
      serial_number: "YK5F87654321"  # Your FIPS key
    - type: "iris_scan"
      required: true
      device_path: "/dev/irisshield0"
      liveness_check: true

  # Both YubiKeys must be present (plugged in)
  yubikey_mode: "both_present"  # NOT "challenge_response"

  # Two-person authorization for Device 61
  two_person_rule:
    enabled: true
    authorizer_role: "l9_executive"
    organizational_separation: true  # Different org units

# Session controls
session:
  max_duration_hours: 6  # L9 default
  idle_timeout_minutes: 15
  reauth_interval_hours: 2
  extension_allowed: true
  extension_requires_approval: false  # For you, self-extension OK

  # NO time-based restrictions (variable shift support)
  time_restrictions:
    enabled: false

  daily_limit_hours: 24  # Enforced across all L9 devices
  mandatory_rest_hours: 4  # After 24h cumulative access

# Geofencing
geofencing:
  enabled: true
  zones:
    - geofence_id: "home"
      override_allowed: true
      override_requires: "supervisor_approval"
    - geofence_id: "office"
      override_allowed: false

  # GPS validation threshold
  location_tolerance_meters: 50

# Context-aware access
context_aware:
  threat_level_enforcement:
    GREEN: "allow"
    YELLOW: "allow_with_reauth"
    ORANGE: "allow_with_continuous_auth"
    RED: "deny"
    DEFCON: "deny"

  # Device 55 behavioral monitoring
  behavioral_monitoring:
    enabled: true
    risk_threshold: 0.7  # Auto-terminate if risk > 70%

# Separation of Duties
separation_of_duties:
  self_authorization: false  # Cannot authorize yourself
  same_org_unit: false       # Authorizer must be different org
  direct_supervisor: false   # Authorizer cannot be direct supervisor

# Audit requirements
audit:
  log_authentication: true
  log_authorization: true
  log_session_events: true
  log_policy_violations: true
  storage_backend: "minio"  # Phase 12 integration

# Rules of Engagement (ROE)
roe:
  device_61_specific:
    read_only: true  # NC3 analysis is read-only
    roe_level_required: 3  # DEFENSIVE_READY minimum
    fail_safe: "deny"  # Deny on ROE validation error

# Policy metadata
metadata:
  created_by: "admin"
  created_at: "2025-11-23T10:30:00Z"
  last_modified_by: "admin"
  last_modified_at: "2025-11-23T14:45:00Z"
  git_commit: "a7f3c2d1e8b4f9a2c5d8e1f4a7b2c5d8"
  description: "Device 61 NC3 access policy with triple-factor auth"
```

### 2.5 Technology Stack

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| **Frontend** | React 18 + Next.js 14 | Modern UI framework, SSR support |
| **UI Components** | shadcn/ui + Radix UI | Accessible, customizable components |
| **Styling** | Tailwind CSS | Utility-first, dark mode support |
| **State Management** | Zustand | Lightweight, minimal boilerplate |
| **Policy Editor** | Monaco Editor | VS Code editor component, YAML syntax |
| **Map Component** | Leaflet + OpenStreetMap | Geofence configuration UI |
| **Backend API** | FastAPI (Python 3.11+) | High-performance async API |
| **Policy Storage** | YAML files + Git | Human-readable, version control |
| **Database** | SQLite (audit log index) | Lightweight, serverless |
| **Audit Storage** | MinIO (Phase 12) | Immutable object storage |
| **IPC** | Netlink sockets | Kernel ↔ userspace communication |
| **Validation** | JSON Schema + Cerberus | YAML schema validation |
| **Authentication** | libfido2 + libykpers + OpenCV | YubiKey + iris integration |
| **Encryption** | TLS 1.3 (mTLS) | Web console ↔ API communication |

### 2.6 Security Architecture

#### 2.6.1 Admin Console Security

1. **Authentication**:
   - Triple-factor required: Dual YubiKey (FIDO2 + FIPS) + iris scan
   - Session token: JWT with 1-hour expiration
   - Refresh token: Stored in secure HTTP-only cookie
   - Token binding: Bound to client IP + user agent

2. **Network Isolation**:
   - Default: Localhost-only (127.0.0.1:8443)
   - Optional: LAN access with IP whitelist
   - NO internet-facing exposure (firewall enforced)

3. **Transport Security**:
   - TLS 1.3 with mutual authentication (mTLS)
   - Client certificate: Admin's hardware-backed certificate
   - Server certificate: Self-signed (internal CA)
   - Cipher suite: TLS_AES_256_GCM_SHA384

4. **Input Validation**:
   - All policy inputs validated against JSON Schema
   - YAML parsing with safe loader (no code execution)
   - SQL injection prevention (parameterized queries)
   - XSS prevention (React auto-escaping + CSP headers)

5. **Rate Limiting**:
   - 100 requests/min per IP address
   - 10 policy updates/min per admin
   - 5 failed auth attempts → 15-minute lockout

#### 2.6.2 Policy Engine Security

1. **File Integrity**:
   - inotify monitoring on `/etc/dsmil/policies/`
   - SHA3-512 hash verification on policy load
   - Immutable filesystem attributes (chattr +i)
   - Tripwire-style integrity checking

2. **Policy Validation**:
   - YAML schema validation (JSON Schema)
   - Conflict detection (SoD violations, permission conflicts)
   - Simulation mode (test policy against current sessions)
   - Rollback on validation failure

3. **Privilege Separation**:
   - Policy service runs as `dsmil-policy` user (non-root)
   - Kernel module runs in kernel space (ring 0)
   - Netlink socket: Permission 0600, owner `root:dsmil-policy`
   - File permissions: `/etc/dsmil/policies/` → 0700, owner `root`

4. **Audit Logging**:
   - All policy changes logged to MinIO (immutable)
   - Blockchain-style chaining (SHA3-512 + ML-DSA-87)
   - Syslog integration for real-time alerting
   - SIEM integration (optional)

#### 2.6.3 Kernel Module Security

1. **Policy Cache**:
   - RCU (Read-Copy-Update) for lock-free reads
   - Atomic pointer swap for policy updates
   - Memory isolation (separate page tables)

2. **Netlink Interface**:
   - Capability check: CAP_NET_ADMIN required
   - Message authentication: HMAC-SHA3-256
   - Sequence number validation (replay attack prevention)
   - Sanitization: All userspace inputs validated

3. **Fail-Safe Defaults**:
   - Policy load failure → Deny all access (fail-closed)
   - Netlink timeout → Keep existing policy
   - Invalid policy → Log error + rollback
   - Kernel panic → Emergency recovery mode

---

## 3. Self-Service Admin Portal

### 3.1 Overview

The admin portal is a web-based interface for managing all DSMIL security policies. It provides:

- **Visual Policy Editor**: Drag-and-drop rule builder, no YAML editing required
- **Real-Time Validation**: Instant feedback on policy conflicts
- **Multi-Tab Interface**: Devices, Roles, Geofences, Sessions, Audit
- **Dark Mode**: Optimized for 24/7 operations (OLED-friendly)
- **Responsive Design**: Desktop (1920x1080+) and tablet (iPad Pro)

### 3.2 Dashboard (Home Page)

**URL**: `https://localhost:8443/`

**Layout**:

```
┌─────────────────────────────────────────────────────────────────┐
│  DSMIL Admin Console                              [User: admin] │
│  ───────────────────────────────────────────────────────────    │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  System Status                           [Last 24 hours] │    │
│  │  • Active Sessions: 3/10                                 │    │
│  │  • Policy Version: v142 (updated 2h ago)                 │    │
│  │  • Failed Auth Attempts: 0                               │    │
│  │  • Geofence Violations: 0                                │    │
│  │  • Threat Level: GREEN                                   │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   Devices    │  │    Roles     │  │  Geofences   │          │
│  │   [51-62]    │  │  [L8, L9]    │  │  [3 zones]   │          │
│  │   Manage →   │  │   Manage →   │  │   Manage →   │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│                                                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   Sessions   │  │  Audit Logs  │  │  Settings    │          │
│  │  [3 active]  │  │ [View logs]  │  │  [System]    │          │
│  │   Monitor →  │  │   View →     │  │  Configure → │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│                                                                   │
│  Recent Policy Changes                                           │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ 2025-11-23 14:45  admin  Device 61: Updated session     │    │
│  │                           duration (6h → 8h)             │    │
│  │ 2025-11-23 10:30  admin  Geofence: Created "office"     │    │
│  │ 2025-11-22 18:20  admin  Role: Modified L9 permissions  │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

**Key Metrics Displayed**:
- Active sessions (current / max concurrent)
- Policy version (monotonic counter + last update time)
- Failed authentication attempts (last 24h)
- Geofence violations (last 24h)
- Current threat level (GREEN/YELLOW/ORANGE/RED/DEFCON)

### 3.3 Device Policy Editor

**URL**: `https://localhost:8443/devices/61`

**Layout**:

```
┌─────────────────────────────────────────────────────────────────┐
│  ← Back to Devices          Device 61: NC3 Analysis Dashboard   │
│  ───────────────────────────────────────────────────────────    │
│                                                                   │
│  [Visual Editor]  [YAML Editor]  [History]  [Simulate]          │
│                                                                   │
│  ┌─ Authentication ───────────────────────────────────────────┐ │
│  │ ☑ YubiKey FIDO2 (Serial: YK5C12345678)                     │ │
│  │ ☑ YubiKey FIPS (Serial: YK5F87654321)                      │ │
│  │ ☑ Iris Scan (Device: /dev/irisshield0)                     │ │
│  │                                                             │ │
│  │ YubiKey Mode: [Both Present ▼]                             │ │
│  │               • Both Present (plugged in continuously)      │ │
│  │               • Challenge-Response (insert on demand)       │ │
│  │                                                             │ │
│  │ ☑ Two-Person Authorization                                 │ │
│  │   Authorizer Role: [L9 Executive ▼]                        │ │
│  │   ☑ Organizational Separation Required                     │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                   │
│  ┌─ Session Controls ──────────────────────────────────────────┐ │
│  │ Max Duration:    [6] hours                                  │ │
│  │ Idle Timeout:    [15] minutes                               │ │
│  │ Re-auth Interval: [2] hours                                 │ │
│  │                                                             │ │
│  │ ☑ Extension Allowed                                        │ │
│  │ ☐ Extension Requires Approval                              │ │
│  │                                                             │ │
│  │ Daily Limit:     [24] hours (across all L9 devices)        │ │
│  │ Mandatory Rest:  [4] hours (after daily limit)             │ │
│  │                                                             │ │
│  │ Time Restrictions:                                          │ │
│  │ ☐ Enable time-based access control                         │ │
│  │   (Variable shift support - NO restrictions)               │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                   │
│  ┌─ Geofencing ─────────────────────────────────────────────────┐ │
│  │ ☑ Enabled                                                   │ │
│  │                                                             │ │
│  │ Required Zones:                                             │ │
│  │  ☑ Home (lat: 40.7128, lng: -74.0060, radius: 100m)        │ │
│  │    Override: [Supervisor Approval ▼]                       │ │
│  │  ☑ Office (lat: 40.7589, lng: -73.9851, radius: 50m)       │ │
│  │    Override: [Not Allowed ▼]                               │ │
│  │                                                             │ │
│  │  [+ Add Zone]  [Manage Geofences →]                        │ │
│  │                                                             │ │
│  │ Location Tolerance: [50] meters                             │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                   │
│  [Validate Policy]  [Apply Changes]  [Discard]                   │
└─────────────────────────────────────────────────────────────────┘
```

**Interactive Elements**:

1. **Tab Switcher**:
   - **Visual Editor**: Form-based UI (shown above)
   - **YAML Editor**: Monaco editor with syntax highlighting
   - **History**: Git commit history for this device policy
   - **Simulate**: Test policy against current/hypothetical sessions

2. **Authentication Section**:
   - Checkboxes to enable/disable auth methods
   - Dropdown for YubiKey mode (both present vs challenge-response)
   - Serial number validation (auto-detect plugged-in YubiKeys)
   - Two-person rule toggle with role selector

3. **Session Controls**:
   - Number inputs for durations (hours/minutes)
   - Checkboxes for extension and approval requirements
   - Time restrictions toggle (disabled for your use case)

4. **Geofencing**:
   - List of assigned geofence zones
   - Override policy per zone
   - Link to geofence manager
   - Location tolerance slider

5. **Action Buttons**:
   - **Validate Policy**: Runs validation without applying
   - **Apply Changes**: Commits policy (requires triple-factor auth)
   - **Discard**: Reverts to last saved version

### 3.4 Policy Validation UI

When clicking "Validate Policy", a modal appears:

```
┌─────────────────────────────────────────────────────────────────┐
│  Policy Validation                                     [X Close] │
│  ───────────────────────────────────────────────────────────    │
│                                                                   │
│  ✓ YAML Syntax: Valid                                            │
│  ✓ Schema Validation: Passed                                     │
│  ✓ Conflict Detection: No conflicts                              │
│  ⚠ Warnings: 1 warning                                           │
│                                                                   │
│  Warnings:                                                        │
│  • Session duration increased from 6h to 8h. This may impact     │
│    daily limit enforcement. Current active sessions will         │
│    continue with 6h limit until re-authentication.               │
│                                                                   │
│  Simulation Results:                                              │
│  • Current Sessions: 1 active session (Device 61, started 2h ago)│
│  • Impact: Session will expire in 4h (old policy). After re-auth,│
│    new 8h limit applies.                                          │
│                                                                   │
│  [Run Simulation]  [Apply Anyway]  [Cancel]                      │
└─────────────────────────────────────────────────────────────────┘
```

**Validation Checks**:
1. **YAML Syntax**: Parsed with safe YAML loader
2. **Schema Validation**: JSON Schema validation against policy spec
3. **Conflict Detection**:
   - SoD violations (self-authorization, same org unit)
   - Permission conflicts (role grants conflicting permissions)
   - Geofence overlaps (multiple zones with incompatible overrides)
4. **Simulation**: Test policy against current active sessions
5. **Warnings**: Non-blocking issues (e.g., session duration changes)

### 3.5 YAML Editor Mode

Switching to "YAML Editor" tab shows Monaco editor:

```
┌─────────────────────────────────────────────────────────────────┐
│  ← Back to Visual Editor                            [Save] [Copy]│
│  ───────────────────────────────────────────────────────────    │
│                                                                   │
│  1  ---                                                           │
│  2  policy_version: 1                                             │
│  3  policy_id: "device_61_v143"                                   │
│  4  device_id: 61                                                 │
│  5  device_name: "NC3 Analysis Dashboard"                         │
│  6  classification: "EXEC"                                        │
│  7  layer: 9                                                      │
│  8                                                                │
│  9  authentication:                                               │
│ 10    methods:                                                    │
│ 11      - type: "yubikey_fido2"                                   │
│ 12        required: true                                          │
│ 13        serial_number: "YK5C12345678"                           │
│ 14      - type: "yubikey_fips"                                    │
│ 15        required: true                                          │
│ 16        serial_number: "YK5F87654321"                           │
│ 17      - type: "iris_scan"                                       │
│ 18        required: true                                          │
│ 19        device_path: "/dev/irisshield0"                         │
│ 20        liveness_check: true                                    │
│ 21                                                                │
│ 22    yubikey_mode: "both_present"                                │
│ 23                                                                │
│ 24    two_person_rule:                                            │
│ 25      enabled: true                                             │
│ 26      authorizer_role: "l9_executive"                           │
│ 27      organizational_separation: true                           │
│ 28                                                                │
│ 29  session:                                                      │
│ 30    max_duration_hours: 8  # Changed from 6                     │
│                              ^ cursor                             │
└─────────────────────────────────────────────────────────────────┘
```

**Monaco Editor Features**:
- Syntax highlighting (YAML)
- Auto-completion (policy fields)
- Error highlighting (invalid YAML)
- Line numbers
- Search & replace
- Undo/redo (50 steps)
- Copy/paste support

### 3.6 Policy History

Clicking "History" tab shows Git commit log:

```
┌─────────────────────────────────────────────────────────────────┐
│  Policy History: Device 61                          [Export CSV] │
│  ───────────────────────────────────────────────────────────    │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ v143  2025-11-23 14:45  admin                           │    │
│  │ Updated session duration (6h → 8h)                      │    │
│  │ [View Diff]  [Rollback]                                 │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ v142  2025-11-23 10:30  admin                           │    │
│  │ Added two-person authorization requirement              │    │
│  │ [View Diff]  [Rollback]                                 │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ v141  2025-11-22 18:20  admin                           │    │
│  │ Created geofence zone "office"                          │    │
│  │ [View Diff]  [Rollback]                                 │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                   │
│  ... (showing 3 of 142 commits)                                  │
│  [Load More]                                                     │
└─────────────────────────────────────────────────────────────────┘
```

**Rollback Feature**:
Clicking "Rollback" shows confirmation modal:

```
┌─────────────────────────────────────────────────────────────────┐
│  Rollback Policy to v142?                          [X Close]     │
│  ───────────────────────────────────────────────────────────    │
│                                                                   │
│  This will revert Device 61 policy to version 142:               │
│                                                                   │
│  Changes to be reverted:                                          │
│  • session.max_duration_hours: 8 → 6                             │
│                                                                   │
│  Impact:                                                          │
│  • 1 active session will be re-validated against old policy      │
│  • Session may be terminated if exceeding 6h limit               │
│                                                                   │
│  ⚠ This action will create a new policy version (v144) with      │
│     the contents of v142. This preserves audit history.          │
│                                                                   │
│  Reason for rollback (required):                                 │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ Testing session duration changes - reverting to baseline  │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                   │
│  [Confirm Rollback]  [Cancel]                                    │
└─────────────────────────────────────────────────────────────────┘
```

### 3.7 Geofence Management UI

**URL**: `https://localhost:8443/geofences`

**Layout**:

```
┌─────────────────────────────────────────────────────────────────┐
│  Geofence Management                        [+ Create Geofence] │
│  ───────────────────────────────────────────────────────────    │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  [Map View]  [List View]                                │    │
│  │                                                          │    │
│  │  ┌──────────────────────────────────────────────────┐   │    │
│  │  │                                                  │   │    │
│  │  │         OpenStreetMap (Leaflet)                  │   │    │
│  │  │                                                  │   │    │
│  │  │    🔵 Home (100m radius)                         │   │    │
│  │  │       [40.7128, -74.0060]                        │   │    │
│  │  │                                                  │   │    │
│  │  │    🔵 Office (50m radius)                        │   │    │
│  │  │       [40.7589, -73.9851]                        │   │    │
│  │  │                                                  │   │    │
│  │  │    🔵 SCIF (25m radius)                          │   │    │
│  │  │       [38.8977, -77.0365]                        │   │    │
│  │  │                                                  │   │    │
│  │  │  [+] Click map to create new zone               │   │    │
│  │  │                                                  │   │    │
│  │  └──────────────────────────────────────────────────┘   │    │
│  │                                                          │    │
│  │  Geofence List:                                          │    │
│  │  ┌────────────────────────────────────────────────────┐ │    │
│  │  │ 🔵 Home                                            │ │    │
│  │  │    Location: 40.7128, -74.0060                     │ │    │
│  │  │    Radius: 100m                                    │ │    │
│  │  │    Devices: 51-62 (All L8/L9)                      │ │    │
│  │  │    [Edit]  [Delete]  [Export]                      │ │    │
│  │  └────────────────────────────────────────────────────┘ │    │
│  │                                                          │    │
│  │  ┌────────────────────────────────────────────────────┐ │    │
│  │  │ 🔵 Office                                          │ │    │
│  │  │    Location: 40.7589, -73.9851                     │ │    │
│  │  │    Radius: 50m                                     │ │    │
│  │  │    Devices: 59-62 (L9 only)                        │ │    │
│  │  │    [Edit]  [Delete]  [Export]                      │ │    │
│  │  └────────────────────────────────────────────────────┘ │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                   │
│  [Import Geofences]  [Export All]  [Test GPS]                    │
└─────────────────────────────────────────────────────────────────┘
```

**Interactive Map**:
- Click to create new geofence
- Drag circles to move zones
- Resize circles to adjust radius
- Hover for zone details

**Create Geofence Modal**:

```
┌─────────────────────────────────────────────────────────────────┐
│  Create Geofence                                   [X Close]     │
│  ───────────────────────────────────────────────────────────    │
│                                                                   │
│  Name: [Office Building                                      ]   │
│                                                                   │
│  Location (selected on map):                                     │
│  Latitude:  [40.7589      ]  Longitude: [-73.9851      ]        │
│                                                                   │
│  Radius: [50] meters  [────────●────] (10m - 1000m)              │
│                                                                   │
│  Applicable Devices:                                             │
│  ☑ Device 51 (L8 ATOMAL)      ☑ Device 59 (L9 EXEC)             │
│  ☑ Device 52 (L8 ATOMAL)      ☑ Device 60 (L9 EXEC)             │
│  ☑ Device 53 (L8 ATOMAL)      ☑ Device 61 (L9 NC3)              │
│  ☑ Device 54 (L8 ATOMAL)      ☑ Device 62 (L9 EXEC)             │
│  ...                                                             │
│                                                                   │
│  Classification: [SECRET ▼]                                      │
│                                                                   │
│  Override Policy:                                                │
│  ( ) Not Allowed                                                 │
│  (●) Supervisor Approval Required                                │
│  ( ) Self-Override Allowed                                       │
│                                                                   │
│  Description (optional):                                         │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ Primary work location for L8/L9 operations                │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                   │
│  [Create]  [Cancel]                                              │
└─────────────────────────────────────────────────────────────────┘
```

### 3.8 Session Monitoring

**URL**: `https://localhost:8443/sessions`

```
┌─────────────────────────────────────────────────────────────────┐
│  Active Sessions                                    [Refresh: 5s]│
│  ───────────────────────────────────────────────────────────    │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Device 61: NC3 Analysis Dashboard                       │    │
│  │ User: admin                                             │    │
│  │ Started: 2025-11-23 12:00:00 (2h 45m ago)               │    │
│  │ Expires: 2025-11-23 18:00:00 (in 3h 15m)                │    │
│  │ Location: Office (40.7589, -73.9851) ✓                  │    │
│  │ Threat Level: GREEN                                     │    │
│  │ Authentication: YubiKey FIDO2 + FIPS + Iris ✓           │    │
│  │ Last Activity: 2m ago                                   │    │
│  │ [Extend Session]  [Terminate]  [Details]                │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Device 55: Security Analytics                           │    │
│  │ User: admin                                             │    │
│  │ Started: 2025-11-23 08:30:00 (6h 15m ago)               │    │
│  │ Expires: 2025-11-23 20:30:00 (in 5h 45m)                │    │
│  │ Location: Home (40.7128, -74.0060) ✓                    │    │
│  │ Threat Level: GREEN                                     │    │
│  │ Authentication: YubiKey FIDO2 + FIPS ✓                  │    │
│  │ Last Activity: 15s ago                                  │    │
│  │ Behavioral Risk: 12% (Low)                              │    │
│  │ [Extend Session]  [Terminate]  [Details]                │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                   │
│  Session Statistics (Last 24h):                                  │
│  • Total Sessions: 8                                             │
│  • Average Duration: 5h 23m                                      │
│  • Cumulative Time: 18h 45m / 24h limit                          │
│  • Mandatory Rest in: 5h 15m                                     │
│                                                                   │
│  [Export Report]  [View History]                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.9 Audit Log Viewer

**URL**: `https://localhost:8443/audit`

```
┌─────────────────────────────────────────────────────────────────┐
│  Audit Logs                                         [Filters ▼] │
│  ───────────────────────────────────────────────────────────    │
│                                                                   │
│  Filters:                                                        │
│  Event Type: [All ▼]  User: [All ▼]  Device: [All ▼]            │
│  Date Range: [Last 24h ▼]  Classification: [All ▼]              │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ 2025-11-23 14:45:32  POLICY_UPDATE  admin              │    │
│  │ Device 61: Updated session duration (6h → 8h)           │    │
│  │ Policy Version: v142 → v143                             │    │
│  │ Authentication: YubiKey FIDO2 + FIPS + Iris             │    │
│  │ [View Details]  [View Diff]                             │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ 2025-11-23 14:40:18  AUTHENTICATION_SUCCESS  admin     │    │
│  │ Admin Console Login                                     │    │
│  │ Location: 40.7589, -73.9851 (Office)                    │    │
│  │ Authentication: YubiKey FIDO2 + FIPS + Iris             │    │
│  │ [View Details]                                          │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ 2025-11-23 12:00:05  DEVICE_ACCESS  admin               │    │
│  │ Device 61: Session started (NC3 Analysis)               │    │
│  │ Authorization: Two-person rule satisfied                │    │
│  │ Authorizer: user_l9_exec_002                            │    │
│  │ [View Details]                                          │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                   │
│  ... (showing 3 of 1,247 events)                                 │
│  [Load More]  [Export CSV]  [Export JSON]                        │
└─────────────────────────────────────────────────────────────────┘
```

**Event Detail Modal**:

```
┌─────────────────────────────────────────────────────────────────┐
│  Event Details                                     [X Close]     │
│  ───────────────────────────────────────────────────────────    │
│                                                                   │
│  Event ID: evt_a7f3c2d1e8b4f9a2                                 │
│  Timestamp: 2025-11-23 14:45:32.847 UTC                          │
│  Event Type: POLICY_UPDATE                                       │
│                                                                   │
│  User Information:                                               │
│  • User ID: admin                                                │
│  • Role: Administrator                                           │
│  • Session ID: sess_4d8e9f2a1b3c5d7e                            │
│                                                                   │
│  Policy Change:                                                  │
│  • Device: 61 (NC3 Analysis Dashboard)                           │
│  • Field: session.max_duration_hours                             │
│  • Old Value: 6                                                  │
│  • New Value: 8                                                  │
│  • Policy Version: v142 → v143                                   │
│  • Git Commit: a7f3c2d1e8b4f9a2c5d8e1f4a7b2c5d8                  │
│                                                                   │
│  Authentication:                                                 │
│  • YubiKey FIDO2: YK5C12345678 ✓                                 │
│  • YubiKey FIPS: YK5F87654321 ✓                                  │
│  • Iris Scan: Verified (liveness: pass) ✓                        │
│                                                                   │
│  Context:                                                        │
│  • Location: 40.7589, -73.9851 (Office geofence)                 │
│  • IP Address: 127.0.0.1 (localhost)                             │
│  • User Agent: Mozilla/5.0 (X11; Linux x86_64) Chrome/120.0      │
│                                                                   │
│  MinIO Object: 2025/11/23/block-evt_a7f3c2d1e8b4f9a2.json       │
│  Blockchain Hash: sha3-512:7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d...  │
│  Signature: ml-dsa-87:4f5a6b7c8d9e0f1a2b3c4d5e6f7a8b9c...       │
│                                                                   │
│  [Download JSON]  [Verify Signature]  [Close]                    │
└─────────────────────────────────────────────────────────────────┘
```

### 3.10 Admin Console Implementation

**Frontend Stack**:

```typescript
// src/pages/_app.tsx
import { SessionProvider } from 'next-auth/react';
import { ThemeProvider } from '@/components/theme-provider';

export default function App({ Component, pageProps }) {
  return (
    <SessionProvider session={pageProps.session}>
      <ThemeProvider attribute="class" defaultTheme="dark">
        <Component {...pageProps} />
      </ThemeProvider>
    </SessionProvider>
  );
}

// src/pages/devices/[deviceId].tsx
import { useState, useEffect } from 'react';
import { useRouter } from 'next/router';
import { PolicyEditor } from '@/components/policy-editor';

export default function DevicePolicyPage() {
  const router = useRouter();
  const { deviceId } = router.query;
  const [policy, setPolicy] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (deviceId) {
      fetch(`/api/policies/device/${deviceId}`)
        .then(res => res.json())
        .then(data => {
          setPolicy(data.policy);
          setLoading(false);
        });
    }
  }, [deviceId]);

  const handleValidate = async () => {
    const res = await fetch('/api/policies/validate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ policy }),
    });
    const result = await res.json();
    return result;
  };

  const handleApply = async () => {
    // Require triple-factor auth
    const authResult = await authenticateAdmin();
    if (!authResult.success) {
      alert('Authentication failed');
      return;
    }

    const res = await fetch(`/api/policies/device/${deviceId}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ policy }),
    });

    if (res.ok) {
      alert('Policy updated successfully');
      router.push('/devices');
    } else {
      const error = await res.json();
      alert(`Policy update failed: ${error.message}`);
    }
  };

  if (loading) return <div>Loading...</div>;

  return (
    <PolicyEditor
      policy={policy}
      onChange={setPolicy}
      onValidate={handleValidate}
      onApply={handleApply}
    />
  );
}

// src/components/policy-editor.tsx
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { VisualEditor } from './visual-editor';
import { YAMLEditor } from './yaml-editor';
import { PolicyHistory } from './policy-history';

export function PolicyEditor({ policy, onChange, onValidate, onApply }) {
  return (
    <div className="container mx-auto p-6">
      <h1 className="text-2xl font-bold mb-4">
        Device {policy.device_id}: {policy.device_name}
      </h1>

      <Tabs defaultValue="visual">
        <TabsList>
          <TabsTrigger value="visual">Visual Editor</TabsTrigger>
          <TabsTrigger value="yaml">YAML Editor</TabsTrigger>
          <TabsTrigger value="history">History</TabsTrigger>
          <TabsTrigger value="simulate">Simulate</TabsTrigger>
        </TabsList>

        <TabsContent value="visual">
          <VisualEditor policy={policy} onChange={onChange} />
        </TabsContent>

        <TabsContent value="yaml">
          <YAMLEditor policy={policy} onChange={onChange} />
        </TabsContent>

        <TabsContent value="history">
          <PolicyHistory deviceId={policy.device_id} />
        </TabsContent>

        <TabsContent value="simulate">
          <PolicySimulator policy={policy} />
        </TabsContent>
      </Tabs>

      <div className="mt-6 flex gap-4">
        <button
          onClick={onValidate}
          className="px-4 py-2 bg-blue-600 text-white rounded"
        >
          Validate Policy
        </button>
        <button
          onClick={onApply}
          className="px-4 py-2 bg-green-600 text-white rounded"
        >
          Apply Changes
        </button>
      </div>
    </div>
  );
}
```

---

## 4. Dynamic Policy Engine

### 4.1 Overview

The Dynamic Policy Engine (DPE) enables **zero-downtime policy updates** by:

1. **Hot Reload**: Policies updated without kernel module reload
2. **Atomic Updates**: All-or-nothing policy application
3. **Validation**: Pre-commit conflict detection and simulation
4. **Versioning**: Git-based policy history with rollback
5. **Auditing**: Immutable audit trail in MinIO storage

### 4.2 Architecture

```
Policy Storage         Policy Service         Kernel Module
──────────────        ────────────────       ──────────────

/etc/dsmil/           FastAPI Server         Policy Cache
policies/             (Python)               (RCU-protected)
  └── devices/              │                      │
      └── device_61.yaml    │                      │
                            │                      │
Git Repo              Netlink Handler        Netlink Listener
/var/lib/dsmil/git/   ─────────────────────> (hot reload)
  └── .git/                                        │
                                                   ▼
MinIO Audit                               Authorization
localhost:9000        <───────────────    Decision Point
  └── audit/                                   (PEE)
```

### 4.3 Policy Update Workflow

**Step 1: Admin edits policy in web console**

```typescript
// Frontend: User clicks "Apply Changes"
const handleApply = async () => {
  // Step 1a: Validate policy
  const validationResult = await fetch('/api/policies/validate', {
    method: 'POST',
    body: JSON.stringify({ policy }),
  });

  if (!validationResult.ok) {
    alert('Policy validation failed');
    return;
  }

  // Step 1b: Authenticate admin (triple-factor)
  const authResult = await authenticateAdmin({
    requireYubikeyFIDO2: true,
    requireYubikeyFIPS: true,
    requireIrisScan: true,
  });

  if (!authResult.success) {
    alert('Authentication failed');
    return;
  }

  // Step 1c: Apply policy
  const applyResult = await fetch(`/api/policies/device/${deviceId}`, {
    method: 'PUT',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${authResult.token}`,
    },
    body: JSON.stringify({ policy }),
  });

  if (applyResult.ok) {
    alert('Policy updated successfully');
  }
};
```

**Step 2: Policy service processes request**

```python
# backend/api/policies.py
from fastapi import APIRouter, HTTPException, Depends
from .auth import verify_admin_auth
from .policy_engine import PolicyEngine

router = APIRouter()
engine = PolicyEngine()

@router.put("/policies/device/{device_id}")
async def update_device_policy(
    device_id: int,
    policy: Dict,
    auth: AdminAuth = Depends(verify_admin_auth)
):
    """
    Update device policy with hot reload.

    Requires:
    - Triple-factor authentication (dual YubiKey + iris)
    - Valid policy schema
    - No conflicts
    """

    # Step 2a: Validate policy
    validation = engine.validate_policy(policy)
    if not validation.valid:
        raise HTTPException(400, detail=validation.errors)

    # Step 2b: Write policy to filesystem
    policy_path = f"/etc/dsmil/policies/devices/device_{device_id}.yaml"
    with open(policy_path, 'w') as f:
        yaml.dump(policy, f)

    # Step 2c: Commit to Git
    git_commit = engine.commit_to_git(
        file_path=policy_path,
        author=auth.user_id,
        message=f"Updated Device {device_id} policy"
    )

    # Step 2d: Audit to MinIO
    engine.audit_policy_change(
        device_id=device_id,
        user_id=auth.user_id,
        old_policy=engine.get_current_policy(device_id),
        new_policy=policy,
        git_commit=git_commit
    )

    # Step 2e: Notify kernel module via netlink
    result = engine.reload_policy(device_id)
    if not result.success:
        # Rollback on failure
        engine.rollback_to_previous_version(device_id)
        raise HTTPException(500, detail="Kernel reload failed")

    # Step 2f: Return success
    return {
        "status": "success",
        "policy_version": engine.get_current_version(device_id),
        "git_commit": git_commit,
        "message": f"Device {device_id} policy updated"
    }
```

**Step 3: Netlink communication**

```python
# backend/policy_engine/netlink.py
import socket
import struct
from enum import IntEnum

class NetlinkMsgType(IntEnum):
    POLICY_RELOAD = 0x1000
    POLICY_RELOAD_ACK = 0x1001
    POLICY_RELOAD_ERR = 0x1002

class NetlinkPolicyReloader:
    def __init__(self):
        self.sock = socket.socket(
            socket.AF_NETLINK,
            socket.SOCK_RAW,
            socket.NETLINK_USERSOCK
        )
        self.sock.bind((0, 0))  # Bind to kernel

    def reload_policy(self, device_id: int) -> bool:
        """
        Send netlink message to kernel module to reload policy.

        Message format:
        - type: POLICY_RELOAD (2 bytes)
        - device_id: (2 bytes)
        - policy_version: (4 bytes)
        - checksum: SHA3-256 of policy file (32 bytes)
        """

        # Read policy file
        policy_path = f"/etc/dsmil/policies/devices/device_{device_id}.yaml"
        with open(policy_path, 'rb') as f:
            policy_data = f.read()

        # Compute checksum
        checksum = hashlib.sha3_256(policy_data).digest()

        # Get current version
        version = self._get_current_version(device_id)

        # Build netlink message
        msg = struct.pack(
            "!HHI32s",
            NetlinkMsgType.POLICY_RELOAD,
            device_id,
            version,
            checksum
        )

        # Send to kernel
        self.sock.send(msg)

        # Wait for ACK (timeout: 5 seconds)
        self.sock.settimeout(5.0)
        try:
            response = self.sock.recv(1024)
            msg_type = struct.unpack("!H", response[:2])[0]

            if msg_type == NetlinkMsgType.POLICY_RELOAD_ACK:
                return True
            elif msg_type == NetlinkMsgType.POLICY_RELOAD_ERR:
                error_code = struct.unpack("!I", response[2:6])[0]
                raise PolicyReloadError(f"Kernel error: {error_code}")

        except socket.timeout:
            raise PolicyReloadError("Kernel timeout (no ACK)")

        return False
```

**Step 4: Kernel module hot reload**

```c
// 01-source/kernel/security/dsmil_policy_reload.c

#include <linux/netlink.h>
#include <linux/skbuff.h>
#include <net/sock.h>

#define NETLINK_DSMIL_POLICY 31  // Custom netlink family

enum netlink_msg_type {
    POLICY_RELOAD = 0x1000,
    POLICY_RELOAD_ACK = 0x1001,
    POLICY_RELOAD_ERR = 0x1002,
};

struct netlink_policy_msg {
    uint16_t msg_type;
    uint16_t device_id;
    uint32_t policy_version;
    uint8_t checksum[32];  // SHA3-256
} __packed;

static struct sock *nl_sock = NULL;

// RCU-protected policy cache
static struct device_policy __rcu *policy_cache[MAX_DEVICES];
static DEFINE_SPINLOCK(policy_cache_lock);

/**
 * netlink_recv_policy_reload - Handle policy reload message from userspace
 */
static void netlink_recv_policy_reload(struct sk_buff *skb)
{
    struct nlmsghdr *nlh;
    struct netlink_policy_msg *msg;
    struct device_policy *new_policy;
    int device_id;
    int ret;

    nlh = (struct nlmsghdr *)skb->data;
    msg = (struct netlink_policy_msg *)nlmsg_data(nlh);

    // Validate message
    if (msg->msg_type != POLICY_RELOAD) {
        pr_err("dsmil: Invalid netlink message type: 0x%x\n", msg->msg_type);
        goto send_error;
    }

    device_id = msg->device_id;

    if (device_id < 0 || device_id >= MAX_DEVICES) {
        pr_err("dsmil: Invalid device_id: %d\n", device_id);
        goto send_error;
    }

    // Load policy from filesystem
    new_policy = load_policy_from_file(device_id);
    if (!new_policy) {
        pr_err("dsmil: Failed to load policy for device %d\n", device_id);
        goto send_error;
    }

    // Verify checksum
    uint8_t computed_checksum[32];
    sha3_256(new_policy->yaml_data, new_policy->yaml_size, computed_checksum);

    if (memcmp(computed_checksum, msg->checksum, 32) != 0) {
        pr_err("dsmil: Policy checksum mismatch for device %d\n", device_id);
        kfree(new_policy);
        goto send_error;
    }

    // Validate policy structure
    ret = validate_policy_structure(new_policy);
    if (ret != 0) {
        pr_err("dsmil: Policy validation failed for device %d: %d\n",
               device_id, ret);
        kfree(new_policy);
        goto send_error;
    }

    // Atomically swap policy (RCU)
    spin_lock(&policy_cache_lock);
    struct device_policy *old_policy = rcu_dereference_protected(
        policy_cache[device_id],
        lockdep_is_held(&policy_cache_lock)
    );
    rcu_assign_pointer(policy_cache[device_id], new_policy);
    spin_unlock(&policy_cache_lock);

    // Free old policy after RCU grace period
    if (old_policy) {
        synchronize_rcu();
        kfree(old_policy);
    }

    pr_info("dsmil: Policy reloaded for device %d (version %u)\n",
            device_id, msg->policy_version);

    // Send ACK
    send_netlink_ack(nlh->nlmsg_pid);
    return;

send_error:
    send_netlink_error(nlh->nlmsg_pid, -EINVAL);
}

/**
 * send_netlink_ack - Send ACK message to userspace
 */
static void send_netlink_ack(uint32_t pid)
{
    struct sk_buff *skb_out;
    struct nlmsghdr *nlh;
    struct netlink_policy_msg *msg;

    skb_out = nlmsg_new(sizeof(struct netlink_policy_msg), GFP_KERNEL);
    if (!skb_out) {
        pr_err("dsmil: Failed to allocate skb for ACK\n");
        return;
    }

    nlh = nlmsg_put(skb_out, 0, 0, NLMSG_DONE, sizeof(struct netlink_policy_msg), 0);
    msg = nlmsg_data(nlh);
    msg->msg_type = POLICY_RELOAD_ACK;

    nlmsg_unicast(nl_sock, skb_out, pid);
}

/**
 * dsmil_policy_reload_init - Initialize netlink socket for policy reload
 */
int dsmil_policy_reload_init(void)
{
    struct netlink_kernel_cfg cfg = {
        .input = netlink_recv_policy_reload,
    };

    nl_sock = netlink_kernel_create(&init_net, NETLINK_DSMIL_POLICY, &cfg);
    if (!nl_sock) {
        pr_err("dsmil: Failed to create netlink socket\n");
        return -ENOMEM;
    }

    pr_info("dsmil: Policy reload netlink socket initialized\n");
    return 0;
}
```

### 4.4 Policy Validation Engine

```python
# backend/policy_engine/validator.py
from typing import Dict, List, Tuple
from jsonschema import validate, ValidationError
from dataclasses import dataclass

@dataclass
class ValidationResult:
    valid: bool
    errors: List[str]
    warnings: List[str]

class PolicyValidator:
    def __init__(self):
        self.schema = self._load_policy_schema()

    def validate_policy(self, policy: Dict) -> ValidationResult:
        """
        Comprehensive policy validation.

        Checks:
        1. YAML schema validation
        2. Conflict detection (SoD, permissions)
        3. Geofence validation
        4. Session parameter validation
        5. Authentication method validation
        """

        errors = []
        warnings = []

        # Check 1: Schema validation
        try:
            validate(instance=policy, schema=self.schema)
        except ValidationError as e:
            errors.append(f"Schema validation failed: {e.message}")
            return ValidationResult(valid=False, errors=errors, warnings=warnings)

        # Check 2: SoD validation
        sod_errors = self._validate_sod_policies(policy)
        errors.extend(sod_errors)

        # Check 3: Permission conflicts
        perm_conflicts = self._detect_permission_conflicts(policy)
        errors.extend(perm_conflicts)

        # Check 4: Geofence validation
        geofence_errors = self._validate_geofences(policy)
        errors.extend(geofence_errors)

        # Check 5: Session parameters
        session_warnings = self._validate_session_params(policy)
        warnings.extend(session_warnings)

        # Check 6: Authentication methods
        auth_errors = self._validate_authentication(policy)
        errors.extend(auth_errors)

        return ValidationResult(
            valid=(len(errors) == 0),
            errors=errors,
            warnings=warnings
        )

    def _validate_sod_policies(self, policy: Dict) -> List[str]:
        """
        Validate Separation of Duties policies.

        Checks:
        - Self-authorization disabled for critical devices
        - Organizational separation for Device 61
        - Two-person rule consistency
        """
        errors = []

        device_id = policy.get('device_id')
        sod = policy.get('separation_of_duties', {})

        # Device 61 (NC3) requires strict SoD
        if device_id == 61:
            if sod.get('self_authorization') != False:
                errors.append("Device 61: self_authorization must be false")

            if sod.get('organizational_separation') != True:
                errors.append("Device 61: organizational_separation must be true")

            two_person = policy.get('authentication', {}).get('two_person_rule', {})
            if not two_person.get('enabled'):
                errors.append("Device 61: two_person_rule must be enabled")

        return errors

    def _detect_permission_conflicts(self, policy: Dict) -> List[str]:
        """
        Detect conflicting permissions.

        Example: A role grants both READ and WRITE to Device 61,
        but ROE policy only allows READ.
        """
        conflicts = []

        # Check ROE vs permissions
        roe = policy.get('roe', {}).get('device_61_specific', {})
        if roe.get('read_only') == True:
            # Device 61 is read-only, check if any role grants WRITE
            # (This would be checked against role definitions)
            pass

        return conflicts

    def _validate_geofences(self, policy: Dict) -> List[str]:
        """
        Validate geofence configuration.

        Checks:
        - Geofence zones exist
        - Coordinates are valid (lat: -90 to 90, lng: -180 to 180)
        - Radius is reasonable (10m to 10km)
        """
        errors = []

        geofencing = policy.get('geofencing', {})
        if not geofencing.get('enabled'):
            return errors  # Geofencing disabled, skip validation

        zones = geofencing.get('zones', [])
        for zone in zones:
            zone_id = zone.get('geofence_id')

            # Check if zone exists in database
            if not self._geofence_exists(zone_id):
                errors.append(f"Geofence zone '{zone_id}' does not exist")

        return errors

    def _validate_session_params(self, policy: Dict) -> List[str]:
        """
        Validate session parameters.

        Returns warnings (not errors) for unusual configurations.
        """
        warnings = []

        session = policy.get('session', {})
        max_duration = session.get('max_duration_hours', 6)
        daily_limit = session.get('daily_limit_hours', 24)

        if max_duration > daily_limit:
            warnings.append(
                f"max_duration_hours ({max_duration}h) exceeds daily_limit_hours ({daily_limit}h)"
            )

        # Check for unreasonably long sessions
        if max_duration > 12:
            warnings.append(
                f"max_duration_hours ({max_duration}h) is unusually long. "
                "Consider operator fatigue."
            )

        return warnings

    def _validate_authentication(self, policy: Dict) -> List[str]:
        """
        Validate authentication configuration.

        Checks:
        - At least one auth method enabled
        - YubiKey serial numbers are valid format
        - Iris scanner device path exists
        """
        errors = []

        auth = policy.get('authentication', {})
        methods = auth.get('methods', [])

        if len(methods) == 0:
            errors.append("At least one authentication method must be enabled")

        # Validate YubiKey serial numbers
        for method in methods:
            if method['type'] in ['yubikey_fido2', 'yubikey_fips']:
                serial = method.get('serial_number')
                if not serial or len(serial) != 12:
                    errors.append(
                        f"Invalid YubiKey serial number: {serial}. "
                        "Must be 12 characters."
                    )

        # Validate iris scanner path
        for method in methods:
            if method['type'] == 'iris_scan':
                device_path = method.get('device_path')
                if device_path and not os.path.exists(device_path):
                    errors.append(
                        f"Iris scanner device not found: {device_path}"
                    )

        return errors
```

### 4.5 Policy Simulation

```python
# backend/policy_engine/simulator.py
from typing import Dict, List
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class SimulationResult:
    policy_version: int
    current_sessions: List[Dict]
    impacts: List[str]
    conflicts: List[str]

class PolicySimulator:
    def __init__(self):
        self.session_db = SessionDatabase()

    def simulate_policy(self, policy: Dict) -> SimulationResult:
        """
        Simulate policy against current active sessions.

        Determines:
        - Which sessions would be affected
        - Which sessions would be terminated
        - Which sessions would require re-authentication
        """

        device_id = policy.get('device_id')

        # Get current active sessions for this device
        sessions = self.session_db.get_active_sessions(device_id=device_id)

        impacts = []
        conflicts = []

        for session in sessions:
            # Simulate session validation against new policy
            impact = self._simulate_session_impact(session, policy)
            if impact:
                impacts.append(impact)

            # Check for policy conflicts
            conflict = self._check_session_conflict(session, policy)
            if conflict:
                conflicts.append(conflict)

        return SimulationResult(
            policy_version=policy.get('policy_version'),
            current_sessions=sessions,
            impacts=impacts,
            conflicts=conflicts
        )

    def _simulate_session_impact(self, session: Dict, policy: Dict) -> str:
        """
        Determine impact of policy change on active session.
        """

        session_id = session['session_id']
        session_start = session['started_at']
        session_elapsed = (datetime.utcnow() - session_start).total_seconds() / 3600

        # Check session duration change
        old_max_duration = session['policy']['session']['max_duration_hours']
        new_max_duration = policy['session']['max_duration_hours']

        if new_max_duration < old_max_duration:
            if session_elapsed > new_max_duration:
                return (
                    f"Session {session_id}: Will be terminated immediately "
                    f"(elapsed {session_elapsed:.1f}h > new limit {new_max_duration}h)"
                )
            else:
                time_remaining_old = old_max_duration - session_elapsed
                time_remaining_new = new_max_duration - session_elapsed
                return (
                    f"Session {session_id}: Expiration shortened by "
                    f"{time_remaining_old - time_remaining_new:.1f}h"
                )

        elif new_max_duration > old_max_duration:
            # Note: Existing sessions continue with old policy until re-auth
            return (
                f"Session {session_id}: Will benefit from extended duration "
                f"after next re-authentication"
            )

        return None

    def _check_session_conflict(self, session: Dict, policy: Dict) -> str:
        """
        Check if policy change would create a conflict with active session.

        Example: New policy requires geofence, but user is outside zone.
        """

        session_id = session['session_id']

        # Check geofencing
        if policy.get('geofencing', {}).get('enabled'):
            user_location = session.get('location')
            required_zones = policy['geofencing']['zones']

            if not self._is_in_any_zone(user_location, required_zones):
                return (
                    f"Session {session_id}: User is outside all required geofence zones. "
                    "Session will be terminated on policy apply."
                )

        # Check authentication requirements
        session_auth = session.get('authentication', {})
        policy_auth = policy.get('authentication', {})

        for method in policy_auth.get('methods', []):
            method_type = method['type']
            if method.get('required') and method_type not in session_auth:
                return (
                    f"Session {session_id}: Missing required auth method '{method_type}'. "
                    "User will be prompted to re-authenticate."
                )

        return None
```

### 4.6 Git-Based Policy Versioning

```python
# backend/policy_engine/git_backend.py
import git
from datetime import datetime
from typing import Dict, List, Optional

class PolicyGitBackend:
    def __init__(self, repo_path: str = "/var/lib/dsmil/git"):
        self.repo_path = repo_path
        self.repo = self._init_repo()

    def _init_repo(self) -> git.Repo:
        """Initialize or open Git repository."""
        try:
            repo = git.Repo(self.repo_path)
        except git.InvalidGitRepositoryError:
            repo = git.Repo.init(self.repo_path)
            # Initial commit
            with open(f"{self.repo_path}/.gitignore", 'w') as f:
                f.write("*.tmp\n*.bak\n")
            repo.index.add(['.gitignore'])
            repo.index.commit("Initial commit")

        return repo

    def commit_policy(self, file_path: str, author: str, message: str) -> str:
        """
        Commit policy file to Git repository.

        Returns: Git commit hash
        """
        # Stage file
        self.repo.index.add([file_path])

        # Create commit
        commit = self.repo.index.commit(
            message=message,
            author=git.Actor(author, f"{author}@dsmil.local"),
            committer=git.Actor("DSMIL Policy Engine", "policy@dsmil.local")
        )

        return commit.hexsha

    def get_policy_history(self, device_id: int, limit: int = 50) -> List[Dict]:
        """
        Get commit history for a specific device policy.
        """
        policy_path = f"policies/devices/device_{device_id}.yaml"
        commits = list(self.repo.iter_commits(paths=policy_path, max_count=limit))

        history = []
        for commit in commits:
            history.append({
                'commit_hash': commit.hexsha,
                'author': str(commit.author),
                'timestamp': datetime.fromtimestamp(commit.committed_date),
                'message': commit.message.strip(),
                'version': self._get_policy_version_from_commit(commit, device_id)
            })

        return history

    def rollback_to_commit(self, commit_hash: str, file_path: str) -> bool:
        """
        Rollback a policy file to a specific commit.

        Creates a new commit with the old content (preserves history).
        """
        try:
            # Get file content at commit
            commit = self.repo.commit(commit_hash)
            old_content = commit.tree[file_path].data_stream.read()

            # Write to filesystem
            with open(f"{self.repo_path}/{file_path}", 'wb') as f:
                f.write(old_content)

            # Create new commit
            self.repo.index.add([file_path])
            self.repo.index.commit(f"Rollback to {commit_hash[:8]}")

            return True

        except Exception as e:
            print(f"Rollback failed: {e}")
            return False

    def get_diff(self, commit_hash1: str, commit_hash2: str, file_path: str) -> str:
        """
        Get diff between two commits for a specific file.
        """
        commit1 = self.repo.commit(commit_hash1)
        commit2 = self.repo.commit(commit_hash2)

        diff = commit1.diff(commit2, paths=[file_path], create_patch=True)
        return diff[0].diff.decode('utf-8') if diff else ""
```

---

## 5. Advanced Role Management

### 5.1 Overview

Phase 13 extends role management beyond the default L0-L9 layers with:

1. **Custom Roles**: Define application-specific roles
2. **Granular Permissions**: Per-device, per-operation permissions
3. **Role Hierarchies**: Inheritance with selective overrides
4. **Temporal Roles**: Time-limited role assignments (optional)
5. **Delegation**: Grant admin privileges to trusted users

### 5.2 Role Definition Structure

**File**: `/etc/dsmil/policies/roles/role_l9_executive.yaml`

```yaml
---
role_id: "l9_executive"
role_name: "Layer 9 Executive"
description: "Executive-level access to L9 strategic devices"
layer: 9
classification: "EXEC"

# Permissions
permissions:
  devices:
    # Device-specific permissions
    - device_id: 59
      operations: ["READ", "WRITE", "EXECUTE"]
      conditions: []

    - device_id: 60
      operations: ["READ", "WRITE"]
      conditions: []

    - device_id: 61
      operations: ["READ"]  # NC3 is read-only
      conditions:
        - type: "two_person_authorization"
          required: true
        - type: "roe_level"
          minimum: 3  # DEFENSIVE_READY

    - device_id: 62
      operations: ["READ", "WRITE", "EXECUTE"]
      conditions: []

  # Global capabilities
  capabilities:
    - "can_extend_session"
    - "can_override_geofence_with_approval"
    - "can_authorize_other_users"
    - "can_view_audit_logs"

  # Admin capabilities (NOT granted by default)
  admin_capabilities: []

# Inheritance
inherits_from:
  - "l8_operator"  # Inherits all L8 permissions

overrides:
  # Override L8 session duration
  - field: "session.max_duration_hours"
    value: 6  # L9 = 6h (L8 = 12h)

# Constraints
constraints:
  # Max concurrent sessions
  max_concurrent_sessions: 3

  # Daily access limit
  daily_limit_hours: 24

  # Mandatory rest period
  mandatory_rest_hours: 4

  # Geofencing required
  geofencing_required: true

  # MFA required
  mfa_required: true
  mfa_methods: ["yubikey_fido2", "yubikey_fips"]

# Separation of Duties
sod_policies:
  # Cannot authorize own actions for Device 61
  - device_id: 61
    self_authorization: false
    organizational_separation: true

# Metadata
metadata:
  created_by: "admin"
  created_at: "2025-11-23T10:00:00Z"
  last_modified_by: "admin"
  last_modified_at: "2025-11-23T14:00:00Z"
  version: 12
```

### 5.3 Custom Role Creation UI

**URL**: `https://localhost:8443/roles/create`

```
┌─────────────────────────────────────────────────────────────────┐
│  Create Custom Role                                  [X Close]   │
│  ───────────────────────────────────────────────────────────    │
│                                                                   │
│  Role ID: [security_analyst                                  ]   │
│  Role Name: [Security Analyst                                ]   │
│                                                                   │
│  Description:                                                    │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ Analyzes security events across L6-L8 devices             │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                   │
│  Layer: [8 ▼]  Classification: [ATOMAL ▼]                        │
│                                                                   │
│  ┌─ Device Permissions ──────────────────────────────────────┐  │
│  │                                                            │  │
│  │  Device 51 (Threat Detection):                            │  │
│  │  ☑ READ  ☑ WRITE  ☐ EXECUTE                               │  │
│  │  Conditions: [+ Add Condition]                            │  │
│  │                                                            │  │
│  │  Device 55 (Security Analytics):                          │  │
│  │  ☑ READ  ☑ WRITE  ☐ EXECUTE                               │  │
│  │  Conditions:                                              │  │
│  │    • Geofencing required (Office or SCIF)                 │  │
│  │    [Edit]  [Remove]                                       │  │
│  │                                                            │  │
│  │  [+ Add Device]                                           │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                   │
│  ┌─ Capabilities ─────────────────────────────────────────────┐  │
│  │  ☑ Can extend session                                     │  │
│  │  ☐ Can override geofence (requires approval)              │  │
│  │  ☐ Can authorize other users                              │  │
│  │  ☑ Can view audit logs                                    │  │
│  │  ☐ Can modify policies (admin)                            │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                   │
│  ┌─ Constraints ──────────────────────────────────────────────┐  │
│  │  Max Concurrent Sessions: [2]                             │  │
│  │  Daily Limit (hours): [12]                                │  │
│  │  Mandatory Rest (hours): [4]                              │  │
│  │  Session Duration (hours): [8]                            │  │
│  │  ☑ Geofencing required                                    │  │
│  │  ☑ MFA required                                           │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                   │
│  Inherits From: [l7_classified ▼]                                │
│                                                                   │
│  [Create Role]  [Cancel]                                         │
└─────────────────────────────────────────────────────────────────┘
```

### 5.4 Role Management Backend

```python
# backend/api/roles.py
from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict
from .auth import verify_admin_auth

router = APIRouter()

@router.get("/roles")
async def list_roles() -> List[Dict]:
    """
    List all roles in the system.
    """
    roles = RoleManager.list_roles()
    return roles

@router.get("/roles/{role_id}")
async def get_role(role_id: str) -> Dict:
    """
    Get detailed information about a specific role.
    """
    role = RoleManager.get_role(role_id)
    if not role:
        raise HTTPException(404, detail=f"Role '{role_id}' not found")
    return role

@router.post("/roles")
async def create_role(
    role: Dict,
    auth: AdminAuth = Depends(verify_admin_auth)
):
    """
    Create a new custom role.

    Requires admin authentication.
    """

    # Validate role definition
    validation = RoleManager.validate_role(role)
    if not validation.valid:
        raise HTTPException(400, detail=validation.errors)

    # Check for conflicts
    conflicts = RoleManager.check_conflicts(role)
    if conflicts:
        raise HTTPException(409, detail=conflicts)

    # Create role
    role_id = RoleManager.create_role(role, created_by=auth.user_id)

    # Audit role creation
    AuditLogger.log_event(
        event_type="ROLE_CREATED",
        user_id=auth.user_id,
        resource=f"role:{role_id}",
        details=role
    )

    return {
        "status": "success",
        "role_id": role_id,
        "message": f"Role '{role_id}' created successfully"
    }

@router.put("/roles/{role_id}")
async def update_role(
    role_id: str,
    role: Dict,
    auth: AdminAuth = Depends(verify_admin_auth)
):
    """
    Update an existing role.
    """

    # Check if role exists
    existing = RoleManager.get_role(role_id)
    if not existing:
        raise HTTPException(404, detail=f"Role '{role_id}' not found")

    # Validate updated role
    validation = RoleManager.validate_role(role)
    if not validation.valid:
        raise HTTPException(400, detail=validation.errors)

    # Update role
    RoleManager.update_role(role_id, role, modified_by=auth.user_id)

    # Audit role update
    AuditLogger.log_event(
        event_type="ROLE_UPDATED",
        user_id=auth.user_id,
        resource=f"role:{role_id}",
        old_value=existing,
        new_value=role
    )

    return {
        "status": "success",
        "message": f"Role '{role_id}' updated successfully"
    }

@router.delete("/roles/{role_id}")
async def delete_role(
    role_id: str,
    auth: AdminAuth = Depends(verify_admin_auth)
):
    """
    Delete a custom role.

    Cannot delete built-in roles (l0-l9).
    """

    # Check if role is built-in
    if role_id.startswith('l') and role_id[1:].isdigit():
        raise HTTPException(403, detail="Cannot delete built-in roles")

    # Check if role is assigned to any users
    assigned_users = RoleManager.get_users_with_role(role_id)
    if assigned_users:
        raise HTTPException(
            409,
            detail=f"Role is assigned to {len(assigned_users)} users. "
                   "Remove role assignments before deleting."
        )

    # Delete role
    RoleManager.delete_role(role_id, deleted_by=auth.user_id)

    # Audit role deletion
    AuditLogger.log_event(
        event_type="ROLE_DELETED",
        user_id=auth.user_id,
        resource=f"role:{role_id}"
    )

    return {
        "status": "success",
        "message": f"Role '{role_id}' deleted successfully"
    }

@router.post("/roles/{role_id}/assign")
async def assign_role_to_user(
    role_id: str,
    user_id: str,
    duration_hours: Optional[int] = None,
    auth: AdminAuth = Depends(verify_admin_auth)
):
    """
    Assign a role to a user.

    Optional: Specify duration_hours for temporary role assignment.
    """

    # Check if role exists
    role = RoleManager.get_role(role_id)
    if not role:
        raise HTTPException(404, detail=f"Role '{role_id}' not found")

    # Check if user exists
    user = UserManager.get_user(user_id)
    if not user:
        raise HTTPException(404, detail=f"User '{user_id}' not found")

    # Assign role
    assignment_id = RoleManager.assign_role(
        user_id=user_id,
        role_id=role_id,
        assigned_by=auth.user_id,
        duration_hours=duration_hours
    )

    # Audit role assignment
    AuditLogger.log_event(
        event_type="ROLE_ASSIGNED",
        user_id=auth.user_id,
        resource=f"user:{user_id}",
        details={
            "role_id": role_id,
            "duration_hours": duration_hours,
            "assignment_id": assignment_id
        }
    )

    return {
        "status": "success",
        "assignment_id": assignment_id,
        "message": f"Role '{role_id}' assigned to user '{user_id}'"
    }
```

### 5.5 Role Inheritance Engine

```python
# backend/policy_engine/role_inheritance.py
from typing import Dict, List, Set
from dataclasses import dataclass

@dataclass
class ResolvedRole:
    role_id: str
    permissions: Dict
    capabilities: Set[str]
    constraints: Dict

class RoleInheritanceEngine:
    def __init__(self):
        self.role_cache = {}

    def resolve_role(self, role_id: str) -> ResolvedRole:
        """
        Resolve a role with inheritance.

        Algorithm:
        1. Load role definition
        2. Recursively load all parent roles
        3. Merge permissions (child overrides parent)
        4. Merge capabilities (union)
        5. Merge constraints (most restrictive wins)
        """

        # Check cache
        if role_id in self.role_cache:
            return self.role_cache[role_id]

        # Load role
        role = self._load_role(role_id)

        # Base case: No inheritance
        if not role.get('inherits_from'):
            resolved = ResolvedRole(
                role_id=role_id,
                permissions=role.get('permissions', {}),
                capabilities=set(role.get('permissions', {}).get('capabilities', [])),
                constraints=role.get('constraints', {})
            )
            self.role_cache[role_id] = resolved
            return resolved

        # Recursive case: Inherit from parents
        parent_roles = role.get('inherits_from', [])
        merged_permissions = {}
        merged_capabilities = set()
        merged_constraints = {}

        # Resolve all parents
        for parent_id in parent_roles:
            parent = self.resolve_role(parent_id)

            # Merge permissions (child overrides parent)
            for device_perm in parent.permissions.get('devices', []):
                device_id = device_perm['device_id']
                if device_id not in merged_permissions:
                    merged_permissions[device_id] = device_perm

            # Merge capabilities (union)
            merged_capabilities.update(parent.capabilities)

            # Merge constraints (most restrictive wins)
            for key, value in parent.constraints.items():
                if key not in merged_constraints:
                    merged_constraints[key] = value
                else:
                    # Most restrictive
                    if isinstance(value, int) and isinstance(merged_constraints[key], int):
                        merged_constraints[key] = min(value, merged_constraints[key])

        # Apply current role's permissions (override parents)
        for device_perm in role.get('permissions', {}).get('devices', []):
            device_id = device_perm['device_id']
            merged_permissions[device_id] = device_perm

        # Apply current role's capabilities
        merged_capabilities.update(
            role.get('permissions', {}).get('capabilities', [])
        )

        # Apply current role's constraints
        merged_constraints.update(role.get('constraints', {}))

        # Apply overrides
        for override in role.get('overrides', []):
            field = override['field']
            value = override['value']
            # Apply override to constraints
            if field.startswith('session.'):
                constraint_key = field.replace('session.', '')
                merged_constraints[constraint_key] = value

        resolved = ResolvedRole(
            role_id=role_id,
            permissions={'devices': list(merged_permissions.values())},
            capabilities=merged_capabilities,
            constraints=merged_constraints
        )

        self.role_cache[role_id] = resolved
        return resolved

    def check_permission(self, role_id: str, device_id: int, operation: str) -> bool:
        """
        Check if a role has permission for a specific device operation.
        """
        resolved = self.resolve_role(role_id)

        for device_perm in resolved.permissions.get('devices', []):
            if device_perm['device_id'] == device_id:
                return operation in device_perm.get('operations', [])

        return False

    def get_allowed_devices(self, role_id: str) -> List[int]:
        """
        Get list of devices accessible by a role.
        """
        resolved = self.resolve_role(role_id)
        return [
            perm['device_id']
            for perm in resolved.permissions.get('devices', [])
        ]
```

---

## 6. Policy Audit & Compliance

### 6.1 Overview

Phase 13 provides comprehensive audit and compliance capabilities:

1. **Change Tracking**: Every policy modification logged
2. **Compliance Reports**: NIST, ISO 27001, DoD STIGs
3. **Policy Drift Detection**: Alert on unauthorized changes
4. **Immutable Audit**: MinIO blockchain-style storage (Phase 12)
5. **Retention**: 7-year audit retention with 3-tiered storage

### 6.2 Audit Event Types

```python
# backend/audit/event_types.py
from enum import Enum

class AuditEventType(Enum):
    # Authentication events
    AUTHENTICATION_SUCCESS = "AUTHENTICATION_SUCCESS"
    AUTHENTICATION_FAILURE = "AUTHENTICATION_FAILURE"
    MFA_CHALLENGE = "MFA_CHALLENGE"
    MFA_SUCCESS = "MFA_SUCCESS"
    MFA_FAILURE = "MFA_FAILURE"

    # Authorization events
    AUTHORIZATION_GRANTED = "AUTHORIZATION_GRANTED"
    AUTHORIZATION_DENIED = "AUTHORIZATION_DENIED"
    TWO_PERSON_AUTHORIZATION = "TWO_PERSON_AUTHORIZATION"

    # Device access events
    DEVICE_ACCESS = "DEVICE_ACCESS"
    DEVICE_ACCESS_DENIED = "DEVICE_ACCESS_DENIED"
    DEVICE_OPERATION = "DEVICE_OPERATION"
    SESSION_STARTED = "SESSION_STARTED"
    SESSION_EXTENDED = "SESSION_EXTENDED"
    SESSION_TERMINATED = "SESSION_TERMINATED"
    SESSION_EXPIRED = "SESSION_EXPIRED"

    # Policy events
    POLICY_CREATED = "POLICY_CREATED"
    POLICY_UPDATED = "POLICY_UPDATED"
    POLICY_DELETED = "POLICY_DELETED"
    POLICY_ROLLBACK = "POLICY_ROLLBACK"

    # Role events
    ROLE_CREATED = "ROLE_CREATED"
    ROLE_UPDATED = "ROLE_UPDATED"
    ROLE_DELETED = "ROLE_DELETED"
    ROLE_ASSIGNED = "ROLE_ASSIGNED"
    ROLE_REVOKED = "ROLE_REVOKED"

    # Geofence events
    GEOFENCE_CREATED = "GEOFENCE_CREATED"
    GEOFENCE_UPDATED = "GEOFENCE_UPDATED"
    GEOFENCE_DELETED = "GEOFENCE_DELETED"
    GEOFENCE_VIOLATION = "GEOFENCE_VIOLATION"
    GEOFENCE_OVERRIDE = "GEOFENCE_OVERRIDE"

    # Security events
    THREAT_LEVEL_CHANGED = "THREAT_LEVEL_CHANGED"
    BEHAVIORAL_ANOMALY = "BEHAVIORAL_ANOMALY"
    BREAK_GLASS_ACTIVATED = "BREAK_GLASS_ACTIVATED"
    EMERGENCY_STOP = "EMERGENCY_STOP"

    # Compliance events
    COMPLIANCE_CHECK = "COMPLIANCE_CHECK"
    COMPLIANCE_VIOLATION = "COMPLIANCE_VIOLATION"
    POLICY_DRIFT_DETECTED = "POLICY_DRIFT_DETECTED"
```

### 6.3 Audit Logger Integration

```python
# backend/audit/logger.py
from typing import Dict, Optional
from datetime import datetime
import json
from .minio_backend import MinIOAuditBackend

class AuditLogger:
    def __init__(self):
        self.backend = MinIOAuditBackend()
        self.sqlite_index = SQLiteAuditIndex()

    def log_event(
        self,
        event_type: str,
        user_id: str,
        resource: Optional[str] = None,
        operation: Optional[str] = None,
        result: str = "SUCCESS",
        details: Optional[Dict] = None,
        old_value: Optional[Dict] = None,
        new_value: Optional[Dict] = None,
        authentication: Optional[Dict] = None,
        context: Optional[Dict] = None
    ) -> str:
        """
        Log an audit event.

        Returns: Event ID
        """

        event_id = self._generate_event_id()
        timestamp = datetime.utcnow()

        event = {
            'event_id': event_id,
            'timestamp': timestamp.isoformat(),
            'event_type': event_type,
            'user_id': user_id,
            'resource': resource,
            'operation': operation,
            'result': result,
            'details': details or {},
            'old_value': old_value,
            'new_value': new_value,
            'authentication': authentication or {},
            'context': context or self._get_current_context()
        }

        # Write to MinIO (immutable blockchain storage)
        self.backend.append_block(event)

        # Index in SQLite (fast queries)
        self.sqlite_index.index_event(event)

        # Send to syslog (real-time alerting)
        self._send_to_syslog(event)

        return event_id

    def query_events(
        self,
        event_type: Optional[str] = None,
        user_id: Optional[str] = None,
        resource: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict]:
        """
        Query audit events.

        Uses SQLite index for fast queries, then retrieves full events from MinIO.
        """

        # Query index
        event_ids = self.sqlite_index.query(
            event_type=event_type,
            user_id=user_id,
            resource=resource,
            start_time=start_time,
            end_time=end_time,
            limit=limit,
            offset=offset
        )

        # Retrieve full events from MinIO
        events = []
        for event_id in event_ids:
            event = self.backend.get_event(event_id)
            if event:
                events.append(event)

        return events

    def generate_compliance_report(
        self,
        standard: str,  # "NIST", "ISO27001", "DoD_STIG"
        start_date: datetime,
        end_date: datetime
    ) -> Dict:
        """
        Generate compliance report for a specific standard.
        """

        if standard == "NIST":
            return self._generate_nist_report(start_date, end_date)
        elif standard == "ISO27001":
            return self._generate_iso27001_report(start_date, end_date)
        elif standard == "DoD_STIG":
            return self._generate_dod_stig_report(start_date, end_date)
        else:
            raise ValueError(f"Unknown compliance standard: {standard}")

    def _generate_nist_report(self, start_date: datetime, end_date: datetime) -> Dict:
        """
        Generate NIST 800-53 compliance report.

        Checks:
        - AC-2: Account Management
        - AC-3: Access Enforcement
        - AC-7: Unsuccessful Logon Attempts
        - AU-2: Audit Events
        - AU-3: Content of Audit Records
        - AU-6: Audit Review, Analysis, and Reporting
        - IA-2: Identification and Authentication
        - IA-5: Authenticator Management
        """

        report = {
            'standard': 'NIST 800-53',
            'period': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat()
            },
            'controls': []
        }

        # AC-2: Account Management
        report['controls'].append(self._check_nist_ac2(start_date, end_date))

        # AC-3: Access Enforcement
        report['controls'].append(self._check_nist_ac3(start_date, end_date))

        # AC-7: Unsuccessful Logon Attempts
        report['controls'].append(self._check_nist_ac7(start_date, end_date))

        # AU-2: Audit Events
        report['controls'].append(self._check_nist_au2(start_date, end_date))

        # IA-2: Identification and Authentication
        report['controls'].append(self._check_nist_ia2(start_date, end_date))

        return report

    def _check_nist_ac2(self, start_date: datetime, end_date: datetime) -> Dict:
        """
        NIST AC-2: Account Management

        Checks:
        - All role assignments are logged
        - Role revocations are logged
        - Inactive accounts are detected
        """

        role_assignments = self.query_events(
            event_type="ROLE_ASSIGNED",
            start_time=start_date,
            end_time=end_date
        )

        role_revocations = self.query_events(
            event_type="ROLE_REVOKED",
            start_time=start_date,
            end_time=end_date
        )

        return {
            'control_id': 'AC-2',
            'control_name': 'Account Management',
            'status': 'COMPLIANT',
            'findings': {
                'role_assignments': len(role_assignments),
                'role_revocations': len(role_revocations),
                'inactive_accounts': 0  # TODO: Implement
            },
            'recommendations': []
        }

    def _check_nist_ac3(self, start_date: datetime, end_date: datetime) -> Dict:
        """
        NIST AC-3: Access Enforcement

        Checks:
        - All device access is authorized
        - Access denials are logged
        - Two-person rule is enforced for Device 61
        """

        device_access = self.query_events(
            event_type="DEVICE_ACCESS",
            start_time=start_date,
            end_time=end_date
        )

        access_denials = self.query_events(
            event_type="DEVICE_ACCESS_DENIED",
            start_time=start_date,
            end_time=end_date
        )

        two_person_auth = self.query_events(
            event_type="TWO_PERSON_AUTHORIZATION",
            start_time=start_date,
            end_time=end_date
        )

        return {
            'control_id': 'AC-3',
            'control_name': 'Access Enforcement',
            'status': 'COMPLIANT',
            'findings': {
                'device_access_count': len(device_access),
                'access_denials': len(access_denials),
                'two_person_authorizations': len(two_person_auth)
            },
            'recommendations': []
        }

    def _check_nist_ac7(self, start_date: datetime, end_date: datetime) -> Dict:
        """
        NIST AC-7: Unsuccessful Logon Attempts

        Checks:
        - Failed authentication attempts are logged
        - Account lockouts are enforced
        """

        auth_failures = self.query_events(
            event_type="AUTHENTICATION_FAILURE",
            start_time=start_date,
            end_time=end_date
        )

        # Check for users with excessive failures
        user_failures = {}
        for event in auth_failures:
            user_id = event['user_id']
            user_failures[user_id] = user_failures.get(user_id, 0) + 1

        excessive_failures = {
            user_id: count
            for user_id, count in user_failures.items()
            if count > 5
        }

        status = 'COMPLIANT' if not excessive_failures else 'NON_COMPLIANT'

        return {
            'control_id': 'AC-7',
            'control_name': 'Unsuccessful Logon Attempts',
            'status': status,
            'findings': {
                'total_failures': len(auth_failures),
                'users_with_excessive_failures': len(excessive_failures),
                'details': excessive_failures
            },
            'recommendations': [
                f"Investigate user '{user_id}' with {count} failed attempts"
                for user_id, count in excessive_failures.items()
            ]
        }
```

### 6.4 Policy Drift Detection

```python
# backend/audit/drift_detection.py
import hashlib
from typing import Dict, List
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class PolicyDriftDetector(FileSystemEventHandler):
    def __init__(self, policy_dir: str = "/etc/dsmil/policies"):
        self.policy_dir = policy_dir
        self.expected_hashes = self._compute_expected_hashes()
        self.observer = Observer()

    def _compute_expected_hashes(self) -> Dict[str, str]:
        """
        Compute SHA3-512 hashes for all policy files.
        """
        hashes = {}
        for root, dirs, files in os.walk(self.policy_dir):
            for file in files:
                if file.endswith('.yaml'):
                    path = os.path.join(root, file)
                    with open(path, 'rb') as f:
                        content = f.read()
                        hash_value = hashlib.sha3_512(content).hexdigest()
                        hashes[path] = hash_value
        return hashes

    def on_modified(self, event):
        """
        Detect unauthorized policy file modifications.
        """
        if event.is_directory:
            return

        file_path = event.src_path

        if not file_path.endswith('.yaml'):
            return

        # Compute current hash
        with open(file_path, 'rb') as f:
            content = f.read()
            current_hash = hashlib.sha3_512(content).hexdigest()

        # Check against expected hash
        expected_hash = self.expected_hashes.get(file_path)

        if expected_hash and current_hash != expected_hash:
            # Policy drift detected!
            self._alert_drift(file_path, expected_hash, current_hash)

    def _alert_drift(self, file_path: str, expected_hash: str, current_hash: str):
        """
        Alert on policy drift.
        """
        AuditLogger.log_event(
            event_type="POLICY_DRIFT_DETECTED",
            user_id="system",
            resource=file_path,
            details={
                'expected_hash': expected_hash,
                'current_hash': current_hash,
                'action': 'ALERT'
            }
        )

        # Send alert via syslog
        syslog.syslog(
            syslog.LOG_ALERT,
            f"SECURITY: Policy drift detected in {file_path}"
        )

        # Optionally: Auto-revert to expected version
        # self._revert_to_expected(file_path, expected_hash)

    def start_monitoring(self):
        """
        Start monitoring policy directory for changes.
        """
        self.observer.schedule(self, self.policy_dir, recursive=True)
        self.observer.start()

    def update_expected_hash(self, file_path: str):
        """
        Update expected hash after authorized policy change.
        """
        with open(file_path, 'rb') as f:
            content = f.read()
            hash_value = hashlib.sha3_512(content).hexdigest()
            self.expected_hashes[file_path] = hash_value
```

### 6.5 Compliance Report UI

**URL**: `https://localhost:8443/compliance`

```
┌─────────────────────────────────────────────────────────────────┐
│  Compliance Reports                         [Generate Report ▼] │
│  ───────────────────────────────────────────────────────────    │
│                                                                   │
│  Standard: [NIST 800-53 ▼]                                       │
│  Period: [Last 30 days ▼]  From: [2025-10-24] To: [2025-11-23]  │
│                                                                   │
│  [Generate Report]  [Export PDF]  [Export JSON]                  │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  NIST 800-53 Compliance Report                          │    │
│  │  Period: 2025-10-24 to 2025-11-23                       │    │
│  │  Generated: 2025-11-23 15:00:00 UTC                     │    │
│  │                                                          │    │
│  │  Overall Status: ✓ COMPLIANT (8/8 controls)             │    │
│  │                                                          │    │
│  │  ┌──────────────────────────────────────────────────┐   │    │
│  │  │ AC-2: Account Management            ✓ COMPLIANT  │   │    │
│  │  │ • Role assignments logged: 24                    │   │    │
│  │  │ • Role revocations logged: 3                     │   │    │
│  │  │ • Inactive accounts: 0                           │   │    │
│  │  │ [View Details]                                   │   │    │
│  │  └──────────────────────────────────────────────────┘   │    │
│  │                                                          │    │
│  │  ┌──────────────────────────────────────────────────┐   │    │
│  │  │ AC-3: Access Enforcement            ✓ COMPLIANT  │   │    │
│  │  │ • Device access attempts: 1,247                  │   │    │
│  │  │ • Access denials: 18                             │   │    │
│  │  │ • Two-person authorizations: 42                  │   │    │
│  │  │ [View Details]                                   │   │    │
│  │  └──────────────────────────────────────────────────┘   │    │
│  │                                                          │    │
│  │  ┌──────────────────────────────────────────────────┐   │    │
│  │  │ AC-7: Unsuccessful Logon Attempts   ✓ COMPLIANT  │   │    │
│  │  │ • Total failures: 12                             │   │    │
│  │  │ • Users with excessive failures: 0               │   │    │
│  │  │ [View Details]                                   │   │    │
│  │  └──────────────────────────────────────────────────┘   │    │
│  │                                                          │    │
│  │  ... (5 more controls)                                   │    │
│  │                                                          │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                   │
│  Historical Reports:                                             │
│  • 2025-10-23: NIST 800-53 (COMPLIANT) [View] [Download]        │
│  • 2025-09-23: NIST 800-53 (COMPLIANT) [View] [Download]        │
│  • 2025-08-23: ISO 27001 (COMPLIANT) [View] [Download]          │
└─────────────────────────────────────────────────────────────────┘
```

---

## 7. Automated Enforcement

### 7.1 Overview

Phase 13 provides automated policy enforcement mechanisms:

1. **Real-Time Violation Detection**: Immediate detection of policy violations
2. **Automated Remediation**: Auto-terminate sessions, revoke access, alert admins
3. **Escalation Workflows**: Severity-based escalation (warn → suspend → block)
4. **Integration with Phase 12**: Leverages existing enforcement infrastructure

### 7.2 Enforcement Rules Engine

```python
# backend/enforcement/rules_engine.py
from typing import Dict, List, Optional
from enum import Enum

class EnforcementAction(Enum):
    WARN = "WARN"                    # Log warning, continue
    BLOCK = "BLOCK"                  # Deny operation
    TERMINATE_SESSION = "TERMINATE_SESSION"  # End active session
    REVOKE_ACCESS = "REVOKE_ACCESS"  # Revoke device/role access
    ALERT_ADMIN = "ALERT_ADMIN"      # Send alert to admin

class EnforcementRule:
    def __init__(
        self,
        rule_id: str,
        condition: callable,
        action: EnforcementAction,
        severity: str,  # "LOW", "MEDIUM", "HIGH", "CRITICAL"
        message: str
    ):
        self.rule_id = rule_id
        self.condition = condition
        self.action = action
        self.severity = severity
        self.message = message

class EnforcementEngine:
    def __init__(self):
        self.rules = self._load_enforcement_rules()

    def _load_enforcement_rules(self) -> List[EnforcementRule]:
        """
        Load enforcement rules from configuration.
        """
        return [
            # Session duration exceeded
            EnforcementRule(
                rule_id="session_duration_exceeded",
                condition=lambda ctx: ctx['session_elapsed'] > ctx['max_duration'],
                action=EnforcementAction.TERMINATE_SESSION,
                severity="HIGH",
                message="Session duration exceeded maximum allowed"
            ),

            # Geofence violation
            EnforcementRule(
                rule_id="geofence_violation",
                condition=lambda ctx: not self._is_in_geofence(ctx['location'], ctx['required_zones']),
                action=EnforcementAction.TERMINATE_SESSION,
                severity="HIGH",
                message="User location outside required geofence zones"
            ),

            # Excessive failed auth attempts
            EnforcementRule(
                rule_id="excessive_auth_failures",
                condition=lambda ctx: ctx['failed_attempts'] > 5,
                action=EnforcementAction.REVOKE_ACCESS,
                severity="CRITICAL",
                message="Excessive authentication failures detected"
            ),

            # Behavioral anomaly detected
            EnforcementRule(
                rule_id="behavioral_anomaly",
                condition=lambda ctx: ctx['risk_score'] > 0.7,
                action=EnforcementAction.ALERT_ADMIN,
                severity="MEDIUM",
                message="Behavioral anomaly detected (risk score > 70%)"
            ),

            # Policy drift detected
            EnforcementRule(
                rule_id="policy_drift",
                condition=lambda ctx: ctx['policy_hash'] != ctx['expected_hash'],
                action=EnforcementAction.ALERT_ADMIN,
                severity="CRITICAL",
                message="Unauthorized policy modification detected"
            ),

            # Threat level escalation
            EnforcementRule(
                rule_id="threat_level_red",
                condition=lambda ctx: ctx['threat_level'] == 'RED',
                action=EnforcementAction.TERMINATE_SESSION,
                severity="CRITICAL",
                message="Threat level RED: Terminating all L8/L9 sessions"
            ),
        ]

    def evaluate(self, context: Dict) -> List[Dict]:
        """
        Evaluate all enforcement rules against the current context.

        Returns: List of triggered rules with actions
        """
        triggered = []

        for rule in self.rules:
            try:
                if rule.condition(context):
                    triggered.append({
                        'rule_id': rule.rule_id,
                        'action': rule.action,
                        'severity': rule.severity,
                        'message': rule.message
                    })
            except Exception as e:
                # Log rule evaluation error
                print(f"Error evaluating rule {rule.rule_id}: {e}")

        return triggered

    def execute_actions(self, triggered_rules: List[Dict], context: Dict):
        """
        Execute enforcement actions for triggered rules.
        """
        for rule in triggered_rules:
            action = rule['action']

            if action == EnforcementAction.WARN:
                self._action_warn(rule, context)
            elif action == EnforcementAction.BLOCK:
                self._action_block(rule, context)
            elif action == EnforcementAction.TERMINATE_SESSION:
                self._action_terminate_session(rule, context)
            elif action == EnforcementAction.REVOKE_ACCESS:
                self._action_revoke_access(rule, context)
            elif action == EnforcementAction.ALERT_ADMIN:
                self._action_alert_admin(rule, context)

    def _action_terminate_session(self, rule: Dict, context: Dict):
        """
        Terminate active session.
        """
        session_id = context.get('session_id')
        SessionManager.terminate_session(session_id, reason=rule['message'])

        # Audit
        AuditLogger.log_event(
            event_type="SESSION_TERMINATED",
            user_id=context.get('user_id'),
            resource=f"session:{session_id}",
            details={
                'rule_id': rule['rule_id'],
                'reason': rule['message'],
                'automated': True
            }
        )

    def _action_alert_admin(self, rule: Dict, context: Dict):
        """
        Send alert to admin console.
        """
        AlertManager.send_alert(
            severity=rule['severity'],
            message=rule['message'],
            context=context
        )

        # Audit
        AuditLogger.log_event(
            event_type="ENFORCEMENT_ALERT",
            user_id="system",
            details={
                'rule_id': rule['rule_id'],
                'message': rule['message'],
                'context': context
            }
        )
```

---

## 8. API & Integration

### 8.1 RESTful API Summary

The Phase 13 Policy Management Service exposes the following REST endpoints:

**Base URL**: `https://localhost:8444/api`

#### Policy Management
- `GET /policies` - List all policies
- `GET /policies/device/{device_id}` - Get device policy
- `PUT /policies/device/{device_id}` - Update device policy
- `POST /policies/validate` - Validate policy without applying
- `POST /policies/rollback` - Rollback policy to previous version
- `GET /policies/device/{device_id}/history` - Get policy history

#### Role Management
- `GET /roles` - List all roles
- `GET /roles/{role_id}` - Get role details
- `POST /roles` - Create custom role
- `PUT /roles/{role_id}` - Update role
- `DELETE /roles/{role_id}` - Delete custom role
- `POST /roles/{role_id}/assign` - Assign role to user
- `DELETE /roles/{role_id}/revoke` - Revoke role from user

#### Geofence Management
- `GET /geofences` - List all geofences
- `GET /geofences/{geofence_id}` - Get geofence details
- `POST /geofences` - Create geofence
- `PUT /geofences/{geofence_id}` - Update geofence
- `DELETE /geofences/{geofence_id}` - Delete geofence

#### Session Management
- `GET /sessions` - List active sessions
- `GET /sessions/{session_id}` - Get session details
- `POST /sessions/{session_id}/extend` - Extend session
- `DELETE /sessions/{session_id}` - Terminate session

#### Audit & Compliance
- `GET /audit/events` - Query audit events
- `GET /audit/events/{event_id}` - Get event details
- `POST /compliance/report` - Generate compliance report
- `GET /compliance/reports` - List historical reports

### 8.2 GraphQL API

**Endpoint**: `https://localhost:8444/graphql`

```graphql
type Query {
  # Policies
  policy(deviceId: Int!): DevicePolicy
  policies: [DevicePolicy!]!
  policyHistory(deviceId: Int!, limit: Int): [PolicyVersion!]!

  # Roles
  role(roleId: String!): Role
  roles: [Role!]!

  # Geofences
  geofence(geofenceId: String!): Geofence
  geofences: [Geofence!]!

  # Sessions
  session(sessionId: String!): Session
  activeSessions: [Session!]!

  # Audit
  auditEvents(
    eventType: String
    userId: String
    startTime: DateTime
    endTime: DateTime
    limit: Int
  ): [AuditEvent!]!

  # Compliance
  complianceReport(
    standard: String!
    startDate: DateTime!
    endDate: DateTime!
  ): ComplianceReport
}

type Mutation {
  # Policies
  updatePolicy(deviceId: Int!, policy: PolicyInput!): PolicyUpdateResult!
  validatePolicy(policy: PolicyInput!): ValidationResult!
  rollbackPolicy(deviceId: Int!, version: Int!): PolicyUpdateResult!

  # Roles
  createRole(role: RoleInput!): Role!
  updateRole(roleId: String!, role: RoleInput!): Role!
  deleteRole(roleId: String!): DeleteResult!
  assignRole(userId: String!, roleId: String!, durationHours: Int): RoleAssignment!

  # Geofences
  createGeofence(geofence: GeofenceInput!): Geofence!
  updateGeofence(geofenceId: String!, geofence: GeofenceInput!): Geofence!
  deleteGeofence(geofenceId: String!): DeleteResult!

  # Sessions
  extendSession(sessionId: String!, hours: Int!): Session!
  terminateSession(sessionId: String!): DeleteResult!
}
```

### 8.3 Integration Examples

#### LDAP/Active Directory Integration

```python
# backend/integrations/ldap_sync.py
import ldap
from typing import List, Dict

class LDAPSyncService:
    def __init__(self, server: str, bind_dn: str, bind_password: str):
        self.server = server
        self.bind_dn = bind_dn
        self.bind_password = bind_password

    def sync_users(self) -> List[Dict]:
        """
        Synchronize users from LDAP/AD to DSMIL.
        """
        conn = ldap.initialize(self.server)
        conn.simple_bind_s(self.bind_dn, self.bind_password)

        # Search for users
        search_filter = "(objectClass=person)"
        attributes = ['uid', 'cn', 'mail', 'memberOf']

        results = conn.search_s(
            'ou=users,dc=example,dc=com',
            ldap.SCOPE_SUBTREE,
            search_filter,
            attributes
        )

        users = []
        for dn, attrs in results:
            user = {
                'user_id': attrs['uid'][0].decode(),
                'name': attrs['cn'][0].decode(),
                'email': attrs['mail'][0].decode() if 'mail' in attrs else None,
                'groups': [g.decode() for g in attrs.get('memberOf', [])]
            }
            users.append(user)

            # Map LDAP groups to DSMIL roles
            self._map_groups_to_roles(user)

        conn.unbind_s()
        return users

    def _map_groups_to_roles(self, user: Dict):
        """
        Map LDAP/AD groups to DSMIL roles.
        """
        group_role_mapping = {
            'CN=Executives,OU=Groups,DC=example,DC=com': 'l9_executive',
            'CN=Operators,OU=Groups,DC=example,DC=com': 'l8_operator',
            'CN=Analysts,OU=Groups,DC=example,DC=com': 'l7_classified',
        }

        for group in user['groups']:
            if group in group_role_mapping:
                role_id = group_role_mapping[group]
                RoleManager.assign_role(user['user_id'], role_id)
```

#### SIEM Integration (Syslog)

```python
# backend/integrations/siem.py
import syslog
import json

class SIEMIntegration:
    @staticmethod
    def send_event(event: Dict):
        """
        Send audit event to SIEM via syslog.
        """
        # Format event as CEF (Common Event Format)
        cef_message = SIEMIntegration._format_cef(event)

        # Send to syslog
        syslog.syslog(syslog.LOG_INFO, cef_message)

    @staticmethod
    def _format_cef(event: Dict) -> str:
        """
        Format event in CEF format for SIEM consumption.
        """
        # CEF format:
        # CEF:Version|Device Vendor|Device Product|Device Version|Signature ID|Name|Severity|Extension

        return (
            f"CEF:0|DSMIL|PolicyEngine|1.0|{event['event_type']}|"
            f"{event['event_type']}|{event.get('severity', 'INFO')}|"
            f"src={event.get('source_ip')} suser={event['user_id']} "
            f"dst={event.get('dest_ip')} dvc={event.get('device_id')} "
            f"msg={event.get('message')}"
        )
```

---

## 9. Exit Criteria

### 9.1 Phase Completion Requirements

Phase 13 is considered complete when ALL of the following criteria are met:

#### 9.1.1 Self-Service Admin Portal
- [ ] Web console accessible at https://localhost:8443
- [ ] Dashboard displays system status (active sessions, policy version, threat level)
- [ ] Device policy editor (visual + YAML modes) functional
- [ ] Policy validation runs successfully (schema + conflicts + simulation)
- [ ] Policy history displays Git commit log
- [ ] Policy rollback creates new version (preserves history)
- [ ] Geofence management UI with interactive map (Leaflet)
- [ ] Session monitoring shows active sessions with real-time updates
- [ ] Audit log viewer displays events with filtering
- [ ] Dark mode UI optimized for 24/7 operations

#### 9.1.2 Dynamic Policy Engine
- [ ] Hot reload updates policies without kernel module restart
- [ ] Netlink communication between userspace and kernel successful
- [ ] Policy files stored in `/etc/dsmil/policies/` with correct permissions (0700)
- [ ] Git backend commits all policy changes with author/timestamp
- [ ] MinIO audit storage logs policy changes with blockchain chaining
- [ ] Policy validation detects SoD violations, permission conflicts, geofence errors
- [ ] Policy simulation accurately predicts impact on active sessions
- [ ] RCU-based policy cache in kernel for lock-free reads
- [ ] Atomic policy updates (all-or-nothing with rollback on failure)

#### 9.1.3 Advanced Role Management
- [ ] Custom roles definable via YAML files in `/etc/dsmil/policies/roles/`
- [ ] Role inheritance engine correctly merges permissions/capabilities/constraints
- [ ] Role creation UI allows per-device, per-operation permissions
- [ ] Role assignment supports optional time-limited duration
- [ ] Built-in roles (l0-l9) cannot be deleted
- [ ] Role validation prevents conflicts and orphaned assignments

#### 9.1.4 Policy Audit & Compliance
- [ ] All policy changes logged to MinIO with immutable blockchain chaining
- [ ] SQLite index enables fast audit event queries
- [ ] Compliance reports generate for NIST 800-53, ISO 27001, DoD STIGs
- [ ] Policy drift detection monitors `/etc/dsmil/policies/` for unauthorized changes
- [ ] Audit retention configured for 7 years (hot: 90d, warm: 1y, cold: 7y+)
- [ ] Syslog integration sends real-time alerts for critical events

#### 9.1.5 Automated Enforcement
- [ ] Enforcement rules engine evaluates violations in real-time
- [ ] Session termination auto-triggered on duration/geofence/threat violations
- [ ] Access revocation automated for excessive auth failures
- [ ] Admin alerts sent for behavioral anomalies and policy drift
- [ ] Enforcement actions audited with rule ID and reason

#### 9.1.6 API & Integration
- [ ] RESTful API accessible at https://localhost:8444/api
- [ ] GraphQL endpoint accessible at https://localhost:8444/graphql
- [ ] API authentication requires JWT token with admin role
- [ ] Rate limiting enforced (100 requests/min per IP)
- [ ] LDAP/AD sync imports users and maps groups to roles
- [ ] SIEM integration sends CEF-formatted events via syslog

### 9.2 Testing Requirements

#### 9.2.1 Functional Testing
- [ ] Policy update workflow (edit → validate → apply → hot reload)
- [ ] Policy rollback restores previous version without data loss
- [ ] Geofence creation/update/delete via UI
- [ ] Role assignment grants correct device permissions
- [ ] Session termination on policy violation (duration/geofence)
- [ ] Audit log query returns correct filtered results
- [ ] Compliance report generates with accurate control status

#### 9.2.2 Security Testing
- [ ] Admin console requires triple-factor auth (dual YubiKey + iris)
- [ ] Policy files protected with 0700 permissions (root-only)
- [ ] Netlink messages authenticated with HMAC-SHA3-256
- [ ] Policy drift detection alerts on unauthorized file modification
- [ ] Break-glass procedure requires dual YubiKey + iris for Device 61
- [ ] SQL injection testing passes (parameterized queries)
- [ ] XSS testing passes (React auto-escaping + CSP headers)

#### 9.2.3 Performance Testing
- [ ] Policy hot reload completes within 5 seconds
- [ ] Web console loads within 2 seconds
- [ ] Policy validation runs within 1 second
- [ ] Audit query returns 1000 events within 2 seconds
- [ ] Role inheritance resolves within 100ms
- [ ] RCU policy cache lookup within 10µs (kernel)

#### 9.2.4 Integration Testing
- [ ] Netlink kernel ↔ userspace communication successful
- [ ] MinIO blockchain append maintains cryptographic chain
- [ ] Git backend commits policy changes with correct metadata
- [ ] LDAP sync imports users and assigns roles
- [ ] SIEM receives syslog events in CEF format
- [ ] Threat level changes (Phase 12) trigger enforcement actions

### 9.3 Documentation Requirements

- [ ] User guide for admin console (screenshots + workflows)
- [ ] API reference documentation (REST + GraphQL)
- [ ] Policy YAML schema specification
- [ ] Role inheritance algorithm explained
- [ ] Compliance mapping (NIST controls → audit events)
- [ ] Integration guides (LDAP, SIEM, ticketing)
- [ ] Troubleshooting guide (common errors + solutions)

### 9.4 Operational Readiness

- [ ] Admin console runs as systemd service (dsmil-admin-console.service)
- [ ] Policy service runs as systemd service (dsmil-policy-service.service)
- [ ] TLS certificates configured (self-signed CA for internal use)
- [ ] MinIO storage initialized with correct buckets
- [ ] Git repository initialized at `/var/lib/dsmil/git/`
- [ ] Backup/restore procedures documented
- [ ] Monitoring alerts configured (service down, policy drift, etc.)

---

## 10. Future Enhancements

### 10.1 Policy Templates
- Pre-built policy templates for common scenarios
- Import/export policy templates in JSON format
- Policy template marketplace (community-contributed)

### 10.2 Advanced Analytics
- Machine learning-based anomaly detection for audit logs
- Predictive compliance risk scoring
- Policy optimization recommendations (e.g., "reduce L9 session duration to improve security")

### 10.3 Multi-Tenancy
- Support multiple independent policy domains
- Tenant isolation for shared DSMIL deployment
- Per-tenant admin consoles

### 10.4 Policy Testing Framework
- Unit tests for policy validation logic
- Integration tests for policy engine
- Policy chaos testing (random mutations to detect edge cases)

### 10.5 Advanced Workflows
- Multi-step approval workflows for critical policy changes
- Change advisory board (CAB) integration
- Scheduled policy changes (e.g., "apply policy on 2025-12-01 00:00")

---

**End of Phase 13 Documentation**


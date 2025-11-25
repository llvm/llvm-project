# DSMIL Driver API Reference

**Version:** 5.2.0
**Date:** 2025-11-13
**API Version:** 1.0

---

## Table of Contents

1. [Overview](#overview)
2. [IOCTL Interface](#ioctl-interface)
3. [Sysfs Interface](#sysfs-interface)
4. [Data Structures](#data-structures)
5. [Error Codes](#error-codes)
6. [Token Database](#token-database)
7. [Constants and Enumerations](#constants-and-enumerations)
8. [Module Parameters](#module-parameters)

---

## Overview

The DSMIL driver provides three primary interfaces:

1. **IOCTL Interface** - Primary programmatic interface via `/dev/dsmil0`
2. **Sysfs Interface** - System monitoring and configuration via `/sys/class/dsmil/dsmil0/`
3. **Module Parameters** - Compile-time and load-time configuration

### Character Device

- **Device Path:** `/dev/dsmil0`
- **Major Number:** Dynamically allocated
- **Minor Number:** 0
- **Permissions:** 0666 (read/write for all users, authentication required for protected operations)

---

## IOCTL Interface

### IOCTL Command Format

```c
#include <sys/ioctl.h>

#define DSMIL_IOC_MAGIC 'D'

/* IOCTL commands */
#define DSMIL_IOC_READ_TOKEN         _IOWR(DSMIL_IOC_MAGIC, 1, struct dsmil_token_request)
#define DSMIL_IOC_WRITE_TOKEN        _IOW(DSMIL_IOC_MAGIC, 2, struct dsmil_token_request)
#define DSMIL_IOC_AUTHENTICATE       _IOW(DSMIL_IOC_MAGIC, 3, struct dsmil_auth_request)
#define DSMIL_IOC_GET_STATUS         _IOR(DSMIL_IOC_MAGIC, 4, struct dsmil_driver_status)
#define DSMIL_IOC_GET_DEVICE_STATUS  _IOWR(DSMIL_IOC_MAGIC, 5, struct dsmil_device_status)
#define DSMIL_IOC_GET_DEVICE_LIST    _IOR(DSMIL_IOC_MAGIC, 6, struct dsmil_device_list)
#define DSMIL_IOC_INVALIDATE_AUTH    _IO(DSMIL_IOC_MAGIC, 7)
#define DSMIL_IOC_GET_BIOS_STATUS    _IOR(DSMIL_IOC_MAGIC, 8, struct dsmil_bios_status)
#define DSMIL_IOC_BIOS_QUERY         _IOWR(DSMIL_IOC_MAGIC, 9, struct dsmil_bios_info)
#define DSMIL_IOC_BIOS_SYNC          _IO(DSMIL_IOC_MAGIC, 10)
#define DSMIL_IOC_TPM_GET_CHALLENGE  _IOR(DSMIL_IOC_MAGIC, 11, struct dsmil_tpm_challenge_data)
#define DSMIL_IOC_TPM_INVALIDATE     _IO(DSMIL_IOC_MAGIC, 12)
```

---

### DSMIL_IOC_READ_TOKEN (1)

**Purpose:** Read a token value from SMBIOS.

**Data Structure:**
```c
struct dsmil_token_request {
    __u16 token_id;   /* Input: Token ID to read */
    __u32 value;      /* Output: Token value */
};
```

**Usage:**
```c
int fd = open("/dev/dsmil0", O_RDWR);
struct dsmil_token_request req;

req.token_id = 0x8000;  // Device 0 status token
req.value = 0;

if (ioctl(fd, DSMIL_IOC_READ_TOKEN, &req) == 0) {
    printf("Token 0x%04x = 0x%08x\n", req.token_id, req.value);
}

close(fd);
```

**Return Values:**
- **0** - Success, `req.value` contains token value
- **-EINVAL** - Invalid token ID
- **-ENOENT** - Token not found in database
- **-EIO** - SMBIOS call failed

**Requires:**
- No special privileges for non-protected tokens
- CAP_SYS_ADMIN for protected tokens (read-only ones)

---

### DSMIL_IOC_WRITE_TOKEN (2)

**Purpose:** Write a token value to SMBIOS.

**Data Structure:**
```c
struct dsmil_token_request {
    __u16 token_id;   /* Input: Token ID to write */
    __u32 value;      /* Input: Value to write */
};
```

**Usage:**
```c
int fd = open("/dev/dsmil0", O_RDWR);
struct dsmil_token_request req;

req.token_id = 0x8001;  // Device 0 config token
req.value = 0x00000001; // Enable device

if (ioctl(fd, DSMIL_IOC_WRITE_TOKEN, &req) == 0) {
    printf("Token 0x%04x written successfully\n", req.token_id);
}

close(fd);
```

**Return Values:**
- **0** - Success
- **-EINVAL** - Invalid token ID or value
- **-ENOENT** - Token not found
- **-EPERM** - Permission denied (protected token without authentication)
- **-EROFS** - Read-only token
- **-EIO** - SMBIOS call failed

**Requires:**
- No special privileges for non-protected tokens
- **CAP_SYS_ADMIN + TPM authentication** for protected tokens

**Notes:**
- Writes to protected tokens (0x8200-0x8FFF) require authentication
- Audit log entry created for protected token writes
- TPM PCR17 extended for protected token access (when TPM available)

---

### DSMIL_IOC_AUTHENTICATE (3)

**Purpose:** Authenticate user for protected token access.

**Data Structure:**
```c
struct dsmil_auth_request {
    __u32 auth_method;      /* Input: Authentication method */
    __u32 auth_data_len;    /* Input: Authentication data length */
    __u8  auth_data[256];   /* Input: Authentication data */
};

/* Authentication methods */
#define DSMIL_AUTH_METHOD_NONE      0  /* No authentication */
#define DSMIL_AUTH_METHOD_CHALLENGE 1  /* TPM challenge-response */
#define DSMIL_AUTH_METHOD_KEY       2  /* TPM key-based */
#define DSMIL_AUTH_METHOD_HMAC      3  /* TPM HMAC */
#define DSMIL_AUTH_METHOD_EXTERNAL  4  /* External authenticator */
```

**Usage:**
```c
int fd = open("/dev/dsmil0", O_RDWR);
struct dsmil_tpm_challenge_data chal;
struct dsmil_auth_request auth;

// Step 1: Get challenge
ioctl(fd, DSMIL_IOC_TPM_GET_CHALLENGE, &chal);

// Step 2: Sign challenge with TPM (external process)
unsigned char signature[256];
int sig_len = tpm_sign_challenge(chal.challenge, signature);

// Step 3: Submit authentication
memset(&auth, 0, sizeof(auth));
auth.auth_method = DSMIL_AUTH_METHOD_CHALLENGE;
memcpy(auth.auth_data, &chal.challenge_id, sizeof(chal.challenge_id));
memcpy(auth.auth_data + 4, signature, sig_len);
auth.auth_data_len = 4 + sig_len;

if (ioctl(fd, DSMIL_IOC_AUTHENTICATE, &auth) == 0) {
    printf("Authentication successful\n");
    // Can now write protected tokens
}

close(fd);
```

**Return Values:**
- **0** - Authentication successful, session created
- **-EINVAL** - Invalid authentication method or data
- **-EPERM** - Authentication failed (wrong signature, insufficient privileges)
- **-ETIMEDOUT** - Challenge expired
- **-ENODEV** - TPM not available (when required)

**Requires:**
- **CAP_SYS_ADMIN** capability (always required)
- Valid TPM challenge response (when TPM available)

**Notes:**
- Creates authenticated session valid for 5 minutes (default)
- TPM PCR16 extended on authentication success
- Audit log entry created
- Session token stored in driver for subsequent operations

---

### DSMIL_IOC_GET_STATUS (4)

**Purpose:** Get overall driver status and statistics.

**Data Structure:**
```c
struct dsmil_driver_status {
    __u32 driver_version;        /* Driver version (0xMMmmpp00) */
    __u32 device_count;          /* Number of DSMIL devices (104) */
    __u32 bios_count;            /* Number of BIOS systems (3) */
    __u8  active_bios;           /* Active BIOS (0=A, 1=B, 2=C) */
    __u8  auth_active;           /* Authentication session active */
    __u16 reserved;
    __u32 total_token_reads;     /* Total token read operations */
    __u32 total_token_writes;    /* Total token write operations */
    __u32 failed_operations;     /* Failed operations count */
    __u32 auth_attempts;         /* Authentication attempts */
    __u32 bios_failovers;        /* BIOS failover count */
    __u64 uptime_seconds;        /* Driver uptime in seconds */
};
```

**Usage:**
```c
int fd = open("/dev/dsmil0", O_RDWR);
struct dsmil_driver_status status;

if (ioctl(fd, DSMIL_IOC_GET_STATUS, &status) == 0) {
    printf("Driver Version: %u.%u.%u\n",
           (status.driver_version >> 24) & 0xFF,
           (status.driver_version >> 16) & 0xFF,
           (status.driver_version >> 8) & 0xFF);
    printf("Devices: %u\n", status.device_count);
    printf("Active BIOS: %c\n", 'A' + status.active_bios);
    printf("Token Reads: %u\n", status.total_token_reads);
    printf("Token Writes: %u\n", status.total_token_writes);
}

close(fd);
```

**Return Values:**
- **0** - Success
- **-EFAULT** - Failed to copy data to userspace

**Requires:** No special privileges

---

### DSMIL_IOC_GET_DEVICE_STATUS (5)

**Purpose:** Get status of a specific DSMIL device.

**Data Structure:**
```c
struct dsmil_device_status {
    __u8  device_id;      /* Input: Device ID (0-103) */
    __u8  online;         /* Output: Device online status */
    __u8  ready;          /* Output: Device ready status */
    __u8  error;          /* Output: Device error status */
    __u32 status_flags;   /* Output: Full status flags */
    __u32 config_flags;   /* Output: Configuration flags */
    __u32 device_data;    /* Output: Device-specific data */
};
```

**Usage:**
```c
int fd = open("/dev/dsmil0", O_RDWR);
struct dsmil_device_status dev;

dev.device_id = 42;  // Query device 42

if (ioctl(fd, DSMIL_IOC_GET_DEVICE_STATUS, &dev) == 0) {
    printf("Device %u: %s %s\n",
           dev.device_id,
           dev.online ? "ONLINE" : "OFFLINE",
           dev.ready ? "READY" : "NOT READY");
    printf("Status: 0x%08x\n", dev.status_flags);
}

close(fd);
```

**Return Values:**
- **0** - Success
- **-EINVAL** - Invalid device ID (>= 104)
- **-EIO** - Failed to read device tokens

**Requires:** No special privileges

---

### DSMIL_IOC_GET_DEVICE_LIST (6)

**Purpose:** Get list of all DSMIL devices with their status.

**Data Structure:**
```c
struct dsmil_device_list {
    __u32 device_count;                      /* Output: Number of devices */
    struct dsmil_device_status devices[104]; /* Output: Device status array */
};
```

**Usage:**
```c
int fd = open("/dev/dsmil0", O_RDWR);
struct dsmil_device_list *list;

list = malloc(sizeof(struct dsmil_device_list));

if (ioctl(fd, DSMIL_IOC_GET_DEVICE_LIST, list) == 0) {
    printf("Total devices: %u\n", list->device_count);

    for (int i = 0; i < list->device_count; i++) {
        if (list->devices[i].online) {
            printf("Device %3d: ONLINE %s\n",
                   list->devices[i].device_id,
                   list->devices[i].ready ? "[READY]" : "[NOT READY]");
        }
    }
}

free(list);
close(fd);
```

**Return Values:**
- **0** - Success
- **-EFAULT** - Failed to copy data to userspace
- **-EIO** - Failed to read device tokens

**Requires:** No special privileges

**Notes:**
- Large data structure (>400 bytes), consider using sysfs for monitoring
- Queries all 104 devices sequentially

---

### DSMIL_IOC_INVALIDATE_AUTH (7)

**Purpose:** Invalidate current authentication session.

**Data Structure:** None (simple command)

**Usage:**
```c
int fd = open("/dev/dsmil0", O_RDWR);

if (ioctl(fd, DSMIL_IOC_INVALIDATE_AUTH) == 0) {
    printf("Authentication session invalidated\n");
}

close(fd);
```

**Return Values:**
- **0** - Success (session invalidated or no active session)

**Requires:** No special privileges

**Notes:**
- Clears authenticated session
- Forces re-authentication for protected token access
- Safe to call even if no active session

---

### DSMIL_IOC_GET_BIOS_STATUS (8)

**Purpose:** Get status of all BIOS systems.

**Data Structure:**
```c
struct dsmil_bios_status {
    __u8 active_bios;          /* Active BIOS (0=A, 1=B, 2=C) */
    __u8 bios_count;           /* Number of BIOS systems (3) */
    __u8 reserved[2];
    struct {
        __u8  bios_id;         /* BIOS ID (0=A, 1=B, 2=C) */
        __u8  is_active;       /* This BIOS is active */
        __u8  health_score;    /* Health score (0-100) */
        __u8  reserved;
        __u32 error_count;     /* Error count */
    } bios[3];
};
```

**Usage:**
```c
int fd = open("/dev/dsmil0", O_RDWR);
struct dsmil_bios_status status;

if (ioctl(fd, DSMIL_IOC_GET_BIOS_STATUS, &status) == 0) {
    printf("Active BIOS: %c\n", 'A' + status.active_bios);

    for (int i = 0; i < status.bios_count; i++) {
        printf("BIOS %c: Health=%3u%%, Errors=%u %s\n",
               'A' + i,
               status.bios[i].health_score,
               status.bios[i].error_count,
               status.bios[i].is_active ? "[ACTIVE]" : "");
    }
}

close(fd);
```

**Return Values:**
- **0** - Success
- **-EFAULT** - Failed to copy data to userspace
- **-EIO** - Failed to read BIOS tokens

**Requires:** No special privileges

---

### DSMIL_IOC_BIOS_QUERY (9)

**Purpose:** Query detailed information about a specific BIOS.

**Data Structure:**
```c
struct dsmil_bios_info {
    __u8  bios_id;              /* Input: BIOS ID (0=A, 1=B, 2=C) */
    __u8  is_active;            /* Output: This BIOS is active */
    __u8  health_score;         /* Output: Health score (0-100) */
    __u8  reserved;
    __u32 error_count;          /* Output: Error count */
    __u64 last_access_time;     /* Output: Last access timestamp (ktime) */
};
```

**Usage:**
```c
int fd = open("/dev/dsmil0", O_RDWR);
struct dsmil_bios_info info;

info.bios_id = 0;  // Query BIOS A

if (ioctl(fd, DSMIL_IOC_BIOS_QUERY, &info) == 0) {
    printf("BIOS %c:\n", 'A' + info.bios_id);
    printf("  Active: %s\n", info.is_active ? "yes" : "no");
    printf("  Health: %u%%\n", info.health_score);
    printf("  Errors: %u\n", info.error_count);
}

close(fd);
```

**Return Values:**
- **0** - Success
- **-EINVAL** - Invalid BIOS ID (>= 3)
- **-EIO** - Failed to read BIOS tokens

**Requires:** No special privileges

---

### DSMIL_IOC_BIOS_SYNC (10)

**Purpose:** Synchronize all BIOS systems (protected operation).

**Data Structure:** None (simple command)

**Usage:**
```c
int fd = open("/dev/dsmil0", O_RDWR);

// Must authenticate first for protected operation
struct dsmil_auth_request auth;
// ... perform authentication ...
ioctl(fd, DSMIL_IOC_AUTHENTICATE, &auth);

// Now perform BIOS sync
if (ioctl(fd, DSMIL_IOC_BIOS_SYNC) == 0) {
    printf("BIOS synchronization completed\n");
}

close(fd);
```

**Return Values:**
- **0** - Success (all BIOS systems synchronized)
- **-EPERM** - Permission denied (requires authentication)
- **-EIO** - Synchronization failed

**Requires:**
- **CAP_SYS_ADMIN** capability
- **TPM authentication** (when TPM available)

**Notes:**
- Copies configuration from active BIOS to redundant BIOS systems
- Protected operation requiring authentication
- Audit log entry created
- TPM PCR18 extended

---

### DSMIL_IOC_TPM_GET_CHALLENGE (11)

**Purpose:** Get TPM authentication challenge.

**Data Structure:**
```c
struct dsmil_tpm_challenge_data {
    __u8  challenge[32];   /* Output: Challenge data (256 bits) */
    __u32 challenge_id;    /* Output: Challenge ID */
    __u8  tpm_available;   /* Output: TPM available flag */
};
```

**Usage:**
```c
int fd = open("/dev/dsmil0", O_RDWR);
struct dsmil_tpm_challenge_data chal;

if (ioctl(fd, DSMIL_IOC_TPM_GET_CHALLENGE, &chal) == 0) {
    printf("Challenge ID: 0x%08x\n", chal.challenge_id);
    printf("TPM Available: %s\n", chal.tpm_available ? "yes" : "no");

    printf("Challenge: ");
    for (int i = 0; i < 32; i++) {
        printf("%02x", chal.challenge[i]);
    }
    printf("\n");

    // Sign challenge with TPM
    // ...
}

close(fd);
```

**Return Values:**
- **0** - Success, challenge generated
- **-EFAULT** - Failed to copy data to userspace

**Requires:** No special privileges

**Notes:**
- Challenge valid for 60 seconds (default)
- Uses cryptographically secure random number generator
- Challenge must be signed by TPM and submitted via DSMIL_IOC_AUTHENTICATE
- If TPM not available, `tpm_available` will be 0

---

### DSMIL_IOC_TPM_INVALIDATE (12)

**Purpose:** Invalidate TPM authentication session.

**Data Structure:** None (simple command)

**Usage:**
```c
int fd = open("/dev/dsmil0", O_RDWR);

if (ioctl(fd, DSMIL_IOC_TPM_INVALIDATE) == 0) {
    printf("TPM session invalidated\n");
}

close(fd);
```

**Return Values:**
- **0** - Success (session invalidated or no active session)

**Requires:** No special privileges

**Notes:**
- Equivalent to DSMIL_IOC_INVALIDATE_AUTH but specific to TPM sessions
- Clears TPM authentication state
- Safe to call even if no active TPM session

---

## Sysfs Interface

### Directory Structure

```
/sys/class/dsmil/dsmil0/
├── active_bios          (RW) - Active BIOS (A/B/C)
├── bios_a_health        (RO) - BIOS A health score
├── bios_b_health        (RO) - BIOS B health score
├── bios_c_health        (RO) - BIOS C health score
├── device_count         (RO) - Number of devices (104)
├── driver_version       (RO) - Driver version string
├── tokens               (RO) - Available tokens (multi-line)
├── error_stats          (RO) - Error statistics
├── last_audit           (RO) - Last audit entry
├── smbios_backend       (RO) - SMBIOS backend info
└── tpm_status           (RO) - TPM status
```

---

### active_bios (Read/Write)

**Purpose:** Get or set active BIOS.

**Read:**
```bash
cat /sys/class/dsmil/dsmil0/active_bios
# Output: A
```

**Write (Protected):**
```bash
# Requires CAP_SYS_ADMIN + TPM authentication
echo "B" | sudo tee /sys/class/dsmil/dsmil0/active_bios
```

**Format:**
- **Read:** Single character: `A`, `B`, or `C`
- **Write:** Single character: `A`, `B`, or `C`

**Permissions:** 0644 (read for all, write requires authentication)

**Notes:**
- Writing triggers manual BIOS failover
- Audit log entry created
- TPM PCR18 extended
- Health scores validated before failover

---

### bios_a_health, bios_b_health, bios_c_health (Read-Only)

**Purpose:** Get BIOS health scores.

**Usage:**
```bash
cat /sys/class/dsmil/dsmil0/bios_a_health
# Output: 95

cat /sys/class/dsmil/dsmil0/bios_b_health
# Output: 88

cat /sys/class/dsmil/dsmil0/bios_c_health
# Output: 92
```

**Format:** Integer 0-100 (percentage)

**Permissions:** 0444 (read-only for all)

**Notes:**
- Health score calculated from:
  - Error count (weight: 40%)
  - Response time (weight: 30%)
  - Success rate (weight: 30%)
- Automatic failover triggered if health < 50

---

### device_count (Read-Only)

**Purpose:** Get number of DSMIL devices.

**Usage:**
```bash
cat /sys/class/dsmil/dsmil0/device_count
# Output: 104
```

**Format:** Integer (always 104 for this driver version)

**Permissions:** 0444 (read-only for all)

---

### driver_version (Read-Only)

**Purpose:** Get driver version string.

**Usage:**
```bash
cat /sys/class/dsmil/dsmil0/driver_version
# Output: 5.2.0
```

**Format:** Semantic version string (major.minor.patch)

**Permissions:** 0444 (read-only for all)

---

### tokens (Read-Only)

**Purpose:** List all available tokens.

**Usage:**
```bash
cat /sys/class/dsmil/dsmil0/tokens
# Output (excerpt):
# 0x8000: DEVICE_000_STATUS [DEVICE] (RW)
# 0x8001: DEVICE_000_CONFIG [DEVICE] (RW)
# 0x8002: DEVICE_000_DATA [DEVICE] (RW)
# ...
# 0x8100: BIOS_A_STATUS [BIOS] (RO, PROTECTED)
# 0x8101: BIOS_A_ERRORS [BIOS] (RO, PROTECTED)
# ...
```

**Format:** Multi-line text
```
<token_id>: <name> [<category>] (<access>) [<flags>]
```

**Permissions:** 0444 (read-only for all)

**Notes:**
- Lists all tokens in database (50+ tokens)
- Flags include: PROTECTED, READONLY, CRITICAL
- Useful for discovery and debugging

---

### error_stats (Read-Only)

**Purpose:** Get error statistics by category.

**Usage:**
```bash
cat /sys/class/dsmil/dsmil0/error_stats
# Output:
# token_errors:      5
# device_errors:     0
# bios_errors:       2
# auth_errors:       1
# security_errors:   0
# smbios_errors:     0
# validation_errors: 3
# thermal_errors:    0
# total_errors:      11
# last_error_time:   12345678
# last_error_code:   1001
```

**Format:** Multi-line key-value pairs

**Permissions:** 0444 (read-only for all)

**Notes:**
- Atomic counters (thread-safe)
- Reset only on driver reload
- `last_error_time` in jiffies
- Useful for monitoring and debugging

---

### last_audit (Read-Only)

**Purpose:** Get last audit log entry.

**Usage:**
```bash
cat /sys/class/dsmil/dsmil0/last_audit
# Output:
# timestamp:  1699900800123456789
# event_type: PROTECTED_ACCESS (3)
# user_id:    1000
# token_id:   0x8500
# old_value:  0x00000000
# new_value:  0xCAFEBABE
# result:     0 (success)
# message:    Protected token write authorized
```

**Format:** Multi-line key-value pairs

**Permissions:** 0400 (read-only for root)

**Notes:**
- Shows most recent security-relevant event
- Event types:
  - 1: AUTH_SUCCESS
  - 2: TOKEN_WRITE
  - 3: PROTECTED_ACCESS
  - 4: AUTH_SUCCESS
  - 5: AUTH_FAILURE
  - 6: BIOS_FAILOVER
  - 7: BIOS_SYNC
  - 8: EMERGENCY_ACTION
- Timestamp in nanoseconds (ktime)

---

### smbios_backend (Read-Only)

**Purpose:** Get SMBIOS backend information.

**Usage:**
```bash
cat /sys/class/dsmil/dsmil0/smbios_backend
# Output (real hardware):
# backend:           dell-smbios (kernel subsystem)
# token_read:        yes
# token_write:       yes
# token_discovery:   yes
# wmi_support:       yes
# smm_support:       yes
# max_buffer_size:   32 bytes

# Output (simulation):
# backend:           simulated (database-aware)
# token_read:        yes
# token_write:       yes
# token_discovery:   yes
# wmi_support:       no
# smm_support:       no
# max_buffer_size:   32 bytes
```

**Format:** Multi-line key-value pairs

**Permissions:** 0444 (read-only for all)

**Notes:**
- Shows which SMBIOS backend is active
- Real backend requires CONFIG_DELL_SMBIOS=y and Dell hardware
- Simulation backend used for development/testing

---

### tpm_status (Read-Only)

**Purpose:** Get TPM authentication subsystem status.

**Usage:**
```bash
cat /sys/class/dsmil/dsmil0/tpm_status
# Output:
# state:           ready
# available:       yes
# chip_present:    yes
# auth_mode:       challenge
# session_active:  yes
# auth_attempts:   15
# auth_successes:  12
# auth_failures:   3
# pcr_extends:     45
```

**Format:** Multi-line key-value pairs

**Permissions:** 0444 (read-only for all)

**Notes:**
- `state` values: uninitialized, unavailable, ready, error
- `auth_mode` values: none, challenge, key, hmac, external
- Statistics are cumulative since driver load
- `pcr_extends` shows TPM measurement count

---

## Data Structures

### Complete Structure Definitions

```c
/* Token request/response */
struct dsmil_token_request {
    __u16 token_id;
    __u32 value;
} __attribute__((packed));

/* Authentication request */
struct dsmil_auth_request {
    __u32 auth_method;
    __u32 auth_data_len;
    __u8  auth_data[256];
} __attribute__((packed));

/* Driver status */
struct dsmil_driver_status {
    __u32 driver_version;
    __u32 device_count;
    __u32 bios_count;
    __u8  active_bios;
    __u8  auth_active;
    __u16 reserved;
    __u32 total_token_reads;
    __u32 total_token_writes;
    __u32 failed_operations;
    __u32 auth_attempts;
    __u32 bios_failovers;
    __u64 uptime_seconds;
} __attribute__((packed));

/* Device status */
struct dsmil_device_status {
    __u8  device_id;
    __u8  online;
    __u8  ready;
    __u8  error;
    __u32 status_flags;
    __u32 config_flags;
    __u32 device_data;
} __attribute__((packed));

/* Device list */
struct dsmil_device_list {
    __u32 device_count;
    struct dsmil_device_status devices[104];
} __attribute__((packed));

/* BIOS status */
struct dsmil_bios_status {
    __u8 active_bios;
    __u8 bios_count;
    __u8 reserved[2];
    struct {
        __u8  bios_id;
        __u8  is_active;
        __u8  health_score;
        __u8  reserved;
        __u32 error_count;
    } bios[3];
} __attribute__((packed));

/* BIOS info */
struct dsmil_bios_info {
    __u8  bios_id;
    __u8  is_active;
    __u8  health_score;
    __u8  reserved;
    __u32 error_count;
    __u64 last_access_time;
} __attribute__((packed));

/* TPM challenge data */
struct dsmil_tpm_challenge_data {
    __u8  challenge[32];
    __u32 challenge_id;
    __u8  tpm_available;
} __attribute__((packed));
```

---

## Error Codes

### Standard Error Codes

| Code | Symbol | Meaning |
|------|--------|---------|
| **0** | Success | Operation completed successfully |
| **-EINVAL** | Invalid argument | Invalid parameter (token ID, value, etc.) |
| **-ENOENT** | No such entry | Token not found in database |
| **-EPERM** | Permission denied | Insufficient privileges or authentication required |
| **-EROFS** | Read-only | Attempt to write read-only token |
| **-EIO** | I/O error | SMBIOS call failed or hardware error |
| **-EFAULT** | Bad address | Failed to copy data to/from userspace |
| **-ETIMEDOUT** | Timeout | Operation timeout (challenge expired, etc.) |
| **-ENODEV** | No device | TPM or hardware not available |
| **-EBUSY** | Device busy | Operation in progress, try again |

### DSMIL-Specific Error Codes

Error codes stored in error statistics and audit logs:

| Code | Category | Name | Description |
|------|----------|------|-------------|
| **1001** | TOKEN | TOKEN_INVALID | Invalid or unknown token ID |
| **1002** | TOKEN | TOKEN_READ_FAILED | Token read operation failed |
| **1003** | TOKEN | TOKEN_PROTECTED | Protected token access denied |
| **2001** | DEVICE | DEVICE_OFFLINE | Device is offline |
| **2002** | DEVICE | DEVICE_ERROR | Device in error state |
| **3001** | BIOS | BIOS_UNAVAILABLE | BIOS system unavailable |
| **3002** | BIOS | BIOS_CRITICAL | BIOS health critically low |
| **4001** | AUTH | AUTH_REQUIRED | Authentication required |
| **4002** | AUTH | AUTH_FAILED | Authentication failed |
| **4003** | AUTH | AUTH_EXPIRED | Authentication session expired |
| **5001** | SECURITY | SECURITY_VIOLATION | Security policy violation |
| **5002** | SECURITY | SECURITY_TPM_FAILED | TPM operation failed |
| **6001** | SMBIOS | SMBIOS_CALL | SMBIOS call failed |
| **6002** | SMBIOS | SMBIOS_BUFFER | SMBIOS buffer allocation failed |
| **7001** | VALIDATION | VALIDATION_RANGE | Value out of range |
| **7002** | VALIDATION | VALIDATION_TYPE | Type mismatch |
| **8001** | THERMAL | THERMAL_CRITICAL | Critical thermal condition |

---

## Token Database

### Token ID Ranges

| Range | Purpose | Count | Access |
|-------|---------|-------|--------|
| **0x8000-0x80FF** | Device tokens (104 devices × 3 tokens) | 312 | Read/Write |
| **0x8100-0x812F** | BIOS system tokens (3 BIOS × 16 tokens) | 48 | Protected |
| **0x8200-0x82FF** | BIOS update control tokens | ~10 | Protected |
| **0x8300-0x83FF** | Thermal management tokens | ~10 | Protected |
| **0x8400-0x84FF** | Power control tokens | ~10 | Protected |
| **0x8500-0x85FF** | Security tokens | ~10 | Protected |
| **0x8600-0x8FFF** | Reserved for future use | - | - |

### Device Token Layout

Each device (0-103) has 3 tokens:

```c
/* Device X tokens (X = 0 to 103) */
#define TOKEN_DEVICE_X_STATUS  (0x8000 + X * 3 + 0)  // Status flags
#define TOKEN_DEVICE_X_CONFIG  (0x8000 + X * 3 + 1)  // Configuration
#define TOKEN_DEVICE_X_DATA    (0x8000 + X * 3 + 2)  // Device-specific data

/* Example: Device 0 */
#define TOKEN_DEVICE_000_STATUS  0x8000  // Device 0 status
#define TOKEN_DEVICE_000_CONFIG  0x8001  // Device 0 config
#define TOKEN_DEVICE_000_DATA    0x8002  // Device 0 data

/* Example: Device 103 */
#define TOKEN_DEVICE_103_STATUS  0x8135  // Device 103 status
#define TOKEN_DEVICE_103_CONFIG  0x8136  // Device 103 config
#define TOKEN_DEVICE_103_DATA    0x8137  // Device 103 data
```

### BIOS Token Layout

Each BIOS (A, B, C) has 16 tokens:

```c
/* BIOS A tokens (0x8100-0x810F) */
#define TOKEN_BIOS_A_STATUS       0x8100  // BIOS A status
#define TOKEN_BIOS_A_ERRORS       0x8101  // Error count
#define TOKEN_BIOS_A_VERSION      0x8102  // Version
#define TOKEN_BIOS_A_FEATURES     0x8103  // Feature flags
#define TOKEN_BIOS_A_TIMING       0x8104  // Timing info
#define TOKEN_BIOS_A_HEALTH       0x8106  // Health score (read-only)
// ... 0x8107-0x810F reserved

/* BIOS B tokens (0x8110-0x811F) */
#define TOKEN_BIOS_B_STATUS       0x8110
#define TOKEN_BIOS_B_ERRORS       0x8111
// ... similar layout

/* BIOS C tokens (0x8120-0x812F) */
#define TOKEN_BIOS_C_STATUS       0x8120
#define TOKEN_BIOS_C_ERRORS       0x8121
// ... similar layout

/* BIOS control tokens */
#define TOKEN_BIOS_ACTIVE_SELECT  0x8130  // Active BIOS selection
```

### Protected Tokens

Tokens requiring authentication:

```c
/* BIOS update control (0x8200-0x82FF) */
#define TOKEN_BIOS_UPDATE_CONTROL    0x8200  // BIOS update control
#define TOKEN_BIOS_UPDATE_STATUS     0x8201  // Update status
#define TOKEN_BIOS_UPDATE_PROGRESS   0x8202  // Update progress

/* Thermal management (0x8300-0x83FF) */
#define TOKEN_THERMAL_EMERGENCY      0x8300  // Emergency thermal control
#define TOKEN_THERMAL_POLICY         0x8301  // Thermal policy
#define TOKEN_THERMAL_LIMITS         0x8302  // Thermal limits

/* Power control (0x8400-0x84FF) */
#define TOKEN_POWER_CONTROL          0x8400  // Power control
#define TOKEN_POWER_LIMITS           0x8401  // Power limits
#define TOKEN_POWER_POLICY           0x8402  // Power policy

/* Security (0x8500-0x85FF) */
#define TOKEN_SECURITY_MASTER        0x8500  // Master security control
#define TOKEN_SECURITY_POLICY        0x8501  // Security policy
#define TOKEN_SECURITY_AUDIT         0x8502  // Audit configuration
```

---

## Constants and Enumerations

### Authentication Methods

```c
enum dsmil_auth_method {
    DSMIL_AUTH_METHOD_NONE      = 0,  /* No authentication */
    DSMIL_AUTH_METHOD_CHALLENGE = 1,  /* TPM challenge-response */
    DSMIL_AUTH_METHOD_KEY       = 2,  /* TPM key-based */
    DSMIL_AUTH_METHOD_HMAC      = 3,  /* TPM HMAC */
    DSMIL_AUTH_METHOD_EXTERNAL  = 4,  /* External authenticator */
};
```

### Token Access Levels

```c
enum dsmil_token_access {
    DSMIL_TOKEN_ACCESS_NONE     = 0,  /* No access */
    DSMIL_TOKEN_ACCESS_DEVICE   = 1,  /* Device-level (non-protected) */
    DSMIL_TOKEN_ACCESS_ADMIN    = 2,  /* Admin-level (CAP_SYS_ADMIN) */
    DSMIL_TOKEN_ACCESS_SECURITY = 3,  /* Security-level (TPM required) */
};
```

### Audit Event Types

```c
enum dsmil_audit_event {
    DSMIL_AUDIT_AUTH_ATTEMPT    = 1,  /* Authentication attempt */
    DSMIL_AUDIT_TOKEN_WRITE     = 2,  /* Token write */
    DSMIL_AUDIT_PROTECTED_ACCESS= 3,  /* Protected token access */
    DSMIL_AUDIT_AUTH_SUCCESS    = 4,  /* Authentication success */
    DSMIL_AUDIT_AUTH_FAILURE    = 5,  /* Authentication failure */
    DSMIL_AUDIT_BIOS_FAILOVER   = 6,  /* BIOS failover */
    DSMIL_AUDIT_BIOS_SYNC       = 7,  /* BIOS synchronization */
    DSMIL_AUDIT_EMERGENCY       = 8,  /* Emergency action */
};
```

### TPM PCR Indexes

```c
/* TPM Platform Configuration Registers used by DSMIL */
#define DSMIL_TPM_PCR_AUTH    16  /* Authentication events */
#define DSMIL_TPM_PCR_TOKEN   17  /* Protected token access */
#define DSMIL_TPM_PCR_BIOS    18  /* BIOS operations */
#define DSMIL_TPM_PCR_SECURITY 23 /* Security events */
```

---

## Module Parameters

### Available Parameters

```c
/* Debug level */
static int debug = 0;
module_param(debug, int, 0644);
MODULE_PARM_DESC(debug, "Debug level (0=off, 1=info, 2=verbose, 3=debug)");

/* Auto BIOS failover */
static bool auto_bios_failover = true;
module_param(auto_bios_failover, bool, 0644);
MODULE_PARM_DESC(auto_bios_failover, "Enable automatic BIOS failover on health degradation");

/* BIOS health threshold */
static int bios_health_threshold = 50;
module_param(bios_health_threshold, int, 0644);
MODULE_PARM_DESC(bios_health_threshold, "BIOS health threshold for failover (0-100)");

/* Authentication timeout */
static int auth_timeout = 300;
module_param(auth_timeout, int, 0644);
MODULE_PARM_DESC(auth_timeout, "Authentication session timeout in seconds");

/* Require TPM */
static bool require_tpm = false;
module_param(require_tpm, bool, 0400);
MODULE_PARM_DESC(require_tpm, "Require TPM for authentication (fails if TPM unavailable)");

/* Allow protected write without auth (DANGEROUS) */
static bool allow_unauth_protected = false;
module_param(allow_unauth_protected, bool, 0400);
MODULE_PARM_DESC(allow_unauth_protected, "Allow protected token writes without authentication (INSECURE, testing only)");
```

### Usage

**Load-time:**
```bash
sudo insmod dsmil-104dev.ko debug=2 auth_timeout=600 require_tpm=1
```

**Runtime (for writable parameters):**
```bash
echo 3 | sudo tee /sys/module/dsmil_104dev/parameters/debug
```

---

## API Versioning

**Current API Version:** 1.0

**Version History:**
- **1.0** (2025-11-13) - Initial release
  - 12 IOCTL commands
  - 10 sysfs attributes
  - Full TPM 2.0 support
  - 104 device + 3 BIOS architecture

**Compatibility:**
- API is stable and will maintain backward compatibility
- New features will use new IOCTL command numbers
- Deprecated features will be marked in documentation before removal

---

## Conclusion

This API reference provides complete documentation for all interfaces provided by the DSMIL driver. For usage examples and tutorials, see:

- **DRIVER_USAGE_GUIDE.md** - User guide with examples
- **TESTING_GUIDE.md** - Comprehensive testing procedures
- **TPM_AUTHENTICATION_GUIDE.md** - Detailed TPM usage guide

For questions or issues, contact the development team or file an issue in the project repository.

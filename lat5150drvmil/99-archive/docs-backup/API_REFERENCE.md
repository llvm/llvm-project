# DSMIL Control System - API Reference

## Overview

This document provides comprehensive API documentation for the DSMIL Control System, including kernel module interfaces, security framework APIs, web service endpoints, and database schemas. All interfaces implement military-grade security with comprehensive audit logging.

## Kernel Module API

### Core Device Interface

#### Device Structure
```c
struct dsmil_device {
    uint16_t token_id;          // Device token (0x8000-0x806B)
    char     name[32];          // Device name
    uint8_t  access_level;      // Required security clearance
    uint32_t capabilities;      // Device capability flags
    bool     quarantined;       // Quarantine status
    uint64_t last_accessed;     // Last access timestamp
    struct   dsmil_ops *ops;    // Device operation structure
    void     *private_data;     // Device-specific data
};

struct dsmil_ops {
    int (*probe)(struct dsmil_device *dev);
    int (*remove)(struct dsmil_device *dev);
    int (*read)(struct dsmil_device *dev, void *buf, size_t len, loff_t *offset);
    int (*write)(struct dsmil_device *dev, const void *buf, size_t len, loff_t *offset);
    long (*ioctl)(struct dsmil_device *dev, unsigned int cmd, unsigned long arg);
    int (*suspend)(struct dsmil_device *dev);
    int (*resume)(struct dsmil_device *dev);
};
```

#### Device Registration API
```c
/**
 * dsmil_register_device - Register a new DSMIL device
 * @dev: Device structure to register
 * @ops: Device operations structure
 * 
 * Returns: 0 on success, negative error code on failure
 */
int dsmil_register_device(struct dsmil_device *dev, 
                         const struct dsmil_ops *ops);

/**
 * dsmil_unregister_device - Unregister a DSMIL device
 * @dev: Device to unregister
 */
void dsmil_unregister_device(struct dsmil_device *dev);

/**
 * dsmil_get_device - Get device by token ID
 * @token_id: Device token (0x8000-0x806B)
 * 
 * Returns: Device pointer or NULL if not found
 */
struct dsmil_device *dsmil_get_device(uint16_t token_id);
```

### SMI Interface API

#### SMI Command Structure
```c
#define DSMIL_SMI_CMD_PORT     0x164E
#define DSMIL_SMI_DATA_PORT    0x164F

struct smi_command {
    uint16_t device_id;        // Target device token
    uint8_t  operation;        // Operation type
    uint8_t  flags;           // Operation flags
    uint32_t data_length;     // Data payload size
    uint64_t data_address;    // Data buffer address
    uint32_t timeout_ms;      // Operation timeout
    uint32_t checksum;        // Command integrity check
} __attribute__((packed));

// SMI Operation Types
#define SMI_OP_READ         1
#define SMI_OP_WRITE        2
#define SMI_OP_IOCTL        3
#define SMI_OP_STATUS       4
#define SMI_OP_RESET        5

// SMI Flags
#define SMI_FLAG_URGENT     0x01
#define SMI_FLAG_ENCRYPTED  0x02
#define SMI_FLAG_SIGNED     0x04
#define SMI_FLAG_COMPRESSED 0x08
```

#### SMI Interface Functions
```c
/**
 * dsmil_smi_command - Execute SMI command
 * @cmd: SMI command structure
 * @response: Response buffer
 * @response_size: Size of response buffer
 * 
 * Returns: Number of bytes received, or negative error code
 */
int dsmil_smi_command(const struct smi_command *cmd, 
                     void *response, size_t response_size);

/**
 * dsmil_smi_read - Read data from device via SMI
 * @device_id: Target device token
 * @offset: Read offset within device
 * @buffer: Data buffer
 * @length: Number of bytes to read
 * 
 * Returns: Number of bytes read, or negative error code
 */
int dsmil_smi_read(uint16_t device_id, uint32_t offset,
                  void *buffer, size_t length);

/**
 * dsmil_smi_write - Write data to device via SMI
 * @device_id: Target device token
 * @offset: Write offset within device
 * @buffer: Data to write
 * @length: Number of bytes to write
 * 
 * Returns: Number of bytes written, or negative error code
 */
int dsmil_smi_write(uint16_t device_id, uint32_t offset,
                   const void *buffer, size_t length);
```

### Safety Validation API

#### Safety Check Functions
```c
/**
 * dsmil_validate_access - Validate device access permissions
 * @device_id: Target device token
 * @operation: Requested operation
 * @user_clearance: User security clearance level
 * 
 * Returns: 0 if access permitted, negative error code if denied
 */
int dsmil_validate_access(uint16_t device_id, uint8_t operation,
                         enum dsmil_clearance user_clearance);

/**
 * dsmil_is_quarantined - Check if device is quarantined
 * @device_id: Device token to check
 * 
 * Returns: true if quarantined, false otherwise
 */
bool dsmil_is_quarantined(uint16_t device_id);

/**
 * dsmil_emergency_stop - Execute emergency stop
 * @level: Emergency level (device, group, or system-wide)
 * @device_id: Target device (0 for system-wide)
 * 
 * Returns: 0 on success, negative error code on failure
 */
int dsmil_emergency_stop(enum emergency_level level, uint16_t device_id);
```

### IOCTL Interface

#### IOCTL Commands
```c
#define DSMIL_IOC_MAGIC 'D'

#define DSMIL_IOC_GET_STATUS        _IOR(DSMIL_IOC_MAGIC, 1, struct dsmil_status)
#define DSMIL_IOC_SET_MODE          _IOW(DSMIL_IOC_MAGIC, 2, int)
#define DSMIL_IOC_GET_DEVICE_INFO   _IOWR(DSMIL_IOC_MAGIC, 3, struct device_info)
#define DSMIL_IOC_EMERGENCY_STOP    _IOW(DSMIL_IOC_MAGIC, 4, int)
#define DSMIL_IOC_GET_QUARANTINE    _IOR(DSMIL_IOC_MAGIC, 5, struct quarantine_status)
#define DSMIL_IOC_VALIDATE_ACCESS   _IOWR(DSMIL_IOC_MAGIC, 6, struct access_request)
#define DSMIL_IOC_GET_AUDIT_LOG     _IOWR(DSMIL_IOC_MAGIC, 7, struct audit_query)

struct dsmil_status {
    uint32_t module_version;    // Kernel module version
    uint32_t device_count;      // Total registered devices
    uint32_t active_devices;    // Currently active devices
    uint32_t quarantined_count; // Number of quarantined devices
    uint64_t uptime;           // Module uptime in milliseconds
    uint32_t error_count;      // Total error count
};

struct device_info {
    uint16_t token_id;         // Device token
    char     name[32];         // Device name
    uint8_t  access_level;     // Required clearance
    uint32_t capabilities;     // Capability flags
    bool     quarantined;      // Quarantine status
    uint64_t last_accessed;    // Last access time
    uint32_t access_count;     // Total access count
    uint32_t error_count;      // Device error count
};
```

#### IOCTL Usage Example
```c
#include <sys/ioctl.h>
#include <fcntl.h>

int fd = open("/dev/dsmil", O_RDWR);
if (fd < 0) {
    perror("Failed to open DSMIL device");
    return -1;
}

// Get system status
struct dsmil_status status;
if (ioctl(fd, DSMIL_IOC_GET_STATUS, &status) == 0) {
    printf("DSMIL Module Version: %u\n", status.module_version);
    printf("Active Devices: %u/%u\n", status.active_devices, status.device_count);
    printf("Quarantined Devices: %u\n", status.quarantined_count);
}

// Get device information
struct device_info info;
info.token_id = 0x8005;  // TPM Interface Controller
if (ioctl(fd, DSMIL_IOC_GET_DEVICE_INFO, &info) == 0) {
    printf("Device: %s (0x%04X)\n", info.name, info.token_id);
    printf("Quarantined: %s\n", info.quarantined ? "YES" : "NO");
}

close(fd);
```

## Security Framework API

### Authentication API

#### Multi-Factor Authentication
```c
enum dsmil_clearance {
    CLEARANCE_UNCLASSIFIED = 0,
    CLEARANCE_CONFIDENTIAL = 1,
    CLEARANCE_SECRET       = 2,
    CLEARANCE_TOP_SECRET   = 3,
    CLEARANCE_SCI          = 4
};

struct mfa_context {
    char user_id[32];          // User identifier
    enum dsmil_clearance level; // Required clearance level
    uint8_t biometric_hash[32]; // Biometric authentication hash
    uint32_t totp_code;        // Time-based one-time password
    uint64_t valid_until;      // Authentication validity period
    uint8_t signature[64];     // Authentication signature
};

/**
 * dsmil_authenticate_user - Authenticate user with MFA
 * @ctx: MFA authentication context
 * 
 * Returns: 0 on success, negative error code on failure
 */
int dsmil_authenticate_user(const struct mfa_context *ctx);

/**
 * dsmil_validate_clearance - Validate user clearance level
 * @user_id: User identifier
 * @required_level: Required clearance level
 * 
 * Returns: 0 if clearance sufficient, negative error code otherwise
 */
int dsmil_validate_clearance(const char *user_id, 
                            enum dsmil_clearance required_level);
```

### Audit Framework API

#### Audit Logging
```c
enum audit_event {
    AUDIT_DEVICE_ACCESS     = 1,
    AUDIT_AUTHENTICATION    = 2,
    AUDIT_QUARANTINE_ACCESS = 3,
    AUDIT_EMERGENCY_STOP    = 4,
    AUDIT_SECURITY_VIOLATION = 5,
    AUDIT_CONFIGURATION     = 6,
    AUDIT_ERROR            = 7
};

struct audit_entry {
    uint64_t timestamp;        // Event timestamp (nanoseconds)
    uint32_t sequence_number;  // Sequential entry number
    enum audit_event event;    // Event type
    uint16_t device_id;       // Associated device (if applicable)
    char user_id[32];         // User identifier
    uint8_t operation;        // Operation performed
    uint8_t result;          // Operation result
    char details[256];        // Additional event details
    uint8_t signature[64];    // Cryptographic signature
    uint32_t checksum;        // Entry integrity checksum
} __attribute__((packed));

/**
 * dsmil_audit_log - Log security event
 * @event: Event type
 * @device_id: Associated device (0 if not device-specific)
 * @user_id: User identifier
 * @details: Additional event details
 * 
 * Returns: 0 on success, negative error code on failure
 */
int dsmil_audit_log(enum audit_event event, uint16_t device_id,
                   const char *user_id, const char *details);

/**
 * dsmil_audit_query - Query audit log entries
 * @start_time: Start timestamp for query
 * @end_time: End timestamp for query
 * @buffer: Buffer for audit entries
 * @buffer_size: Size of buffer
 * 
 * Returns: Number of entries retrieved, or negative error code
 */
int dsmil_audit_query(uint64_t start_time, uint64_t end_time,
                     struct audit_entry *buffer, size_t buffer_size);
```

### Threat Detection API

#### Threat Detection Engine
```c
enum threat_level {
    THREAT_LEVEL_LOW    = 1,
    THREAT_LEVEL_MEDIUM = 2,
    THREAT_LEVEL_HIGH   = 3,
    THREAT_LEVEL_CRITICAL = 4
};

struct threat_event {
    uint64_t timestamp;        // Detection timestamp
    enum threat_level level;   // Threat severity level
    uint16_t device_id;       // Associated device
    char threat_type[32];     // Type of threat detected
    char description[256];    // Threat description
    uint32_t confidence;      // Detection confidence (0-100)
    bool auto_response;       // Automatic response triggered
    char response_action[128]; // Response action taken
};

/**
 * dsmil_register_threat_handler - Register threat detection handler
 * @handler: Threat event handler function
 * 
 * Returns: 0 on success, negative error code on failure
 */
typedef int (*threat_handler_t)(const struct threat_event *event);
int dsmil_register_threat_handler(threat_handler_t handler);

/**
 * dsmil_report_threat - Report detected threat
 * @event: Threat event details
 * 
 * Returns: 0 on success, negative error code on failure
 */
int dsmil_report_threat(const struct threat_event *event);
```

## Web Service API

### REST API Endpoints

#### Base Configuration
```
Base URL: http://localhost:8000/api/v1
Authentication: Bearer token (JWT)
Content-Type: application/json
```

#### Authentication Endpoints

##### POST /auth/login
Authenticate user and obtain access token.

**Request:**
```json
{
    "user_id": "string",
    "password": "string",
    "totp_code": "string",
    "biometric_data": "base64_string"
}
```

**Response:**
```json
{
    "access_token": "jwt_token_string",
    "token_type": "bearer",
    "expires_in": 3600,
    "clearance_level": 3,
    "permissions": ["read", "write", "admin"]
}
```

##### POST /auth/logout
Invalidate current session token.

**Response:**
```json
{
    "message": "Successfully logged out"
}
```

#### Device Management Endpoints

##### GET /devices
List accessible DSMIL devices.

**Query Parameters:**
- `clearance_level` (optional): Filter by required clearance level
- `quarantined` (optional): Filter by quarantine status
- `group` (optional): Filter by device group (0-6)

**Response:**
```json
{
    "devices": [
        {
            "id": 32773,
            "token": "0x8005",
            "name": "TPM Interface Controller",
            "group": 0,
            "clearance_required": 2,
            "quarantined": false,
            "status": "active",
            "capabilities": ["read", "ioctl"],
            "last_accessed": "2025-09-02T12:00:00Z",
            "access_count": 42
        }
    ],
    "total": 84,
    "accessible": 28,
    "quarantined": 5
}
```

##### GET /devices/{device_id}
Get detailed information about specific device.

**Response:**
```json
{
    "id": 32773,
    "token": "0x8005",
    "name": "TPM Interface Controller",
    "group": 0,
    "description": "Trusted Platform Module interface and control",
    "clearance_required": 2,
    "quarantined": false,
    "status": "active",
    "capabilities": {
        "read": true,
        "write": false,
        "ioctl": true,
        "interrupt": false
    },
    "performance_metrics": {
        "avg_response_time": "0.8ms",
        "success_rate": "100%",
        "last_error": null
    },
    "security_info": {
        "threat_level": "low",
        "last_threat_scan": "2025-09-02T11:55:00Z",
        "access_violations": 0
    }
}
```

##### POST /devices/{device_id}/read
Read data from device.

**Request:**
```json
{
    "offset": 0,
    "length": 1024,
    "format": "hex"
}
```

**Response:**
```json
{
    "device_id": 32773,
    "data": "48656c6c6f20576f726c64",
    "length": 11,
    "timestamp": "2025-09-02T12:00:00Z",
    "checksum": "a1b2c3d4"
}
```

##### POST /devices/{device_id}/emergency-stop
Execute emergency stop for device.

**Request:**
```json
{
    "reason": "Security threat detected",
    "force": false
}
```

**Response:**
```json
{
    "device_id": 32773,
    "stopped": true,
    "timestamp": "2025-09-02T12:00:00Z",
    "response_time": "85ms"
}
```

#### System Management Endpoints

##### GET /system/status
Get overall system status.

**Response:**
```json
{
    "status": "operational",
    "uptime": "72h 15m 30s",
    "performance": {
        "cpu_usage": "2.9%",
        "memory_usage": "10.3%",
        "disk_usage": "45.2%"
    },
    "devices": {
        "total": 84,
        "active": 79,
        "quarantined": 5,
        "offline": 0
    },
    "security": {
        "threat_level": "low",
        "active_sessions": 3,
        "recent_violations": 0
    },
    "health_score": 95.8
}
```

##### POST /system/emergency-stop
Execute system-wide emergency stop.

**Request:**
```json
{
    "level": "all_devices",
    "reason": "Critical security incident",
    "authorization": "emergency_override_code"
}
```

**Response:**
```json
{
    "emergency_stop_executed": true,
    "timestamp": "2025-09-02T12:00:00Z",
    "devices_stopped": 79,
    "response_time": "85ms",
    "incident_id": "EMRG-2025090212000001"
}
```

#### Audit and Logging Endpoints

##### GET /audit/logs
Query audit log entries.

**Query Parameters:**
- `start_time`: Start timestamp (ISO 8601)
- `end_time`: End timestamp (ISO 8601)
- `event_type`: Filter by event type
- `user_id`: Filter by user
- `device_id`: Filter by device
- `limit`: Maximum entries to return
- `offset`: Pagination offset

**Response:**
```json
{
    "entries": [
        {
            "id": 12345,
            "timestamp": "2025-09-02T12:00:00.123456Z",
            "event_type": "device_access",
            "user_id": "admin",
            "device_id": 32773,
            "operation": "read",
            "result": "success",
            "details": "Read 1024 bytes from device 0x8005",
            "verified": true
        }
    ],
    "total": 10847,
    "page": 1,
    "per_page": 50
}
```

##### GET /audit/integrity
Verify audit log integrity.

**Query Parameters:**
- `start_time`: Start timestamp for verification
- `end_time`: End timestamp for verification

**Response:**
```json
{
    "integrity_verified": true,
    "entries_checked": 1247,
    "signature_failures": 0,
    "checksum_failures": 0,
    "sequence_gaps": 0,
    "verification_time": "2025-09-02T12:00:00Z"
}
```

### WebSocket API

#### Real-Time Device Monitoring
```javascript
// Connect to WebSocket
const ws = new WebSocket('ws://localhost:8000/ws/devices');

// Device status updates
ws.onmessage = function(event) {
    const update = JSON.parse(event.data);
    
    switch(update.type) {
        case 'device_status':
            handleDeviceStatusUpdate(update.data);
            break;
        case 'security_alert':
            handleSecurityAlert(update.data);
            break;
        case 'emergency_stop':
            handleEmergencyStop(update.data);
            break;
        case 'threat_detection':
            handleThreatDetection(update.data);
            break;
    }
};

// Subscribe to specific device updates
ws.send(JSON.stringify({
    action: 'subscribe',
    device_id: 32773,
    events: ['status_change', 'access', 'error']
}));
```

#### Message Formats

##### Device Status Update
```json
{
    "type": "device_status",
    "timestamp": "2025-09-02T12:00:00Z",
    "data": {
        "device_id": 32773,
        "status": "active",
        "performance": {
            "response_time": "0.8ms",
            "throughput": "1.2MB/s"
        },
        "last_operation": {
            "type": "read",
            "user": "admin",
            "result": "success"
        }
    }
}
```

##### Security Alert
```json
{
    "type": "security_alert",
    "timestamp": "2025-09-02T12:00:00Z",
    "data": {
        "alert_level": "high",
        "device_id": 32773,
        "threat_type": "unauthorized_access",
        "description": "Multiple failed authentication attempts",
        "auto_response": "session_locked",
        "requires_attention": true
    }
}
```

## Database Schema

### Core Tables

#### devices table
```sql
CREATE TABLE devices (
    id SERIAL PRIMARY KEY,
    token_id INTEGER UNIQUE NOT NULL,
    name VARCHAR(64) NOT NULL,
    group_id INTEGER NOT NULL,
    description TEXT,
    classification security_level NOT NULL,
    quarantined BOOLEAN DEFAULT FALSE,
    capabilities JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_devices_token_id ON devices(token_id);
CREATE INDEX idx_devices_classification ON devices(classification);
CREATE INDEX idx_devices_quarantined ON devices(quarantined);
```

#### audit_log table
```sql
CREATE TABLE audit_log (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    sequence_number BIGINT UNIQUE NOT NULL,
    event_type VARCHAR(32) NOT NULL,
    user_id VARCHAR(32) NOT NULL,
    device_id INTEGER REFERENCES devices(id),
    operation VARCHAR(32),
    result VARCHAR(16) NOT NULL,
    details JSONB,
    signature BYTEA NOT NULL,
    checksum INTEGER NOT NULL,
    verified BOOLEAN DEFAULT FALSE
);

CREATE INDEX idx_audit_timestamp ON audit_log(timestamp);
CREATE INDEX idx_audit_user_id ON audit_log(user_id);
CREATE INDEX idx_audit_device_id ON audit_log(device_id);
CREATE INDEX idx_audit_event_type ON audit_log(event_type);
```

#### users table
```sql
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(32) UNIQUE NOT NULL,
    clearance_level security_level NOT NULL,
    active BOOLEAN DEFAULT TRUE,
    last_login TIMESTAMP,
    failed_attempts INTEGER DEFAULT 0,
    locked_until TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_users_user_id ON users(user_id);
CREATE INDEX idx_users_clearance ON users(clearance_level);
```

#### sessions table
```sql
CREATE TABLE sessions (
    id SERIAL PRIMARY KEY,
    session_token VARCHAR(256) UNIQUE NOT NULL,
    user_id VARCHAR(32) REFERENCES users(user_id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP NOT NULL,
    last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ip_address INET,
    user_agent TEXT,
    active BOOLEAN DEFAULT TRUE
);

CREATE INDEX idx_sessions_token ON sessions(session_token);
CREATE INDEX idx_sessions_user_id ON sessions(user_id);
CREATE INDEX idx_sessions_expires ON sessions(expires_at);
```

### Database Functions

#### Audit Log Verification
```sql
CREATE OR REPLACE FUNCTION verify_audit_integrity(
    start_time TIMESTAMP,
    end_time TIMESTAMP
) RETURNS TABLE(
    total_entries BIGINT,
    signature_failures BIGINT,
    checksum_failures BIGINT,
    sequence_gaps BIGINT,
    integrity_verified BOOLEAN
) AS $$
BEGIN
    -- Implementation of audit trail integrity verification
    -- Returns verification results
END;
$$ LANGUAGE plpgsql;
```

#### Device Access Validation
```sql
CREATE OR REPLACE FUNCTION validate_device_access(
    p_user_id VARCHAR(32),
    p_device_id INTEGER,
    p_operation VARCHAR(32)
) RETURNS BOOLEAN AS $$
DECLARE
    user_clearance security_level;
    device_clearance security_level;
    device_quarantined BOOLEAN;
BEGIN
    -- Get user clearance level
    SELECT clearance_level INTO user_clearance
    FROM users WHERE user_id = p_user_id AND active = TRUE;
    
    -- Get device requirements
    SELECT classification, quarantined 
    INTO device_clearance, device_quarantined
    FROM devices WHERE id = p_device_id;
    
    -- Check quarantine status
    IF device_quarantined THEN
        RETURN FALSE;
    END IF;
    
    -- Check clearance level
    IF user_clearance < device_clearance THEN
        RETURN FALSE;
    END IF;
    
    RETURN TRUE;
END;
$$ LANGUAGE plpgsql;
```

## Error Codes and Messages

### Kernel Module Error Codes
```c
#define DSMIL_SUCCESS           0       // Success
#define DSMIL_ERROR_INVALID     -EINVAL // Invalid parameter
#define DSMIL_ERROR_ACCESS      -EACCES // Access denied
#define DSMIL_ERROR_PERM        -EPERM  // Operation not permitted
#define DSMIL_ERROR_NODEV       -ENODEV // Device not found
#define DSMIL_ERROR_BUSY        -EBUSY  // Device busy
#define DSMIL_ERROR_TIMEOUT     -ETIMEDOUT // Operation timeout
#define DSMIL_ERROR_QUARANTINE  -EACCES // Device quarantined
#define DSMIL_ERROR_CLEARANCE   -EPERM  // Insufficient clearance
#define DSMIL_ERROR_SIGNATURE   -EBADMSG // Invalid signature
#define DSMIL_ERROR_CHECKSUM    -EILSEQ  // Checksum failure
```

### Web API Error Responses
```json
{
    "error": {
        "code": "DEVICE_QUARANTINED",
        "message": "Device 0x8009 is permanently quarantined",
        "details": "Access to destructive device is permanently prohibited",
        "timestamp": "2025-09-02T12:00:00Z",
        "request_id": "req_12345"
    }
}
```

### HTTP Status Codes
- `200`: Success
- `400`: Bad Request - Invalid parameters
- `401`: Unauthorized - Authentication required
- `403`: Forbidden - Insufficient permissions or quarantined device
- `404`: Not Found - Device or resource not found
- `429`: Too Many Requests - Rate limit exceeded
- `500`: Internal Server Error - System error
- `503`: Service Unavailable - Emergency stop active

---

**API Version**: 2.0  
**Last Updated**: September 2, 2025  
**Compatibility**: Kernel 6.14+, Python 3.9+, React 18+  
**Security Level**: Military Grade with FIPS 140-2 compliance  
**Documentation Status**: Complete and validated
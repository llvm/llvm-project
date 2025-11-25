# DSMIL Control System API Architecture - Phase 3
## Multi-Client API Design Specification

**Version:** 3.0  
**Classification:** RESTRICTED  
**Date:** 2025-01-15  

---

## Executive Summary

This document outlines the comprehensive API architecture for Phase 3 of the DSMIL control system, expanding beyond the current web interface to support multiple client types while maintaining military-grade security and performance requirements.

## System Overview

### Current System (Phase 2)
- **84 DSMIL devices** with device IDs 0x8000-0x806B (32768-32875)
- **5 quarantined devices**: 0x8009, 0x800A, 0x800B, 0x8019, 0x8029
- **FastAPI backend** with PostgreSQL database
- **React web interface** with military theme
- **Multi-level security** with clearance-based authorization
- **Real-time WebSocket communication**

### Phase 3 Expansion Goals
- Support for **multiple client types**:
  - Web interface (React - existing)
  - Python clients (data analysis, automation)  
  - C++ native client (high-performance local control)
  - Future mobile clients (iOS/Android)
- **Enhanced security protocols**
- **Performance optimization** for concurrent access
- **Comprehensive SDK development**

---

## API Architecture Design

### 1. Core API Layers

```
┌─────────────────────────────────────────────┐
│                API Gateway                  │
│         (Load Balancer + Security)         │
└─────────────────────────────────────────────┘
                         │
┌─────────────────────────────────────────────┐
│              Authentication                 │
│        (JWT + MFA + Clearance)             │
└─────────────────────────────────────────────┘
                         │
┌─────────────────────────────────────────────┐
│              Authorization                  │
│       (RBAC + Device Access Control)       │
└─────────────────────────────────────────────┘
                         │
┌─────────────────────────────────────────────┐
│               Rate Limiting                 │
│        (Per-Client + Per-Operation)        │
└─────────────────────────────────────────────┘
                         │
┌─────────────────────────────────────────────┐
│                Core API                     │
│         (Business Logic Layer)             │
└─────────────────────────────────────────────┘
                         │
┌─────────────────────────────────────────────┐
│              Data Layer                     │
│         (PostgreSQL + Device I/O)          │
└─────────────────────────────────────────────┘
```

### 2. API Versioning Strategy

- **Version Format**: `/api/v{major}.{minor}`
- **Current Version**: `/api/v1.0` (Phase 2 compatibility)
- **Phase 3 Version**: `/api/v2.0` (new multi-client features)
- **Backward Compatibility**: v1.0 maintained for existing web interface

---

## RESTful API Endpoints

### 2.1 Authentication & Authorization

#### POST /api/v2/auth/login
```json
{
  "username": "string",
  "password": "string",
  "client_type": "web|python|cpp|mobile",
  "client_version": "string",
  "mfa_token": "string (optional)"
}
```

**Response:**
```json
{
  "access_token": "jwt_token",
  "refresh_token": "jwt_token", 
  "token_type": "bearer",
  "expires_in": 3600,
  "user_context": {
    "user_id": "string",
    "username": "string",
    "clearance_level": "CONFIDENTIAL|SECRET|TOP_SECRET|SCI",
    "authorized_devices": [32768, 32769, "..."],
    "permissions": ["DEVICE_READ", "DEVICE_WRITE", "..."],
    "compartment_access": ["DSMIL", "SCI"]
  },
  "api_capabilities": {
    "max_requests_per_minute": 120,
    "max_concurrent_operations": 5,
    "websocket_enabled": true,
    "bulk_operations_enabled": false
  }
}
```

#### POST /api/v2/auth/refresh
```json
{
  "refresh_token": "jwt_token"
}
```

#### POST /api/v2/auth/logout
```json
{
  "session_id": "string (optional - logout specific session)"
}
```

#### GET /api/v2/auth/sessions
List active sessions for current user

### 2.2 System Information

#### GET /api/v2/system/status
**Enhanced system status with client-specific filtering**
```json
{
  "timestamp": "2025-01-15T10:30:00Z",
  "overall_status": "NORMAL|WARNING|CRITICAL|EMERGENCY",
  "api_version": "2.0",
  "server_capabilities": {
    "max_clients": 100,
    "current_clients": 15,
    "supported_client_types": ["web", "python", "cpp", "mobile"],
    "rate_limiting": true,
    "bulk_operations": true,
    "streaming": true
  },
  "device_summary": {
    "total_devices": 84,
    "active_devices": 79,
    "quarantined_devices": 5,
    "error_devices": 0
  },
  "security_status": {
    "threat_level": "LOW|MEDIUM|HIGH|CRITICAL", 
    "active_sessions": 15,
    "failed_auth_attempts": 2,
    "emergency_stop_active": false
  },
  "performance_metrics": {
    "avg_response_time_ms": 45,
    "requests_per_second": 25.5,
    "cpu_usage_percent": 15.2,
    "memory_usage_percent": 32.1
  }
}
```

#### GET /api/v2/system/capabilities
```json
{
  "api_version": "2.0",
  "supported_operations": [
    "DEVICE_READ", "DEVICE_WRITE", "DEVICE_CONFIG", 
    "BULK_READ", "STREAMING_READ", "BATCH_OPERATIONS"
  ],
  "client_sdks": {
    "python": {
      "version": "2.0.1",
      "download_url": "/api/v2/sdk/python",
      "documentation": "/docs/python-sdk"
    },
    "cpp": {
      "version": "2.0.1", 
      "download_url": "/api/v2/sdk/cpp",
      "documentation": "/docs/cpp-sdk"
    }
  },
  "rate_limits": {
    "requests_per_minute": 120,
    "concurrent_operations": 5,
    "bulk_operation_max_devices": 20
  }
}
```

### 2.3 Device Management

#### GET /api/v2/devices
**Enhanced device listing with filtering and pagination**
```
Query Parameters:
- include_quarantined: boolean (default: false)
- device_group: integer (filter by group 0-11)
- risk_level: string (LOW|MEDIUM|HIGH|CRITICAL)
- status: string (ACTIVE|INACTIVE|ERROR|QUARANTINED)
- limit: integer (default: 50, max: 100)
- offset: integer (default: 0)
- fields: string (comma-separated field list for optimization)
```

**Response:**
```json
{
  "devices": [
    {
      "device_id": 32768,
      "device_name": "DSMIL_Master_Security_0",
      "device_group": 0,
      "device_index": 0,
      "risk_level": "CRITICAL",
      "security_classification": "TOP_SECRET",
      "required_clearance": "TOP_SECRET",
      "is_active": true,
      "is_quarantined": false,
      "capabilities": ["READ", "WRITE", "CONFIG"],
      "constraints": {
        "max_operations_per_minute": 10,
        "requires_dual_auth": true,
        "emergency_stop_priority": 1
      },
      "hardware_info": {
        "firmware_version": "2.1.5",
        "last_calibration": "2025-01-10T08:00:00Z"
      },
      "performance_metrics": {
        "average_response_time_ms": 23,
        "success_rate_percent": 99.8,
        "last_accessed": "2025-01-15T09:15:00Z"
      }
    }
  ],
  "pagination": {
    "total": 84,
    "returned": 50,
    "offset": 0,
    "has_more": true
  },
  "meta": {
    "accessible_devices": 79,
    "quarantined_devices": 5,
    "user_authorized_devices": 60
  }
}
```

#### GET /api/v2/devices/{device_id}
**Detailed device information**

#### GET /api/v2/devices/{device_id}/history
**Device operation history and performance metrics**
```
Query Parameters:
- start_date: ISO datetime
- end_date: ISO datetime  
- operation_type: string
- limit: integer
```

### 2.4 Device Operations

#### POST /api/v2/devices/{device_id}/operations
**Single device operation**
```json
{
  "operation_type": "READ|WRITE|CONFIG|RESET|ACTIVATE|DEACTIVATE",
  "operation_data": {
    "register": "STATUS|CONFIG|DATA",
    "offset": 0,
    "length": 4,
    "value": "0x12345678 (for WRITE operations)"
  },
  "justification": "Required for high-risk operations",
  "dual_auth_required": false,
  "priority": "LOW|NORMAL|HIGH|EMERGENCY",
  "timeout_ms": 5000
}
```

**Response:**
```json
{
  "operation_id": "uuid",
  "device_id": 32768,
  "operation_type": "READ",
  "status": "SUCCESS|PENDING|FAILED|DENIED|TIMEOUT",
  "result": {
    "data": "0xABCD1234",
    "register_status": "VALID",
    "device_response_time_ms": 23
  },
  "timestamp": "2025-01-15T10:30:00Z",
  "execution_time_ms": 45,
  "user_id": "admin_001",
  "audit_trail_id": "audit_uuid",
  "authorization": {
    "clearance_verified": true,
    "device_access_verified": true,
    "dual_auth_completed": false
  }
}
```

### 2.5 Bulk Operations (New in v2.0)

#### POST /api/v2/operations/bulk
**Bulk operations across multiple devices**
```json
{
  "operations": [
    {
      "device_id": 32768,
      "operation_type": "READ",
      "operation_data": {"register": "STATUS"}
    },
    {
      "device_id": 32769, 
      "operation_type": "READ",
      "operation_data": {"register": "STATUS"}
    }
  ],
  "execution_mode": "PARALLEL|SEQUENTIAL",
  "max_concurrency": 5,
  "timeout_ms": 10000,
  "justification": "Bulk system health check",
  "stop_on_first_error": false
}
```

**Response:**
```json
{
  "bulk_operation_id": "uuid",
  "total_operations": 2,
  "status": "COMPLETED|PARTIAL|FAILED",
  "execution_time_ms": 156,
  "results": [
    {
      "device_id": 32768,
      "status": "SUCCESS",
      "result": {"data": "0x12345678"},
      "execution_time_ms": 78
    },
    {
      "device_id": 32769,
      "status": "SUCCESS", 
      "result": {"data": "0x87654321"},
      "execution_time_ms": 82
    }
  ],
  "summary": {
    "successful": 2,
    "failed": 0,
    "denied": 0,
    "timeouts": 0
  }
}
```

#### GET /api/v2/operations/bulk/{bulk_operation_id}
**Get bulk operation status and results**

### 2.6 Streaming Operations (New in v2.0)

#### GET /api/v2/devices/{device_id}/stream
**Server-Sent Events for real-time device data**
```
Query Parameters:
- registers: comma-separated list (STATUS,TEMP,VOLTAGE)
- interval_ms: integer (default: 1000, min: 100)
- duration_s: integer (max streaming duration)
```

**Stream Format:**
```
event: device_data
data: {"device_id": 32768, "timestamp": "...", "data": {"STATUS": "0x1234"}}

event: device_error
data: {"device_id": 32768, "error": "Communication timeout"}
```

### 2.7 Emergency Controls

#### POST /api/v2/emergency/stop
**Enhanced emergency stop with client notification**
```json
{
  "justification": "Security breach detected",
  "scope": "ALL|DEVICE_GROUP|SINGLE_DEVICE",
  "target_devices": [32768, 32769],
  "notify_all_clients": true,
  "escalation_level": "IMMEDIATE|SCHEDULED"
}
```

#### GET /api/v2/emergency/status
#### POST /api/v2/emergency/release
#### GET /api/v2/emergency/history

---

## WebSocket Protocol v2.0

### Connection Endpoint
```
WSS://<host>/api/v2/ws?token=<jwt_token>&client_type=<type>
```

### Message Format
```json
{
  "id": "message_uuid",
  "type": "MESSAGE_TYPE",
  "data": {},
  "timestamp": "2025-01-15T10:30:00Z",
  "source": "server|client",
  "target": "broadcast|user_id|client_type",
  "priority": "LOW|NORMAL|HIGH|EMERGENCY"
}
```

### Enhanced Message Types

#### Client → Server Messages
- **SUBSCRIBE_ADVANCED**: Enhanced subscription with filters
```json
{
  "type": "SUBSCRIBE_ADVANCED",
  "data": {
    "subscription": "device_updates",
    "filters": {
      "device_ids": [32768, 32769],
      "operation_types": ["READ", "WRITE"],
      "risk_levels": ["HIGH", "CRITICAL"]
    },
    "rate_limit": "MAX_1_PER_SECOND"
  }
}
```

- **CLIENT_HEARTBEAT**: Enhanced heartbeat with status
```json
{
  "type": "CLIENT_HEARTBEAT",
  "data": {
    "client_type": "python",
    "client_version": "2.0.1",
    "active_operations": 3,
    "resource_usage": {
      "cpu_percent": 15.2,
      "memory_mb": 256
    }
  }
}
```

#### Server → Client Messages
- **DEVICE_STATE_UPDATE**: Enhanced device updates
- **BULK_OPERATION_COMPLETE**: Bulk operation notifications
- **SYSTEM_MAINTENANCE**: Scheduled maintenance notifications
- **CLIENT_RATE_LIMITED**: Rate limiting notifications
- **EMERGENCY_BROADCAST**: Priority emergency messages

---

## Client SDK Specifications

### 3.1 Python SDK

#### Installation
```bash
pip install dsmil-control-client==2.0.1
```

#### Basic Usage
```python
from dsmil_client import DSMILClient
import asyncio

async def main():
    client = DSMILClient(
        base_url="https://dsmil-control.mil",
        api_version="2.0"
    )
    
    # Authenticate
    await client.authenticate(
        username="operator",
        password="secure_password",
        client_type="python"
    )
    
    # Single device operation
    result = await client.read_device(32768, register="STATUS")
    print(f"Device status: {result.data}")
    
    # Bulk operations
    devices = [32768, 32769, 32770]
    results = await client.bulk_read(devices, register="STATUS")
    
    # WebSocket streaming
    async for update in client.stream_devices(devices):
        print(f"Device {update.device_id}: {update.data}")

if __name__ == "__main__":
    asyncio.run(main())
```

#### Advanced Features
```python
# Rate limiting aware client
client.configure_rate_limiting(
    requests_per_minute=100,
    burst_size=10
)

# Automatic retry with exponential backoff
client.configure_retry_policy(
    max_retries=3,
    backoff_strategy="exponential"
)

# Connection pooling
client.configure_connection_pool(
    max_connections=10,
    keepalive=True
)
```

### 3.2 C++ SDK

#### CMake Integration
```cmake
find_package(DSMILClient 2.0 REQUIRED)
target_link_libraries(your_app DSMILClient::DSMILClient)
```

#### Basic Usage
```cpp
#include <dsmil/client.hpp>
#include <iostream>

int main() {
    dsmil::Client client("https://dsmil-control.mil", "2.0");
    
    // Authenticate
    auto auth_result = client.authenticate(
        "operator", "secure_password", dsmil::ClientType::Cpp
    );
    
    if (!auth_result.success) {
        std::cerr << "Authentication failed\n";
        return 1;
    }
    
    // High-performance device operations
    auto result = client.read_device_sync(0x8000, dsmil::Register::STATUS);
    if (result.success) {
        std::cout << "Device status: 0x" << std::hex << result.data << "\n";
    }
    
    // Async operations for high throughput
    auto future = client.read_device_async(0x8000, dsmil::Register::STATUS);
    // ... do other work ...
    auto async_result = future.get();
    
    return 0;
}
```

#### Performance Features
```cpp
// Connection pooling
client.configure_pool(10 /* max_connections */);

// Bulk operations with callback
std::vector<uint16_t> device_ids = {0x8000, 0x8001, 0x8002};
client.bulk_read_async(device_ids, dsmil::Register::STATUS, 
    [](const dsmil::BulkResult& result) {
        // Handle results as they arrive
        for (const auto& device_result : result.results) {
            std::cout << "Device " << device_result.device_id 
                      << ": " << device_result.data << "\n";
        }
    }
);

// Real-time streaming
client.stream_devices(device_ids, 
    [](const dsmil::DeviceUpdate& update) {
        // Low-latency callback for real-time data
        process_update(update);
    }
);
```

---

## Security Architecture

### 4.1 Authentication Flow

```
1. Client Request → API Gateway
2. API Gateway → Authentication Service
3. Authentication Service → User Database
4. MFA Challenge (if enabled)
5. JWT Token Generation
6. Client Receives Token + Capabilities
7. Token Used for All Subsequent Requests
```

### 4.2 Authorization Matrix

| Operation Type | Clearance Required | Device Access | Dual Auth | Audit Level |
|----------------|-------------------|---------------|-----------|-------------|
| DEVICE_READ    | CONFIDENTIAL      | Per-device    | No        | Standard    |
| DEVICE_WRITE   | SECRET            | Per-device    | Risk-based| Enhanced    |
| DEVICE_CONFIG  | SECRET            | Per-device    | Yes       | Enhanced    |
| QUARANTINE_ACCESS | TOP_SECRET     | Specific      | Yes       | Maximum     |
| EMERGENCY_STOP | CONFIDENTIAL      | N/A           | No        | Maximum     |
| BULK_OPERATIONS| SECRET            | All devices   | Risk-based| Enhanced    |

### 4.3 Rate Limiting Strategy

```yaml
Global Limits:
  requests_per_minute: 1000
  concurrent_connections: 100
  
Per-User Limits:
  CONFIDENTIAL: 60 req/min, 3 concurrent
  SECRET: 120 req/min, 5 concurrent  
  TOP_SECRET: 300 req/min, 10 concurrent
  
Per-Operation Limits:
  DEVICE_READ: 2/second
  DEVICE_WRITE: 1/second
  BULK_OPERATIONS: 1 per 5 seconds
  
Per-Device Limits:
  CRITICAL devices: 10 operations/minute
  HIGH risk devices: 30 operations/minute
  NORMAL devices: 60 operations/minute
```

### 4.4 Quarantine Protection

The 5 quarantined devices (0x8009, 0x800A, 0x800B, 0x8019, 0x8029) have additional protection:

- **Access Requirements**: TOP_SECRET clearance minimum
- **Dual Authorization**: Always required
- **Operation Logging**: Maximum audit detail
- **Rate Limiting**: 5 operations per minute maximum
- **Emergency Override**: Only via physical access
- **Automatic Monitoring**: All access immediately reported

---

## Error Handling & Safety Protocols

### 5.1 HTTP Status Codes

| Code | Meaning | Usage |
|------|---------|--------|
| 200  | Success | Operation completed successfully |
| 201  | Created | Resource created (sessions, etc.) |
| 400  | Bad Request | Invalid request format/parameters |
| 401  | Unauthorized | Authentication required/failed |
| 403  | Forbidden | Insufficient clearance/permissions |
| 404  | Not Found | Device/resource not found |
| 409  | Conflict | Device in use/conflicting operation |
| 423  | Locked | Account locked/device quarantined |
| 429  | Rate Limited | Rate limit exceeded |
| 500  | Server Error | Internal server error |
| 503  | Service Unavailable | Emergency stop active |

### 5.2 Error Response Format

```json
{
  "error": {
    "code": "DEVICE_QUARANTINED",
    "message": "Device 0x8009 is quarantined and requires TOP_SECRET clearance",
    "details": {
      "device_id": 32777,
      "required_clearance": "TOP_SECRET",
      "user_clearance": "SECRET",
      "help_url": "/docs/quarantine-procedures"
    },
    "timestamp": "2025-01-15T10:30:00Z",
    "request_id": "req_12345",
    "retry_after": 300
  }
}
```

### 5.3 Safety Protocols

1. **Device State Validation**: All operations verify device is in safe state
2. **Operation Queuing**: Prevent conflicting operations on same device
3. **Automatic Rollback**: Failed operations automatically rolled back
4. **Emergency Stop Integration**: All clients notified of emergency stops
5. **Hardware Monitoring**: Continuous monitoring of device health
6. **Audit Trail**: Every operation logged with full context

---

## Performance Requirements & Monitoring

### 6.1 Performance Targets

- **API Response Time**: <100ms for 95% of requests
- **Concurrent Clients**: Support 100+ simultaneous clients
- **Bulk Operations**: Process 50+ devices in <2 seconds
- **WebSocket Latency**: <50ms for real-time updates
- **Throughput**: 1000+ operations per minute system-wide
- **Availability**: 99.9% uptime (8.77 hours downtime/year)

### 6.2 Monitoring & Metrics

#### System Metrics
- Request latency (P50, P95, P99)
- Request throughput (requests/second)
- Error rates by endpoint and client type
- Authentication success/failure rates
- Device operation success rates
- WebSocket connection counts and stability

#### Security Metrics
- Failed authentication attempts
- Authorization denials by clearance level
- Quarantine access attempts
- Emergency stop activations
- Unusual client behavior patterns

#### Client SDK Metrics
- Client connection success rates
- SDK error rates by version
- Client-reported performance metrics
- Feature utilization statistics

### 6.3 Monitoring Endpoints

#### GET /api/v2/monitoring/health
```json
{
  "status": "healthy",
  "components": {
    "database": "healthy",
    "device_controllers": "healthy", 
    "authentication": "healthy",
    "websockets": "degraded"
  },
  "metrics": {
    "response_time_ms": 45,
    "active_clients": 23,
    "operations_per_second": 15.5
  }
}
```

#### GET /api/v2/monitoring/metrics
**Detailed system metrics for administrators**

---

## Implementation Roadmap

### Phase 3.1: Core Multi-Client API (4 weeks)
- [ ] API versioning implementation
- [ ] Enhanced authentication with client type support
- [ ] Bulk operations endpoint
- [ ] Enhanced WebSocket protocol
- [ ] Rate limiting by client type
- [ ] Basic Python SDK

### Phase 3.2: C++ SDK & Performance (3 weeks)
- [ ] High-performance C++ SDK
- [ ] Connection pooling
- [ ] Streaming operations
- [ ] Advanced error handling
- [ ] Performance optimization

### Phase 3.3: Advanced Features (3 weeks)  
- [ ] Server-sent events streaming
- [ ] Advanced subscription filtering
- [ ] Enhanced monitoring
- [ ] Mobile client preparation
- [ ] Documentation and testing

### Phase 3.4: Security Hardening (2 weeks)
- [ ] Advanced threat detection
- [ ] Enhanced audit logging
- [ ] Penetration testing
- [ ] Security documentation
- [ ] Deployment preparation

---

## Integration with Existing System

### Backward Compatibility
- All existing v1.0 endpoints remain functional
- Current React web interface unchanged
- Database schema extensions (no breaking changes)
- WebSocket v1.0 protocol maintained alongside v2.0

### Migration Strategy
1. **Parallel Deployment**: v2.0 API runs alongside v1.0
2. **Feature Flag**: Toggle new features per user/client
3. **Gradual Migration**: Move clients to v2.0 over time
4. **Monitoring**: Track usage of both versions
5. **Sunset Planning**: v1.0 deprecation timeline

---

## Conclusion

This API architecture provides a robust, secure, and performant foundation for Phase 3 of the DSMIL control system. The design maintains backward compatibility while enabling new client types and advanced features, ensuring the system can scale to meet future requirements while preserving the critical security and safety standards required for military applications.

---

**Document Classification**: RESTRICTED  
**Review Date**: 2025-04-15  
**Next Version**: 3.1 (Mobile client specifications)
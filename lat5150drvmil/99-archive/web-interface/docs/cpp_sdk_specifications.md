# DSMIL C++ SDK Technical Specifications
## High-Performance Native Client Library

**Version:** 2.0.1  
**Classification:** RESTRICTED  
**Date:** 2025-01-15  

---

## Overview

The DSMIL C++ SDK provides high-performance, low-latency access to the DSMIL control system for C++ applications requiring maximum throughput and minimal overhead. Designed for real-time control systems, data acquisition, and high-frequency trading-style operations.

## Architecture

### Core Components

```cpp
namespace dsmil {
    class Client;              // Main API client
    class ConnectionPool;      // HTTP connection management  
    class WebSocketClient;     // Real-time streaming
    class BulkOperator;        // Batch operations
    class SecurityManager;    // Authentication & encryption
    class DeviceRegistry;     // Device metadata cache
    class PerformanceMonitor; // Metrics and diagnostics
}
```

### Threading Model

```
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│   User Thread   │  │  I/O Thread     │  │ Callback Thread │
│                 │  │                 │  │                 │
│ • API calls     │◄─┤ • HTTP requests │◄─┤ • Async results │
│ • Sync results  │  │ • WebSocket     │  │ • Event handling│
│ • Error handling│  │ • Connection    │  │ • Stream updates│
└─────────────────┘  └─────────────────┘  └─────────────────┘
```

## Core API

### 2.1 Client Initialization

```cpp
#include <dsmil/client.hpp>

// Basic client setup
dsmil::Client client("https://dsmil-control.mil", "2.0");

// Advanced configuration
dsmil::ClientConfig config;
config.base_url = "https://dsmil-control.mil";
config.api_version = "2.0";
config.timeout_ms = 5000;
config.max_retries = 3;
config.connection_pool_size = 10;
config.enable_compression = true;
config.verify_ssl = true;
config.client_cert_path = "/path/to/client.pem";

dsmil::Client client(config);
```

### 2.2 Authentication

```cpp
// Simple authentication
auto auth_result = client.authenticate("operator", "password", dsmil::ClientType::Cpp);

// MFA authentication  
auto mfa_result = client.authenticate_mfa("operator", "password", "123456");

// Certificate-based authentication
auto cert_result = client.authenticate_certificate("/path/to/cert.p12", "cert_password");

// Check authentication result
if (!auth_result.success) {
    std::cerr << "Auth failed: " << auth_result.error_message << std::endl;
    return -1;
}

// Access user context
const auto& user = auth_result.user_context;
std::cout << "Logged in as: " << user.username 
          << " (clearance: " << user.clearance_level << ")" << std::endl;
```

### 2.3 Device Operations

#### Synchronous Operations
```cpp
// Read device register
auto result = client.read_device_sync(0x8000, dsmil::Register::STATUS);
if (result.success) {
    std::cout << "Status: 0x" << std::hex << result.data << std::endl;
} else {
    std::cerr << "Read failed: " << result.error_message << std::endl;
}

// Write device register
dsmil::WriteRequest write_req;
write_req.device_id = 0x8001;
write_req.register_type = dsmil::Register::CONFIG;
write_req.offset = 0;
write_req.data = 0x12345678;
write_req.justification = "Routine configuration update";

auto write_result = client.write_device_sync(write_req);

// Complex operations
dsmil::ConfigRequest config_req;
config_req.device_id = 0x8002;
config_req.config_data = {
    {"threshold", 1024},
    {"mode", "continuous"},
    {"calibration", true}
};
config_req.justification = "Performance optimization";

auto config_result = client.configure_device_sync(config_req);
```

#### Asynchronous Operations
```cpp
// Future-based async operations
auto future = client.read_device_async(0x8000, dsmil::Register::STATUS);

// Do other work...
std::this_thread::sleep_for(std::chrono::milliseconds(10));

// Get result when ready
auto result = future.get();  // Blocks until complete

// Callback-based async operations
client.read_device_async(0x8000, dsmil::Register::STATUS,
    [](const dsmil::DeviceResult& result) {
        if (result.success) {
            std::cout << "Async read result: 0x" << std::hex << result.data << std::endl;
        }
    }
);

// Timeout handling
auto future_with_timeout = client.read_device_async(0x8000, dsmil::Register::STATUS);
if (future_with_timeout.wait_for(std::chrono::seconds(2)) == std::future_status::timeout) {
    std::cerr << "Operation timed out" << std::endl;
    client.cancel_operation(future_with_timeout.get_operation_id());
}
```

### 2.4 Bulk Operations

```cpp
// Bulk read operations
std::vector<uint16_t> device_ids = {0x8000, 0x8001, 0x8002, 0x8003};

// Synchronous bulk read
auto bulk_result = client.bulk_read_sync(device_ids, dsmil::Register::STATUS);

for (const auto& device_result : bulk_result.results) {
    std::cout << "Device 0x" << std::hex << device_result.device_id
              << ": 0x" << device_result.data << std::endl;
}

// Asynchronous bulk operations with callback
client.bulk_read_async(device_ids, dsmil::Register::STATUS,
    [](const dsmil::BulkResult& result) {
        std::cout << "Bulk operation completed: " 
                  << result.summary.successful << "/" 
                  << result.summary.total << " successful" << std::endl;
        
        for (const auto& device_result : result.results) {
            if (device_result.success) {
                process_device_data(device_result.device_id, device_result.data);
            } else {
                handle_device_error(device_result.device_id, device_result.error_message);
            }
        }
    }
);

// Streaming bulk operations
client.bulk_read_streaming(device_ids, dsmil::Register::STATUS,
    [](const dsmil::DeviceResult& result) {
        // Called as each device responds
        handle_individual_result(result);
    },
    [](const dsmil::BulkSummary& summary) {
        // Called when all operations complete
        std::cout << "All operations completed" << std::endl;
    }
);
```

### 2.5 Real-Time Streaming

#### WebSocket Streaming
```cpp
// Connect to WebSocket
auto ws_client = client.create_websocket_client();
ws_client->connect();

// Subscribe to device updates
std::vector<uint16_t> monitored_devices = {0x8000, 0x8001, 0x8002};

ws_client->subscribe_device_updates(monitored_devices,
    [](const dsmil::DeviceUpdate& update) {
        std::cout << "Device 0x" << std::hex << update.device_id 
                  << " updated: " << update.data << std::endl;
        
        // High-frequency processing
        process_real_time_data(update);
    }
);

// Subscribe to system events
ws_client->subscribe_system_events(
    [](const dsmil::SystemEvent& event) {
        if (event.type == dsmil::EventType::EMERGENCY_STOP) {
            // Immediate response to emergency stop
            handle_emergency_stop(event);
        }
    }
);
```

#### Server-Sent Events (SSE) Streaming
```cpp
// High-throughput streaming for data acquisition
auto sse_stream = client.create_device_stream(device_ids);

sse_stream->set_interval(std::chrono::milliseconds(100));  // 10 Hz
sse_stream->set_registers({dsmil::Register::STATUS, dsmil::Register::DATA});

sse_stream->start([](const dsmil::StreamUpdate& update) {
    // Low-latency callback for real-time data
    auto timestamp = std::chrono::high_resolution_clock::now();
    
    for (const auto& device_data : update.devices) {
        // Process high-frequency data
        data_acquisition_pipeline(device_data, timestamp);
    }
});
```

## Advanced Features

### 3.1 Connection Management

```cpp
// Connection pool configuration
dsmil::PoolConfig pool_config;
pool_config.max_connections = 20;
pool_config.keep_alive_timeout = std::chrono::minutes(5);
pool_config.connection_timeout = std::chrono::seconds(10);
pool_config.idle_timeout = std::chrono::minutes(2);

client.configure_connection_pool(pool_config);

// Connection health monitoring
client.set_connection_health_callback(
    [](const dsmil::ConnectionHealth& health) {
        if (health.failed_connections > 3) {
            // Implement failover logic
            switch_to_backup_server();
        }
    }
);

// Manual connection management
auto connection = client.acquire_connection();
// Use connection for multiple operations...
client.release_connection(connection);
```

### 3.2 Performance Optimization

```cpp
// Batch multiple operations in single HTTP request
dsmil::BatchRequest batch;
batch.add_read(0x8000, dsmil::Register::STATUS);
batch.add_read(0x8001, dsmil::Register::TEMP);
batch.add_write(0x8002, dsmil::Register::CONFIG, 0x12345678);

auto batch_result = client.execute_batch_sync(batch);

// Request pipelining
client.enable_http_pipelining(true);
client.set_pipeline_depth(10);

// Data compression
client.enable_compression(dsmil::CompressionType::GZIP);

// Response caching for read operations
client.enable_response_cache(std::chrono::seconds(30));

// Pre-fetch device metadata
client.prefetch_device_registry();
```

### 3.3 Error Handling & Resilience

```cpp
// Configure retry policy
dsmil::RetryPolicy retry_policy;
retry_policy.max_attempts = 5;
retry_policy.base_delay = std::chrono::milliseconds(100);
retry_policy.max_delay = std::chrono::seconds(5);
retry_policy.backoff_multiplier = 2.0;
retry_policy.jitter = true;

// Retry conditions
retry_policy.retry_on_status = {
    dsmil::StatusCode::INTERNAL_SERVER_ERROR,
    dsmil::StatusCode::BAD_GATEWAY,
    dsmil::StatusCode::SERVICE_UNAVAILABLE
};

client.set_retry_policy(retry_policy);

// Circuit breaker pattern
dsmil::CircuitBreakerConfig cb_config;
cb_config.failure_threshold = 5;
cb_config.recovery_timeout = std::chrono::seconds(30);
cb_config.half_open_max_calls = 3;

client.configure_circuit_breaker(cb_config);

// Custom error handling
client.set_error_handler([](const dsmil::Error& error) {
    switch (error.code) {
        case dsmil::ErrorCode::DEVICE_QUARANTINED:
            // Log security event
            log_security_event(error);
            break;
        case dsmil::ErrorCode::RATE_LIMITED:
            // Implement backoff
            implement_backoff_strategy();
            break;
        case dsmil::ErrorCode::EMERGENCY_STOP:
            // Immediate halt
            emergency_shutdown();
            break;
        default:
            // General error handling
            handle_general_error(error);
    }
});
```

### 3.4 Security Features

```cpp
// Certificate-based mutual authentication
dsmil::SecurityConfig security;
security.client_cert_path = "/secure/client.pem";
security.client_key_path = "/secure/client.key";
security.ca_cert_path = "/secure/ca.pem";
security.verify_hostname = true;
security.cipher_suites = {
    "TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384",
    "TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256"
};

client.configure_security(security);

// Token refresh handling
client.set_token_refresh_callback([](const std::string& new_token) {
    // Store new token securely
    secure_token_storage.store(new_token);
});

// Security event monitoring
client.set_security_event_handler(
    [](const dsmil::SecurityEvent& event) {
        if (event.severity >= dsmil::Severity::HIGH) {
            // Immediate notification
            security_operations_center.notify(event);
        }
    }
);
```

### 3.5 Diagnostics & Monitoring

```cpp
// Performance monitoring
client.enable_performance_monitoring(true);

auto metrics = client.get_performance_metrics();
std::cout << "Average latency: " << metrics.avg_latency_ms << "ms" << std::endl;
std::cout << "Success rate: " << metrics.success_rate << "%" << std::endl;
std::cout << "Operations/sec: " << metrics.operations_per_second << std::endl;

// Custom metrics collection
client.set_metrics_callback([](const dsmil::Metrics& metrics) {
    prometheus_registry.record_latency(metrics.avg_latency_ms);
    prometheus_registry.record_throughput(metrics.operations_per_second);
    prometheus_registry.record_error_rate(metrics.error_rate);
});

// Debug logging
client.set_log_level(dsmil::LogLevel::DEBUG);
client.set_log_callback([](dsmil::LogLevel level, const std::string& message) {
    std::cout << "[" << to_string(level) << "] " << message << std::endl;
});

// Health check
auto health = client.check_health();
if (!health.is_healthy) {
    std::cerr << "Client unhealthy: " << health.issues << std::endl;
}
```

## Data Types & Structures

### 4.1 Core Data Types

```cpp
namespace dsmil {
    // Device identifier
    using DeviceId = uint16_t;  // 0x8000 - 0x806B
    
    // Register types
    enum class Register : uint8_t {
        STATUS = 0x00,
        CONFIG = 0x01,
        DATA = 0x02,
        TEMP = 0x03,
        VOLTAGE = 0x04,
        ERROR = 0xFF
    };
    
    // Operation results
    struct DeviceResult {
        bool success;
        DeviceId device_id;
        Register register_type;
        uint32_t data;
        std::chrono::milliseconds execution_time;
        std::string error_message;
        std::string operation_id;
    };
    
    // Bulk operation results
    struct BulkResult {
        bool overall_success;
        std::vector<DeviceResult> results;
        BulkSummary summary;
        std::chrono::milliseconds total_execution_time;
        std::string bulk_operation_id;
    };
    
    struct BulkSummary {
        size_t total;
        size_t successful; 
        size_t failed;
        size_t denied;
        size_t timeouts;
    };
    
    // Real-time updates
    struct DeviceUpdate {
        DeviceId device_id;
        Register register_type;
        uint32_t data;
        std::chrono::system_clock::time_point timestamp;
        uint32_t sequence_number;
    };
    
    // System events
    struct SystemEvent {
        EventType type;
        std::string message;
        std::chrono::system_clock::time_point timestamp;
        Severity severity;
        std::unordered_map<std::string, std::string> metadata;
    };
}
```

### 4.2 Configuration Structures

```cpp
struct ClientConfig {
    std::string base_url = "https://dsmil-control.mil";
    std::string api_version = "2.0";
    std::chrono::milliseconds timeout = std::chrono::seconds(5);
    uint32_t max_retries = 3;
    uint32_t connection_pool_size = 10;
    bool enable_compression = true;
    bool verify_ssl = true;
    std::string user_agent = "DSMIL-CPP-SDK/2.0.1";
    
    // Security settings
    std::string client_cert_path;
    std::string client_key_path;
    std::string ca_cert_path;
    
    // Performance settings
    bool enable_http2 = true;
    bool enable_pipelining = false;
    uint32_t pipeline_depth = 5;
    
    // Logging
    LogLevel log_level = LogLevel::INFO;
    std::string log_file_path;
};

struct PoolConfig {
    uint32_t max_connections = 10;
    std::chrono::seconds keep_alive_timeout{300};
    std::chrono::seconds connection_timeout{10};
    std::chrono::seconds idle_timeout{120};
    uint32_t max_requests_per_connection = 1000;
    bool enable_connection_reuse = true;
};
```

## Compilation & Linking

### 5.1 CMake Integration

```cmake
# Find the DSMIL SDK
find_package(DSMILClient 2.0 REQUIRED)

# Create your executable
add_executable(your_app main.cpp)

# Link against DSMIL SDK
target_link_libraries(your_app 
    DSMILClient::DSMILClient
    # SDK automatically brings in dependencies:
    # - OpenSSL::SSL
    # - OpenSSL::Crypto  
    # - CURL::CURL
    # - Threads::Threads
)

# Optional: Enable C++17 features
target_compile_features(your_app PRIVATE cxx_std_17)
```

### 5.2 Manual Compilation

```bash
# Include directories
INCLUDES="-I/usr/local/include/dsmil -I/usr/local/include"

# Library directories and libraries
LIBS="-L/usr/local/lib -ldsmil_client -lcurl -lssl -lcrypto -pthread"

# Compile your application
g++ -std=c++17 $INCLUDES -O3 -DNDEBUG main.cpp -o your_app $LIBS
```

### 5.3 Dependencies

**Required:**
- C++17 compatible compiler (GCC 7+, Clang 6+, MSVC 2019+)
- OpenSSL 1.1.1+ (for TLS/SSL support)
- libcurl 7.60+ (for HTTP/HTTPS communication)
- pthreads (for threading support)

**Optional:**
- zlib (for compression support)
- Boost.Beast (for WebSocket performance optimization)
- jemalloc (for memory allocation optimization)

## Performance Benchmarks

### 6.1 Latency Benchmarks (Local Network)

| Operation Type | P50 | P95 | P99 | Max |
|----------------|-----|-----|-----|-----|
| Single Read    | 2ms | 5ms | 12ms| 45ms|
| Single Write   | 8ms | 15ms| 28ms| 78ms|
| Bulk Read (10) | 15ms| 32ms| 58ms|120ms|
| Bulk Write (10)| 45ms| 89ms|156ms|280ms|

### 6.2 Throughput Benchmarks

| Client Config | Ops/sec | CPU Usage | Memory |
|---------------|---------|-----------|---------|
| 1 thread, sync| 125     | 15%       | 25MB    |
| 4 threads, sync| 450    | 45%       | 40MB    |
| 1 thread, async| 850    | 25%       | 35MB    |
| 4 threads, async| 2800  | 78%       | 85MB    |

### 6.3 Memory Usage

- **Base client**: ~15MB
- **Per connection**: ~2MB
- **Per WebSocket**: ~1.5MB
- **Bulk operation**: ~500KB per 100 devices
- **Streaming cache**: ~10KB per device

## Examples

### 7.1 Simple Monitoring Application

```cpp
#include <dsmil/client.hpp>
#include <iostream>
#include <thread>
#include <chrono>

int main() {
    dsmil::Client client("https://dsmil-control.mil", "2.0");
    
    // Authenticate
    auto auth_result = client.authenticate("monitor", "secure_pass", dsmil::ClientType::Cpp);
    if (!auth_result.success) {
        std::cerr << "Authentication failed" << std::endl;
        return 1;
    }
    
    // Monitor critical devices
    std::vector<uint16_t> critical_devices = {0x8000, 0x8001, 0x8002};
    
    while (true) {
        auto bulk_result = client.bulk_read_sync(critical_devices, dsmil::Register::STATUS);
        
        for (const auto& result : bulk_result.results) {
            if (!result.success) {
                std::cout << "WARNING: Device 0x" << std::hex << result.device_id
                         << " failed: " << result.error_message << std::endl;
            } else if (result.data & 0x80000000) {  // Error bit
                std::cout << "ALERT: Device 0x" << std::hex << result.device_id
                         << " reports error status: 0x" << result.data << std::endl;
            }
        }
        
        std::this_thread::sleep_for(std::chrono::seconds(10));
    }
    
    return 0;
}
```

### 7.2 High-Performance Data Acquisition

```cpp
#include <dsmil/client.hpp>
#include <fstream>
#include <atomic>
#include <queue>
#include <mutex>

class DataAcquisitionSystem {
    dsmil::Client client_;
    std::atomic<bool> running_{false};
    std::queue<dsmil::DeviceUpdate> data_queue_;
    std::mutex queue_mutex_;
    std::ofstream data_file_;
    
public:
    DataAcquisitionSystem() : client_("https://dsmil-control.mil", "2.0") {}
    
    bool start() {
        // Authenticate
        auto auth_result = client_.authenticate("dataacq", "secure_pass", dsmil::ClientType::Cpp);
        if (!auth_result.success) return false;
        
        // Open data file
        data_file_.open("device_data.csv");
        data_file_ << "timestamp,device_id,register,value\n";
        
        // Configure for high performance
        client_.enable_compression(false);  // Reduce CPU overhead
        client_.configure_connection_pool({.max_connections = 20});
        
        // Start WebSocket streaming
        auto ws_client = client_.create_websocket_client();
        ws_client->connect();
        
        std::vector<uint16_t> all_devices(84);
        std::iota(all_devices.begin(), all_devices.end(), 0x8000);
        
        ws_client->subscribe_device_updates(all_devices,
            [this](const dsmil::DeviceUpdate& update) {
                std::lock_guard<std::mutex> lock(queue_mutex_);
                data_queue_.push(update);
            }
        );
        
        // Start data processing thread
        running_ = true;
        std::thread data_thread(&DataAcquisitionSystem::process_data, this);
        data_thread.detach();
        
        return true;
    }
    
private:
    void process_data() {
        while (running_) {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            
            while (!data_queue_.empty()) {
                const auto& update = data_queue_.front();
                
                // Write to file
                data_file_ << std::chrono::duration_cast<std::chrono::milliseconds>(
                    update.timestamp.time_since_epoch()).count()
                    << "," << update.device_id
                    << "," << static_cast<int>(update.register_type)
                    << "," << update.data << "\n";
                
                data_queue_.pop();
            }
            
            data_file_.flush();
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
};
```

### 7.3 Emergency Response System

```cpp
#include <dsmil/client.hpp>
#include <signal.h>

class EmergencyResponseSystem {
    dsmil::Client client_;
    std::atomic<bool> emergency_active_{false};
    
public:
    EmergencyResponseSystem() : client_("https://dsmil-control.mil", "2.0") {}
    
    void initialize() {
        // High-privilege authentication
        auto auth_result = client_.authenticate("emergency", "ultra_secure", dsmil::ClientType::Cpp);
        if (!auth_result.success) {
            throw std::runtime_error("Emergency system authentication failed");
        }
        
        // Subscribe to emergency events
        auto ws_client = client_.create_websocket_client();
        ws_client->subscribe_system_events(
            [this](const dsmil::SystemEvent& event) {
                if (event.type == dsmil::EventType::EMERGENCY_STOP) {
                    handle_emergency_stop(event);
                }
            }
        );
        
        // Set up signal handler for manual emergency stop
        signal(SIGUSR1, [](int) {
            // This would need proper signal handling in real code
            // trigger_manual_emergency_stop();
        });
    }
    
    void trigger_emergency_stop(const std::string& reason) {
        dsmil::EmergencyStopRequest request;
        request.justification = reason;
        request.scope = dsmil::EmergencyScope::ALL;
        request.notify_all_clients = true;
        request.escalation_level = dsmil::EscalationLevel::IMMEDIATE;
        
        auto result = client_.trigger_emergency_stop_sync(request);
        if (result.success) {
            emergency_active_ = true;
            std::cout << "EMERGENCY STOP ACTIVATED: " << reason << std::endl;
            
            // Notify external systems
            notify_external_systems(reason);
        }
    }
    
private:
    void handle_emergency_stop(const dsmil::SystemEvent& event) {
        emergency_active_ = true;
        
        // Log emergency event
        std::cout << "EMERGENCY STOP RECEIVED: " << event.message << std::endl;
        
        // Implement emergency procedures
        shutdown_local_systems();
        notify_operations_center();
        
        // Wait for release
        monitor_emergency_release();
    }
    
    void notify_external_systems(const std::string& reason) {
        // Implementation would notify external monitoring systems
        // via separate communication channels
    }
    
    void shutdown_local_systems() {
        // Implementation would safely shutdown local control systems
    }
    
    void notify_operations_center() {
        // Implementation would alert human operators
    }
    
    void monitor_emergency_release() {
        // Implementation would monitor for emergency stop release
    }
};
```

---

## Conclusion

The DSMIL C++ SDK provides comprehensive, high-performance access to the DSMIL control system with enterprise-grade features including connection pooling, automatic retry logic, real-time streaming, and comprehensive security features. The SDK is designed for mission-critical applications requiring maximum performance and reliability.

---

**Document Classification**: RESTRICTED  
**Review Date**: 2025-04-15  
**Next Version**: 2.1 (Mobile platform extensions)
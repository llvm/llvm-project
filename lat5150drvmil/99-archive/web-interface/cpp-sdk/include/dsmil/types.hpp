#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include <chrono>
#include <memory>
#include <future>
#include <functional>

namespace dsmil {

// Device identifier type
using DeviceId = uint16_t;  // 0x8000 - 0x806B (32768-32875)

// Register types
enum class Register : uint8_t {
    STATUS = 0x00,
    CONFIG = 0x01,
    DATA = 0x02,
    TEMP = 0x03,
    VOLTAGE = 0x04,
    ERROR = 0xFF
};

// Client types
enum class ClientType {
    Web,
    Python, 
    Cpp,
    Mobile
};

// Operation types
enum class OperationType {
    READ,
    WRITE,
    CONFIG,
    RESET,
    ACTIVATE,
    DEACTIVATE
};

// Risk levels
enum class RiskLevel {
    SAFE,
    LOW,
    MODERATE,
    HIGH,
    CRITICAL,
    QUARANTINED
};

// Clearance levels
enum class ClearanceLevel {
    RESTRICTED,
    CONFIDENTIAL,
    SECRET,
    TOP_SECRET,
    SCI
};

// Security classification
enum class SecurityClassification {
    UNCLASSIFIED,
    RESTRICTED,
    CONFIDENTIAL,
    SECRET,
    TOP_SECRET
};

// Status codes
enum class StatusCode {
    SUCCESS = 200,
    CREATED = 201,
    BAD_REQUEST = 400,
    UNAUTHORIZED = 401,
    FORBIDDEN = 403,
    NOT_FOUND = 404,
    CONFLICT = 409,
    LOCKED = 423,
    RATE_LIMITED = 429,
    INTERNAL_SERVER_ERROR = 500,
    BAD_GATEWAY = 502,
    SERVICE_UNAVAILABLE = 503
};

// Error codes
enum class ErrorCode {
    SUCCESS,
    DEVICE_NOT_FOUND,
    DEVICE_QUARANTINED,
    DEVICE_BUSY,
    PERMISSION_DENIED,
    AUTHENTICATION_FAILED,
    RATE_LIMITED,
    TIMEOUT,
    NETWORK_ERROR,
    PROTOCOL_ERROR,
    INVALID_PARAMETER,
    INTERNAL_ERROR,
    EMERGENCY_STOP,
    HARDWARE_ERROR,
    SECURITY_VIOLATION,
    KERNEL_MODULE_ERROR
};

// Event types
enum class EventType {
    DEVICE_UPDATE,
    SYSTEM_STATUS,
    SECURITY_EVENT,
    EMERGENCY_STOP,
    RATE_LIMIT_WARNING,
    AUTHENTICATION_EVENT,
    PERFORMANCE_ALERT,
    HARDWARE_ALERT
};

// Severity levels
enum class Severity {
    INFO,
    WARNING,
    HIGH,
    CRITICAL
};

// Execution modes
enum class ExecutionMode {
    SYNCHRONOUS,
    ASYNCHRONOUS,
    PARALLEL,
    SEQUENTIAL
};

// Compression types
enum class CompressionType {
    NONE,
    GZIP,
    DEFLATE
};

// Log levels
enum class LogLevel {
    TRACE,
    DEBUG,
    INFO,
    WARNING,
    ERROR,
    CRITICAL
};

// Emergency scope
enum class EmergencyScope {
    ALL,
    DEVICE_GROUP,
    SINGLE_DEVICE
};

// Escalation levels
enum class EscalationLevel {
    NORMAL,
    HIGH,
    IMMEDIATE
};

// Forward declarations
class Client;
class ConnectionPool;
class WebSocketClient;
class BulkOperator;
class SecurityManager;
class DeviceRegistry;
class PerformanceMonitor;

// Basic data structures

struct DeviceInfo {
    DeviceId device_id;
    std::string device_name;
    uint8_t device_group;
    uint8_t device_index;
    RiskLevel risk_level;
    SecurityClassification security_classification;
    ClearanceLevel required_clearance;
    bool is_active;
    bool is_quarantined;
    std::vector<std::string> capabilities;
    std::unordered_map<std::string, std::string> constraints;
    std::unordered_map<std::string, std::string> hardware_info;
    std::unordered_map<std::string, double> performance_metrics;
};

struct DeviceResult {
    bool success;
    DeviceId device_id;
    Register register_type;
    uint32_t data;
    std::chrono::milliseconds execution_time;
    std::string error_message;
    std::string operation_id;
    std::chrono::system_clock::time_point timestamp;
};

struct BulkSummary {
    size_t total;
    size_t successful;
    size_t failed;
    size_t denied;
    size_t timeouts;
};

struct BulkResult {
    bool overall_success;
    std::vector<DeviceResult> results;
    BulkSummary summary;
    std::chrono::milliseconds total_execution_time;
    std::string bulk_operation_id;
};

struct DeviceUpdate {
    DeviceId device_id;
    Register register_type;
    uint32_t data;
    std::chrono::system_clock::time_point timestamp;
    uint32_t sequence_number;
};

struct SystemEvent {
    EventType type;
    std::string message;
    std::chrono::system_clock::time_point timestamp;
    Severity severity;
    std::unordered_map<std::string, std::string> metadata;
};

struct UserContext {
    std::string user_id;
    std::string username;
    ClearanceLevel clearance_level;
    std::vector<DeviceId> authorized_devices;
    std::vector<std::string> permissions;
    std::vector<std::string> compartment_access;
};

struct ApiCapabilities {
    uint32_t max_requests_per_minute;
    uint32_t max_concurrent_operations;
    bool websocket_enabled;
    bool bulk_operations_enabled;
};

struct AuthResult {
    bool success;
    std::string access_token;
    std::string refresh_token;
    std::string token_type;
    std::chrono::seconds expires_in;
    UserContext user_context;
    ApiCapabilities api_capabilities;
    std::string error_message;
};

struct WriteRequest {
    DeviceId device_id;
    Register register_type;
    uint32_t offset;
    uint32_t data;
    std::string justification;
};

struct ConfigRequest {
    DeviceId device_id;
    std::unordered_map<std::string, std::string> config_data;
    std::string justification;
};

struct EmergencyStopRequest {
    std::string justification;
    EmergencyScope scope;
    std::vector<DeviceId> target_devices;
    bool notify_all_clients;
    EscalationLevel escalation_level;
};

struct ConnectionHealth {
    bool is_healthy;
    uint32_t active_connections;
    uint32_t failed_connections;
    std::chrono::milliseconds average_latency;
    double success_rate;
    std::string issues;
};

struct PerformanceMetrics {
    std::chrono::milliseconds avg_latency_ms;
    std::chrono::milliseconds p95_latency_ms;
    std::chrono::milliseconds p99_latency_ms;
    double operations_per_second;
    double success_rate;
    double error_rate;
    uint64_t total_operations;
    uint64_t successful_operations;
    uint64_t failed_operations;
};

struct RetryPolicy {
    uint32_t max_attempts;
    std::chrono::milliseconds base_delay;
    std::chrono::milliseconds max_delay;
    double backoff_multiplier;
    bool jitter;
    std::vector<StatusCode> retry_on_status;
};

struct CircuitBreakerConfig {
    uint32_t failure_threshold;
    std::chrono::seconds recovery_timeout;
    uint32_t half_open_max_calls;
};

struct SecurityEvent {
    EventType type;
    Severity severity;
    std::string message;
    std::unordered_map<std::string, std::string> context;
    std::chrono::system_clock::time_point timestamp;
};

struct Error {
    ErrorCode code;
    std::string message;
    std::string details;
    std::chrono::system_clock::time_point timestamp;
    std::string request_id;
    std::chrono::seconds retry_after;
};

// Callback types
using DeviceUpdateCallback = std::function<void(const DeviceUpdate&)>;
using SystemEventCallback = std::function<void(const SystemEvent&)>;
using BulkResultCallback = std::function<void(const BulkResult&)>;
using DeviceResultCallback = std::function<void(const DeviceResult&)>;
using SecurityEventCallback = std::function<void(const SecurityEvent&)>;
using ErrorCallback = std::function<void(const Error&)>;
using MetricsCallback = std::function<void(const PerformanceMetrics&)>;
using LogCallback = std::function<void(LogLevel, const std::string&)>;
using ConnectionHealthCallback = std::function<void(const ConnectionHealth&)>;
using TokenRefreshCallback = std::function<void(const std::string&)>;

// Future types for async operations
using DeviceResultFuture = std::future<DeviceResult>;
using BulkResultFuture = std::future<BulkResult>;
using AuthResultFuture = std::future<AuthResult>;

} // namespace dsmil
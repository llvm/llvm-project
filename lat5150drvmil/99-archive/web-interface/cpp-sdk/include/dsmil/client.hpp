#pragma once

#include "types.hpp"
#include "config.hpp"
#include "exceptions.hpp"
#include <memory>
#include <string>
#include <vector>
#include <future>

namespace dsmil {

// Forward declarations
class ConnectionPool;
class WebSocketClient;
class BulkOperator;
class SecurityManager;
class DeviceRegistry;
class PerformanceMonitor;

/**
 * @brief High-performance C++ client for DSMIL control system
 * 
 * The Client class provides comprehensive access to the DSMIL control system
 * with enterprise-grade features including connection pooling, automatic retry
 * logic, real-time streaming, and comprehensive security features.
 */
class Client {
public:
    /**
     * @brief Construct a new Client with basic configuration
     * 
     * @param base_url Base URL of the DSMIL control system
     * @param api_version API version to use (default: "2.0")
     */
    explicit Client(const std::string& base_url, const std::string& api_version = "2.0");
    
    /**
     * @brief Construct a new Client with advanced configuration
     * 
     * @param config Advanced client configuration
     */
    explicit Client(const ClientConfig& config);
    
    /**
     * @brief Destroy the Client and cleanup resources
     */
    ~Client();
    
    // Non-copyable but movable
    Client(const Client&) = delete;
    Client& operator=(const Client&) = delete;
    Client(Client&&) noexcept;
    Client& operator=(Client&&) noexcept;
    
    // Authentication methods
    
    /**
     * @brief Authenticate with username/password
     * 
     * @param username Username
     * @param password Password
     * @param client_type Client type identifier
     * @return AuthResult Authentication result with tokens and user context
     */
    AuthResult authenticate(
        const std::string& username,
        const std::string& password,
        ClientType client_type
    );
    
    /**
     * @brief Authenticate with MFA token
     * 
     * @param username Username
     * @param password Password
     * @param mfa_token MFA token
     * @param client_type Client type identifier
     * @return AuthResult Authentication result
     */
    AuthResult authenticate_mfa(
        const std::string& username,
        const std::string& password,
        const std::string& mfa_token,
        ClientType client_type = ClientType::Cpp
    );
    
    /**
     * @brief Authenticate with client certificate
     * 
     * @param cert_path Path to client certificate
     * @param cert_password Certificate password
     * @return AuthResult Authentication result
     */
    AuthResult authenticate_certificate(
        const std::string& cert_path,
        const std::string& cert_password
    );
    
    /**
     * @brief Refresh authentication token
     * 
     * @param refresh_token Refresh token
     * @return AuthResult New authentication result
     */
    AuthResult refresh_token(const std::string& refresh_token);
    
    /**
     * @brief Logout and invalidate session
     * 
     * @param session_id Optional session ID to logout specific session
     */
    void logout(const std::string& session_id = "");
    
    // Synchronous device operations
    
    /**
     * @brief Read device register synchronously
     * 
     * @param device_id Device ID
     * @param register_type Register to read
     * @return DeviceResult Operation result with data
     */
    DeviceResult read_device_sync(DeviceId device_id, Register register_type);
    
    /**
     * @brief Write to device register synchronously
     * 
     * @param request Write request with device, register, and data
     * @return DeviceResult Operation result
     */
    DeviceResult write_device_sync(const WriteRequest& request);
    
    /**
     * @brief Configure device synchronously
     * 
     * @param request Configuration request
     * @return DeviceResult Operation result
     */
    DeviceResult configure_device_sync(const ConfigRequest& request);
    
    /**
     * @brief Reset device synchronously
     * 
     * @param device_id Device ID to reset
     * @return DeviceResult Operation result
     */
    DeviceResult reset_device_sync(DeviceId device_id);
    
    // Asynchronous device operations
    
    /**
     * @brief Read device register asynchronously
     * 
     * @param device_id Device ID
     * @param register_type Register to read
     * @return std::future<DeviceResult> Future for async result
     */
    DeviceResultFuture read_device_async(DeviceId device_id, Register register_type);
    
    /**
     * @brief Read device register asynchronously with callback
     * 
     * @param device_id Device ID
     * @param register_type Register to read
     * @param callback Callback function for result
     */
    void read_device_async(
        DeviceId device_id, 
        Register register_type, 
        DeviceResultCallback callback
    );
    
    /**
     * @brief Write to device register asynchronously
     * 
     * @param request Write request
     * @return std::future<DeviceResult> Future for async result
     */
    DeviceResultFuture write_device_async(const WriteRequest& request);
    
    /**
     * @brief Write to device register asynchronously with callback
     * 
     * @param request Write request
     * @param callback Callback function for result
     */
    void write_device_async(const WriteRequest& request, DeviceResultCallback callback);
    
    // Bulk operations
    
    /**
     * @brief Bulk read from multiple devices synchronously
     * 
     * @param device_ids Vector of device IDs
     * @param register_type Register to read from all devices
     * @return BulkResult Bulk operation results
     */
    BulkResult bulk_read_sync(
        const std::vector<DeviceId>& device_ids, 
        Register register_type
    );
    
    /**
     * @brief Bulk read from multiple devices asynchronously
     * 
     * @param device_ids Vector of device IDs
     * @param register_type Register to read from all devices
     * @param callback Callback function for results
     */
    void bulk_read_async(
        const std::vector<DeviceId>& device_ids,
        Register register_type,
        BulkResultCallback callback
    );
    
    /**
     * @brief Streaming bulk read with individual device callbacks
     * 
     * @param device_ids Vector of device IDs
     * @param register_type Register to read
     * @param device_callback Callback for individual device results
     * @param completion_callback Callback when all operations complete
     */
    void bulk_read_streaming(
        const std::vector<DeviceId>& device_ids,
        Register register_type,
        DeviceResultCallback device_callback,
        BulkResultCallback completion_callback
    );
    
    // Real-time streaming
    
    /**
     * @brief Create WebSocket client for real-time communication
     * 
     * @return std::shared_ptr<WebSocketClient> WebSocket client instance
     */
    std::shared_ptr<WebSocketClient> create_websocket_client();
    
    /**
     * @brief Create device data stream using Server-Sent Events
     * 
     * @param device_ids Vector of device IDs to monitor
     * @return std::unique_ptr<DeviceStream> Device stream instance
     */
    std::unique_ptr<class DeviceStream> create_device_stream(
        const std::vector<DeviceId>& device_ids
    );
    
    // Emergency controls
    
    /**
     * @brief Trigger emergency stop synchronously
     * 
     * @param request Emergency stop request
     * @return DeviceResult Operation result
     */
    DeviceResult trigger_emergency_stop_sync(const EmergencyStopRequest& request);
    
    /**
     * @brief Trigger emergency stop asynchronously
     * 
     * @param request Emergency stop request
     * @param callback Callback for result
     */
    void trigger_emergency_stop_async(
        const EmergencyStopRequest& request,
        DeviceResultCallback callback
    );
    
    // Configuration and management
    
    /**
     * @brief Configure connection pool
     * 
     * @param config Pool configuration
     */
    void configure_connection_pool(const PoolConfig& config);
    
    /**
     * @brief Configure security settings
     * 
     * @param config Security configuration
     */
    void configure_security(const SecurityConfig& config);
    
    /**
     * @brief Set retry policy for operations
     * 
     * @param policy Retry policy configuration
     */
    void set_retry_policy(const RetryPolicy& policy);
    
    /**
     * @brief Configure circuit breaker
     * 
     * @param config Circuit breaker configuration
     */
    void configure_circuit_breaker(const CircuitBreakerConfig& config);
    
    /**
     * @brief Enable/disable response compression
     * 
     * @param compression_type Type of compression to use
     */
    void enable_compression(CompressionType compression_type);
    
    /**
     * @brief Enable/disable HTTP/2 pipelining
     * 
     * @param enable Enable pipelining
     */
    void enable_http_pipelining(bool enable);
    
    /**
     * @brief Set HTTP pipeline depth
     * 
     * @param depth Pipeline depth (number of concurrent requests)
     */
    void set_pipeline_depth(uint32_t depth);
    
    /**
     * @brief Enable response caching for read operations
     * 
     * @param cache_duration Cache duration
     */
    void enable_response_cache(std::chrono::seconds cache_duration);
    
    /**
     * @brief Prefetch device registry for improved performance
     */
    void prefetch_device_registry();
    
    // Monitoring and diagnostics
    
    /**
     * @brief Enable/disable performance monitoring
     * 
     * @param enable Enable monitoring
     */
    void enable_performance_monitoring(bool enable);
    
    /**
     * @brief Get current performance metrics
     * 
     * @return PerformanceMetrics Current metrics
     */
    PerformanceMetrics get_performance_metrics() const;
    
    /**
     * @brief Set metrics collection callback
     * 
     * @param callback Callback for metrics updates
     */
    void set_metrics_callback(MetricsCallback callback);
    
    /**
     * @brief Set log level for client logging
     * 
     * @param level Log level
     */
    void set_log_level(LogLevel level);
    
    /**
     * @brief Set log callback for custom log handling
     * 
     * @param callback Log callback function
     */
    void set_log_callback(LogCallback callback);
    
    /**
     * @brief Check client health status
     * 
     * @return ConnectionHealth Health status information
     */
    ConnectionHealth check_health() const;
    
    // Event handlers and callbacks
    
    /**
     * @brief Set connection health monitoring callback
     * 
     * @param callback Callback for connection health events
     */
    void set_connection_health_callback(ConnectionHealthCallback callback);
    
    /**
     * @brief Set error handler callback
     * 
     * @param callback Callback for error handling
     */
    void set_error_handler(ErrorCallback callback);
    
    /**
     * @brief Set security event handler
     * 
     * @param callback Callback for security events
     */
    void set_security_event_handler(SecurityEventCallback callback);
    
    /**
     * @brief Set token refresh callback
     * 
     * @param callback Callback when token is refreshed
     */
    void set_token_refresh_callback(TokenRefreshCallback callback);
    
    // Connection management
    
    /**
     * @brief Manually acquire a connection from the pool
     * 
     * @return Connection handle (implementation-specific)
     */
    std::shared_ptr<class Connection> acquire_connection();
    
    /**
     * @brief Release a connection back to the pool
     * 
     * @param connection Connection to release
     */
    void release_connection(std::shared_ptr<class Connection> connection);
    
    /**
     * @brief Cancel an ongoing operation
     * 
     * @param operation_id Operation ID to cancel
     * @return bool True if operation was cancelled
     */
    bool cancel_operation(const std::string& operation_id);
    
    // Advanced features
    
    /**
     * @brief Execute batch of operations in a single request
     * 
     * @param batch Batch request with multiple operations
     * @return BulkResult Batch execution results
     */
    BulkResult execute_batch_sync(const class BatchRequest& batch);
    
    /**
     * @brief Get device registry information
     * 
     * @return std::vector<DeviceInfo> List of all accessible devices
     */
    std::vector<DeviceInfo> get_device_registry() const;
    
    /**
     * @brief Get specific device information
     * 
     * @param device_id Device ID
     * @return DeviceInfo Device information
     */
    DeviceInfo get_device_info(DeviceId device_id) const;

private:
    // Private implementation (PIMPL pattern)
    class Impl;
    std::unique_ptr<Impl> pimpl_;
};

} // namespace dsmil
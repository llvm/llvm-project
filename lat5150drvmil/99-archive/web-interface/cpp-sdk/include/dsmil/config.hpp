#pragma once

#include "types.hpp"
#include <string>
#include <vector>
#include <chrono>

namespace dsmil {

/**
 * @brief Client configuration structure
 */
struct ClientConfig {
    // Basic configuration
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
    std::vector<std::string> cipher_suites;
    
    // Performance settings
    bool enable_http2 = true;
    bool enable_pipelining = false;
    uint32_t pipeline_depth = 5;
    bool enable_keep_alive = true;
    std::chrono::seconds keep_alive_timeout = std::chrono::seconds(300);
    
    // Hardware security
    bool enable_hardware_security = false;
    std::string tpm_device_path = "/dev/tpm0";
    std::string hardware_security_module_path;
    
    // Kernel module integration
    std::string kernel_module_path = "/dev/dsmil_control";
    std::string sysfs_device_path = "/sys/class/dsmil";
    bool enable_kernel_bypass = false;
    
    // Logging
    LogLevel log_level = LogLevel::INFO;
    std::string log_file_path;
    bool enable_structured_logging = false;
    
    // Monitoring
    bool enable_metrics = true;
    std::chrono::seconds metrics_interval = std::chrono::seconds(60);
    bool enable_tracing = false;
};

/**
 * @brief Connection pool configuration
 */
struct PoolConfig {
    uint32_t max_connections = 10;
    uint32_t min_connections = 2;
    std::chrono::seconds keep_alive_timeout = std::chrono::seconds(300);
    std::chrono::seconds connection_timeout = std::chrono::seconds(10);
    std::chrono::seconds idle_timeout = std::chrono::seconds(120);
    uint32_t max_requests_per_connection = 1000;
    bool enable_connection_reuse = true;
    bool enable_connection_pooling = true;
    
    // Advanced settings
    std::chrono::milliseconds acquire_timeout = std::chrono::seconds(5);
    uint32_t max_queue_size = 100;
    bool enable_health_checks = true;
    std::chrono::seconds health_check_interval = std::chrono::seconds(30);
};

/**
 * @brief Security configuration
 */
struct SecurityConfig {
    // TLS/SSL settings
    std::string client_cert_path;
    std::string client_key_path;
    std::string ca_cert_path;
    bool verify_hostname = true;
    std::vector<std::string> cipher_suites;
    std::string min_tls_version = "1.2";
    bool enable_client_cert_auth = false;
    
    // Authentication
    std::chrono::minutes token_refresh_threshold = std::chrono::minutes(5);
    bool enable_automatic_token_refresh = true;
    std::string token_storage_path;
    bool secure_token_storage = true;
    
    // Hardware security
    bool enable_tpm_integration = false;
    std::string tpm_device_path = "/dev/tpm0";
    bool enable_hardware_crypto = false;
    std::string hsm_library_path;
    
    // Audit and logging
    bool enable_security_audit = true;
    std::string audit_log_path;
    bool log_sensitive_operations = false;
    
    // Rate limiting
    bool respect_rate_limits = true;
    std::chrono::seconds rate_limit_backoff = std::chrono::seconds(1);
};

/**
 * @brief Batch request configuration
 */
struct BatchConfig {
    uint32_t max_operations_per_batch = 50;
    std::chrono::milliseconds batch_timeout = std::chrono::seconds(30);
    ExecutionMode execution_mode = ExecutionMode::PARALLEL;
    uint32_t max_concurrency = 10;
    bool stop_on_first_error = false;
    bool enable_partial_results = true;
};

/**
 * @brief WebSocket configuration
 */
struct WebSocketConfig {
    std::chrono::seconds ping_interval = std::chrono::seconds(30);
    std::chrono::seconds ping_timeout = std::chrono::seconds(10);
    std::chrono::seconds connect_timeout = std::chrono::seconds(30);
    uint32_t max_message_size = 1024 * 1024; // 1MB
    bool enable_compression = true;
    bool enable_auto_reconnect = true;
    std::chrono::seconds reconnect_interval = std::chrono::seconds(5);
    uint32_t max_reconnect_attempts = 10;
};

/**
 * @brief Device stream configuration
 */
struct StreamConfig {
    std::chrono::milliseconds interval = std::chrono::seconds(1);
    std::chrono::seconds max_duration = std::chrono::minutes(30);
    std::vector<Register> registers = {Register::STATUS};
    uint32_t buffer_size = 1000;
    bool enable_backpressure_control = true;
    uint32_t max_queue_size = 10000;
};

/**
 * @brief Performance monitoring configuration
 */
struct MonitoringConfig {
    bool enable_metrics = true;
    std::chrono::seconds metrics_collection_interval = std::chrono::seconds(60);
    bool enable_histogram_metrics = true;
    bool enable_performance_alerts = true;
    std::chrono::milliseconds latency_alert_threshold = std::chrono::milliseconds(200);
    double error_rate_alert_threshold = 0.05; // 5%
    
    // Prometheus integration
    bool enable_prometheus_metrics = false;
    std::string prometheus_endpoint = "/metrics";
    uint16_t prometheus_port = 9090;
};

/**
 * @brief Threading configuration
 */
struct ThreadingConfig {
    uint32_t io_thread_count = 4;
    uint32_t worker_thread_count = 8;
    uint32_t callback_thread_count = 2;
    
    // Thread affinity (Linux only)
    bool enable_thread_affinity = false;
    std::vector<uint32_t> io_thread_affinity;
    std::vector<uint32_t> worker_thread_affinity;
    
    // Thread priorities
    bool enable_thread_priorities = false;
    int io_thread_priority = 0;
    int worker_thread_priority = 0;
    int callback_thread_priority = 0;
};

/**
 * @brief Memory configuration
 */
struct MemoryConfig {
    size_t connection_buffer_size = 64 * 1024; // 64KB
    size_t response_buffer_size = 1024 * 1024; // 1MB
    uint32_t max_cached_responses = 1000;
    size_t max_memory_pool_size = 100 * 1024 * 1024; // 100MB
    
    // Memory pool settings
    bool enable_memory_pool = true;
    size_t small_block_size = 1024;
    size_t medium_block_size = 8192;
    size_t large_block_size = 65536;
    uint32_t blocks_per_chunk = 64;
};

/**
 * @brief Default configuration factory
 */
class ConfigFactory {
public:
    static ClientConfig create_default_config();
    static ClientConfig create_high_performance_config();
    static ClientConfig create_secure_config();
    static ClientConfig create_embedded_config();
    
    static PoolConfig create_default_pool_config();
    static SecurityConfig create_default_security_config();
    static WebSocketConfig create_default_websocket_config();
    static MonitoringConfig create_default_monitoring_config();
};

} // namespace dsmil
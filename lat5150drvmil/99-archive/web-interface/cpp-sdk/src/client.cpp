#include "dsmil/client.hpp"
#include "dsmil/exceptions.hpp"
#include "connection_pool.hpp"
#include "websocket_client.hpp"
#include "bulk_operator.hpp"
#include "security_manager.hpp"
#include "device_registry.hpp"
#include "performance_monitor.hpp"
#include "kernel_interface.hpp"
#include "async_executor.hpp"

#include <curl/curl.h>
#include <openssl/ssl.h>
#include <json/json.h>
#include <memory>
#include <mutex>
#include <atomic>
#include <thread>

namespace dsmil {

// Implementation details hidden from public interface
class Client::Impl {
public:
    explicit Impl(const ClientConfig& config)
        : config_(config)
        , connection_pool_(std::make_unique<ConnectionPool>(config))
        , security_manager_(std::make_unique<SecurityManager>(config))
        , device_registry_(std::make_unique<DeviceRegistry>(config))
        , performance_monitor_(std::make_unique<PerformanceMonitor>(config))
        , bulk_operator_(std::make_unique<BulkOperator>(*connection_pool_))
        , async_executor_(std::make_unique<AsyncExecutor>(config.worker_thread_count))
        , initialized_(false)
        , authenticated_(false)
        , emergency_stop_active_(false)
    {
        initialize();
    }
    
    ~Impl() {
        cleanup();
    }
    
    void initialize() {
        // Initialize libcurl
        curl_global_init(CURL_GLOBAL_ALL);
        
        // Initialize OpenSSL
        SSL_library_init();
        SSL_load_error_strings();
        
        // Initialize connection pool
        connection_pool_->initialize();
        
        // Initialize kernel module interface if enabled
        if (!config_.kernel_module_path.empty()) {
            kernel_interface_ = std::make_unique<KernelInterface>(config_);
            kernel_available_ = kernel_interface_->initialize();
        }
        
        // Start performance monitoring if enabled
        if (config_.enable_metrics) {
            performance_monitor_->start();
        }
        
        initialized_ = true;
    }
    
    void cleanup() {
        if (!initialized_) return;
        
        // Stop background tasks
        async_executor_.reset();
        
        // Cleanup performance monitoring
        if (performance_monitor_) {
            performance_monitor_->stop();
        }
        
        // Cleanup WebSocket connections
        websocket_clients_.clear();
        
        // Cleanup connection pool
        if (connection_pool_) {
            connection_pool_->shutdown();
        }
        
        // Cleanup kernel interface
        if (kernel_interface_) {
            kernel_interface_->cleanup();
        }
        
        // Cleanup libcurl
        curl_global_cleanup();
        
        initialized_ = false;
    }
    
    AuthResult authenticate(const std::string& username, const std::string& password, ClientType client_type) {
        if (!initialized_) {
            throw DSMILException("Client not initialized");
        }
        
        try {
            Json::Value request_body;
            request_body["username"] = username;
            request_body["password"] = password;
            request_body["client_type"] = client_type_to_string(client_type);
            request_body["client_version"] = DSMIL_VERSION_STRING;
            
            auto connection = connection_pool_->acquire_connection();
            auto response = connection->post("/api/v2/auth/login", request_body);
            
            if (response.status_code == 200) {
                // Parse successful authentication response
                auto auth_result = parse_auth_response(response.body);
                
                // Store authentication state
                access_token_ = auth_result.access_token;
                refresh_token_ = auth_result.refresh_token;
                user_context_ = auth_result.user_context;
                authenticated_ = true;
                
                // Configure connection pool with auth token
                connection_pool_->set_auth_token(access_token_);
                
                // Update device registry with user permissions
                device_registry_->update_authorized_devices(user_context_.authorized_devices);
                
                return auth_result;
            } else {
                return AuthResult{
                    .success = false,
                    .error_message = "Authentication failed with HTTP " + std::to_string(response.status_code)
                };
            }
            
        } catch (const std::exception& e) {
            return AuthResult{
                .success = false,
                .error_message = std::string("Authentication error: ") + e.what()
            };
        }
    }
    
    DeviceResult read_device_sync(DeviceId device_id, Register register_type) {
        if (!authenticated_) {
            throw AuthenticationException("Not authenticated");
        }
        
        auto start_time = std::chrono::high_resolution_clock::now();
        std::string operation_id = generate_operation_id();
        
        try {
            // Check device access permissions
            if (!device_registry_->is_device_accessible(device_id)) {
                throw DeviceException(ErrorCode::PERMISSION_DENIED, "Device not accessible", device_id);
            }
            
            // Check if device is quarantined
            if (device_registry_->is_device_quarantined(device_id)) {
                throw QuarantineException(device_id);
            }
            
            DeviceResult result;
            
            // Try kernel module interface first if available
            if (kernel_available_ && kernel_interface_) {
                result = kernel_interface_->read_device(device_id, register_type);
            } else {
                // Fall back to HTTP API
                result = read_device_via_http(device_id, register_type);
            }
            
            // Update performance metrics
            auto execution_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now() - start_time
            );
            
            performance_monitor_->record_operation("read", execution_time, true);
            
            result.operation_id = operation_id;
            result.execution_time = execution_time;
            result.timestamp = std::chrono::system_clock::now();
            
            return result;
            
        } catch (const std::exception& e) {
            auto execution_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now() - start_time
            );
            
            performance_monitor_->record_operation("read", execution_time, false);
            
            return DeviceResult{
                .success = false,
                .device_id = device_id,
                .register_type = register_type,
                .data = 0,
                .execution_time = execution_time,
                .error_message = e.what(),
                .operation_id = operation_id,
                .timestamp = std::chrono::system_clock::now()
            };
        }
    }
    
    DeviceResult write_device_sync(const WriteRequest& request) {
        if (!authenticated_) {
            throw AuthenticationException("Not authenticated");
        }
        
        auto start_time = std::chrono::high_resolution_clock::now();
        std::string operation_id = generate_operation_id();
        
        try {
            // Enhanced security checks for write operations
            if (!device_registry_->is_device_accessible(request.device_id)) {
                throw DeviceException(ErrorCode::PERMISSION_DENIED, "Device not accessible", request.device_id);
            }
            
            if (device_registry_->is_device_quarantined(request.device_id)) {
                throw QuarantineException(request.device_id, "Write operations prohibited on quarantined devices");
            }
            
            // Check write permissions based on clearance level
            auto required_clearance = device_registry_->get_required_clearance(request.device_id, "WRITE");
            if (user_context_.clearance_level < required_clearance) {
                throw PermissionException(
                    "Insufficient clearance for write operation", 
                    required_clearance, 
                    user_context_.clearance_level
                );
            }
            
            DeviceResult result;
            
            // Use kernel module for direct hardware access if available
            if (kernel_available_ && kernel_interface_) {
                result = kernel_interface_->write_device(request);
            } else {
                // Fall back to HTTP API with additional safety checks
                result = write_device_via_http(request);
            }
            
            // Update performance metrics
            auto execution_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now() - start_time
            );
            
            performance_monitor_->record_operation("write", execution_time, true);
            
            result.operation_id = operation_id;
            result.execution_time = execution_time;
            result.timestamp = std::chrono::system_clock::now();
            
            return result;
            
        } catch (const std::exception& e) {
            auto execution_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now() - start_time
            );
            
            performance_monitor_->record_operation("write", execution_time, false);
            
            return DeviceResult{
                .success = false,
                .device_id = request.device_id,
                .register_type = request.register_type,
                .data = 0,
                .execution_time = execution_time,
                .error_message = e.what(),
                .operation_id = operation_id,
                .timestamp = std::chrono::system_clock::now()
            };
        }
    }
    
    DeviceResultFuture read_device_async(DeviceId device_id, Register register_type) {
        return async_executor_->submit([this, device_id, register_type]() {
            return read_device_sync(device_id, register_type);
        });
    }
    
    void read_device_async(DeviceId device_id, Register register_type, DeviceResultCallback callback) {
        async_executor_->submit([this, device_id, register_type, callback]() {
            auto result = read_device_sync(device_id, register_type);
            callback(result);
        });
    }
    
    BulkResult bulk_read_sync(const std::vector<DeviceId>& device_ids, Register register_type) {
        return bulk_operator_->bulk_read_sync(device_ids, register_type);
    }
    
    void bulk_read_async(const std::vector<DeviceId>& device_ids, Register register_type, BulkResultCallback callback) {
        bulk_operator_->bulk_read_async(device_ids, register_type, callback);
    }
    
    std::shared_ptr<WebSocketClient> create_websocket_client() {
        auto ws_client = std::make_shared<WebSocketClient>(config_, access_token_);
        websocket_clients_.push_back(ws_client);
        return ws_client;
    }
    
    PerformanceMetrics get_performance_metrics() const {
        if (performance_monitor_) {
            return performance_monitor_->get_metrics();
        }
        return PerformanceMetrics{};
    }
    
    std::vector<DeviceInfo> get_device_registry() const {
        return device_registry_->get_all_devices();
    }
    
    DeviceInfo get_device_info(DeviceId device_id) const {
        return device_registry_->get_device_info(device_id);
    }

private:
    ClientConfig config_;
    std::unique_ptr<ConnectionPool> connection_pool_;
    std::unique_ptr<SecurityManager> security_manager_;
    std::unique_ptr<DeviceRegistry> device_registry_;
    std::unique_ptr<PerformanceMonitor> performance_monitor_;
    std::unique_ptr<BulkOperator> bulk_operator_;
    std::unique_ptr<AsyncExecutor> async_executor_;
    std::unique_ptr<KernelInterface> kernel_interface_;
    
    std::vector<std::shared_ptr<WebSocketClient>> websocket_clients_;
    
    // State management
    std::atomic<bool> initialized_;
    std::atomic<bool> authenticated_;
    std::atomic<bool> emergency_stop_active_;
    std::atomic<bool> kernel_available_{false};
    
    // Authentication state
    std::string access_token_;
    std::string refresh_token_;
    UserContext user_context_;
    
    // Thread safety
    mutable std::mutex state_mutex_;
    
    // Utility methods
    std::string generate_operation_id() const {
        static std::atomic<uint64_t> counter{0};
        return "op_" + std::to_string(++counter) + "_" + 
               std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(
                   std::chrono::steady_clock::now().time_since_epoch()
               ).count());
    }
    
    std::string client_type_to_string(ClientType type) const {
        switch (type) {
            case ClientType::Web: return "web";
            case ClientType::Python: return "python";
            case ClientType::Cpp: return "cpp";
            case ClientType::Mobile: return "mobile";
            default: return "cpp";
        }
    }
    
    AuthResult parse_auth_response(const std::string& response_body) {
        Json::Reader reader;
        Json::Value root;
        
        if (!reader.parse(response_body, root)) {
            throw ProtocolException("Failed to parse authentication response");
        }
        
        AuthResult result;
        result.success = true;
        result.access_token = root.get("access_token", "").asString();
        result.refresh_token = root.get("refresh_token", "").asString();
        result.token_type = root.get("token_type", "bearer").asString();
        result.expires_in = std::chrono::seconds(root.get("expires_in", 3600).asInt());
        
        // Parse user context
        Json::Value user_ctx = root["user_context"];
        result.user_context.user_id = user_ctx.get("user_id", "").asString();
        result.user_context.username = user_ctx.get("username", "").asString();
        
        // Parse clearance level
        std::string clearance = user_ctx.get("clearance_level", "RESTRICTED").asString();
        result.user_context.clearance_level = parse_clearance_level(clearance);
        
        // Parse authorized devices
        Json::Value devices = user_ctx["authorized_devices"];
        for (const auto& device : devices) {
            result.user_context.authorized_devices.push_back(device.asUInt());
        }
        
        // Parse API capabilities
        Json::Value capabilities = root["api_capabilities"];
        result.api_capabilities.max_requests_per_minute = capabilities.get("max_requests_per_minute", 100).asUInt();
        result.api_capabilities.max_concurrent_operations = capabilities.get("max_concurrent_operations", 5).asUInt();
        result.api_capabilities.websocket_enabled = capabilities.get("websocket_enabled", true).asBool();
        result.api_capabilities.bulk_operations_enabled = capabilities.get("bulk_operations_enabled", true).asBool();
        
        return result;
    }
    
    ClearanceLevel parse_clearance_level(const std::string& clearance) const {
        if (clearance == "RESTRICTED") return ClearanceLevel::RESTRICTED;
        if (clearance == "CONFIDENTIAL") return ClearanceLevel::CONFIDENTIAL;
        if (clearance == "SECRET") return ClearanceLevel::SECRET;
        if (clearance == "TOP_SECRET") return ClearanceLevel::TOP_SECRET;
        if (clearance == "SCI") return ClearanceLevel::SCI;
        return ClearanceLevel::RESTRICTED;
    }
    
    DeviceResult read_device_via_http(DeviceId device_id, Register register_type) {
        auto connection = connection_pool_->acquire_connection();
        
        std::string endpoint = "/api/v2/devices/" + std::to_string(device_id) + "/operations";
        
        Json::Value request_body;
        request_body["operation_type"] = "READ";
        request_body["operation_data"]["register"] = register_to_string(register_type);
        
        auto response = connection->post(endpoint, request_body);
        
        if (response.status_code != 200) {
            throw NetworkException("HTTP request failed", response.status_code);
        }
        
        return parse_device_result(response.body);
    }
    
    DeviceResult write_device_via_http(const WriteRequest& request) {
        auto connection = connection_pool_->acquire_connection();
        
        std::string endpoint = "/api/v2/devices/" + std::to_string(request.device_id) + "/operations";
        
        Json::Value request_body;
        request_body["operation_type"] = "WRITE";
        request_body["operation_data"]["register"] = register_to_string(request.register_type);
        request_body["operation_data"]["offset"] = request.offset;
        request_body["operation_data"]["value"] = "0x" + std::to_string(request.data);
        request_body["justification"] = request.justification;
        
        auto response = connection->post(endpoint, request_body);
        
        if (response.status_code != 200) {
            throw NetworkException("HTTP request failed", response.status_code);
        }
        
        return parse_device_result(response.body);
    }
    
    std::string register_to_string(Register reg) const {
        switch (reg) {
            case Register::STATUS: return "STATUS";
            case Register::CONFIG: return "CONFIG";
            case Register::DATA: return "DATA";
            case Register::TEMP: return "TEMP";
            case Register::VOLTAGE: return "VOLTAGE";
            case Register::ERROR: return "ERROR";
            default: return "STATUS";
        }
    }
    
    DeviceResult parse_device_result(const std::string& response_body) {
        Json::Reader reader;
        Json::Value root;
        
        if (!reader.parse(response_body, root)) {
            throw ProtocolException("Failed to parse device result response");
        }
        
        DeviceResult result;
        result.success = root.get("status", "FAILED").asString() == "SUCCESS";
        result.device_id = root.get("device_id", 0).asUInt();
        result.operation_id = root.get("operation_id", "").asString();
        result.error_message = root.get("error_message", "").asString();
        
        Json::Value result_data = root["result"];
        if (!result_data.isNull()) {
            std::string data_str = result_data.get("data", "0x0").asString();
            if (data_str.substr(0, 2) == "0x") {
                result.data = std::stoul(data_str.substr(2), nullptr, 16);
            } else {
                result.data = std::stoul(data_str);
            }
        }
        
        return result;
    }
};

// Client implementation

Client::Client(const std::string& base_url, const std::string& api_version)
    : pimpl_(std::make_unique<Impl>(ClientConfig{
        .base_url = base_url,
        .api_version = api_version
    })) {}

Client::Client(const ClientConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

Client::~Client() = default;

Client::Client(Client&&) noexcept = default;
Client& Client::operator=(Client&&) noexcept = default;

AuthResult Client::authenticate(const std::string& username, const std::string& password, ClientType client_type) {
    return pimpl_->authenticate(username, password, client_type);
}

DeviceResult Client::read_device_sync(DeviceId device_id, Register register_type) {
    return pimpl_->read_device_sync(device_id, register_type);
}

DeviceResult Client::write_device_sync(const WriteRequest& request) {
    return pimpl_->write_device_sync(request);
}

DeviceResultFuture Client::read_device_async(DeviceId device_id, Register register_type) {
    return pimpl_->read_device_async(device_id, register_type);
}

void Client::read_device_async(DeviceId device_id, Register register_type, DeviceResultCallback callback) {
    pimpl_->read_device_async(device_id, register_type, callback);
}

BulkResult Client::bulk_read_sync(const std::vector<DeviceId>& device_ids, Register register_type) {
    return pimpl_->bulk_read_sync(device_ids, register_type);
}

void Client::bulk_read_async(const std::vector<DeviceId>& device_ids, Register register_type, BulkResultCallback callback) {
    pimpl_->bulk_read_async(device_ids, register_type, callback);
}

std::shared_ptr<WebSocketClient> Client::create_websocket_client() {
    return pimpl_->create_websocket_client();
}

PerformanceMetrics Client::get_performance_metrics() const {
    return pimpl_->get_performance_metrics();
}

std::vector<DeviceInfo> Client::get_device_registry() const {
    return pimpl_->get_device_registry();
}

DeviceInfo Client::get_device_info(DeviceId device_id) const {
    return pimpl_->get_device_info(device_id);
}

} // namespace dsmil
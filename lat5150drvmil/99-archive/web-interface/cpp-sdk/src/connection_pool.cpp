#include "connection_pool.hpp"
#include "dsmil/exceptions.hpp"
#include <curl/curl.h>
#include <json/json.h>
#include <algorithm>
#include <future>

namespace dsmil {

// Connection implementation
class Connection::Impl {
public:
    explicit Impl(const ClientConfig& config)
        : config_(config)
        , curl_handle_(nullptr)
        , request_count_(0)
        , total_response_time_(0)
        , successful_requests_(0)
        , failed_requests_(0)
        , last_used_(std::chrono::system_clock::now())
    {
        initialize();
    }
    
    ~Impl() {
        cleanup();
    }
    
    void initialize() {
        curl_handle_ = curl_easy_init();
        if (!curl_handle_) {
            throw NetworkException("Failed to initialize CURL handle");
        }
        
        // Set basic options
        curl_easy_setopt(curl_handle_, CURLOPT_USERAGENT, config_.user_agent.c_str());
        curl_easy_setopt(curl_handle_, CURLOPT_TIMEOUT_MS, static_cast<long>(config_.timeout.count()));
        curl_easy_setopt(curl_handle_, CURLOPT_CONNECTTIMEOUT_MS, 10000);
        
        // SSL/TLS configuration
        if (config_.verify_ssl) {
            curl_easy_setopt(curl_handle_, CURLOPT_SSL_VERIFYPEER, 1L);
            curl_easy_setopt(curl_handle_, CURLOPT_SSL_VERIFYHOST, 2L);
            
            if (!config_.ca_cert_path.empty()) {
                curl_easy_setopt(curl_handle_, CURLOPT_CAINFO, config_.ca_cert_path.c_str());
            }
            
            if (!config_.client_cert_path.empty()) {
                curl_easy_setopt(curl_handle_, CURLOPT_SSLCERT, config_.client_cert_path.c_str());
            }
            
            if (!config_.client_key_path.empty()) {
                curl_easy_setopt(curl_handle_, CURLOPT_SSLKEY, config_.client_key_path.c_str());
            }
        } else {
            curl_easy_setopt(curl_handle_, CURLOPT_SSL_VERIFYPEER, 0L);
            curl_easy_setopt(curl_handle_, CURLOPT_SSL_VERIFYHOST, 0L);
        }
        
        // HTTP/2 support
        if (config_.enable_http2) {
            curl_easy_setopt(curl_handle_, CURLOPT_HTTP_VERSION, CURL_HTTP_VERSION_2_0);
        }
        
        // Compression support
        if (config_.enable_compression) {
            curl_easy_setopt(curl_handle_, CURLOPT_ACCEPT_ENCODING, "gzip,deflate");
        }
        
        // Keep-alive
        if (config_.enable_keep_alive) {
            curl_easy_setopt(curl_handle_, CURLOPT_TCP_KEEPALIVE, 1L);
            curl_easy_setopt(curl_handle_, CURLOPT_TCP_KEEPIDLE, config_.keep_alive_timeout.count());
        }
        
        // Callback for response data
        curl_easy_setopt(curl_handle_, CURLOPT_WRITEFUNCTION, write_callback);
        curl_easy_setopt(curl_handle_, CURLOPT_WRITEDATA, &response_buffer_);
        
        // Callback for headers
        curl_easy_setopt(curl_handle_, CURLOPT_HEADERFUNCTION, header_callback);
        curl_easy_setopt(curl_handle_, CURLOPT_HEADERDATA, &headers_);
        
        initialized_ = true;
    }
    
    void cleanup() {
        if (curl_handle_) {
            curl_easy_cleanup(curl_handle_);
            curl_handle_ = nullptr;
        }
        initialized_ = false;
    }
    
    HttpResponse execute_request() {
        if (!initialized_ || !curl_handle_) {
            return HttpResponse{
                .status_code = 0,
                .error_message = "Connection not initialized"
            };
        }
        
        // Clear previous response data
        response_buffer_.clear();
        headers_.clear();
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        CURLcode res = curl_easy_perform(curl_handle_);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto response_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        // Update statistics
        request_count_++;
        total_response_time_ += response_time.count();
        last_used_ = std::chrono::system_clock::now();
        
        HttpResponse response;
        response.response_time = response_time;
        
        if (res != CURLE_OK) {
            failed_requests_++;
            response.status_code = 0;
            response.error_message = curl_easy_strerror(res);
            return response;
        }
        
        // Get HTTP status code
        long http_code;
        curl_easy_getinfo(curl_handle_, CURLINFO_RESPONSE_CODE, &http_code);
        response.status_code = static_cast<int>(http_code);
        response.body = response_buffer_;
        
        if (http_code >= 200 && http_code < 300) {
            successful_requests_++;
        } else {
            failed_requests_++;
        }
        
        return response;
    }
    
    HttpResponse get(const std::string& endpoint) {
        std::string url = config_.base_url + endpoint;
        
        curl_easy_setopt(curl_handle_, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl_handle_, CURLOPT_HTTPGET, 1L);
        curl_easy_setopt(curl_handle_, CURLOPT_HTTPHEADER, create_headers());
        
        return execute_request();
    }
    
    HttpResponse post(const std::string& endpoint, const Json::Value& body) {
        std::string url = config_.base_url + endpoint;
        std::string json_body = json_writer_.write(body);
        
        curl_easy_setopt(curl_handle_, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl_handle_, CURLOPT_POSTFIELDS, json_body.c_str());
        curl_easy_setopt(curl_handle_, CURLOPT_POSTFIELDSIZE, json_body.length());
        
        struct curl_slist* headers = create_headers();
        headers = curl_slist_append(headers, "Content-Type: application/json");
        curl_easy_setopt(curl_handle_, CURLOPT_HTTPHEADER, headers);
        
        auto response = execute_request();
        curl_slist_free_all(headers);
        return response;
    }
    
    HttpResponse put(const std::string& endpoint, const Json::Value& body) {
        std::string url = config_.base_url + endpoint;
        std::string json_body = json_writer_.write(body);
        
        curl_easy_setopt(curl_handle_, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl_handle_, CURLOPT_CUSTOMREQUEST, "PUT");
        curl_easy_setopt(curl_handle_, CURLOPT_POSTFIELDS, json_body.c_str());
        curl_easy_setopt(curl_handle_, CURLOPT_POSTFIELDSIZE, json_body.length());
        
        struct curl_slist* headers = create_headers();
        headers = curl_slist_append(headers, "Content-Type: application/json");
        curl_easy_setopt(curl_handle_, CURLOPT_HTTPHEADER, headers);
        
        auto response = execute_request();
        curl_slist_free_all(headers);
        return response;
    }
    
    HttpResponse delete_(const std::string& endpoint) {
        std::string url = config_.base_url + endpoint;
        
        curl_easy_setopt(curl_handle_, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl_handle_, CURLOPT_CUSTOMREQUEST, "DELETE");
        curl_easy_setopt(curl_handle_, CURLOPT_HTTPHEADER, create_headers());
        
        return execute_request();
    }
    
    bool is_healthy() const {
        if (!initialized_) return false;
        
        // Check if connection is too old or has too many requests
        auto now = std::chrono::system_clock::now();
        auto age = std::chrono::duration_cast<std::chrono::minutes>(now - last_used_);
        
        if (age > std::chrono::minutes(30)) return false;  // Connection too old
        if (request_count_ > 10000) return false;  // Too many requests
        
        return success_rate() > 0.9;  // At least 90% success rate
    }
    
    void set_auth_token(const std::string& token) {
        auth_token_ = "Bearer " + token;
    }
    
    std::chrono::system_clock::time_point last_used() const {
        return last_used_;
    }
    
    uint64_t request_count() const {
        return request_count_;
    }
    
    std::chrono::milliseconds average_response_time() const {
        if (request_count_ == 0) return std::chrono::milliseconds(0);
        return std::chrono::milliseconds(total_response_time_ / request_count_);
    }
    
    double success_rate() const {
        if (request_count_ == 0) return 1.0;
        return static_cast<double>(successful_requests_) / request_count_;
    }

private:
    ClientConfig config_;
    CURL* curl_handle_;
    bool initialized_ = false;
    
    // Authentication
    std::string auth_token_;
    
    // Response data
    std::string response_buffer_;
    std::vector<std::string> headers_;
    Json::StreamWriterBuilder json_writer_builder_;
    Json::StreamWriter* json_writer_ = json_writer_builder_.newStreamWriter();
    
    // Statistics
    std::atomic<uint64_t> request_count_;
    std::atomic<uint64_t> total_response_time_;
    std::atomic<uint64_t> successful_requests_;
    std::atomic<uint64_t> failed_requests_;
    std::chrono::system_clock::time_point last_used_;
    
    // CURL callbacks
    static size_t write_callback(void* contents, size_t size, size_t nmemb, std::string* userp) {
        size_t total_size = size * nmemb;
        userp->append(static_cast<char*>(contents), total_size);
        return total_size;
    }
    
    static size_t header_callback(char* buffer, size_t size, size_t nitems, std::vector<std::string>* userdata) {
        size_t total_size = size * nitems;
        userdata->emplace_back(buffer, total_size);
        return total_size;
    }
    
    struct curl_slist* create_headers() {
        struct curl_slist* headers = nullptr;
        
        if (!auth_token_.empty()) {
            std::string auth_header = "Authorization: " + auth_token_;
            headers = curl_slist_append(headers, auth_header.c_str());
        }
        
        headers = curl_slist_append(headers, "Accept: application/json");
        
        return headers;
    }
};

// Connection public interface
Connection::Connection(const ClientConfig& config) 
    : pimpl_(std::make_unique<Impl>(config)) {}

Connection::~Connection() = default;

Connection::Connection(Connection&&) noexcept = default;
Connection& Connection::operator=(Connection&&) noexcept = default;

HttpResponse Connection::get(const std::string& endpoint) {
    return pimpl_->get(endpoint);
}

HttpResponse Connection::post(const std::string& endpoint, const Json::Value& body) {
    return pimpl_->post(endpoint, body);
}

HttpResponse Connection::put(const std::string& endpoint, const Json::Value& body) {
    return pimpl_->put(endpoint, body);
}

HttpResponse Connection::delete_(const std::string& endpoint) {
    return pimpl_->delete_(endpoint);
}

bool Connection::is_healthy() const {
    return pimpl_->is_healthy();
}

void Connection::set_auth_token(const std::string& token) {
    pimpl_->set_auth_token(token);
}

std::chrono::system_clock::time_point Connection::last_used() const {
    return pimpl_->last_used();
}

uint64_t Connection::request_count() const {
    return pimpl_->request_count();
}

std::chrono::milliseconds Connection::average_response_time() const {
    return pimpl_->average_response_time();
}

double Connection::success_rate() const {
    return pimpl_->success_rate();
}

// ConnectionPool implementation
class ConnectionPool::Impl {
public:
    explicit Impl(const ClientConfig& config)
        : client_config_(config)
        , pool_config_()
        , shutdown_requested_(false)
        , total_requests_(0)
        , successful_requests_(0)
        , failed_requests_(0)
        , total_wait_time_(0)
        , health_check_running_(false)
    {
        // Set default pool configuration based on client config
        pool_config_.max_connections = config.connection_pool_size;
        pool_config_.keep_alive_timeout = config.keep_alive_timeout;
    }
    
    ~Impl() {
        shutdown();
    }
    
    void initialize() {
        std::lock_guard<std::mutex> lock(pool_mutex_);
        
        if (initialized_) return;
        
        // Create initial connections
        for (uint32_t i = 0; i < pool_config_.min_connections; ++i) {
            auto connection = std::make_shared<Connection>(client_config_);
            if (!auth_token_.empty()) {
                connection->set_auth_token(auth_token_);
            }
            idle_connections_.push(connection);
        }
        
        // Start health check thread
        if (pool_config_.enable_health_checks) {
            health_check_running_ = true;
            health_check_thread_ = std::thread(&Impl::health_check_loop, this);
        }
        
        initialized_ = true;
    }
    
    void shutdown() {
        {
            std::lock_guard<std::mutex> lock(pool_mutex_);
            shutdown_requested_ = true;
        }
        
        // Stop health check thread
        if (health_check_running_) {
            health_check_running_ = false;
            if (health_check_thread_.joinable()) {
                health_check_thread_.join();
            }
        }
        
        // Notify all waiting threads
        connection_available_.notify_all();
        
        // Clear all connections
        std::lock_guard<std::mutex> lock(pool_mutex_);
        while (!idle_connections_.empty()) {
            idle_connections_.pop();
        }
        active_connections_.clear();
        
        initialized_ = false;
    }
    
    std::shared_ptr<Connection> acquire_connection() {
        return acquire_connection(std::chrono::milliseconds(5000));
    }
    
    std::shared_ptr<Connection> acquire_connection(std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(pool_mutex_);
        
        if (shutdown_requested_) {
            throw ConnectionPoolException("Connection pool is shutting down");
        }
        
        auto start_time = std::chrono::steady_clock::now();
        
        // Wait for available connection or timeout
        bool acquired = connection_available_.wait_for(lock, timeout, [this] {
            return !idle_connections_.empty() || 
                   active_connections_.size() < pool_config_.max_connections ||
                   shutdown_requested_;
        });
        
        if (shutdown_requested_) {
            throw ConnectionPoolException("Connection pool is shutting down");
        }
        
        if (!acquired) {
            throw TimeoutException("Timeout acquiring connection from pool", timeout);
        }
        
        std::shared_ptr<Connection> connection;
        
        if (!idle_connections_.empty()) {
            // Use existing idle connection
            connection = idle_connections_.front();
            idle_connections_.pop();
            
            // Check if connection is still healthy
            if (!connection->is_healthy()) {
                // Create new connection if unhealthy
                connection = std::make_shared<Connection>(client_config_);
                if (!auth_token_.empty()) {
                    connection->set_auth_token(auth_token_);
                }
            }
        } else if (active_connections_.size() < pool_config_.max_connections) {
            // Create new connection
            connection = std::make_shared<Connection>(client_config_);
            if (!auth_token_.empty()) {
                connection->set_auth_token(auth_token_);
            }
        }
        
        if (connection) {
            active_connections_.insert(connection);
            
            // Update wait time statistics
            auto wait_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - start_time
            );
            total_wait_time_ += wait_time.count();
            total_requests_++;
        }
        
        return connection;
    }
    
    void release_connection(std::shared_ptr<Connection> connection) {
        if (!connection) return;
        
        std::lock_guard<std::mutex> lock(pool_mutex_);
        
        // Remove from active connections
        active_connections_.erase(connection);
        
        // Add to idle connections if healthy and pool not full
        if (connection->is_healthy() && 
            idle_connections_.size() < pool_config_.max_connections) {
            idle_connections_.push(connection);
        }
        
        // Notify waiting threads
        connection_available_.notify_one();
    }
    
    void configure(const PoolConfig& config) {
        std::lock_guard<std::mutex> lock(pool_mutex_);
        pool_config_ = config;
    }
    
    void set_auth_token(const std::string& token) {
        std::lock_guard<std::mutex> lock(pool_mutex_);
        auth_token_ = token;
        
        // Update all existing connections
        auto temp_idle = std::queue<std::shared_ptr<Connection>>();
        while (!idle_connections_.empty()) {
            auto conn = idle_connections_.front();
            idle_connections_.pop();
            conn->set_auth_token(token);
            temp_idle.push(conn);
        }
        idle_connections_ = std::move(temp_idle);
        
        for (auto& conn : active_connections_) {
            conn->set_auth_token(token);
        }
    }
    
    ConnectionPool::PoolStats get_stats() const {
        std::lock_guard<std::mutex> lock(pool_mutex_);
        
        ConnectionPool::PoolStats stats;
        stats.total_connections = idle_connections_.size() + active_connections_.size();
        stats.active_connections = active_connections_.size();
        stats.idle_connections = idle_connections_.size();
        stats.pending_requests = 0;  // TODO: Track pending requests
        stats.total_requests = total_requests_;
        stats.successful_requests = successful_requests_;
        stats.failed_requests = failed_requests_;
        
        if (total_requests_ > 0) {
            stats.average_wait_time = std::chrono::milliseconds(total_wait_time_ / total_requests_);
            stats.pool_utilization = static_cast<double>(active_connections_.size()) / pool_config_.max_connections;
        }
        
        return stats;
    }
    
    bool is_healthy() const {
        std::lock_guard<std::mutex> lock(pool_mutex_);
        
        if (shutdown_requested_) return false;
        
        // Check if we have minimum number of healthy connections
        uint32_t healthy_connections = 0;
        
        auto temp_idle = idle_connections_;
        while (!temp_idle.empty()) {
            if (temp_idle.front()->is_healthy()) {
                healthy_connections++;
            }
            temp_idle.pop();
        }
        
        for (const auto& conn : active_connections_) {
            if (conn->is_healthy()) {
                healthy_connections++;
            }
        }
        
        return healthy_connections >= pool_config_.min_connections;
    }
    
    void perform_health_check() {
        std::lock_guard<std::mutex> lock(pool_mutex_);
        
        // Check idle connections and remove unhealthy ones
        auto temp_idle = std::queue<std::shared_ptr<Connection>>();
        while (!idle_connections_.empty()) {
            auto conn = idle_connections_.front();
            idle_connections_.pop();
            
            if (conn->is_healthy()) {
                temp_idle.push(conn);
            }
            // Unhealthy connections are simply discarded
        }
        idle_connections_ = std::move(temp_idle);
        
        // Create new connections if needed
        uint32_t total_connections = idle_connections_.size() + active_connections_.size();
        while (total_connections < pool_config_.min_connections) {
            auto connection = std::make_shared<Connection>(client_config_);
            if (!auth_token_.empty()) {
                connection->set_auth_token(auth_token_);
            }
            idle_connections_.push(connection);
            total_connections++;
        }
    }

private:
    ClientConfig client_config_;
    PoolConfig pool_config_;
    
    // Connection storage
    std::queue<std::shared_ptr<Connection>> idle_connections_;
    std::set<std::shared_ptr<Connection>> active_connections_;
    
    // Authentication
    std::string auth_token_;
    
    // Threading
    mutable std::mutex pool_mutex_;
    std::condition_variable connection_available_;
    std::thread health_check_thread_;
    
    // State
    std::atomic<bool> initialized_{false};
    std::atomic<bool> shutdown_requested_;
    std::atomic<bool> health_check_running_;
    
    // Statistics
    std::atomic<uint64_t> total_requests_;
    std::atomic<uint64_t> successful_requests_;
    std::atomic<uint64_t> failed_requests_;
    std::atomic<uint64_t> total_wait_time_;
    
    void health_check_loop() {
        while (health_check_running_) {
            std::this_thread::sleep_for(pool_config_.health_check_interval);
            
            if (!health_check_running_) break;
            
            perform_health_check();
        }
    }
};

// ConnectionPool public interface
ConnectionPool::ConnectionPool(const ClientConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

ConnectionPool::~ConnectionPool() = default;

void ConnectionPool::initialize() {
    pimpl_->initialize();
}

void ConnectionPool::shutdown() {
    pimpl_->shutdown();
}

std::shared_ptr<Connection> ConnectionPool::acquire_connection() {
    return pimpl_->acquire_connection();
}

std::shared_ptr<Connection> ConnectionPool::acquire_connection(std::chrono::milliseconds timeout) {
    return pimpl_->acquire_connection(timeout);
}

void ConnectionPool::release_connection(std::shared_ptr<Connection> connection) {
    pimpl_->release_connection(connection);
}

void ConnectionPool::configure(const PoolConfig& config) {
    pimpl_->configure(config);
}

void ConnectionPool::set_auth_token(const std::string& token) {
    pimpl_->set_auth_token(token);
}

ConnectionPool::PoolStats ConnectionPool::get_stats() const {
    return pimpl_->get_stats();
}

bool ConnectionPool::is_healthy() const {
    return pimpl_->is_healthy();
}

void ConnectionPool::perform_health_check() {
    pimpl_->perform_health_check();
}

} // namespace dsmil
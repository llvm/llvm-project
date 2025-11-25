#pragma once

#include "dsmil/config.hpp"
#include "dsmil/types.hpp"
#include <memory>
#include <string>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <chrono>
#include <thread>

namespace dsmil {

struct HttpResponse {
    int status_code;
    std::string body;
    std::chrono::milliseconds response_time;
    std::string error_message;
};

class Connection {
public:
    Connection(const ClientConfig& config);
    ~Connection();
    
    // Non-copyable but movable
    Connection(const Connection&) = delete;
    Connection& operator=(const Connection&) = delete;
    Connection(Connection&&) noexcept;
    Connection& operator=(Connection&&) noexcept;
    
    // HTTP operations
    HttpResponse get(const std::string& endpoint);
    HttpResponse post(const std::string& endpoint, const Json::Value& body);
    HttpResponse put(const std::string& endpoint, const Json::Value& body);
    HttpResponse delete_(const std::string& endpoint);
    
    // Connection management
    bool is_healthy() const;
    void set_auth_token(const std::string& token);
    std::chrono::system_clock::time_point last_used() const;
    uint64_t request_count() const;
    
    // Performance metrics
    std::chrono::milliseconds average_response_time() const;
    double success_rate() const;

private:
    class Impl;
    std::unique_ptr<Impl> pimpl_;
};

class ConnectionPool {
public:
    explicit ConnectionPool(const ClientConfig& config);
    ~ConnectionPool();
    
    // Pool management
    void initialize();
    void shutdown();
    
    // Connection acquisition and release
    std::shared_ptr<Connection> acquire_connection();
    std::shared_ptr<Connection> acquire_connection(std::chrono::milliseconds timeout);
    void release_connection(std::shared_ptr<Connection> connection);
    
    // Configuration
    void configure(const PoolConfig& config);
    void set_auth_token(const std::string& token);
    
    // Pool statistics
    struct PoolStats {
        uint32_t total_connections;
        uint32_t active_connections;
        uint32_t idle_connections;
        uint32_t pending_requests;
        uint64_t total_requests;
        uint64_t successful_requests;
        uint64_t failed_requests;
        std::chrono::milliseconds average_wait_time;
        std::chrono::milliseconds average_response_time;
        double pool_utilization;
    };
    
    PoolStats get_stats() const;
    
    // Health monitoring
    bool is_healthy() const;
    void perform_health_check();
    
private:
    class Impl;
    std::unique_ptr<Impl> pimpl_;
};

} // namespace dsmil
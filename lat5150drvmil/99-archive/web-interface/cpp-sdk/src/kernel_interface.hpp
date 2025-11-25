#pragma once

#include "dsmil/types.hpp"
#include "dsmil/config.hpp"
#include <memory>
#include <string>

namespace dsmil {

/**
 * @brief Direct kernel module interface for high-performance hardware access
 * 
 * This class provides direct communication with the DSMIL kernel module
 * for maximum performance and minimal latency operations.
 */
class KernelInterface {
public:
    explicit KernelInterface(const ClientConfig& config);
    ~KernelInterface();
    
    // Non-copyable
    KernelInterface(const KernelInterface&) = delete;
    KernelInterface& operator=(const KernelInterface&) = delete;
    
    /**
     * @brief Initialize kernel module interface
     * 
     * @return bool True if initialization successful
     */
    bool initialize();
    
    /**
     * @brief Cleanup kernel module interface
     */
    void cleanup();
    
    /**
     * @brief Check if kernel module is available
     * 
     * @return bool True if kernel module is available
     */
    bool is_available() const;
    
    /**
     * @brief Test communication with kernel module
     * 
     * @return bool True if communication test successful
     */
    bool test_communication();
    
    /**
     * @brief Read from device via kernel module
     * 
     * @param device_id Device ID
     * @param register_type Register to read
     * @return DeviceResult Operation result
     */
    DeviceResult read_device(DeviceId device_id, Register register_type);
    
    /**
     * @brief Write to device via kernel module
     * 
     * @param request Write request
     * @return DeviceResult Operation result
     */
    DeviceResult write_device(const WriteRequest& request);
    
    /**
     * @brief Get device status via kernel module
     * 
     * @param device_id Device ID
     * @return DeviceResult Status result
     */
    DeviceResult get_device_status(DeviceId device_id);
    
    /**
     * @brief Reset device via kernel module
     * 
     * @param device_id Device ID
     * @return DeviceResult Reset result
     */
    DeviceResult reset_device(DeviceId device_id);
    
    /**
     * @brief Bulk read from multiple devices
     * 
     * @param device_ids Vector of device IDs
     * @param register_type Register to read
     * @return BulkResult Bulk operation result
     */
    BulkResult bulk_read_devices(const std::vector<DeviceId>& device_ids, Register register_type);
    
    /**
     * @brief Get kernel module statistics
     */
    struct KernelStats {
        uint64_t total_operations;
        uint64_t successful_operations;
        uint64_t failed_operations;
        uint64_t read_operations;
        uint64_t write_operations;
        std::chrono::milliseconds average_latency;
        std::chrono::milliseconds min_latency;
        std::chrono::milliseconds max_latency;
        bool emergency_mode;
        uint32_t quarantine_violations;
        std::chrono::system_clock::time_point last_operation;
    };
    
    KernelStats get_kernel_stats() const;
    
    /**
     * @brief Set emergency stop via kernel module
     * 
     * @param reason Emergency stop reason
     * @return bool True if emergency stop set successfully
     */
    bool set_emergency_stop(const std::string& reason);
    
    /**
     * @brief Clear emergency stop
     * 
     * @return bool True if emergency stop cleared successfully
     */
    bool clear_emergency_stop();
    
    /**
     * @brief Check if emergency stop is active
     * 
     * @return bool True if emergency stop is active
     */
    bool is_emergency_stop_active() const;

private:
    class Impl;
    std::unique_ptr<Impl> pimpl_;
};

} // namespace dsmil
#include "kernel_interface.hpp"
#include "dsmil/exceptions.hpp"

#include <sys/types.h>
#include <sys/stat.h>
#include <sys/ioctl.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <cstring>
#include <cstdint>

namespace dsmil {

// IOCTL command codes (must match kernel module)
#define DSMIL_IOCTL_MAGIC 'D'
#define DSMIL_IOCTL_READ         _IOWR(DSMIL_IOCTL_MAGIC, 0, struct dsmil_ioctl_data)
#define DSMIL_IOCTL_WRITE        _IOWR(DSMIL_IOCTL_MAGIC, 1, struct dsmil_ioctl_data)
#define DSMIL_IOCTL_STATUS       _IOWR(DSMIL_IOCTL_MAGIC, 2, struct dsmil_ioctl_data)
#define DSMIL_IOCTL_RESET        _IOWR(DSMIL_IOCTL_MAGIC, 3, struct dsmil_ioctl_data)
#define DSMIL_IOCTL_BULK_READ    _IOWR(DSMIL_IOCTL_MAGIC, 4, struct dsmil_bulk_ioctl_data)
#define DSMIL_IOCTL_EMERGENCY    _IOWR(DSMIL_IOCTL_MAGIC, 5, struct dsmil_emergency_data)
#define DSMIL_IOCTL_STATS        _IOR(DSMIL_IOCTL_MAGIC, 6, struct dsmil_stats_data)

// IOCTL data structures (must match kernel module)
struct dsmil_ioctl_data {
    uint32_t device_id;
    uint32_t register_type;
    uint32_t offset;
    uint32_t data_length;
    uint32_t data;
    uint32_t status;
    char justification[256];
    uint64_t timestamp;
} __attribute__((packed));

struct dsmil_bulk_ioctl_data {
    uint32_t device_count;
    uint32_t register_type;
    uint32_t device_ids[84];  // Maximum 84 devices
    uint32_t results[84];
    uint32_t statuses[84];
    uint64_t execution_times[84];
    uint32_t successful_count;
    uint32_t failed_count;
} __attribute__((packed));

struct dsmil_emergency_data {
    uint32_t command;  // 1=set, 0=clear
    char reason[512];
    uint64_t timestamp;
    uint32_t status;
} __attribute__((packed));

struct dsmil_stats_data {
    uint64_t total_operations;
    uint64_t successful_operations;
    uint64_t failed_operations;
    uint64_t read_operations;
    uint64_t write_operations;
    uint32_t average_latency_us;
    uint32_t min_latency_us;
    uint32_t max_latency_us;
    uint32_t emergency_mode;
    uint32_t quarantine_violations;
    uint64_t last_operation_time;
} __attribute__((packed));

class KernelInterface::Impl {
public:
    explicit Impl(const ClientConfig& config)
        : config_(config)
        , device_fd_(-1)
        , available_(false)
        , emergency_stop_active_(false)
    {}
    
    ~Impl() {
        cleanup();
    }
    
    bool initialize() {
        // Check if kernel module device exists
        if (access(config_.kernel_module_path.c_str(), F_OK) != 0) {
            return false;  // Device not found
        }
        
        // Open device file
        device_fd_ = open(config_.kernel_module_path.c_str(), O_RDWR);
        if (device_fd_ < 0) {
            return false;  // Failed to open device
        }
        
        // Test basic communication
        if (!test_communication()) {
            close(device_fd_);
            device_fd_ = -1;
            return false;
        }
        
        available_ = true;
        return true;
    }
    
    void cleanup() {
        if (device_fd_ >= 0) {
            close(device_fd_);
            device_fd_ = -1;
        }
        available_ = false;
    }
    
    bool is_available() const {
        return available_;
    }
    
    bool test_communication() {
        if (device_fd_ < 0) return false;
        
        struct dsmil_ioctl_data test_data = {};
        test_data.device_id = 0;  // Test with device 0
        
        int result = ioctl(device_fd_, DSMIL_IOCTL_STATUS, &test_data);
        return (result == 0);
    }
    
    DeviceResult read_device(DeviceId device_id, Register register_type) {
        if (!available_ || device_fd_ < 0) {
            throw KernelModuleException("Kernel module not available");
        }
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        struct dsmil_ioctl_data ioctl_data = {};
        ioctl_data.device_id = device_id;
        ioctl_data.register_type = static_cast<uint32_t>(register_type);
        ioctl_data.offset = 0;
        ioctl_data.data_length = 4;  // 32-bit read
        
        int result = ioctl(device_fd_, DSMIL_IOCTL_READ, &ioctl_data);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto execution_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        DeviceResult device_result;
        device_result.device_id = device_id;
        device_result.register_type = register_type;
        device_result.execution_time = execution_time;
        device_result.timestamp = std::chrono::system_clock::now();
        
        if (result == 0 && ioctl_data.status == 0) {
            device_result.success = true;
            device_result.data = ioctl_data.data;
        } else {
            device_result.success = false;
            device_result.data = 0;
            
            if (result < 0) {
                device_result.error_message = "IOCTL failed: " + std::string(strerror(errno));
            } else {
                device_result.error_message = "Device operation failed with status: " + std::to_string(ioctl_data.status);
            }
        }
        
        return device_result;
    }
    
    DeviceResult write_device(const WriteRequest& request) {
        if (!available_ || device_fd_ < 0) {
            throw KernelModuleException("Kernel module not available");
        }
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        struct dsmil_ioctl_data ioctl_data = {};
        ioctl_data.device_id = request.device_id;
        ioctl_data.register_type = static_cast<uint32_t>(request.register_type);
        ioctl_data.offset = request.offset;
        ioctl_data.data = request.data;
        ioctl_data.data_length = 4;  // 32-bit write
        
        // Copy justification (truncate if necessary)
        strncpy(ioctl_data.justification, request.justification.c_str(), sizeof(ioctl_data.justification) - 1);
        ioctl_data.justification[sizeof(ioctl_data.justification) - 1] = '\0';
        
        int result = ioctl(device_fd_, DSMIL_IOCTL_WRITE, &ioctl_data);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto execution_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        DeviceResult device_result;
        device_result.device_id = request.device_id;
        device_result.register_type = request.register_type;
        device_result.execution_time = execution_time;
        device_result.timestamp = std::chrono::system_clock::now();
        
        if (result == 0 && ioctl_data.status == 0) {
            device_result.success = true;
            device_result.data = ioctl_data.data;
        } else {
            device_result.success = false;
            device_result.data = 0;
            
            if (result < 0) {
                device_result.error_message = "IOCTL failed: " + std::string(strerror(errno));
                
                // Check for specific error conditions
                if (errno == EPERM) {
                    throw PermissionException("Insufficient permissions for device write operation");
                } else if (errno == EBUSY) {
                    throw DeviceException(ErrorCode::DEVICE_BUSY, "Device is busy", request.device_id);
                } else if (errno == EACCES) {
                    throw QuarantineException(request.device_id, "Device is quarantined");
                }
            } else {
                device_result.error_message = "Device write failed with status: " + std::to_string(ioctl_data.status);
                
                // Interpret kernel module status codes
                switch (ioctl_data.status) {
                    case 1:
                        throw QuarantineException(request.device_id, "Write blocked by quarantine protection");
                    case 2:
                        throw PermissionException("Insufficient clearance for write operation");
                    case 3:
                        throw SecurityViolationException("Security violation detected");
                    case 4:
                        throw EmergencyStopException("Emergency stop is active");
                    default:
                        break;
                }
            }
        }
        
        return device_result;
    }
    
    DeviceResult get_device_status(DeviceId device_id) {
        return read_device(device_id, Register::STATUS);
    }
    
    DeviceResult reset_device(DeviceId device_id) {
        if (!available_ || device_fd_ < 0) {
            throw KernelModuleException("Kernel module not available");
        }
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        struct dsmil_ioctl_data ioctl_data = {};
        ioctl_data.device_id = device_id;
        
        int result = ioctl(device_fd_, DSMIL_IOCTL_RESET, &ioctl_data);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto execution_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        DeviceResult device_result;
        device_result.device_id = device_id;
        device_result.register_type = Register::STATUS;  // Reset operation affects status
        device_result.execution_time = execution_time;
        device_result.timestamp = std::chrono::system_clock::now();
        
        if (result == 0 && ioctl_data.status == 0) {
            device_result.success = true;
            device_result.data = ioctl_data.data;
        } else {
            device_result.success = false;
            device_result.data = 0;
            
            if (result < 0) {
                device_result.error_message = "Reset IOCTL failed: " + std::string(strerror(errno));
            } else {
                device_result.error_message = "Device reset failed with status: " + std::to_string(ioctl_data.status);
            }
        }
        
        return device_result;
    }
    
    BulkResult bulk_read_devices(const std::vector<DeviceId>& device_ids, Register register_type) {
        if (!available_ || device_fd_ < 0) {
            throw KernelModuleException("Kernel module not available");
        }
        
        if (device_ids.size() > 84) {
            throw ConfigurationException("Too many devices for bulk operation", "device_ids");
        }
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        struct dsmil_bulk_ioctl_data bulk_data = {};
        bulk_data.device_count = device_ids.size();
        bulk_data.register_type = static_cast<uint32_t>(register_type);
        
        // Copy device IDs
        for (size_t i = 0; i < device_ids.size(); ++i) {
            bulk_data.device_ids[i] = device_ids[i];
        }
        
        int result = ioctl(device_fd_, DSMIL_IOCTL_BULK_READ, &bulk_data);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto total_execution_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        BulkResult bulk_result;
        bulk_result.total_execution_time = total_execution_time;
        bulk_result.bulk_operation_id = "bulk_kernel_" + std::to_string(
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now().time_since_epoch()
            ).count()
        );
        
        if (result == 0) {
            // Process individual results
            for (size_t i = 0; i < device_ids.size(); ++i) {
                DeviceResult device_result;
                device_result.device_id = device_ids[i];
                device_result.register_type = register_type;
                device_result.timestamp = std::chrono::system_clock::now();
                device_result.execution_time = std::chrono::microseconds(bulk_data.execution_times[i]);
                
                if (bulk_data.statuses[i] == 0) {
                    device_result.success = true;
                    device_result.data = bulk_data.results[i];
                } else {
                    device_result.success = false;
                    device_result.data = 0;
                    device_result.error_message = "Device operation failed with status: " + 
                                                 std::to_string(bulk_data.statuses[i]);
                }
                
                bulk_result.results.push_back(device_result);
            }
            
            // Set summary
            bulk_result.summary.total = device_ids.size();
            bulk_result.summary.successful = bulk_data.successful_count;
            bulk_result.summary.failed = bulk_data.failed_count;
            bulk_result.summary.denied = 0;  // TODO: Track denials separately
            bulk_result.summary.timeouts = 0;  // TODO: Track timeouts separately
            
            bulk_result.overall_success = (bulk_data.failed_count == 0);
            
        } else {
            bulk_result.overall_success = false;
            
            // Create error results for all devices
            for (DeviceId device_id : device_ids) {
                DeviceResult device_result;
                device_result.success = false;
                device_result.device_id = device_id;
                device_result.register_type = register_type;
                device_result.data = 0;
                device_result.execution_time = std::chrono::milliseconds(0);
                device_result.timestamp = std::chrono::system_clock::now();
                device_result.error_message = "Bulk IOCTL failed: " + std::string(strerror(errno));
                
                bulk_result.results.push_back(device_result);
            }
            
            bulk_result.summary.total = device_ids.size();
            bulk_result.summary.successful = 0;
            bulk_result.summary.failed = device_ids.size();
            bulk_result.summary.denied = 0;
            bulk_result.summary.timeouts = 0;
        }
        
        return bulk_result;
    }
    
    KernelInterface::KernelStats get_kernel_stats() const {
        if (!available_ || device_fd_ < 0) {
            throw KernelModuleException("Kernel module not available");
        }
        
        struct dsmil_stats_data stats_data = {};
        int result = ioctl(device_fd_, DSMIL_IOCTL_STATS, &stats_data);
        
        if (result != 0) {
            throw KernelModuleException("Failed to get kernel stats: " + std::string(strerror(errno)));
        }
        
        KernelInterface::KernelStats stats;
        stats.total_operations = stats_data.total_operations;
        stats.successful_operations = stats_data.successful_operations;
        stats.failed_operations = stats_data.failed_operations;
        stats.read_operations = stats_data.read_operations;
        stats.write_operations = stats_data.write_operations;
        stats.average_latency = std::chrono::microseconds(stats_data.average_latency_us);
        stats.min_latency = std::chrono::microseconds(stats_data.min_latency_us);
        stats.max_latency = std::chrono::microseconds(stats_data.max_latency_us);
        stats.emergency_mode = (stats_data.emergency_mode != 0);
        stats.quarantine_violations = stats_data.quarantine_violations;
        stats.last_operation = std::chrono::system_clock::from_time_t(stats_data.last_operation_time);
        
        return stats;
    }
    
    bool set_emergency_stop(const std::string& reason) {
        if (!available_ || device_fd_ < 0) {
            return false;
        }
        
        struct dsmil_emergency_data emergency_data = {};
        emergency_data.command = 1;  // Set emergency stop
        strncpy(emergency_data.reason, reason.c_str(), sizeof(emergency_data.reason) - 1);
        emergency_data.reason[sizeof(emergency_data.reason) - 1] = '\0';
        
        int result = ioctl(device_fd_, DSMIL_IOCTL_EMERGENCY, &emergency_data);
        
        if (result == 0 && emergency_data.status == 0) {
            emergency_stop_active_ = true;
            return true;
        }
        
        return false;
    }
    
    bool clear_emergency_stop() {
        if (!available_ || device_fd_ < 0) {
            return false;
        }
        
        struct dsmil_emergency_data emergency_data = {};
        emergency_data.command = 0;  // Clear emergency stop
        
        int result = ioctl(device_fd_, DSMIL_IOCTL_EMERGENCY, &emergency_data);
        
        if (result == 0 && emergency_data.status == 0) {
            emergency_stop_active_ = false;
            return true;
        }
        
        return false;
    }
    
    bool is_emergency_stop_active() const {
        return emergency_stop_active_;
    }

private:
    ClientConfig config_;
    int device_fd_;
    bool available_;
    std::atomic<bool> emergency_stop_active_;
};

// KernelInterface public interface
KernelInterface::KernelInterface(const ClientConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

KernelInterface::~KernelInterface() = default;

bool KernelInterface::initialize() {
    return pimpl_->initialize();
}

void KernelInterface::cleanup() {
    pimpl_->cleanup();
}

bool KernelInterface::is_available() const {
    return pimpl_->is_available();
}

bool KernelInterface::test_communication() {
    return pimpl_->test_communication();
}

DeviceResult KernelInterface::read_device(DeviceId device_id, Register register_type) {
    return pimpl_->read_device(device_id, register_type);
}

DeviceResult KernelInterface::write_device(const WriteRequest& request) {
    return pimpl_->write_device(request);
}

DeviceResult KernelInterface::get_device_status(DeviceId device_id) {
    return pimpl_->get_device_status(device_id);
}

DeviceResult KernelInterface::reset_device(DeviceId device_id) {
    return pimpl_->reset_device(device_id);
}

BulkResult KernelInterface::bulk_read_devices(const std::vector<DeviceId>& device_ids, Register register_type) {
    return pimpl_->bulk_read_devices(device_ids, register_type);
}

KernelInterface::KernelStats KernelInterface::get_kernel_stats() const {
    return pimpl_->get_kernel_stats();
}

bool KernelInterface::set_emergency_stop(const std::string& reason) {
    return pimpl_->set_emergency_stop(reason);
}

bool KernelInterface::clear_emergency_stop() {
    return pimpl_->clear_emergency_stop();
}

bool KernelInterface::is_emergency_stop_active() const {
    return pimpl_->is_emergency_stop_active();
}

} // namespace dsmil
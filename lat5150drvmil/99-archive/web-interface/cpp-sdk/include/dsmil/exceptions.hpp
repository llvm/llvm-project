#pragma once

#include "types.hpp"
#include <stdexcept>
#include <string>
#include <chrono>

namespace dsmil {

/**
 * @brief Base exception class for DSMIL SDK
 */
class DSMILException : public std::runtime_error {
public:
    explicit DSMILException(const std::string& message)
        : std::runtime_error(message), error_code_(ErrorCode::INTERNAL_ERROR) {}
    
    DSMILException(ErrorCode code, const std::string& message)
        : std::runtime_error(message), error_code_(code) {}
    
    ErrorCode error_code() const noexcept { return error_code_; }
    
private:
    ErrorCode error_code_;
};

/**
 * @brief Authentication related exceptions
 */
class AuthenticationException : public DSMILException {
public:
    explicit AuthenticationException(const std::string& message)
        : DSMILException(ErrorCode::AUTHENTICATION_FAILED, message) {}
};

/**
 * @brief Permission/authorization related exceptions
 */
class PermissionException : public DSMILException {
public:
    explicit PermissionException(const std::string& message)
        : DSMILException(ErrorCode::PERMISSION_DENIED, message) {}
    
    PermissionException(const std::string& message, ClearanceLevel required, ClearanceLevel current)
        : DSMILException(ErrorCode::PERMISSION_DENIED, message)
        , required_clearance_(required)
        , current_clearance_(current) {}
    
    ClearanceLevel required_clearance() const noexcept { return required_clearance_; }
    ClearanceLevel current_clearance() const noexcept { return current_clearance_; }
    
private:
    ClearanceLevel required_clearance_ = ClearanceLevel::RESTRICTED;
    ClearanceLevel current_clearance_ = ClearanceLevel::RESTRICTED;
};

/**
 * @brief Device related exceptions
 */
class DeviceException : public DSMILException {
public:
    explicit DeviceException(const std::string& message)
        : DSMILException(ErrorCode::DEVICE_NOT_FOUND, message) {}
    
    DeviceException(ErrorCode code, const std::string& message, DeviceId device_id)
        : DSMILException(code, message), device_id_(device_id) {}
    
    DeviceId device_id() const noexcept { return device_id_; }
    
private:
    DeviceId device_id_ = 0;
};

/**
 * @brief Device quarantine exceptions
 */
class QuarantineException : public DeviceException {
public:
    explicit QuarantineException(DeviceId device_id)
        : DeviceException(
            ErrorCode::DEVICE_QUARANTINED, 
            "Device " + std::to_string(device_id) + " is quarantined",
            device_id
        ) {}
    
    QuarantineException(DeviceId device_id, const std::string& reason)
        : DeviceException(
            ErrorCode::DEVICE_QUARANTINED,
            "Device " + std::to_string(device_id) + " is quarantined: " + reason,
            device_id
        ) {}
};

/**
 * @brief Network related exceptions
 */
class NetworkException : public DSMILException {
public:
    explicit NetworkException(const std::string& message)
        : DSMILException(ErrorCode::NETWORK_ERROR, message) {}
    
    NetworkException(const std::string& message, int http_status)
        : DSMILException(ErrorCode::NETWORK_ERROR, message)
        , http_status_(http_status) {}
    
    int http_status() const noexcept { return http_status_; }
    
private:
    int http_status_ = 0;
};

/**
 * @brief Timeout related exceptions
 */
class TimeoutException : public DSMILException {
public:
    explicit TimeoutException(const std::string& message)
        : DSMILException(ErrorCode::TIMEOUT, message) {}
    
    TimeoutException(const std::string& message, std::chrono::milliseconds timeout_duration)
        : DSMILException(ErrorCode::TIMEOUT, message)
        , timeout_duration_(timeout_duration) {}
    
    std::chrono::milliseconds timeout_duration() const noexcept { return timeout_duration_; }
    
private:
    std::chrono::milliseconds timeout_duration_{0};
};

/**
 * @brief Rate limiting exceptions
 */
class RateLimitException : public DSMILException {
public:
    explicit RateLimitException(const std::string& message)
        : DSMILException(ErrorCode::RATE_LIMITED, message) {}
    
    RateLimitException(const std::string& message, std::chrono::seconds retry_after)
        : DSMILException(ErrorCode::RATE_LIMITED, message)
        , retry_after_(retry_after) {}
    
    std::chrono::seconds retry_after() const noexcept { return retry_after_; }
    
private:
    std::chrono::seconds retry_after_{0};
};

/**
 * @brief Protocol related exceptions
 */
class ProtocolException : public DSMILException {
public:
    explicit ProtocolException(const std::string& message)
        : DSMILException(ErrorCode::PROTOCOL_ERROR, message) {}
    
    ProtocolException(const std::string& message, const std::string& expected, const std::string& received)
        : DSMILException(ErrorCode::PROTOCOL_ERROR, message)
        , expected_(expected)
        , received_(received) {}
    
    const std::string& expected() const noexcept { return expected_; }
    const std::string& received() const noexcept { return received_; }
    
private:
    std::string expected_;
    std::string received_;
};

/**
 * @brief Hardware related exceptions
 */
class HardwareException : public DSMILException {
public:
    explicit HardwareException(const std::string& message)
        : DSMILException(ErrorCode::HARDWARE_ERROR, message) {}
    
    HardwareException(const std::string& message, const std::string& hardware_component)
        : DSMILException(ErrorCode::HARDWARE_ERROR, message)
        , hardware_component_(hardware_component) {}
    
    const std::string& hardware_component() const noexcept { return hardware_component_; }
    
private:
    std::string hardware_component_;
};

/**
 * @brief Kernel module related exceptions
 */
class KernelModuleException : public DSMILException {
public:
    explicit KernelModuleException(const std::string& message)
        : DSMILException(ErrorCode::KERNEL_MODULE_ERROR, message) {}
    
    KernelModuleException(const std::string& message, int ioctl_error)
        : DSMILException(ErrorCode::KERNEL_MODULE_ERROR, message)
        , ioctl_error_(ioctl_error) {}
    
    int ioctl_error() const noexcept { return ioctl_error_; }
    
private:
    int ioctl_error_ = 0;
};

/**
 * @brief Security violation exceptions
 */
class SecurityViolationException : public DSMILException {
public:
    explicit SecurityViolationException(const std::string& message)
        : DSMILException(ErrorCode::SECURITY_VIOLATION, message) {}
    
    SecurityViolationException(const std::string& message, Severity severity)
        : DSMILException(ErrorCode::SECURITY_VIOLATION, message)
        , severity_(severity) {}
    
    Severity severity() const noexcept { return severity_; }
    
private:
    Severity severity_ = Severity::HIGH;
};

/**
 * @brief Emergency stop exceptions
 */
class EmergencyStopException : public DSMILException {
public:
    explicit EmergencyStopException(const std::string& message)
        : DSMILException(ErrorCode::EMERGENCY_STOP, message) {}
    
    EmergencyStopException(const std::string& message, const std::string& reason)
        : DSMILException(ErrorCode::EMERGENCY_STOP, message)
        , emergency_reason_(reason) {}
    
    const std::string& emergency_reason() const noexcept { return emergency_reason_; }
    
private:
    std::string emergency_reason_;
};

/**
 * @brief Configuration related exceptions
 */
class ConfigurationException : public DSMILException {
public:
    explicit ConfigurationException(const std::string& message)
        : DSMILException(ErrorCode::INVALID_PARAMETER, message) {}
    
    ConfigurationException(const std::string& message, const std::string& parameter_name)
        : DSMILException(ErrorCode::INVALID_PARAMETER, message)
        , parameter_name_(parameter_name) {}
    
    const std::string& parameter_name() const noexcept { return parameter_name_; }
    
private:
    std::string parameter_name_;
};

/**
 * @brief Connection pool related exceptions
 */
class ConnectionPoolException : public DSMILException {
public:
    explicit ConnectionPoolException(const std::string& message)
        : DSMILException(ErrorCode::INTERNAL_ERROR, message) {}
    
    ConnectionPoolException(const std::string& message, uint32_t pool_size, uint32_t active_connections)
        : DSMILException(ErrorCode::INTERNAL_ERROR, message)
        , pool_size_(pool_size)
        , active_connections_(active_connections) {}
    
    uint32_t pool_size() const noexcept { return pool_size_; }
    uint32_t active_connections() const noexcept { return active_connections_; }
    
private:
    uint32_t pool_size_ = 0;
    uint32_t active_connections_ = 0;
};

/**
 * @brief WebSocket related exceptions
 */
class WebSocketException : public DSMILException {
public:
    explicit WebSocketException(const std::string& message)
        : DSMILException(ErrorCode::PROTOCOL_ERROR, message) {}
    
    WebSocketException(const std::string& message, uint16_t close_code)
        : DSMILException(ErrorCode::PROTOCOL_ERROR, message)
        , close_code_(close_code) {}
    
    uint16_t close_code() const noexcept { return close_code_; }
    
private:
    uint16_t close_code_ = 0;
};

/**
 * @brief Utility functions for exception handling
 */
namespace exception_utils {

/**
 * @brief Convert ErrorCode to string
 */
std::string error_code_to_string(ErrorCode code);

/**
 * @brief Convert HTTP status code to ErrorCode
 */
ErrorCode http_status_to_error_code(int http_status);

/**
 * @brief Check if error code is retryable
 */
bool is_retryable_error(ErrorCode code);

/**
 * @brief Get suggested retry delay for error code
 */
std::chrono::milliseconds get_retry_delay(ErrorCode code, uint32_t attempt);

} // namespace exception_utils

} // namespace dsmil
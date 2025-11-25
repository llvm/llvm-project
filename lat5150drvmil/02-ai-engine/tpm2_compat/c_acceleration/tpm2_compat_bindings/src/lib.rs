//! FFI Bindings for TPM2 Compatibility Layer
//!
//! Provides C and Python bindings for seamless integration with existing
//! systems while maintaining maximum performance and security.

#![deny(missing_docs)]
#![deny(clippy::all)]
#![warn(clippy::pedantic)]

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int, c_uint, c_void};
use std::ptr;
use std::sync::Arc;

use tokio::runtime::Runtime;
use zeroize::Zeroize;

use tpm2_compat_common::{
    Tpm2Rc, SecurityLevel, AccelerationFlags, SessionHandle,
    TpmCommand, LibraryConfig, PerformanceMetrics, HardwareCapabilities
};
use tpm2_compat_userspace::{Tpm2CompatService, ServiceConfig};
use tpm2_compat_crypto::{CryptoEngine, CryptoParams, CryptoOperation, CryptoAlgorithm};

pub mod c_bindings;
#[cfg(feature = "python-bindings")]
pub mod python_bindings;

/// C-compatible TPM2 service handle
#[repr(C)]
pub struct Tpm2Service {
    runtime: Runtime,
    service: Arc<Tpm2CompatService>,
}

/// C-compatible configuration structure
#[repr(C)]
#[derive(Debug)]
pub struct Tpm2Config {
    /// Security level (0=Unclassified, 1=Confidential, 2=Secret, 3=TopSecret)
    pub security_level: c_uint,
    /// Hardware acceleration flags
    pub acceleration_flags: c_uint,
    /// Enable profiling
    pub enable_profiling: c_int,
    /// Enable fault detection
    pub enable_fault_detection: c_int,
    /// Maximum sessions
    pub max_sessions: c_uint,
    /// Memory pool size in MB
    pub memory_pool_size_mb: c_uint,
    /// Enable debug mode
    pub enable_debug_mode: c_int,
}

/// C-compatible command structure
#[repr(C)]
pub struct Tpm2CCommand {
    /// Command data pointer
    pub data: *const u8,
    /// Command data length
    pub data_len: usize,
    /// Security level required
    pub security_level: c_uint,
    /// Session handle (0 if no session)
    pub session_handle: c_uint,
}

/// C-compatible response structure
#[repr(C)]
pub struct Tpm2CResponse {
    /// Response data pointer
    pub data: *mut u8,
    /// Response data length
    pub data_len: usize,
    /// Response code
    pub response_code: c_uint,
}

/// C-compatible performance metrics
#[repr(C)]
pub struct Tpm2CMetrics {
    /// Operations per second
    pub ops_per_second: f64,
    /// Average latency in microseconds
    pub avg_latency_us: f64,
    /// Total operations processed
    pub total_operations: u64,
    /// Hardware acceleration usage percentage
    pub acceleration_usage_percent: f32,
}

/// C-compatible hardware capabilities
#[repr(C)]
pub struct Tpm2CHardware {
    /// CPU model string pointer
    pub cpu_model: *const c_char,
    /// Acceleration flags
    pub acceleration_flags: c_uint,
    /// NPU TOPS (0 if not available)
    pub npu_tops: f32,
    /// GNA available (0=false, 1=true)
    pub gna_available: c_int,
    /// CPU core count
    pub cpu_cores: c_uint,
    /// L3 cache size in MB
    pub l3_cache_mb: c_uint,
}

/// Global runtime for async operations
static mut GLOBAL_RUNTIME: Option<Runtime> = None;

/// Initialize the global runtime
fn get_runtime() -> &'static Runtime {
    unsafe {
        if GLOBAL_RUNTIME.is_none() {
            GLOBAL_RUNTIME = Some(
                Runtime::new().expect("Failed to create Tokio runtime")
            );
        }
        GLOBAL_RUNTIME.as_ref().unwrap()
    }
}

/// Convert C config to Rust config
fn convert_config(c_config: &Tpm2Config) -> Result<ServiceConfig, Tpm2Rc> {
    let security_level = match c_config.security_level {
        0 => SecurityLevel::Unclassified,
        1 => SecurityLevel::Confidential,
        2 => SecurityLevel::Secret,
        3 => SecurityLevel::TopSecret,
        _ => return Err(Tpm2Rc::Parameter),
    };

    let acceleration_flags = AccelerationFlags::from_bits(c_config.acceleration_flags)
        .unwrap_or(AccelerationFlags::ALL);

    let library_config = LibraryConfig {
        security_level,
        acceleration_flags,
        enable_profiling: c_config.enable_profiling != 0,
        enable_fault_detection: c_config.enable_fault_detection != 0,
        max_sessions: c_config.max_sessions,
        memory_pool_size_mb: c_config.memory_pool_size_mb,
        enable_debug_mode: c_config.enable_debug_mode != 0,
    };

    Ok(ServiceConfig {
        library_config,
        ..ServiceConfig::default()
    })
}

/// Convert C command to Rust command
fn convert_command(c_command: &Tpm2CCommand) -> Result<TpmCommand, Tpm2Rc> {
    if c_command.data.is_null() || c_command.data_len == 0 {
        return Err(Tpm2Rc::Parameter);
    }

    let data = unsafe {
        std::slice::from_raw_parts(c_command.data, c_command.data_len)
    }.to_vec();

    let security_level = match c_command.security_level {
        0 => SecurityLevel::Unclassified,
        1 => SecurityLevel::Confidential,
        2 => SecurityLevel::Secret,
        3 => SecurityLevel::TopSecret,
        _ => return Err(Tpm2Rc::Parameter),
    };

    let session = if c_command.session_handle != 0 {
        Some(SessionHandle::new(c_command.session_handle))
    } else {
        None
    };

    let mut command = TpmCommand::new(data, security_level);
    if let Some(session_handle) = session {
        command = command.with_session(session_handle);
    }

    Ok(command)
}

/// C API Functions
extern "C" {
    /// Initialize TPM2 compatibility library
    #[no_mangle]
    pub extern "C" fn tpm2_compat_init(config: *const Tpm2Config) -> *mut Tpm2Service;

    /// Process TPM command
    #[no_mangle]
    pub extern "C" fn tpm2_compat_process_command(
        service: *mut Tpm2Service,
        command: *const Tpm2CCommand,
        response: *mut Tpm2CResponse,
    ) -> c_uint;

    /// Create a new session
    #[no_mangle]
    pub extern "C" fn tpm2_compat_create_session(
        service: *mut Tpm2Service,
        security_level: c_uint,
        session_handle: *mut c_uint,
    ) -> c_uint;

    /// Close a session
    #[no_mangle]
    pub extern "C" fn tpm2_compat_close_session(
        service: *mut Tpm2Service,
        session_handle: c_uint,
    ) -> c_uint;

    /// Get performance metrics
    #[no_mangle]
    pub extern "C" fn tpm2_compat_get_metrics(
        service: *mut Tpm2Service,
        metrics: *mut Tpm2CMetrics,
    ) -> c_uint;

    /// Get hardware capabilities
    #[no_mangle]
    pub extern "C" fn tpm2_compat_get_hardware(
        service: *mut Tpm2Service,
        hardware: *mut Tpm2CHardware,
    ) -> c_uint;

    /// Free response data
    #[no_mangle]
    pub extern "C" fn tpm2_compat_free_response(response: *mut Tpm2CResponse);

    /// Free hardware capabilities
    #[no_mangle]
    pub extern "C" fn tpm2_compat_free_hardware(hardware: *mut Tpm2CHardware);

    /// Cleanup and shutdown
    #[no_mangle]
    pub extern "C" fn tpm2_compat_cleanup(service: *mut Tpm2Service) -> c_uint;

    /// Get last error message
    #[no_mangle]
    pub extern "C" fn tpm2_compat_get_last_error() -> *const c_char;
}

/// Initialize TPM2 compatibility library
#[no_mangle]
pub extern "C" fn tpm2_compat_init(config: *const Tpm2Config) -> *mut Tpm2Service {
    if config.is_null() {
        return ptr::null_mut();
    }

    let c_config = unsafe { &*config };
    let service_config = match convert_config(c_config) {
        Ok(config) => config,
        Err(_) => return ptr::null_mut(),
    };

    let runtime = match Runtime::new() {
        Ok(rt) => rt,
        Err(_) => return ptr::null_mut(),
    };

    let service = match runtime.block_on(Tpm2CompatService::new(service_config)) {
        Ok(service) => Arc::new(service),
        Err(_) => return ptr::null_mut(),
    };

    let tpm2_service = Tpm2Service {
        runtime,
        service,
    };

    Box::into_raw(Box::new(tpm2_service))
}

/// Process TPM command
#[no_mangle]
pub extern "C" fn tpm2_compat_process_command(
    service: *mut Tpm2Service,
    command: *const Tpm2CCommand,
    response: *mut Tpm2CResponse,
) -> c_uint {
    if service.is_null() || command.is_null() || response.is_null() {
        return Tpm2Rc::Parameter as c_uint;
    }

    let service = unsafe { &*service };
    let c_command = unsafe { &*command };

    let tpm_command = match convert_command(c_command) {
        Ok(cmd) => cmd,
        Err(e) => return e as c_uint,
    };

    let result = service.runtime.block_on(
        service.service.process_tpm_command(tpm_command)
    );

    match result {
        Ok(response_data) => {
            let response_ptr = unsafe { &mut *response };

            // Allocate memory for response data
            let data_len = response_data.len();
            let data_ptr = unsafe {
                libc::malloc(data_len) as *mut u8
            };

            if data_ptr.is_null() {
                return Tpm2Rc::Memory as c_uint;
            }

            // Copy response data
            unsafe {
                ptr::copy_nonoverlapping(
                    response_data.as_ptr(),
                    data_ptr,
                    data_len,
                );
            }

            response_ptr.data = data_ptr;
            response_ptr.data_len = data_len;
            response_ptr.response_code = Tpm2Rc::Success as c_uint;

            Tpm2Rc::Success as c_uint
        }
        Err(e) => {
            let response_ptr = unsafe { &mut *response };
            response_ptr.data = ptr::null_mut();
            response_ptr.data_len = 0;
            response_ptr.response_code = e as c_uint;

            e as c_uint
        }
    }
}

/// Create a new session
#[no_mangle]
pub extern "C" fn tpm2_compat_create_session(
    service: *mut Tpm2Service,
    security_level: c_uint,
    session_handle: *mut c_uint,
) -> c_uint {
    if service.is_null() || session_handle.is_null() {
        return Tpm2Rc::Parameter as c_uint;
    }

    let service = unsafe { &*service };

    let sec_level = match security_level {
        0 => SecurityLevel::Unclassified,
        1 => SecurityLevel::Confidential,
        2 => SecurityLevel::Secret,
        3 => SecurityLevel::TopSecret,
        _ => return Tpm2Rc::Parameter as c_uint,
    };

    let result = service.runtime.block_on(
        service.service.create_session(sec_level)
    );

    match result {
        Ok(handle) => {
            unsafe {
                *session_handle = handle.raw();
            }
            Tpm2Rc::Success as c_uint
        }
        Err(e) => e as c_uint,
    }
}

/// Close a session
#[no_mangle]
pub extern "C" fn tpm2_compat_close_session(
    service: *mut Tpm2Service,
    session_handle: c_uint,
) -> c_uint {
    if service.is_null() {
        return Tpm2Rc::Parameter as c_uint;
    }

    let service = unsafe { &*service };
    let handle = SessionHandle::new(session_handle);

    let result = service.runtime.block_on(
        service.service.close_session(handle)
    );

    match result {
        Ok(()) => Tpm2Rc::Success as c_uint,
        Err(e) => e as c_uint,
    }
}

/// Get performance metrics
#[no_mangle]
pub extern "C" fn tpm2_compat_get_metrics(
    service: *mut Tpm2Service,
    metrics: *mut Tpm2CMetrics,
) -> c_uint {
    if service.is_null() || metrics.is_null() {
        return Tpm2Rc::Parameter as c_uint;
    }

    let service = unsafe { &*service };

    let perf_metrics = service.runtime.block_on(
        service.service.get_performance_metrics()
    );

    let metrics_ptr = unsafe { &mut *metrics };
    metrics_ptr.ops_per_second = perf_metrics.ops_per_second;
    metrics_ptr.avg_latency_us = perf_metrics.avg_latency_us;
    metrics_ptr.total_operations = perf_metrics.total_operations;
    metrics_ptr.acceleration_usage_percent = perf_metrics.acceleration_usage_percent;

    Tpm2Rc::Success as c_uint
}

/// Get hardware capabilities
#[no_mangle]
pub extern "C" fn tpm2_compat_get_hardware(
    service: *mut Tpm2Service,
    hardware: *mut Tpm2CHardware,
) -> c_uint {
    if service.is_null() || hardware.is_null() {
        return Tpm2Rc::Parameter as c_uint;
    }

    let service = unsafe { &*service };

    let capabilities = service.runtime.block_on(
        service.service.get_hardware_capabilities()
    );

    match capabilities {
        Ok(caps) => {
            let hardware_ptr = unsafe { &mut *hardware };

            // Allocate and copy CPU model string
            let cpu_model_cstr = match CString::new(caps.cpu_model) {
                Ok(cstr) => cstr,
                Err(_) => return Tpm2Rc::Memory as c_uint,
            };

            hardware_ptr.cpu_model = cpu_model_cstr.into_raw();
            hardware_ptr.acceleration_flags = caps.acceleration_flags.bits();
            hardware_ptr.npu_tops = caps.npu_tops.unwrap_or(0.0);
            hardware_ptr.gna_available = if caps.gna_available { 1 } else { 0 };
            hardware_ptr.cpu_cores = caps.cpu_cores;
            hardware_ptr.l3_cache_mb = caps.l3_cache_mb;

            Tpm2Rc::Success as c_uint
        }
        Err(e) => e as c_uint,
    }
}

/// Free response data
#[no_mangle]
pub extern "C" fn tpm2_compat_free_response(response: *mut Tpm2CResponse) {
    if response.is_null() {
        return;
    }

    let response_ptr = unsafe { &mut *response };
    if !response_ptr.data.is_null() {
        unsafe {
            // Clear memory before freeing for security
            let data_slice = std::slice::from_raw_parts_mut(
                response_ptr.data,
                response_ptr.data_len,
            );
            data_slice.zeroize();

            libc::free(response_ptr.data as *mut c_void);
        }
        response_ptr.data = ptr::null_mut();
        response_ptr.data_len = 0;
    }
}

/// Free hardware capabilities
#[no_mangle]
pub extern "C" fn tpm2_compat_free_hardware(hardware: *mut Tpm2CHardware) {
    if hardware.is_null() {
        return;
    }

    let hardware_ptr = unsafe { &mut *hardware };
    if !hardware_ptr.cpu_model.is_null() {
        unsafe {
            let _ = CString::from_raw(hardware_ptr.cpu_model as *mut c_char);
        }
        hardware_ptr.cpu_model = ptr::null();
    }
}

/// Cleanup and shutdown
#[no_mangle]
pub extern "C" fn tpm2_compat_cleanup(service: *mut Tpm2Service) -> c_uint {
    if service.is_null() {
        return Tpm2Rc::Parameter as c_uint;
    }

    let service_box = unsafe { Box::from_raw(service) };

    let result = service_box.runtime.block_on(
        service_box.service.clone().shutdown()
    );

    match result {
        Ok(()) => Tpm2Rc::Success as c_uint,
        Err(e) => e as c_uint,
    }
}

/// Get last error message (placeholder)
#[no_mangle]
pub extern "C" fn tpm2_compat_get_last_error() -> *const c_char {
    static ERROR_MSG: &[u8] = b"No error information available\0";
    ERROR_MSG.as_ptr() as *const c_char
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_conversion() {
        let c_config = Tpm2Config {
            security_level: 1,
            acceleration_flags: AccelerationFlags::ALL.bits(),
            enable_profiling: 1,
            enable_fault_detection: 1,
            max_sessions: 64,
            memory_pool_size_mb: 128,
            enable_debug_mode: 0,
        };

        let rust_config = convert_config(&c_config);
        assert!(rust_config.is_ok());

        let config = rust_config.unwrap();
        assert_eq!(config.library_config.security_level, SecurityLevel::Confidential);
        assert_eq!(config.library_config.max_sessions, 64);
    }

    #[test]
    fn test_command_conversion() {
        let data = vec![0x80, 0x01, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x00, 0x01, 0x43, 0x00, 0x00];
        let c_command = Tpm2CCommand {
            data: data.as_ptr(),
            data_len: data.len(),
            security_level: 0,
            session_handle: 0,
        };

        let rust_command = convert_command(&c_command);
        assert!(rust_command.is_ok());

        let command = rust_command.unwrap();
        assert_eq!(command.data, data);
        assert_eq!(command.security_level, SecurityLevel::Unclassified);
    }
}
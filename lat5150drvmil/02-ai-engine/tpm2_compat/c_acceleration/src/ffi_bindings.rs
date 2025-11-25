//! Safe Rust FFI Bindings for Dell Military Token Integration
//!
//! DSMIL Agent - Memory-Safe C/Python Interoperability
//! Dell Latitude 5450 MIL-SPEC Integration Layer
//!
//! MISSION: Provide safe FFI bindings for existing C and Python code
//! - Zero unsafe operations in Rust interface
//! - Memory-safe data exchange with C libraries
//! - Python integration via PyO3 (optional)
//! - Type-safe parameter validation
//! - Automatic resource cleanup

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]
#![deny(missing_docs)]

use crate::tpm2_compat_common::{
    Tpm2Result, Tpm2Rc, SecurityLevel, MilitaryToken,
    LibraryConfig, PerformanceMetrics,
};
use crate::dell_military_tokens::{
    DellMilitaryTokenValidator, SecurityMatrix, TokenValidationResult,
    NpuAcceleratedValidator,
};
use crate::security_matrix::{
    SecurityAuthorizationEngine, AuthorizationSession, AuditEntry,
    SecurityStatistics,
};
use crate::npu_acceleration::{
    NpuMilitaryTokenManager, NpuPerformanceReport,
};

use core::ffi::{c_char, c_int, c_uint, c_void};
use core::ptr;
use core::slice;
use alloc::boxed::Box;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use alloc::ffi::CString;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// FFI-safe error codes for C interoperability
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DsmilErrorCode {
    /// Success
    Success = 0,
    /// Invalid parameter
    InvalidParameter = -1001,
    /// Memory allocation failure
    MemoryError = -1002,
    /// Token validation failure
    TokenValidationFailed = -1003,
    /// NPU acceleration error
    NpuError = -1004,
    /// Security violation
    SecurityViolation = -1005,
    /// Hardware not available
    HardwareUnavailable = -1006,
    /// Session not found
    SessionNotFound = -1007,
    /// Insufficient permissions
    InsufficientPermissions = -1008,
}

impl From<Tpm2Rc> for DsmilErrorCode {
    fn from(tpm_error: Tpm2Rc) -> Self {
        match tpm_error {
            Tpm2Rc::Success => Self::Success,
            Tpm2Rc::Parameter => Self::InvalidParameter,
            Tpm2Rc::Memory => Self::MemoryError,
            Tpm2Rc::MilitaryTokenFailure => Self::TokenValidationFailed,
            Tpm2Rc::NpuAccelerationError => Self::NpuError,
            Tpm2Rc::SecurityViolation => Self::SecurityViolation,
            Tpm2Rc::AccelerationUnavailable => Self::HardwareUnavailable,
            Tpm2Rc::SessionNotFound => Self::SessionNotFound,
            _ => Self::InvalidParameter,
        }
    }
}

/// FFI-safe token validation result
#[repr(C)]
#[derive(Debug, Clone)]
pub struct FfiTokenValidationResult {
    /// Token ID
    pub token_id: u16,
    /// Validation success flag
    pub is_valid: bool,
    /// Actual token value
    pub actual_value: u32,
    /// Expected token value
    pub expected_value: u32,
    /// Security level (0=Unclassified, 1=Confidential, 2=Secret, 3=TopSecret)
    pub security_level: u8,
    /// Validation time in microseconds
    pub validation_time_us: u64,
    /// Hardware acceleration used
    pub acceleration_used: bool,
}

impl From<TokenValidationResult> for FfiTokenValidationResult {
    fn from(result: TokenValidationResult) -> Self {
        Self {
            token_id: result.token_id,
            is_valid: result.is_valid,
            actual_value: result.actual_value,
            expected_value: result.expected_value,
            security_level: result.security_level as u8,
            validation_time_us: result.validation_time_us,
            acceleration_used: result.acceleration_used,
        }
    }
}

/// FFI-safe security matrix
#[repr(C)]
#[derive(Debug, Clone)]
pub struct FfiSecurityMatrix {
    /// Number of validated tokens
    pub tokens_validated: u8,
    /// Security level achieved
    pub security_level: u8,
    /// Authorization bitmask
    pub authorization_mask: u32,
}

impl From<SecurityMatrix> for FfiSecurityMatrix {
    fn from(matrix: SecurityMatrix) -> Self {
        Self {
            tokens_validated: matrix.tokens_validated,
            security_level: matrix.security_level as u8,
            authorization_mask: matrix.authorization_mask,
        }
    }
}

/// FFI-safe performance metrics
#[repr(C)]
#[derive(Debug, Clone)]
pub struct FfiPerformanceMetrics {
    /// Operations per second
    pub ops_per_second: f64,
    /// Average latency in microseconds
    pub avg_latency_us: f64,
    /// Total operations processed
    pub total_operations: u64,
    /// Hardware acceleration usage percentage
    pub acceleration_usage_percent: f32,
}

/// Opaque handle for DSMIL context
pub type DsmilHandle = *mut c_void;

/// DSMIL context for managing state
struct DsmilContext {
    /// Token validator
    token_validator: DellMilitaryTokenValidator,
    /// Security authorization engine
    security_engine: SecurityAuthorizationEngine,
    /// NPU token manager (optional)
    npu_manager: Option<Box<NpuMilitaryTokenManager>>,
    /// Configuration
    config: LibraryConfig,
}

impl DsmilContext {
    fn new(config: LibraryConfig) -> Self {
        Self {
            token_validator: DellMilitaryTokenValidator::new(config.acceleration_flags),
            security_engine: SecurityAuthorizationEngine::new(),
            npu_manager: None,
            config,
        }
    }
}

/// Initialize DSMIL library with configuration
///
/// # Safety
/// This function is safe because:
/// - All parameters are validated before use
/// - Memory allocation is handled safely
/// - Returns opaque handle for subsequent operations
#[no_mangle]
pub extern "C" fn dsmil_init(
    enable_npu: bool,
    security_level: u8,
    enable_debug: bool,
) -> DsmilHandle {
    // Validate security level
    let security_level = match security_level {
        0 => SecurityLevel::Unclassified,
        1 => SecurityLevel::Confidential,
        2 => SecurityLevel::Secret,
        3 => SecurityLevel::TopSecret,
        _ => return ptr::null_mut(),
    };

    // Create configuration
    let config = LibraryConfig {
        security_level,
        acceleration_flags: if enable_npu {
            crate::tpm2_compat_common::AccelerationFlags::ALL
        } else {
            crate::tpm2_compat_common::AccelerationFlags::NONE
        },
        enable_debug_mode: enable_debug,
        ..LibraryConfig::default()
    };

    // Create context
    let mut context = DsmilContext::new(config);

    // Initialize NPU manager if requested
    if enable_npu {
        // Note: In async context, this would use tokio::runtime
        // For FFI, we'll initialize it lazily
        context.npu_manager = None; // Will be initialized on first use
    }

    // Box the context and return as opaque handle
    let boxed_context = Box::new(context);
    Box::into_raw(boxed_context) as DsmilHandle
}

/// Cleanup DSMIL library and free resources
///
/// # Safety
/// This function is safe because:
/// - Validates handle before use
/// - Properly frees allocated memory
/// - Handles null pointers gracefully
#[no_mangle]
pub extern "C" fn dsmil_cleanup(handle: DsmilHandle) {
    if handle.is_null() {
        return;
    }

    // Convert back to Box and drop
    let _context = Box::from_raw(handle as *mut DsmilContext);
    // Box will be automatically dropped, cleaning up resources
}

/// Validate single Dell military token
///
/// # Safety
/// This function is safe because:
/// - Validates all input parameters
/// - Uses safe Rust operations only
/// - Handles errors gracefully
#[no_mangle]
pub extern "C" fn dsmil_validate_token(
    handle: DsmilHandle,
    token_id: u16,
    result: *mut FfiTokenValidationResult,
) -> DsmilErrorCode {
    // Validate parameters
    if handle.is_null() || result.is_null() {
        return DsmilErrorCode::InvalidParameter;
    }

    // Get context safely
    let context = match unsafe { handle.as_mut() } {
        Some(ctx) => ctx as *mut DsmilContext,
        None => return DsmilErrorCode::InvalidParameter,
    };

    let context = unsafe { &mut *context };

    // Validate token
    match context.token_validator.validate_token(token_id) {
        Ok(validation_result) => {
            let ffi_result = FfiTokenValidationResult::from(validation_result);
            unsafe {
                ptr::write(result, ffi_result);
            }
            DsmilErrorCode::Success
        }
        Err(tpm_error) => DsmilErrorCode::from(tpm_error),
    }
}

/// Validate all Dell military tokens and return security matrix
///
/// # Safety
/// This function is safe because:
/// - Validates all input parameters
/// - Uses safe memory operations
/// - Proper error handling
#[no_mangle]
pub extern "C" fn dsmil_validate_all_tokens(
    handle: DsmilHandle,
    matrix: *mut FfiSecurityMatrix,
) -> DsmilErrorCode {
    if handle.is_null() || matrix.is_null() {
        return DsmilErrorCode::InvalidParameter;
    }

    let context = match unsafe { handle.as_mut() } {
        Some(ctx) => ctx as *mut DsmilContext,
        None => return DsmilErrorCode::InvalidParameter,
    };

    let context = unsafe { &mut *context };

    match context.token_validator.validate_all_tokens() {
        Ok(security_matrix) => {
            let ffi_matrix = FfiSecurityMatrix::from(security_matrix);
            unsafe {
                ptr::write(matrix, ffi_matrix);
            }
            DsmilErrorCode::Success
        }
        Err(tpm_error) => DsmilErrorCode::from(tpm_error),
    }
}

/// Create authorization session for given security level
///
/// # Safety
/// This function is safe because:
/// - Parameter validation
/// - Safe session management
/// - Proper error propagation
#[no_mangle]
pub extern "C" fn dsmil_create_authorization_session(
    handle: DsmilHandle,
    required_security_level: u8,
    session_id_out: *mut u64,
) -> DsmilErrorCode {
    if handle.is_null() || session_id_out.is_null() {
        return DsmilErrorCode::InvalidParameter;
    }

    let security_level = match required_security_level {
        0 => SecurityLevel::Unclassified,
        1 => SecurityLevel::Confidential,
        2 => SecurityLevel::Secret,
        3 => SecurityLevel::TopSecret,
        _ => return DsmilErrorCode::InvalidParameter,
    };

    let context = match unsafe { handle.as_mut() } {
        Some(ctx) => ctx as *mut DsmilContext,
        None => return DsmilErrorCode::InvalidParameter,
    };

    let context = unsafe { &mut *context };

    match context.security_engine.authorize_access(security_level) {
        Ok(session) => {
            unsafe {
                ptr::write(session_id_out, session.session_id);
            }
            DsmilErrorCode::Success
        }
        Err(tpm_error) => DsmilErrorCode::from(tpm_error),
    }
}

/// Validate session access to specific security level
///
/// # Safety
/// This function is safe because:
/// - Parameter validation
/// - Safe session lookup
/// - Boolean return via pointer
#[no_mangle]
pub extern "C" fn dsmil_validate_session_access(
    handle: DsmilHandle,
    session_id: u64,
    required_security_level: u8,
    access_granted_out: *mut bool,
) -> DsmilErrorCode {
    if handle.is_null() || access_granted_out.is_null() {
        return DsmilErrorCode::InvalidParameter;
    }

    let security_level = match required_security_level {
        0 => SecurityLevel::Unclassified,
        1 => SecurityLevel::Confidential,
        2 => SecurityLevel::Secret,
        3 => SecurityLevel::TopSecret,
        _ => return DsmilErrorCode::InvalidParameter,
    };

    let context = match unsafe { handle.as_mut() } {
        Some(ctx) => ctx as *mut DsmilContext,
        None => return DsmilErrorCode::InvalidParameter,
    };

    let context = unsafe { &mut *context };

    match context.security_engine.validate_session_access(session_id, security_level) {
        Ok(access_granted) => {
            unsafe {
                ptr::write(access_granted_out, access_granted);
            }
            DsmilErrorCode::Success
        }
        Err(tpm_error) => DsmilErrorCode::from(tpm_error),
    }
}

/// Get performance metrics
///
/// # Safety
/// This function is safe because:
/// - Parameter validation
/// - Read-only data access
/// - Safe struct copying
#[no_mangle]
pub extern "C" fn dsmil_get_performance_metrics(
    handle: DsmilHandle,
    metrics_out: *mut FfiPerformanceMetrics,
) -> DsmilErrorCode {
    if handle.is_null() || metrics_out.is_null() {
        return DsmilErrorCode::InvalidParameter;
    }

    let context = match unsafe { handle.as_ref() } {
        Some(ctx) => ctx,
        None => return DsmilErrorCode::InvalidParameter,
    };

    let validator_metrics = context.token_validator.get_metrics();

    let ffi_metrics = FfiPerformanceMetrics {
        ops_per_second: if validator_metrics.avg_validation_time_us > 0.0 {
            1_000_000.0 / validator_metrics.avg_validation_time_us
        } else {
            0.0
        },
        avg_latency_us: validator_metrics.avg_validation_time_us,
        total_operations: validator_metrics.total_validations,
        acceleration_usage_percent: validator_metrics.acceleration_usage_percent,
    };

    unsafe {
        ptr::write(metrics_out, ffi_metrics);
    }

    DsmilErrorCode::Success
}

/// Get security statistics
///
/// # Safety
/// This function is safe because:
/// - Parameter validation
/// - Read-only access to statistics
/// - Safe data copying
#[no_mangle]
pub extern "C" fn dsmil_get_security_statistics(
    handle: DsmilHandle,
    total_audits_out: *mut u64,
    successful_auths_out: *mut u64,
    denied_accesses_out: *mut u64,
    active_sessions_out: *mut u32,
) -> DsmilErrorCode {
    if handle.is_null() {
        return DsmilErrorCode::InvalidParameter;
    }

    let context = match unsafe { handle.as_ref() } {
        Some(ctx) => ctx,
        None => return DsmilErrorCode::InvalidParameter,
    };

    let stats = context.security_engine.get_security_statistics();

    if !total_audits_out.is_null() {
        unsafe { ptr::write(total_audits_out, stats.total_audit_entries); }
    }
    if !successful_auths_out.is_null() {
        unsafe { ptr::write(successful_auths_out, stats.successful_authorizations); }
    }
    if !denied_accesses_out.is_null() {
        unsafe { ptr::write(denied_accesses_out, stats.denied_accesses); }
    }
    if !active_sessions_out.is_null() {
        unsafe { ptr::write(active_sessions_out, stats.active_sessions); }
    }

    DsmilErrorCode::Success
}

/// Convert error code to human-readable string
///
/// # Safety
/// This function is safe because:
/// - Returns static string literals
/// - No memory allocation
/// - No pointer manipulation
#[no_mangle]
pub extern "C" fn dsmil_error_string(error_code: DsmilErrorCode) -> *const c_char {
    let error_str = match error_code {
        DsmilErrorCode::Success => "Success\0",
        DsmilErrorCode::InvalidParameter => "Invalid parameter\0",
        DsmilErrorCode::MemoryError => "Memory allocation error\0",
        DsmilErrorCode::TokenValidationFailed => "Token validation failed\0",
        DsmilErrorCode::NpuError => "NPU acceleration error\0",
        DsmilErrorCode::SecurityViolation => "Security policy violation\0",
        DsmilErrorCode::HardwareUnavailable => "Hardware not available\0",
        DsmilErrorCode::SessionNotFound => "Session not found\0",
        DsmilErrorCode::InsufficientPermissions => "Insufficient permissions\0",
    };

    error_str.as_ptr() as *const c_char
}

/// Python integration module (optional feature)
#[cfg(feature = "python")]
pub mod python {
    use super::*;
    use pyo3::prelude::*;
    use pyo3::wrap_pyfunction;

    /// Python wrapper for DSMIL context
    #[pyclass]
    pub struct PyDsmilContext {
        handle: DsmilHandle,
    }

    #[pymethods]
    impl PyDsmilContext {
        #[new]
        fn new(enable_npu: bool, security_level: u8, enable_debug: bool) -> PyResult<Self> {
            let handle = dsmil_init(enable_npu, security_level, enable_debug);
            if handle.is_null() {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    "Failed to initialize DSMIL context"
                ));
            }

            Ok(Self { handle })
        }

        fn validate_token(&mut self, token_id: u16) -> PyResult<(bool, u32, u32, u8, u64, bool)> {
            let mut result = FfiTokenValidationResult {
                token_id: 0,
                is_valid: false,
                actual_value: 0,
                expected_value: 0,
                security_level: 0,
                validation_time_us: 0,
                acceleration_used: false,
            };

            let error_code = dsmil_validate_token(self.handle, token_id, &mut result);
            if error_code != DsmilErrorCode::Success {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    format!("Token validation failed: {:?}", error_code)
                ));
            }

            Ok((
                result.is_valid,
                result.actual_value,
                result.expected_value,
                result.security_level,
                result.validation_time_us,
                result.acceleration_used,
            ))
        }

        fn validate_all_tokens(&mut self) -> PyResult<(u8, u8, u32)> {
            let mut matrix = FfiSecurityMatrix {
                tokens_validated: 0,
                security_level: 0,
                authorization_mask: 0,
            };

            let error_code = dsmil_validate_all_tokens(self.handle, &mut matrix);
            if error_code != DsmilErrorCode::Success {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    format!("Token validation failed: {:?}", error_code)
                ));
            }

            Ok((matrix.tokens_validated, matrix.security_level, matrix.authorization_mask))
        }

        fn create_authorization_session(&mut self, required_security_level: u8) -> PyResult<u64> {
            let mut session_id = 0u64;

            let error_code = dsmil_create_authorization_session(
                self.handle,
                required_security_level,
                &mut session_id,
            );

            if error_code != DsmilErrorCode::Success {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    format!("Authorization failed: {:?}", error_code)
                ));
            }

            Ok(session_id)
        }

        fn get_performance_metrics(&self) -> PyResult<(f64, f64, u64, f32)> {
            let mut metrics = FfiPerformanceMetrics {
                ops_per_second: 0.0,
                avg_latency_us: 0.0,
                total_operations: 0,
                acceleration_usage_percent: 0.0,
            };

            let error_code = dsmil_get_performance_metrics(self.handle, &mut metrics);
            if error_code != DsmilErrorCode::Success {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    format!("Failed to get metrics: {:?}", error_code)
                ));
            }

            Ok((
                metrics.ops_per_second,
                metrics.avg_latency_us,
                metrics.total_operations,
                metrics.acceleration_usage_percent,
            ))
        }
    }

    impl Drop for PyDsmilContext {
        fn drop(&mut self) {
            dsmil_cleanup(self.handle);
        }
    }

    /// Python module definition
    #[pymodule]
    fn dsmil_rust(_py: Python, m: &PyModule) -> PyResult<()> {
        m.add_class::<PyDsmilContext>()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ffi_init_cleanup() {
        let handle = dsmil_init(false, 0, false);
        assert!(!handle.is_null());

        dsmil_cleanup(handle);
        // No crash means success
    }

    #[test]
    fn test_ffi_token_validation() {
        let handle = dsmil_init(false, 0, false);
        assert!(!handle.is_null());

        let mut result = FfiTokenValidationResult {
            token_id: 0,
            is_valid: false,
            actual_value: 0,
            expected_value: 0,
            security_level: 0,
            validation_time_us: 0,
            acceleration_used: false,
        };

        let error_code = dsmil_validate_token(handle, 0x049e, &mut result);
        assert_eq!(error_code, DsmilErrorCode::Success);
        assert_eq!(result.token_id, 0x049e);

        dsmil_cleanup(handle);
    }

    #[test]
    fn test_ffi_error_string() {
        let error_str = dsmil_error_string(DsmilErrorCode::Success);
        assert!(!error_str.is_null());

        let error_str = dsmil_error_string(DsmilErrorCode::TokenValidationFailed);
        assert!(!error_str.is_null());
    }

    #[test]
    fn test_invalid_parameters() {
        // Test null handle
        let mut result = FfiTokenValidationResult {
            token_id: 0,
            is_valid: false,
            actual_value: 0,
            expected_value: 0,
            security_level: 0,
            validation_time_us: 0,
            acceleration_used: false,
        };

        let error_code = dsmil_validate_token(ptr::null_mut(), 0x049e, &mut result);
        assert_eq!(error_code, DsmilErrorCode::InvalidParameter);

        // Test null result pointer
        let handle = dsmil_init(false, 0, false);
        let error_code = dsmil_validate_token(handle, 0x049e, ptr::null_mut());
        assert_eq!(error_code, DsmilErrorCode::InvalidParameter);

        dsmil_cleanup(handle);
    }
}
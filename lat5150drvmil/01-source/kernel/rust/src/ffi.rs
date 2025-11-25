//! C/Rust FFI Bridge
//!
//! Safe FFI bridge between C kernel module and Rust components:
//! - Export Rust functions for C to call
//! - Import kernel APIs safely
//! - Error code translation between C and Rust
//! - Resource lifecycle management

use crate::{
    DsmilError, DsmilResult, DeviceState, TokenPosition,
    get_device_registry,
    smi::{get_smi_controller, init_smi_controller, cleanup_smi_controller},
    memory::discover_dsmil_region,
};

/// C-compatible error codes
#[repr(C)]
pub enum CDsmilError {
    Success = 0,
    InvalidDevice = -22,
    PermissionDenied = -13,
    Busy = -16,
    IoError = -5,
    TimedOut = -110,
    OutOfMemory = -12,
    NotFound = -2,
    HardwareFault = -100,
}

impl From<DsmilError> for CDsmilError {
    fn from(err: DsmilError) -> Self {
        match err {
            DsmilError::InvalidDevice => CDsmilError::InvalidDevice,
            DsmilError::PermissionDenied => CDsmilError::PermissionDenied,
            DsmilError::Busy => CDsmilError::Busy,
            DsmilError::IoError => CDsmilError::IoError,
            DsmilError::TimedOut => CDsmilError::TimedOut,
            DsmilError::OutOfMemory => CDsmilError::OutOfMemory,
            DsmilError::NotFound => CDsmilError::NotFound,
            DsmilError::HardwareFault => CDsmilError::HardwareFault,
        }
    }
}

impl From<CDsmilError> for i32 {
    fn from(err: CDsmilError) -> Self {
        err as i32
    }
}

/// C-compatible device info structure
#[repr(C)]
pub struct CDeviceInfo {
    pub group_id: u8,
    pub device_id: u8,
    pub global_id: u8,
    pub state: u8, // DeviceState as u8
}

/// C-compatible group info structure
#[repr(C)]
pub struct CGroupInfo {
    pub group_id: u8,
    pub state: u8, // GroupState as u8
    pub active_devices: u16,
}

/// C-compatible SMI request structure
#[repr(C)]
pub struct CSmiRequest {
    pub token_id: u16,
    pub command: u8,
    pub status: u8,
    pub data: u32,
    pub reserved: u32,
}

/// C-compatible memory stats structure
#[repr(C)]
pub struct CMemoryStats {
    pub total_size: u64,
    pub mapped_size: u64,
    pub chunk_count: u32,
    pub mapped_chunks: u32,
}

// Helper functions for safe conversions

fn to_c_error(result: DsmilResult<()>) -> i32 {
    match result {
        Ok(()) => 0,
        Err(e) => CDsmilError::from(e) as i32,
    }
}

// Exported functions for C to call

/// Initialize Rust DSMIL subsystem
#[no_mangle]
pub extern "C" fn rust_dsmil_init(enable_smi: bool) -> i32 {
    // Initialize SMI controller
    if let Err(e) = init_smi_controller(enable_smi) {
        return CDsmilError::from(e) as i32;
    }
    
    // Initialize device registry
    let registry = unsafe { get_device_registry() };
    to_c_error(registry.initialize())
}

/// Cleanup Rust DSMIL subsystem
#[no_mangle]
pub extern "C" fn rust_dsmil_cleanup() {
    cleanup_smi_controller();
    // Registry cleanup happens automatically in Drop
}

/// Create new device
#[no_mangle]
pub extern "C" fn rust_dsmil_create_device(
    group_id: u8,
    device_id: u8,
    info: *mut CDeviceInfo,
) -> i32 {
    if info.is_null() {
        return CDsmilError::InvalidDevice as i32;
    }
    
    let registry = unsafe { get_device_registry() };
    
    // Add device to group
    let result = registry.group_mut(group_id)
        .ok_or(DsmilError::NotFound)
        .and_then(|group| group.add_device(device_id));
    
    match result {
        Ok(()) => {
            // Fill device info
            if let Some(device) = registry.device(group_id, device_id) {
                let (gid, did, global_id) = device.identifiers();
                unsafe {
                    (*info) = CDeviceInfo {
                        group_id: gid,
                        device_id: did,
                        global_id,
                        state: device.state() as u8,
                    };
                }
            }
            0
        }
        Err(e) => CDsmilError::from(e) as i32,
    }
}

/// Get device information
#[no_mangle]
pub extern "C" fn rust_dsmil_get_device_info(
    group_id: u8,
    device_id: u8,
    info: *mut CDeviceInfo,
) -> i32 {
    if info.is_null() {
        return CDsmilError::InvalidDevice as i32;
    }
    
    let registry = unsafe { get_device_registry() };
    
    if let Some(device) = registry.device(group_id, device_id) {
        let (gid, did, global_id) = device.identifiers();
        unsafe {
            (*info) = CDeviceInfo {
                group_id: gid,
                device_id: did,
                global_id,
                state: device.state() as u8,
            };
        }
        0
    } else {
        CDsmilError::NotFound as i32
    }
}

/// Set device state
#[no_mangle]
pub extern "C" fn rust_dsmil_set_device_state(
    group_id: u8,
    device_id: u8,
    new_state: u8,
) -> i32 {
    const STATE_MAP: [DeviceState; 6] = [
        DeviceState::Offline,
        DeviceState::Initializing,
        DeviceState::Ready,
        DeviceState::Active,
        DeviceState::Error,
        DeviceState::Locked,
    ];

    let idx = new_state as usize;
    if idx >= STATE_MAP.len() {
        return CDsmilError::InvalidDevice as i32;
    }

    let state = STATE_MAP[idx];
    
    let registry = unsafe { get_device_registry() };
    
    if let Some(device) = registry.device_mut(group_id, device_id) {
        to_c_error(device.transition_state(state))
    } else {
        CDsmilError::NotFound as i32
    }
}

/// Map device MMIO region
#[no_mangle]
pub extern "C" fn rust_dsmil_map_device_mmio(
    group_id: u8,
    device_id: u8,
    base_addr: u64,
    size: u64,
) -> i32 {
    let registry = unsafe { get_device_registry() };
    
    if let Some(device) = registry.device_mut(group_id, device_id) {
        to_c_error(device.map_mmio(base_addr, size as usize))
    } else {
        CDsmilError::NotFound as i32
    }
}

/// Get group information
#[no_mangle]
pub extern "C" fn rust_dsmil_get_group_info(
    group_id: u8,
    info: *mut CGroupInfo,
) -> i32 {
    if info.is_null() {
        return CDsmilError::InvalidDevice as i32;
    }
    
    let registry = unsafe { get_device_registry() };
    
    if let Some(group) = registry.group(group_id) {
        unsafe {
            (*info) = CGroupInfo {
                group_id,
                state: group.state() as u8,
                active_devices: group.active_devices(),
            };
        }
        0
    } else {
        CDsmilError::NotFound as i32
    }
}

/// Activate device in group
#[no_mangle]
pub extern "C" fn rust_dsmil_activate_device(
    group_id: u8,
    device_id: u8,
) -> i32 {
    let registry = unsafe { get_device_registry() };
    
    if let Some(group) = registry.group_mut(group_id) {
        to_c_error(group.activate_device(device_id))
    } else {
        CDsmilError::NotFound as i32
    }
}

/// Deactivate device in group
#[no_mangle]
pub extern "C" fn rust_dsmil_deactivate_device(
    group_id: u8,
    device_id: u8,
) -> i32 {
    let registry = unsafe { get_device_registry() };
    
    if let Some(group) = registry.group_mut(group_id) {
        to_c_error(group.deactivate_device(device_id))
    } else {
        CDsmilError::NotFound as i32
    }
}

/// SMI token read operation
#[no_mangle]
pub extern "C" fn rust_dsmil_smi_read_token(
    position: u8,
    group_id: u8,
    data: *mut u32,
) -> i32 {
    if data.is_null() {
        return CDsmilError::InvalidDevice as i32;
    }
    
    let token_pos = match TokenPosition::from_u8(position) {
        Some(pos) => pos,
        None => return CDsmilError::InvalidDevice as i32,
    };
    
    if let Some(smi) = get_smi_controller() {
        match smi.access_locked_token(token_pos, group_id, true, None) {
            Ok(value) => {
                unsafe { *data = value; }
                0
            }
            Err(e) => CDsmilError::from(e) as i32,
        }
    } else {
        CDsmilError::NotFound as i32
    }
}

/// SMI token write operation
#[no_mangle]
pub extern "C" fn rust_dsmil_smi_write_token(
    position: u8,
    group_id: u8,
    data: u32,
) -> i32 {
    let token_pos = match TokenPosition::from_u8(position) {
        Some(pos) => pos,
        None => return CDsmilError::InvalidDevice as i32,
    };
    
    if let Some(smi) = get_smi_controller() {
        match smi.access_locked_token(token_pos, group_id, false, Some(data)) {
            Ok(_) => 0,
            Err(e) => CDsmilError::from(e) as i32,
        }
    } else {
        CDsmilError::NotFound as i32
    }
}

/// Unlock Meteor Lake region via SMI
#[no_mangle]
pub extern "C" fn rust_dsmil_smi_unlock_region(base_addr: u64) -> i32 {
    if let Some(smi) = get_smi_controller() {
        to_c_error(smi.mtl_unlock_region(base_addr))
    } else {
        CDsmilError::NotFound as i32
    }
}

/// Verify SMI functionality
#[no_mangle]
pub extern "C" fn rust_dsmil_smi_verify() -> i32 {
    if let Some(smi) = get_smi_controller() {
        to_c_error(smi.verify_functionality())
    } else {
        CDsmilError::NotFound as i32
    }
}

/// Check if SMI is currently active
#[no_mangle]
pub extern "C" fn rust_dsmil_smi_is_active(active: *mut bool) -> i32 {
    if active.is_null() {
        return CDsmilError::InvalidDevice as i32;
    }
    
    if let Some(smi) = get_smi_controller() {
        match smi.is_smi_active() {
            Ok(is_active) => {
                unsafe { *active = is_active; }
                0
            }
            Err(e) => CDsmilError::from(e) as i32,
        }
    } else {
        CDsmilError::NotFound as i32
    }
}

/// Emergency SMI abort
#[no_mangle]
pub extern "C" fn rust_dsmil_smi_emergency_abort() -> i32 {
    if let Some(smi) = get_smi_controller() {
        to_c_error(smi.emergency_abort())
    } else {
        CDsmilError::NotFound as i32
    }
}

/// Set SMI timeout
#[no_mangle]
pub extern "C" fn rust_dsmil_smi_set_timeout(timeout_ms: u32) -> i32 {
    if let Some(smi) = get_smi_controller() {
        smi.set_timeout_ms(timeout_ms);
        0
    } else {
        CDsmilError::NotFound as i32
    }
}

/// Enable/disable SMI access
#[no_mangle]
pub extern "C" fn rust_dsmil_smi_set_enabled(enabled: bool) -> i32 {
    if let Some(smi) = get_smi_controller() {
        smi.set_enabled(enabled);
        0
    } else {
        CDsmilError::NotFound as i32
    }
}

/// Discover DSMIL memory region
#[no_mangle]
pub extern "C" fn rust_dsmil_discover_region(
    size: u64,
    base_addr: *mut u64,
) -> i32 {
    if base_addr.is_null() {
        return CDsmilError::InvalidDevice as i32;
    }
    
    match discover_dsmil_region(size as usize) {
        Ok(addr) => {
            unsafe { *base_addr = addr; }
            0
        }
        Err(e) => CDsmilError::from(e) as i32,
    }
}

/// Get total active devices count
#[no_mangle]
pub extern "C" fn rust_dsmil_get_total_active_devices() -> u16 {
    let registry = unsafe { get_device_registry() };
    registry.total_active_devices()
}

/// Get system status
#[no_mangle]
pub extern "C" fn rust_dsmil_get_system_status(
    total_devices: *mut u16,
    smi_enabled: *mut bool,
) -> i32 {
    if total_devices.is_null() || smi_enabled.is_null() {
        return CDsmilError::InvalidDevice as i32;
    }
    
    let registry = unsafe { get_device_registry() };
    unsafe {
        *total_devices = registry.total_active_devices();
        *smi_enabled = get_smi_controller()
            .map(|smi| smi.is_enabled())
            .unwrap_or(false);
    }
    
    0
}

// Imported C functions that Rust can call

extern "C" {
    // Port I/O functions
    pub fn rust_inb(port: u16) -> u8;
    pub fn rust_inl(port: u16) -> u32;
    pub fn rust_outb(value: u8, port: u16);
    pub fn rust_outl(value: u32, port: u16);
    
    // Memory functions (already declared in memory.rs)
    // pub fn kernel_ioremap(phys_addr: u64, size: usize) -> *mut u8;
    // pub fn kernel_iounmap(addr: *mut u8, size: usize);
    
    // Scheduling functions
    pub fn rust_udelay(usecs: u32);
    pub fn rust_need_resched() -> bool;
    pub fn rust_cond_resched();
    
    // Thermal functions
    pub fn rust_get_thermal_temperature() -> i32;
    pub fn rust_thermal_zone_device_update(tzd: *mut core::ffi::c_void) -> i32;
    
    // Workqueue functions
    pub fn rust_schedule_work(work: *mut core::ffi::c_void);
    pub fn rust_cancel_work_sync(work: *mut core::ffi::c_void) -> bool;
    
    // Mutex functions
    pub fn rust_mutex_lock(mutex: *mut core::ffi::c_void);
    pub fn rust_mutex_unlock(mutex: *mut core::ffi::c_void);
    pub fn rust_mutex_trylock(mutex: *mut core::ffi::c_void) -> bool;
    
    // Logging functions
    pub fn rust_printk(level: u8, msg: *const core::ffi::c_char);
}

/// Safe wrapper for kernel logging
pub fn kernel_log(level: u8, msg: &str) {
    let cstr = msg.as_ptr() as *const core::ffi::c_char;
    unsafe {
        rust_printk(level, cstr);
    }
}

/// Kernel log levels
pub mod log_levels {
    pub const KERN_EMERG: u8 = 0;   // Emergency messages
    pub const KERN_ALERT: u8 = 1;   // Alert messages  
    pub const KERN_CRIT: u8 = 2;    // Critical messages
    pub const KERN_ERR: u8 = 3;     // Error messages
    pub const KERN_WARNING: u8 = 4; // Warning messages
    pub const KERN_NOTICE: u8 = 5;  // Notice messages
    pub const KERN_INFO: u8 = 6;    // Informational messages
    pub const KERN_DEBUG: u8 = 7;   // Debug messages
}

/// Helper macros for logging (similar to pr_info, pr_err, etc.)
#[macro_export]
macro_rules! pr_info {
    ($msg:expr) => {
        $crate::ffi::kernel_log($crate::ffi::log_levels::KERN_INFO, $msg)
    };
}

#[macro_export]
macro_rules! pr_err {
    ($msg:expr) => {
        $crate::ffi::kernel_log($crate::ffi::log_levels::KERN_ERR, $msg)
    };
}

#[macro_export]
macro_rules! pr_warn {
    ($msg:expr) => {
        $crate::ffi::kernel_log($crate::ffi::log_levels::KERN_WARNING, $msg)
    };
}

#[macro_export]
macro_rules! pr_debug {
    ($msg:expr) => {
        $crate::ffi::kernel_log($crate::ffi::log_levels::KERN_DEBUG, $msg)
    };
}

/// Thermal monitoring wrapper
pub fn get_thermal_temperature() -> i32 {
    unsafe { rust_get_thermal_temperature() }
}

/// Check if thermal threshold exceeded
pub fn check_thermal_threshold(threshold: i32) -> bool {
    get_thermal_temperature() > threshold
}

/// Safe delay function
pub fn safe_delay_us(microseconds: u32) {
    unsafe {
        rust_udelay(microseconds);
        if rust_need_resched() {
            rust_cond_resched();
        }
    }
}

/// Safe delay in milliseconds
pub fn safe_delay_ms(milliseconds: u32) {
    safe_delay_us(milliseconds * 1000);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_conversion() {
        let rust_err = DsmilError::InvalidDevice;
        let c_err = CDsmilError::from(rust_err);
        let error_code: i32 = c_err.into();
        assert_eq!(error_code, -22);
    }

    #[test]
    fn test_device_info_structure() {
        let info = CDeviceInfo {
            group_id: 1,
            device_id: 5,
            global_id: 17,
            state: DeviceState::Active as u8,
        };
        
        assert_eq!(info.group_id, 1);
        assert_eq!(info.device_id, 5);
        assert_eq!(info.global_id, 17);
        assert_eq!(info.state, 3);
    }
}

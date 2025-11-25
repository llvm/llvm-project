//! DSMIL Rust Safety Layer
//!
//! Provides safe Rust abstractions for the DSMIL kernel module operations:
//! - Safe memory mapping with automatic cleanup
//! - SMI controller with timeout guarantees  
//! - Token state machine with type safety
//! - Device registry with lifetime management

#![no_std]
#![no_main]

use core::ptr::NonNull;
use core::pin::Pin;
use core::marker::PhantomData;

// Panic handler required for no_std kernel modules
#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    // In kernel space, panic should halt the system or reset
    // This is a minimal implementation for kernel module safety
    loop {}
}

// Re-export submodules
pub mod smi;
pub mod memory;
pub mod ffi;

pub use smi::*;
pub use memory::*;
pub use ffi::*;

/// DSMIL Driver Error Types
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[repr(i32)]
pub enum DsmilError {
    InvalidDevice = -22,  // EINVAL
    PermissionDenied = -13, // EACCES
    Busy = -16,           // EBUSY
    IoError = -5,         // EIO
    TimedOut = -110,      // ETIMEDOUT
    OutOfMemory = -12,    // ENOMEM
    NotFound = -2,        // ENOENT
    HardwareFault = -100, // Custom hardware error
}

impl From<DsmilError> for i32 {
    fn from(err: DsmilError) -> i32 {
        err as i32
    }
}

/// DSMIL Result Type
pub type DsmilResult<T> = Result<T, DsmilError>;

/// Device State Machine
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum DeviceState {
    Offline = 0,
    Initializing = 1,
    Ready = 2,
    Active = 3,
    Error = 4,
    Locked = 5,
}

/// Group State Machine
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum GroupState {
    Disabled = 0,
    Initializing = 1,
    Ready = 2,
    Active = 3,
    Error = 4,
    EmergencyStop = 5,
}

/// Token Position for SMI Access
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum TokenPosition {
    PowerMgmt = 0,    // Position 0: Power management tokens
    MemoryCtrl = 1,   // Position 3: Memory control tokens  
    StorageCtrl = 2,  // Position 6: Storage control tokens
    SensorHub = 3,    // Position 9: Sensor hub tokens
}

impl TokenPosition {
    pub fn as_u8(self) -> u8 {
        self as u8
    }
    
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(TokenPosition::PowerMgmt),
            1 => Some(TokenPosition::MemoryCtrl),
            2 => Some(TokenPosition::StorageCtrl),
            3 => Some(TokenPosition::SensorHub),
            _ => None,
        }
    }
}

/// Safe Device Handle with lifetime management
pub struct DsmilDevice {
    group_id: u8,
    device_id: u8,
    global_id: u8,
    state: DeviceState,
    mmio_mapped: bool,
    mmio_base: u64,
    mmio_size: usize,
    _phantom: PhantomData<*const ()>, // !Send + !Sync for kernel context
}

impl DsmilDevice {
    /// Create new device handle (only called from C FFI)
    pub(crate) fn new(group_id: u8, device_id: u8) -> DsmilResult<Self> {
        if group_id >= 6 || device_id >= 12 {
            return Err(DsmilError::InvalidDevice);
        }
        
        let global_id = group_id * 12 + device_id;
        
        Ok(DsmilDevice {
            group_id,
            device_id,
            global_id,
            state: DeviceState::Offline,
            mmio_mapped: false,
            mmio_base: 0,
            mmio_size: 0,
            _phantom: PhantomData,
        })
    }
    
    /// Get device identifiers
    pub fn identifiers(&self) -> (u8, u8, u8) {
        (self.group_id, self.device_id, self.global_id)
    }
    
    /// Get current device state
    pub fn state(&self) -> DeviceState {
        self.state
    }
    
    /// Transition device state with validation
    pub fn transition_state(&mut self, new_state: DeviceState) -> DsmilResult<()> {
        use DeviceState::*;
        
        // State transition validation
        let valid_transition = match (self.state, new_state) {
            (Offline, Initializing) => true,
            (Initializing, Ready) => true,
            (Initializing, Error) => true,
            (Ready, Active) => true,
            (Ready, Error) => true,
            (Active, Ready) => true,
            (Active, Error) => true,
            (Error, Offline) => true,
            (Error, Initializing) => true,
            (_, Locked) => true, // Can always lock
            (Locked, Error) => true, // Can error from locked
            _ => false,
        };
        
        if !valid_transition {
            return Err(DsmilError::InvalidDevice);
        }
        
        self.state = new_state;
        Ok(())
    }
    
    /// Map device MMIO region safely
    pub fn map_mmio(&mut self, base_addr: u64, size: usize) -> DsmilResult<()> {
        if self.mmio_mapped {
            return Err(DsmilError::Busy);
        }
        
        // Validate alignment and size
        if base_addr & 0xFFF != 0 || size == 0 {
            return Err(DsmilError::InvalidDevice);
        }
        
        self.mmio_base = base_addr;
        self.mmio_size = size;
        self.mmio_mapped = true;
        Ok(())
    }
    
    /// Get MMIO mapping info
    pub fn mmio_info(&self) -> Option<(u64, usize)> {
        if self.mmio_mapped {
            Some((self.mmio_base, self.mmio_size))
        } else {
            None
        }
    }
}

impl Drop for DsmilDevice {
    fn drop(&mut self) {
        // Automatic cleanup - reset MMIO mapping
        self.mmio_mapped = false;
        self.mmio_base = 0;
        self.mmio_size = 0;
        self.state = DeviceState::Offline;
    }
}

/// Safe Device Group with coordination
pub struct DsmilGroup {
    group_id: u8,
    state: GroupState,
    devices: [Option<DsmilDevice>; 12],
    active_devices: u16, // Bitmask of active devices
}

impl DsmilGroup {
    /// Create new device group
    pub fn new(group_id: u8) -> DsmilResult<Self> {
        if group_id >= 6 {
            return Err(DsmilError::InvalidDevice);
        }
        
        Ok(DsmilGroup {
            group_id,
            state: GroupState::Disabled,
            devices: Default::default(),
            active_devices: 0,
        })
    }
    
    /// Add device to group
    pub fn add_device(&mut self, device_id: u8) -> DsmilResult<()> {
        if device_id >= 12 {
            return Err(DsmilError::InvalidDevice);
        }
        
        if self.devices[device_id as usize].is_some() {
            return Err(DsmilError::Busy);
        }
        
        let device = DsmilDevice::new(self.group_id, device_id)?;
        self.devices[device_id as usize] = Some(device);
        Ok(())
    }
    
    /// Get device reference
    pub fn device(&self, device_id: u8) -> Option<&DsmilDevice> {
        if device_id >= 12 {
            return None;
        }
        self.devices[device_id as usize].as_ref()
    }
    
    /// Get mutable device reference
    pub fn device_mut(&mut self, device_id: u8) -> Option<&mut DsmilDevice> {
        if device_id >= 12 {
            return None;
        }
        self.devices[device_id as usize].as_mut()
    }
    
    /// Activate device in group
    pub fn activate_device(&mut self, device_id: u8) -> DsmilResult<()> {
        if device_id >= 12 {
            return Err(DsmilError::InvalidDevice);
        }
        
        if let Some(device) = self.device_mut(device_id) {
            device.transition_state(DeviceState::Active)?;
            self.active_devices |= 1 << device_id;
            Ok(())
        } else {
            Err(DsmilError::NotFound)
        }
    }
    
    /// Deactivate device in group  
    pub fn deactivate_device(&mut self, device_id: u8) -> DsmilResult<()> {
        if device_id >= 12 {
            return Err(DsmilError::InvalidDevice);
        }
        
        if let Some(device) = self.device_mut(device_id) {
            device.transition_state(DeviceState::Ready)?;
            self.active_devices &= !(1 << device_id);
            Ok(())
        } else {
            Err(DsmilError::NotFound)
        }
    }
    
    /// Get active devices bitmask
    pub fn active_devices(&self) -> u16 {
        self.active_devices
    }
    
    /// Get group state
    pub fn state(&self) -> GroupState {
        self.state
    }
    
    /// Set group state
    pub fn set_state(&mut self, state: GroupState) {
        self.state = state;
    }
}

/// Global Device Registry with safe access
pub struct DeviceRegistry {
    groups: [Option<DsmilGroup>; 6],
    initialized: bool,
}

impl DeviceRegistry {
    /// Create new device registry
    pub const fn new() -> Self {
        DeviceRegistry {
            groups: [None, None, None, None, None, None],
            initialized: false,
        }
    }
    
    /// Initialize registry
    pub fn initialize(&mut self) -> DsmilResult<()> {
        if self.initialized {
            return Err(DsmilError::Busy);
        }
        
        // Initialize all groups
        for i in 0..6 {
            self.groups[i] = Some(DsmilGroup::new(i as u8)?);
        }
        
        self.initialized = true;
        Ok(())
    }
    
    /// Get group reference
    pub fn group(&self, group_id: u8) -> Option<&DsmilGroup> {
        if !self.initialized || group_id >= 6 {
            return None;
        }
        self.groups[group_id as usize].as_ref()
    }
    
    /// Get mutable group reference
    pub fn group_mut(&mut self, group_id: u8) -> Option<&mut DsmilGroup> {
        if !self.initialized || group_id >= 6 {
            return None;
        }
        self.groups[group_id as usize].as_mut()
    }
    
    /// Get device across all groups
    pub fn device(&self, group_id: u8, device_id: u8) -> Option<&DsmilDevice> {
        self.group(group_id)?.device(device_id)
    }
    
    /// Get mutable device across all groups
    pub fn device_mut(&mut self, group_id: u8, device_id: u8) -> Option<&mut DsmilDevice> {
        self.group_mut(group_id)?.device_mut(device_id)
    }
    
    /// Count total active devices
    pub fn total_active_devices(&self) -> u16 {
        if !self.initialized {
            return 0;
        }
        
        self.groups.iter()
            .flatten()
            .map(|group| group.active_devices().count_ones() as u16)
            .sum()
    }
}

impl Drop for DeviceRegistry {
    fn drop(&mut self) {
        // Cleanup all groups and devices
        for group in &mut self.groups {
            *group = None;
        }
        self.initialized = false;
    }
}

/// Global static registry (initialized by C code)
static mut DEVICE_REGISTRY: DeviceRegistry = DeviceRegistry::new();

/// Get global device registry (unsafe - must be called from kernel context)
pub unsafe fn get_device_registry() -> &'static mut DeviceRegistry {
    &mut DEVICE_REGISTRY
}

/// Constants from C module
pub mod constants {
    pub const DSMIL_GROUP_COUNT: u8 = 6;
    pub const DSMIL_DEVICES_PER_GROUP: u8 = 12;
    pub const DSMIL_TOTAL_DEVICES: u8 = 72;
    
    // Memory mapping constants
    pub const DSMIL_PRIMARY_BASE: u64 = 0x52000000;
    pub const DSMIL_JRTC1_BASE: u64 = 0x58000000;
    pub const DSMIL_EXTENDED_BASE: u64 = 0x5C000000;
    pub const DSMIL_PLATFORM_BASE: u64 = 0x48000000;
    pub const DSMIL_HIGH_BASE: u64 = 0x60000000;
    pub const DSMIL_MEMORY_SIZE: usize = 360 * 1024 * 1024; // 360MB
    pub const DSMIL_CHUNK_SIZE: usize = 4 * 1024 * 1024;    // 4MB chunks
    
    // SMI ports and commands
    pub const SMI_CMD_PORT: u16 = 0xB2;
    pub const SMI_STATUS_PORT: u16 = 0xB3;
    pub const SMI_CMD_TOKEN_READ: u8 = 0x01;
    pub const SMI_CMD_TOKEN_WRITE: u8 = 0x02;
    pub const SMI_CMD_VERIFY: u8 = 0xFF;
    
    // Meteor Lake SMI coordination
    pub const MTL_SMI_COORD_START: u8 = 0xA0;
    pub const MTL_SMI_COORD_SYNC: u8 = 0xA1;
    pub const MTL_SMI_COORD_COMPLETE: u8 = 0xA2;
    
    // Dell Legacy I/O
    pub const DELL_LEGACY_IO_BASE: u16 = 0x164E;
    pub const DELL_LEGACY_IO_DATA: u16 = 0x164F;
    
    // Timeouts
    pub const SMI_TIMEOUT_MS: u32 = 50;
    pub const MTL_UNLOCK_TIMEOUT_MS: u32 = 200;
}
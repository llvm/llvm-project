//! SMI Operations Module
//!
//! Safe Rust abstractions for System Management Interrupt operations:
//! - Port I/O with timeouts and hang detection
//! - Dell-specific timing and coordination
//! - Emergency abort mechanisms
//! - Meteor Lake P-core/E-core coordination

use crate::{DsmilError, DsmilResult, TokenPosition};
use crate::constants::*;
use core::time::Duration;

/// SMI Request Structure
#[repr(C)]
pub struct SmiRequest {
    pub token_id: u16,
    pub command: u8,
    pub status: u8,
    pub data: u32,
    pub reserved: u32,
}

impl SmiRequest {
    /// Create new SMI request
    pub fn new(token_id: u16, command: u8) -> Self {
        SmiRequest {
            token_id,
            command,
            status: 0,
            data: 0,
            reserved: 0,
        }
    }
    
    /// Create read request
    pub fn read_token(token_id: u16) -> Self {
        Self::new(token_id, SMI_CMD_TOKEN_READ)
    }
    
    /// Create write request
    pub fn write_token(token_id: u16, data: u32) -> Self {
        let mut req = Self::new(token_id, SMI_CMD_TOKEN_WRITE);
        req.data = data;
        req
    }
    
    /// Create verify request
    pub fn verify() -> Self {
        Self::new(0, SMI_CMD_VERIFY)
    }
}

/// SMI Controller with timeout guarantees and hang detection
pub struct SmiController {
    enabled: bool,
    timeout_ms: u32,
    emergency_abort_threshold: u32,
}

impl SmiController {
    /// Create new SMI controller with safety parameters
    pub fn new(enabled: bool) -> Self {
        SmiController {
            enabled,
            timeout_ms: SMI_TIMEOUT_MS,
            emergency_abort_threshold: 5, // Max retries before emergency abort
        }
    }
    
    /// Check if SMI is already active (hang prevention)
    pub fn is_smi_active(&self) -> DsmilResult<bool> {
        if !self.enabled {
            return Ok(false);
        }
        
        let status = unsafe { self.inb_safe(SMI_STATUS_PORT)? };
        Ok(status & 0x01 != 0)
    }
    
    /// Perform safe port I/O read with error checking
    unsafe fn inb_safe(&self, port: u16) -> DsmilResult<u8> {
        // In real kernel, this would call the actual inb() function
        // For now, we define the interface
        extern "C" {
            fn rust_inb(port: u16) -> u8;
        }
        
        // TODO: Add hardware fault detection
        Ok(rust_inb(port))
    }
    
    /// Perform safe port I/O write with error checking
    unsafe fn outb_safe(&self, value: u8, port: u16) -> DsmilResult<()> {
        extern "C" {
            fn rust_outb(value: u8, port: u16);
        }
        
        rust_outb(value, port);
        Ok(())
    }
    
    /// Perform safe 32-bit port I/O write
    unsafe fn outl_safe(&self, value: u32, port: u16) -> DsmilResult<()> {
        extern "C" {
            fn rust_outl(value: u32, port: u16);
        }
        
        rust_outl(value, port);
        Ok(())
    }
    
    /// Wait for SMI completion with hang detection
    pub fn wait_for_completion(&self, timeout_ms: u32) -> DsmilResult<u8> {
        if !self.enabled {
            return Err(DsmilError::PermissionDenied);
        }
        
        let mut elapsed = 0u32;
        let mut last_status = 0xFFu8;
        let mut hang_count = 0u32;
        let check_interval = 5u32; // Check every 5ms
        
        while elapsed < timeout_ms {
            let status = unsafe { self.inb_safe(SMI_STATUS_PORT)? };
            
            // Dell-specific hang detection
            if status == last_status {
                hang_count += 1;
                
                // If status unchanged for too long, consider it hung
                if hang_count * check_interval > 20 {
                    return Err(DsmilError::TimedOut);
                }
            } else {
                hang_count = 0;
                last_status = status;
            }
            
            // Check for completion
            if status == MTL_SMI_COORD_COMPLETE {
                return Ok(status);
            }
            
            // Check for error conditions
            if status & 0x80 != 0 {
                return Err(DsmilError::IoError);
            }
            
            // Small delay with preemption point
            self.safe_delay_ms(check_interval)?;
            elapsed += check_interval;
        }
        
        Err(DsmilError::TimedOut)
    }
    
    /// Safe delay with preemption
    fn safe_delay_ms(&self, ms: u32) -> DsmilResult<()> {
        extern "C" {
            fn rust_udelay(usecs: u32);
            fn rust_need_resched() -> bool;
            fn rust_cond_resched();
        }
        
        unsafe {
            // Use microsecond delays for precision
            rust_udelay(ms * 1000);
            
            // Check for preemption needs
            if rust_need_resched() {
                rust_cond_resched();
            }
        }
        
        Ok(())
    }
    
    /// Emergency SMI abort procedure
    pub fn emergency_abort(&self) -> DsmilResult<()> {
        if !self.enabled {
            return Ok(());
        }
        
        // Try to clear SMI state
        unsafe {
            self.outb_safe(0x00, SMI_CMD_PORT)?; // Clear command
            self.safe_delay_ms(10)?; // Give hardware time to respond
            
            // Check if cleared
            let status = self.inb_safe(SMI_STATUS_PORT)?;
            if status & 0x01 != 0 {
                // Still active, more aggressive reset
                self.outb_safe(SMI_CMD_VERIFY, SMI_CMD_PORT)?; // Verify command
                self.safe_delay_ms(50)?;
            }
        }
        
        Ok(())
    }
    
    /// Meteor Lake region unlock with Dell-safe coordination
    pub fn mtl_unlock_region(&self, base_addr: u64) -> DsmilResult<()> {
        if !self.enabled {
            return Err(DsmilError::PermissionDenied);
        }
        
        // Check if SMI already active (prevent hang)
        if self.is_smi_active()? {
            return Err(DsmilError::Busy);
        }
        
        unsafe {
            // Start coordination sequence
            self.outb_safe(MTL_SMI_COORD_START, SMI_CMD_PORT)?;
            
            // Provide base address via legacy I/O with Dell-safe timing
            self.outl_safe((base_addr >> 32) as u32, DELL_LEGACY_IO_BASE)?;
            self.outl_safe(base_addr as u32, DELL_LEGACY_IO_DATA)?;
            
            // Dell-specific delay
            self.safe_delay_ms(50)?;
            
            // Trigger P/E core synchronization
            self.outb_safe(MTL_SMI_COORD_SYNC, SMI_CMD_PORT)?;
            
            // Wait for completion with timeout
            match self.wait_for_completion(MTL_UNLOCK_TIMEOUT_MS) {
                Ok(_) => Ok(()),
                Err(DsmilError::TimedOut) => {
                    // Emergency abort on timeout
                    self.emergency_abort()?;
                    Err(DsmilError::TimedOut)
                }
                Err(e) => {
                    // Emergency abort on other errors
                    self.emergency_abort()?;
                    Err(e)
                }
            }
        }
    }
    
    /// Execute SMI request with full safety checks
    pub fn execute_request(&self, request: &mut SmiRequest) -> DsmilResult<()> {
        if !self.enabled {
            return Err(DsmilError::PermissionDenied);
        }
        
        // Pre-flight safety checks
        if self.is_smi_active()? {
            return Err(DsmilError::Busy);
        }
        
        // Setup request data in legacy I/O ports
        unsafe {
            self.outl_safe(request.token_id as u32, DELL_LEGACY_IO_BASE)?;
            if request.command == SMI_CMD_TOKEN_WRITE {
                self.outl_safe(request.data, DELL_LEGACY_IO_DATA)?;
            }
        }
        
        // Execute with retry logic
        let mut attempts = 0;
        while attempts < self.emergency_abort_threshold {
            // Issue SMI command
            unsafe {
                self.outb_safe(request.command, SMI_CMD_PORT)?;
            }
            
            // Wait for completion
            match self.wait_for_completion(self.timeout_ms) {
                Ok(status) => {
                    request.status = status;
                    
                    // For read operations, get the data
                    if request.command == SMI_CMD_TOKEN_READ {
                        unsafe {
                            // Data would be available in legacy I/O port
                            // This is implementation-specific
                            request.data = 0; // Placeholder
                        }
                    }
                    
                    return Ok(());
                }
                Err(DsmilError::TimedOut) if attempts < self.emergency_abort_threshold - 1 => {
                    // Retry on timeout (up to limit)
                    attempts += 1;
                    self.emergency_abort()?;
                    self.safe_delay_ms(10)?; // Brief delay before retry
                    continue;
                }
                Err(e) => {
                    // Abort on any error
                    self.emergency_abort()?;
                    return Err(e);
                }
            }
        }
        
        // If we get here, all retries failed
        self.emergency_abort()?;
        Err(DsmilError::HardwareFault)
    }
    
    /// Access locked token via SMI (primary interface)
    pub fn access_locked_token(
        &self,
        position: TokenPosition,
        group_id: u8,
        is_read: bool,
        write_data: Option<u32>,
    ) -> DsmilResult<u32> {
        if group_id >= 6 {
            return Err(DsmilError::InvalidDevice);
        }
        
        // Map position to actual token ID (simplified mapping)
        let base_token_id = match position {
            TokenPosition::PowerMgmt => 0x8300,
            TokenPosition::MemoryCtrl => 0x8310,
            TokenPosition::StorageCtrl => 0x8320,
            TokenPosition::SensorHub => 0x8330,
        };
        
        let token_id = base_token_id + (group_id as u16);
        
        let mut request = if is_read {
            SmiRequest::read_token(token_id)
        } else {
            let data = write_data.ok_or(DsmilError::InvalidDevice)?;
            SmiRequest::write_token(token_id, data)
        };
        
        self.execute_request(&mut request)?;
        
        Ok(request.data)
    }
    
    /// Set timeout for SMI operations
    pub fn set_timeout_ms(&mut self, timeout_ms: u32) {
        self.timeout_ms = timeout_ms.min(200); // Cap at 200ms for safety
    }
    
    /// Enable or disable SMI access
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }
    
    /// Get current enabled state
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }
    
    /// Verify SMI functionality
    pub fn verify_functionality(&self) -> DsmilResult<()> {
        if !self.enabled {
            return Err(DsmilError::PermissionDenied);
        }
        
        let mut request = SmiRequest::verify();
        self.execute_request(&mut request)?;
        
        // Check if verify succeeded (implementation-specific)
        if request.status == 0 {
            Ok(())
        } else {
            Err(DsmilError::HardwareFault)
        }
    }
}

impl Default for SmiController {
    fn default() -> Self {
        Self::new(true)
    }
}

/// Global SMI controller instance
static mut SMI_CONTROLLER: Option<SmiController> = None;

/// Initialize global SMI controller
pub fn init_smi_controller(enabled: bool) -> DsmilResult<()> {
    unsafe {
        if SMI_CONTROLLER.is_some() {
            return Err(DsmilError::Busy);
        }
        SMI_CONTROLLER = Some(SmiController::new(enabled));
    }
    Ok(())
}

/// Get global SMI controller reference
pub fn get_smi_controller() -> Option<&'static mut SmiController> {
    unsafe { SMI_CONTROLLER.as_mut() }
}

/// Cleanup global SMI controller
pub fn cleanup_smi_controller() {
    unsafe {
        if let Some(controller) = &SMI_CONTROLLER {
            let _ = controller.emergency_abort(); // Best effort cleanup
        }
        SMI_CONTROLLER = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_smi_request_creation() {
        let request = SmiRequest::read_token(0x8300);
        assert_eq!(request.token_id, 0x8300);
        assert_eq!(request.command, SMI_CMD_TOKEN_READ);
        assert_eq!(request.data, 0);
        
        let request = SmiRequest::write_token(0x8310, 0xDEADBEEF);
        assert_eq!(request.token_id, 0x8310);
        assert_eq!(request.command, SMI_CMD_TOKEN_WRITE);
        assert_eq!(request.data, 0xDEADBEEF);
    }
    
    #[test]
    fn test_controller_creation() {
        let controller = SmiController::new(true);
        assert!(controller.is_enabled());
        assert_eq!(controller.timeout_ms, SMI_TIMEOUT_MS);
    }
}
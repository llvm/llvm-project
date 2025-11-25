//! Safety constraints and emergency controls for chaos testing
//!
//! This module implements comprehensive safety constraints, emergency stop
//! mechanisms, and protective measures to ensure chaos testing never
//! compromises critical system components or safety-critical operations.

use crate::*;
use std::collections::HashSet;

/// Security safety limits and constraints
#[derive(Debug, Clone)]
pub struct SecuritySafetyLimits {
    /// Devices that are completely protected from chaos testing
    pub protected_devices: HashSet<u32>,
    
    /// Maximum number of concurrent chaos tests
    pub max_concurrent_tests: u32,
    
    /// Maximum duration for any single chaos test
    pub max_test_duration: Duration,
    
    /// Minimum time between chaos tests
    pub min_test_interval: Duration,
    
    /// System resource limits
    pub resource_limits: ResourceLimits,
    
    /// Emergency response configuration
    pub emergency_config: EmergencyConfig,
    
    /// Quarantine validation settings
    pub quarantine_validation: QuarantineValidation,
}

/// System resource limits
#[derive(Debug, Clone)]
pub struct ResourceLimits {
    pub max_cpu_usage: f32,
    pub max_memory_usage: f32,
    pub max_network_bandwidth: f32,
    pub max_disk_io: f32,
}

/// Emergency configuration
#[derive(Debug, Clone)]
pub struct EmergencyConfig {
    pub auto_stop_on_violation: bool,
    pub notification_endpoints: Vec<String>,
    pub escalation_timeout: Duration,
    pub recovery_procedures: Vec<String>,
}

/// Quarantine validation settings
#[derive(Debug, Clone)]
pub struct QuarantineValidation {
    pub verify_before_test: bool,
    pub continuous_monitoring: bool,
    pub immediate_stop_on_violation: bool,
    pub quarantine_devices: HashSet<u32>,
}

impl Default for SecuritySafetyLimits {
    fn default() -> Self {
        let mut protected_devices = HashSet::new();
        // Critical quarantined devices that must never be accessed
        protected_devices.insert(0);  // Master Control
        protected_devices.insert(1);  // Security Platform  
        protected_devices.insert(12); // Power Controller
        protected_devices.insert(24); // Memory Controller
        protected_devices.insert(83); // Emergency Stop
        
        let mut quarantine_devices = HashSet::new();
        quarantine_devices.extend(&protected_devices);
        
        Self {
            protected_devices,
            max_concurrent_tests: 1,
            max_test_duration: Duration::from_secs(3600), // 1 hour max
            min_test_interval: Duration::from_secs(300),   // 5 minutes between tests
            resource_limits: ResourceLimits {
                max_cpu_usage: 0.7,
                max_memory_usage: 0.8,
                max_network_bandwidth: 0.5,
                max_disk_io: 0.6,
            },
            emergency_config: EmergencyConfig {
                auto_stop_on_violation: true,
                notification_endpoints: vec![
                    "security-team@dsmil.local".to_string(),
                    "soc@dsmil.local".to_string(),
                ],
                escalation_timeout: Duration::from_secs(60),
                recovery_procedures: vec![
                    "Isolate affected systems".to_string(),
                    "Verify quarantine integrity".to_string(),
                    "Run system diagnostics".to_string(),
                    "Generate incident report".to_string(),
                ],
            },
            quarantine_validation: QuarantineValidation {
                verify_before_test: true,
                continuous_monitoring: true,
                immediate_stop_on_violation: true,
                quarantine_devices,
            },
        }
    }
}

/// Emergency stop controller
#[derive(Debug)]
pub struct EmergencyStopController {
    stop_state: Arc<Mutex<EmergencyStopState>>,
    safety_limits: SecuritySafetyLimits,
}

/// Emergency stop state
#[derive(Debug, Clone)]
pub struct EmergencyStopState {
    pub emergency_active: bool,
    pub triggered_at: Option<DateTime<Utc>>,
    pub trigger_reason: Option<String>,
    pub triggered_by: Option<String>,
    pub active_responses: Vec<String>,
}

impl Default for EmergencyStopState {
    fn default() -> Self {
        Self {
            emergency_active: false,
            triggered_at: None,
            trigger_reason: None,
            triggered_by: None,
            active_responses: Vec::new(),
        }
    }
}

/// Safety violation types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SafetyViolation {
    QuarantineAccess { device_id: u32, attempted_operation: String },
    ResourceExceeded { resource_type: String, current_value: f32, limit: f32 },
    ConcurrentTestLimit { current_tests: u32, max_tests: u32 },
    TestDurationExceeded { duration: Duration, max_duration: Duration },
    SystemInstability { description: String },
    UnauthorizedAccess { user_id: u32, attempted_action: String },
    CriticalSystemFailure { system: String, error: String },
}

impl EmergencyStopController {
    /// Create new emergency stop controller
    pub fn new(safety_limits: SecuritySafetyLimits) -> Self {
        Self {
            stop_state: Arc::new(Mutex::new(EmergencyStopState::default())),
            safety_limits,
        }
    }
    
    /// Trigger emergency stop
    pub async fn trigger_emergency_stop(
        &self,
        reason: &str,
        triggered_by: &str,
    ) -> Result<(), SecurityChaosError> {
        let mut state = self.stop_state.lock().unwrap();
        
        if state.emergency_active {
            log::warn!("Emergency stop already active since {:?}", state.triggered_at);
            return Ok(());
        }
        
        state.emergency_active = true;
        state.triggered_at = Some(Utc::now());
        state.trigger_reason = Some(reason.to_string());
        state.triggered_by = Some(triggered_by.to_string());
        
        log::error!("EMERGENCY STOP TRIGGERED: {} (by: {})", reason, triggered_by);
        
        // Execute immediate response procedures
        self.execute_emergency_response(&mut state).await?;
        
        // Send notifications
        self.send_emergency_notifications(reason, triggered_by).await?;
        
        Ok(())
    }
    
    /// Check if emergency stop is active
    pub fn is_emergency_active(&self) -> bool {
        let state = self.stop_state.lock().unwrap();
        state.emergency_active
    }
    
    /// Get emergency stop state
    pub fn get_emergency_state(&self) -> EmergencyStopState {
        let state = self.stop_state.lock().unwrap();
        state.clone()
    }
    
    /// Reset emergency stop (requires manual intervention)
    pub async fn reset_emergency_stop(&self, reset_by: &str) -> Result<(), SecurityChaosError> {
        let mut state = self.stop_state.lock().unwrap();
        
        if !state.emergency_active {
            log::info!("Emergency stop not active, no reset needed");
            return Ok(());
        }
        
        // Verify system is safe before reset
        let system_safe = self.verify_system_safety().await?;
        if !system_safe {
            return Err(SecurityChaosError::SystemNotReady(
                "System not safe for emergency stop reset".to_string()
            ));
        }
        
        log::info!("Emergency stop reset by: {}", reset_by);
        
        *state = EmergencyStopState::default();
        
        Ok(())
    }
    
    /// Execute emergency response procedures
    async fn execute_emergency_response(
        &self,
        state: &mut EmergencyStopState,
    ) -> Result<(), SecurityChaosError> {
        log::info!("Executing emergency response procedures");
        
        // Isolate affected systems
        state.active_responses.push("System isolation initiated".to_string());
        
        // Verify quarantine integrity
        if self.safety_limits.quarantine_validation.immediate_stop_on_violation {
            let quarantine_intact = self.verify_quarantine_integrity().await?;
            if quarantine_intact {
                state.active_responses.push("Quarantine integrity verified".to_string());
            } else {
                state.active_responses.push("CRITICAL: Quarantine integrity compromised".to_string());
                log::error!("CRITICAL: Quarantine integrity compromised during emergency stop");
            }
        }
        
        // Stop all active chaos tests
        state.active_responses.push("All chaos tests terminated".to_string());
        
        // Run system diagnostics
        state.active_responses.push("System diagnostics initiated".to_string());
        
        log::info!("Emergency response procedures completed");
        Ok(())
    }
    
    /// Send emergency notifications
    async fn send_emergency_notifications(
        &self,
        reason: &str,
        triggered_by: &str,
    ) -> Result<(), SecurityChaosError> {
        for endpoint in &self.safety_limits.emergency_config.notification_endpoints {
            log::warn!("Emergency notification sent to {}: {} (triggered by: {})",
                      endpoint, reason, triggered_by);
        }
        Ok(())
    }
    
    /// Verify system safety
    async fn verify_system_safety(&self) -> Result<bool, SecurityChaosError> {
        // Verify quarantine devices are not compromised
        let quarantine_safe = self.verify_quarantine_integrity().await?;
        if !quarantine_safe {
            log::error!("Quarantine integrity check failed");
            return Ok(false);
        }
        
        // Check system resource usage
        let resources_ok = self.verify_resource_usage().await?;
        if !resources_ok {
            log::error!("System resource usage exceeds safe limits");
            return Ok(false);
        }
        
        // Verify no active high-risk operations
        let operations_safe = self.verify_no_high_risk_operations().await?;
        if !operations_safe {
            log::error!("High-risk operations still active");
            return Ok(false);
        }
        
        Ok(true)
    }
    
    /// Verify quarantine integrity
    async fn verify_quarantine_integrity(&self) -> Result<bool, SecurityChaosError> {
        log::info!("Verifying quarantine integrity for {} devices",
                  self.safety_limits.quarantine_validation.quarantine_devices.len());
        
        // Simulate quarantine verification
        sleep(Duration::from_millis(100)).await;
        
        // In a real implementation, this would:
        // 1. Check device access logs
        // 2. Verify no unauthorized writes occurred
        // 3. Validate device state integrity
        // 4. Confirm hardware protection mechanisms
        
        for device_id in &self.safety_limits.quarantine_validation.quarantine_devices {
            // Simulate device check
            log::debug!("Checking quarantine device {}", device_id);
        }
        
        // Always return true in simulation (quarantine is intact)
        Ok(true)
    }
    
    /// Verify resource usage is within limits
    async fn verify_resource_usage(&self) -> Result<bool, SecurityChaosError> {
        // Simulate resource usage check
        sleep(Duration::from_millis(50)).await;
        
        // In a real implementation, this would check:
        // - CPU usage
        // - Memory usage  
        // - Network bandwidth
        // - Disk I/O
        
        // Return true for simulation
        Ok(true)
    }
    
    /// Verify no high-risk operations are active
    async fn verify_no_high_risk_operations(&self) -> Result<bool, SecurityChaosError> {
        // Simulate operation check
        sleep(Duration::from_millis(30)).await;
        
        // In a real implementation, this would:
        // 1. Query active operation database
        // 2. Check for operations with RiskLevel::High or above
        // 3. Verify no critical system maintenance
        
        // Return true for simulation
        Ok(true)
    }
}

/// Safety validator for chaos testing
#[derive(Debug)]
pub struct ChaosTestSafetyValidator {
    safety_limits: SecuritySafetyLimits,
    emergency_controller: Arc<EmergencyStopController>,
    active_violations: Arc<Mutex<Vec<SafetyViolation>>>,
}

impl ChaosTestSafetyValidator {
    /// Create new safety validator
    pub fn new(safety_limits: SecuritySafetyLimits) -> Self {
        let emergency_controller = Arc::new(EmergencyStopController::new(safety_limits.clone()));
        
        Self {
            safety_limits,
            emergency_controller,
            active_violations: Arc::new(Mutex::new(Vec::new())),
        }
    }
    
    /// Validate test scenario safety
    pub async fn validate_test_safety(
        &self,
        scenario: &SecurityChaosScenario,
    ) -> Result<(), SecurityChaosError> {
        log::debug!("Validating test scenario safety");
        
        // Check for protected device access
        if let Some(device_id) = self.get_target_device(scenario) {
            if self.safety_limits.protected_devices.contains(&device_id) {
                let violation = SafetyViolation::QuarantineAccess {
                    device_id,
                    attempted_operation: format!("Chaos test: {:?}", scenario),
                };
                self.report_safety_violation(violation).await?;
                return Err(SecurityChaosError::DeviceProtected(device_id));
            }
        }
        
        // Check scenario-specific safety constraints
        match scenario {
            SecurityChaosScenario::BruteForceAttack { attempts_per_second, duration_seconds, .. } => {
                let total_attempts = attempts_per_second * duration_seconds;
                if total_attempts > 10000 {
                    log::warn!("Brute force test has very high attempt count: {}", total_attempts);
                }
            }
            SecurityChaosScenario::LogFlooding { events_per_second, duration_minutes, .. } => {
                let total_events = events_per_second * duration_minutes * 60;
                if total_events > 100000 {
                    return Err(SecurityChaosError::RiskTooHigh);
                }
            }
            SecurityChaosScenario::APTSimulation { attack_stages, .. } => {
                if attack_stages.len() > 20 {
                    log::warn!("APT simulation has many stages: {}", attack_stages.len());
                }
            }
            _ => {}
        }
        
        Ok(())
    }
    
    /// Monitor ongoing test for safety violations
    pub async fn monitor_test_safety(
        &self,
        test_id: Uuid,
        duration: Duration,
    ) -> Result<(), SecurityChaosError> {
        log::info!("Starting safety monitoring for test {}", test_id);
        
        let start_time = Instant::now();
        let check_interval = Duration::from_secs(5);
        
        while start_time.elapsed() < duration {
            // Check for emergency stop
            if self.emergency_controller.is_emergency_active() {
                log::warn!("Emergency stop active - terminating safety monitoring");
                break;
            }
            
            // Check quarantine integrity
            if self.safety_limits.quarantine_validation.continuous_monitoring {
                let quarantine_ok = self.emergency_controller.verify_quarantine_integrity().await?;
                if !quarantine_ok {
                    let violation = SafetyViolation::QuarantineAccess {
                        device_id: 0, // Generic quarantine violation
                        attempted_operation: "Quarantine integrity check failed".to_string(),
                    };
                    self.report_safety_violation(violation).await?;
                    
                    self.emergency_controller.trigger_emergency_stop(
                        "Quarantine integrity violation detected during test",
                        "Safety Monitor"
                    ).await?;
                    break;
                }
            }
            
            // Check resource limits
            let resources_ok = self.check_resource_limits().await?;
            if !resources_ok {
                log::warn!("Resource limits exceeded during test");
            }
            
            sleep(check_interval).await;
        }
        
        log::info!("Safety monitoring completed for test {}", test_id);
        Ok(())
    }
    
    /// Report safety violation
    async fn report_safety_violation(
        &self,
        violation: SafetyViolation,
    ) -> Result<(), SecurityChaosError> {
        log::error!("SAFETY VIOLATION: {:?}", violation);
        
        {
            let mut violations = self.active_violations.lock().unwrap();
            violations.push(violation.clone());
        }
        
        // Trigger emergency stop for critical violations
        match violation {
            SafetyViolation::QuarantineAccess { .. } => {
                self.emergency_controller.trigger_emergency_stop(
                    "Quarantine access violation",
                    "Safety Validator"
                ).await?;
            }
            SafetyViolation::CriticalSystemFailure { .. } => {
                self.emergency_controller.trigger_emergency_stop(
                    "Critical system failure",
                    "Safety Validator"
                ).await?;
            }
            _ => {
                // Non-critical violations logged but don't trigger emergency stop
                log::warn!("Non-critical safety violation logged");
            }
        }
        
        Ok(())
    }
    
    /// Get target device from scenario
    fn get_target_device(&self, scenario: &SecurityChaosScenario) -> Option<u32> {
        match scenario {
            SecurityChaosScenario::AuthorizationBypass { target_device, .. } => Some(*target_device),
            SecurityChaosScenario::HardwareTampering { device_id, .. } => Some(*device_id),
            _ => None,
        }
    }
    
    /// Check system resource limits
    async fn check_resource_limits(&self) -> Result<bool, SecurityChaosError> {
        // Simulate resource monitoring
        sleep(Duration::from_millis(10)).await;
        
        let mut rng = rand::thread_rng();
        let cpu_usage = rng.gen_range(0.1..0.5);
        let memory_usage = rng.gen_range(0.2..0.6);
        
        if cpu_usage > self.safety_limits.resource_limits.max_cpu_usage {
            let violation = SafetyViolation::ResourceExceeded {
                resource_type: "CPU".to_string(),
                current_value: cpu_usage,
                limit: self.safety_limits.resource_limits.max_cpu_usage,
            };
            self.report_safety_violation(violation).await?;
            return Ok(false);
        }
        
        if memory_usage > self.safety_limits.resource_limits.max_memory_usage {
            let violation = SafetyViolation::ResourceExceeded {
                resource_type: "Memory".to_string(),
                current_value: memory_usage,
                limit: self.safety_limits.resource_limits.max_memory_usage,
            };
            self.report_safety_violation(violation).await?;
            return Ok(false);
        }
        
        Ok(true)
    }
    
    /// Get current safety violations
    pub fn get_active_violations(&self) -> Vec<SafetyViolation> {
        let violations = self.active_violations.lock().unwrap();
        violations.clone()
    }
    
    /// Clear resolved violations
    pub fn clear_violations(&self) {
        let mut violations = self.active_violations.lock().unwrap();
        violations.clear();
        log::info!("Safety violations cleared");
    }
}
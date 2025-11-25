//! Security chaos scenario implementations
//!
//! This module contains the actual implementation of various security chaos
//! testing scenarios, including brute force attacks, privilege escalation,
//! authorization bypass, audit tampering, APT simulations, hardware tampering,
//! MFA bypass, data exfiltration, and EDR evasion.

use rand::Rng;
use std::{
    cmp,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};
use tokio::time::sleep;

// ==========================================================================
// Core types
// ==========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ClearanceLevel {
    Unclassified,
    Confidential,
    Secret,
    TopSecret,
}

#[derive(Debug, Clone)]
pub enum SecurityChaosError {
    InvalidScenario,
    SecurityViolation(String),
    AuthError(String),
    DeviceProtected(u32),
    Other(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuthError {
    InvalidCredentials,
    AccountLocked,
    RateLimited,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EscalationMethod {
    PrivilegeAbuse,
    CredentialTheft,
    VulnerabilityExploit,
    SocialEngineering,
    InsiderThreat,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BypassMethod {
    AuthorizationSkip,
    TokenManipulation,
    SessionHijacking,
    PrivilegeEscalation,
    BufferOverflow,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TamperMethod {
    EntryDeletion,
    EntryModification,
    TimestampManipulation,
    HashCollision,
    ChainBreak,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TamperType {
    PhysicalAccess,
    PowerAnalysis,
    ElectromagneticFault,
    TimingAttack,
    VoltageGlitch,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StealthLevel {
    High,
    Medium,
    Low,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum APTStageType {
    InitialAccess,
    Persistence,
    PrivilegeEscalation,
    DefenseEvasion,
    CredentialAccess,
    Discovery,
    LateralMovement,
    Collection,
    CommandAndControl,
    Exfiltration,
    Impact,
}

#[derive(Debug, Clone)]
pub struct APTStage {
    pub stage_name: String,
    pub stage_type: APTStageType,
    pub success_probability: f32,
    pub detection_probability: f32,
    pub duration_minutes: u32,
}

#[derive(Debug, Clone)]
pub struct APTStageResult {
    pub stage_name: String,
    pub successful: bool,
    pub detected: bool,
    pub execution_time_ms: u64,
    pub detection_method: Option<String>,
    pub artifacts_left: Vec<String>,
}

// MFA bypass

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MfaBypassVector {
    PushFatigue,
    TotpCapture,
    DeviceSwap,
    SimSwap,
    BackupCodeTheft,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MfaBypassOutcome {
    Blocked,
    UserFatigued,
    Bypass,
}

// Data exfiltration

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExfilChannel {
    DnsTunneling,
    HttpsBulk,
    CloudStorage,
    Sneakernet,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataSensitivity {
    Low,
    Medium,
    High,
    Regulated, // e.g. PCI / PHI
}

// EDR evasion

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdrEvasionMethod {
    KillAndDisable,
    UserModeInjection,
    LolbinAbuse,
    DriverAbuse,
}

// ==========================================================================
// Scenario + result enums
// ==========================================================================

#[derive(Debug, Clone)]
pub enum SecurityChaosScenario {
    BruteForceAttack {
        target_user: String,
        attempts_per_second: u32,
        duration_seconds: u32,
    },
    PrivilegeEscalation {
        source_clearance: ClearanceLevel,
        target_clearance: ClearanceLevel,
        attack_method: EscalationMethod,
    },
    AuthorizationBypass {
        target_device: u32,
        bypass_method: BypassMethod,
        risk_level: String,
    },
    AuditTampering {
        tamper_method: TamperMethod,
        target_entries: u32,
    },
    APTSimulation {
        campaign_name: String,
        attack_stages: Vec<APTStage>,
        stealth_level: StealthLevel,
    },
    HardwareTampering {
        device_id: u32,
        tamper_type: TamperType,
    },
    MfaBypass {
        user_id: String,
        vector: MfaBypassVector,
        max_attempts: u32,
    },
    DataExfiltration {
        channel: ExfilChannel,
        data_volume_mb: u32,
        sensitivity_level: DataSensitivity,
    },
    EdrEvasion {
        host_id: u32,
        method: EdrEvasionMethod,
        max_duration_seconds: u32,
    },
}

#[derive(Debug, Clone)]
pub enum ChaosExecutionResult {
    BruteForceResult {
        total_attempts: u32,
        successful_attempts: u32,
        lockout_triggered: bool,
        rate_limiting_effective: bool,
    },
    PrivilegeEscalationResult {
        escalation_attempted: bool,
        escalation_successful: bool,
        detection_time_ms: u64,
        mitigation_effective: bool,
    },
    AuthorizationBypassResult {
        bypass_attempted: bool,
        bypass_successful: bool,
        detection_time_ms: u64,
        devices_compromised: Vec<u32>,
    },
    AuditTamperingResult {
        tamper_attempted: bool,
        tamper_detected: bool,
        tampered_entries: u32,
    },
    APTResult {
        campaign_name: String,
        total_stages: usize,
        successful_stages: usize,
        detected_stages: usize,
        stage_results: Vec<APTStageResult>,
    },
    HardwareTamperResult {
        tamper_detected: bool,
        detection_time_ms: u64,
        system_response: String,
        integrity_maintained: bool,
    },
    MfaBypassResult {
        user_id: String,
        vector: MfaBypassVector,
        attempts: u32,
        bypass_successful: bool,
        user_fatigued: bool,
        detection_time_ms: u64,
    },
    DataExfiltrationResult {
        channel: ExfilChannel,
        data_volume_mb: u32,
        sensitivity_level: DataSensitivity,
        exfil_successful: bool,
        detection_time_ms: u64,
        detection_vector: String,
    },
    EdrEvasionResult {
        host_id: u32,
        method: EdrEvasionMethod,
        evasion_successful: bool,
        detection_time_ms: u64,
        detection_reason: String,
    },
}

// ==========================================================================
// Engine + constraints
// ==========================================================================

#[derive(Debug, Default, Clone)]
pub struct SafetyConstraints {
    pub protected_devices: Vec<u32>,
}

#[derive(Debug)]
pub struct SecurityChaosEngine {
    pub safety_constraints: SafetyConstraints,
    emergency_stop: Arc<AtomicBool>,
}

impl SecurityChaosEngine {
    pub fn new(safety_constraints: SafetyConstraints) -> Self {
        Self {
            safety_constraints,
            emergency_stop: Arc::new(AtomicBool::new(false)),
        }
    }

    pub fn set_emergency_stop(&self, stop: bool) {
        self.emergency_stop.store(stop, Ordering::Relaxed);
    }

    fn is_emergency_stopped(&self) -> bool {
        self.emergency_stop.load(Ordering::Relaxed)
    }

    // ======================================================================
    // Brute force
    // ======================================================================

    /// Execute brute force attack simulation
    pub async fn execute_brute_force_test(
        &self,
        scenario: &SecurityChaosScenario,
    ) -> Result<ChaosExecutionResult, SecurityChaosError> {
        if let SecurityChaosScenario::BruteForceAttack {
            target_user,
            attempts_per_second,
            duration_seconds,
        } = scenario
        {
            log::info!(
                "Starting brute force attack simulation against user {}",
                target_user
            );

            let mut successful_attempts = 0;
            let mut failed_attempts = 0;
            let mut lockout_triggered = false;
            let mut rate_limiting_detected = false;

            let total_attempts = attempts_per_second.saturating_mul(*duration_seconds);
            let delay_between_attempts = if *attempts_per_second == 0 {
                Duration::from_millis(0)
            } else {
                Duration::from_millis(1000 / *attempts_per_second as u64)
            };

            for attempt in 0..total_attempts {
                if self.is_emergency_stopped() {
                    log::warn!("Brute force test stopped due to emergency stop");
                    break;
                }

                let test_password = self.generate_test_password(attempt);

                let auth_result = self.test_authentication(target_user, &test_password).await;

                match auth_result {
                    Ok(_) => {
                        successful_attempts += 1;
                        log::error!(
                            "CRITICAL: Brute force attack succeeded - security failure!"
                        );
                        return Err(SecurityChaosError::SecurityViolation(
                            "Brute force attack succeeded - critical security failure"
                                .to_string(),
                        ));
                    }
                    Err(AuthError::InvalidCredentials) => {
                        failed_attempts += 1;
                    }
                    Err(AuthError::AccountLocked) => {
                        lockout_triggered = true;
                        log::info!(
                            "Account lockout triggered after {} attempts",
                            attempt + 1
                        );
                        break;
                    }
                    Err(AuthError::RateLimited) => {
                        rate_limiting_detected = true;
                        log::info!(
                            "Rate limiting activated after {} attempts",
                            attempt + 1
                        );
                        sleep(Duration::from_secs(2)).await;
                    }
                }

                if !delay_between_attempts.is_zero() {
                    sleep(delay_between_attempts).await;
                }
            }

            log::info!(
                "Brute force test completed: {} failed attempts, lockout: {}, rate limiting: {}",
                failed_attempts,
                lockout_triggered,
                rate_limiting_detected
            );

            Ok(ChaosExecutionResult::BruteForceResult {
                total_attempts: failed_attempts,
                successful_attempts,
                lockout_triggered,
                // Effective if we ever saw rate limiting or the account locked
                rate_limiting_effective: rate_limining_detected || lockout_triggered,
            })
        } else {
            Err(SecurityChaosError::InvalidScenario)
        }
    }

    // ======================================================================
    // Privilege escalation
    // ======================================================================

    /// Execute privilege escalation test
    pub async fn execute_privilege_escalation_test(
        &self,
        scenario: &SecurityChaosScenario,
    ) -> Result<ChaosExecutionResult, SecurityChaosError> {
        if let SecurityChaosScenario::PrivilegeEscalation {
            source_clearance,
            target_clearance,
            attack_method,
        } = scenario
        {
            log::info!(
                "Starting privilege escalation test: {:?} -> {:?} using {:?}",
                source_clearance,
                target_clearance,
                attack_method
            );

            let start_time = Instant::now();
            let mut escalation_successful = false;
            let mut mitigation_effective = true;

            match attack_method {
                EscalationMethod::PrivilegeAbuse => {
                    let access_result = self
                        .attempt_clearance_access(*source_clearance, *target_clearance)
                        .await;
                    escalation_successful = access_result.is_ok();
                }
                EscalationMethod::CredentialTheft => {
                    let theft_result = self.simulate_credential_theft().await;
                    escalation_successful = theft_result.is_ok();
                }
                EscalationMethod::VulnerabilityExploit => {
                    let exploit_result = self.simulate_vulnerability_exploit().await;
                    escalation_successful = exploit_result.is_ok();
                }
                EscalationMethod::SocialEngineering => {
                    escalation_successful = false;
                }
                EscalationMethod::InsiderThreat => {
                    let insider_result = self.simulate_insider_threat().await;
                    escalation_successful = insider_result.is_ok();
                }
            }

            let detection_time_ms = start_time.elapsed().as_millis() as u64;

            if escalation_successful {
                log::error!("CRITICAL: Privilege escalation succeeded - security failure!");
                mitigation_effective = false;
            } else {
                log::info!("Privilege escalation properly blocked");
            }

            Ok(ChaosExecutionResult::PrivilegeEscalationResult {
                escalation_attempted: true,
                escalation_successful,
                detection_time_ms,
                mitigation_effective,
            })
        } else {
            Err(SecurityChaosError::InvalidScenario)
        }
    }

    // ======================================================================
    // Authorization bypass
    // ======================================================================

    /// Execute authorization bypass test
    pub async fn execute_authorization_bypass_test(
        &self,
        scenario: &SecurityChaosScenario,
    ) -> Result<ChaosExecutionResult, SecurityChaosError> {
        if let SecurityChaosScenario::AuthorizationBypass {
            target_device,
            bypass_method,
            risk_level,
        } = scenario
        {
            log::info!(
                "Starting authorization bypass test on device {} using {:?} (risk: {:?})",
                target_device,
                bypass_method,
                risk_level
            );

            let start_time = Instant::now();
            let mut bypass_successful = false;
            let mut devices_compromised = Vec::new();

            let quarantined_devices = [0, 1, 12, 24, 83];
            if quarantined_devices.contains(target_device) {
                log::info!(
                    "Attempting bypass on quarantined device {} - should be blocked",
                    target_device
                );
            }

            match bypass_method {
                BypassMethod::AuthorizationSkip => {
                    bypass_successful = self
                        .attempt_authorization_skip(*target_device)
                        .await
                        .is_ok();
                }
                BypassMethod::TokenManipulation => {
                    bypass_successful = self.attempt_token_manipulation().await.is_ok();
                }
                BypassMethod::SessionHijacking => {
                    bypass_successful = self.attempt_session_hijack().await.is_ok();
                }
                BypassMethod::PrivilegeEscalation => {
                    bypass_successful = self.attempt_privilege_bypass().await.is_ok();
                }
                BypassMethod::BufferOverflow => {
                    bypass_successful = self.attempt_buffer_overflow().await.is_ok();
                }
            }

            let detection_time_ms = start_time.elapsed().as_millis() as u64;

            if bypass_successful {
                log::error!(
                    "CRITICAL: Authorization bypass succeeded on device {} - security failure!",
                    target_device
                );
                devices_compromised.push(*target_device);
            } else {
                log::info!(
                    "Authorization bypass properly blocked for device {}",
                    target_device
                );
            }

            Ok(ChaosExecutionResult::AuthorizationBypassResult {
                bypass_attempted: true,
                bypass_successful,
                detection_time_ms,
                devices_compromised,
            })
        } else {
            Err(SecurityChaosError::InvalidScenario)
        }
    }

    // ======================================================================
    // Audit tampering
    // ======================================================================

    /// Execute audit tampering test
    pub async fn execute_audit_tampering_test(
        &self,
        scenario: &SecurityChaosScenario,
    ) -> Result<ChaosExecutionResult, SecurityChaosError> {
        if let SecurityChaosScenario::AuditTampering {
            tamper_method,
            target_entries,
        } = scenario
        {
            log::info!(
                "Starting audit tampering test using {:?} on {} entries",
                tamper_method,
                target_entries
            );

            let tamper_detected = match tamper_method {
                TamperMethod::EntryDeletion => {
                    self.simulate_entry_deletion(*target_entries).await
                }
                TamperMethod::EntryModification => {
                    self.simulate_entry_modification(*target_entries).await
                }
                TamperMethod::TimestampManipulation => {
                    self.simulate_timestamp_tampering(*target_entries).await
                }
                TamperMethod::HashCollision => {
                    self.simulate_hash_collision_attack().await
                }
                TamperMethod::ChainBreak => {
                    self.simulate_chain_break_attack().await
                }
            };

            if tamper_detected {
                log::info!("Audit tampering properly detected and blocked");
                Ok(ChaosExecutionResult::AuditTamperingResult {
                    tamper_attempted: true,
                    tamper_detected: true,
                    tampered_entries: *target_entries,
                })
            } else {
                log::error!("CRITICAL: Audit tampering not detected - security failure!");
                Err(SecurityChaosError::SecurityViolation(
                    "Audit tampering went undetected".to_string(),
                ))
            }
        } else {
            Err(SecurityChaosError::InvalidScenario)
        }
    }

    // ======================================================================
    // APT simulation
    // ======================================================================

    /// Execute Advanced Persistent Threat (APT) simulation
    pub async fn execute_apt_simulation(
        &self,
        scenario: &SecurityChaosScenario,
    ) -> Result<ChaosExecutionResult, SecurityChaosError> {
        if let SecurityChaosScenario::APTSimulation {
            campaign_name,
            attack_stages,
            stealth_level,
        } = scenario
        {
            log::info!(
                "Starting APT simulation '{}' with {} stages (stealth: {:?})",
                campaign_name,
                attack_stages.len(),
                stealth_level
            );

            let mut stage_results = Vec::new();
            let mut detected_stages = 0;
            let mut successful_stages = 0;

            for (stage_num, stage) in attack_stages.iter().enumerate() {
                if self.is_emergency_stopped() {
                    log::warn!(
                        "APT simulation stopped due to emergency stop at stage {}",
                        stage_num
                    );
                    break;
                }

                let stage_delay = match stealth_level {
                    StealthLevel::High => Duration::from_secs(3600),
                    StealthLevel::Medium => Duration::from_secs(900),
                    StealthLevel::Low => Duration::from_secs(60),
                };

                if stage_num > 0 {
                    log::info!(
                        "APT stealth delay: waiting {} seconds before stage {}",
                        stage_delay.as_secs(),
                        stage_num + 1
                    );
                    sleep(stage_delay).await;
                }

                let stage_result = self.execute_apt_stage(stage.clone()).await?;

                if stage_result.detected {
                    detected_stages += 1;
                    log::info!(
                        "APT stage '{}' was detected by security systems",
                        stage.stage_name
                    );
                } else if stage_result.successful {
                    successful_stages += 1;
                    log::warn!(
                        "APT stage '{}' executed successfully",
                        stage.stage_name
                    );
                }

                stage_results.push(stage_result);

                if detected_stages > 0 && *stealth_level == StealthLevel::High {
                    log::info!("APT aborting due to detection (high stealth mode)");
                    break;
                }
            }

            log::info!(
                "APT simulation '{}' completed: {}/{} stages successful, {} detected",
                campaign_name,
                successful_stages,
                attack_stages.len(),
                detected_stages
            );

            Ok(ChaosExecutionResult::APTResult {
                campaign_name: campaign_name.clone(),
                total_stages: attack_stages.len(),
                successful_stages,
                detected_stages,
                stage_results,
            })
        } else {
            Err(SecurityChaosError::InvalidScenario)
        }
    }

    // ======================================================================
    // Hardware tampering
    // ======================================================================

    /// Execute hardware tampering test
    pub async fn execute_hardware_tampering_test(
        &self,
        scenario: &SecurityChaosScenario,
    ) -> Result<ChaosExecutionResult, SecurityChaosError> {
        if let SecurityChaosScenario::HardwareTampering { device_id, tamper_type } = scenario {
            log::info!(
                "Starting hardware tampering test on device {} using {:?}",
                device_id,
                tamper_type
            );

            let start_time = Instant::now();

            if self.safety_constraints.protected_devices.contains(device_id) {
                return Err(SecurityChaosError::DeviceProtected(*device_id));
            }

            let tamper_detected = match tamper_type {
                TamperType::PhysicalAccess => self.simulate_physical_tamper(*device_id).await,
                TamperType::PowerAnalysis => self.simulate_power_analysis(*device_id).await,
                TamperType::ElectromagneticFault => {
                    self.simulate_em_fault_injection(*device_id).await
                }
                TamperType::TimingAttack => self.simulate_timing_attack(*device_id).await,
                TamperType::VoltageGlitch => self.simulate_voltage_glitch(*device_id).await,
            };

            let detection_time_ms = start_time.elapsed().as_millis() as u64;

            let system_response = if tamper_detected {
                "Tamper detected, device isolated, security protocols activated".to_string()
            } else {
                "No tamper detection - potential security vulnerability".to_string()
            };

            if tamper_detected {
                log::info!("Hardware tampering on device {} properly detected", device_id);
            } else {
                log::warn!("Hardware tampering on device {} went undetected", device_id);
            }

            Ok(ChaosExecutionResult::HardwareTamperResult {
                tamper_detected,
                detection_time_ms,
                system_response,
                integrity_maintained: tamper_detected,
            })
        } else {
            Err(SecurityChaosError::InvalidScenario)
        }
    }

    // ======================================================================
    // MFA bypass
    // ======================================================================

    pub async fn execute_mfa_bypass_test(
        &self,
        scenario: &SecurityChaosScenario,
    ) -> Result<ChaosExecutionResult, SecurityChaosError> {
        if let SecurityChaosScenario::MfaBypass {
            user_id,
            vector,
            max_attempts,
        } = scenario
        {
            log::info!(
                "Starting MFA bypass test for user {} using {:?} (max attempts: {})",
                user_id,
                vector,
                max_attempts
            );

            let start_time = Instant::now();
            let mut attempts = 0u32;
            let mut bypass_successful = false;
            let mut user_fatigued = false;

            while attempts < *max_attempts {
                if self.is_emergency_stopped() {
                    log::warn!("MFA bypass test stopped due to emergency stop");
                    break;
                }

                attempts += 1;

                let result = self.simulate_mfa_bypass(&user_id, vector).await;

                match result {
                    Ok(MfaBypassOutcome::Blocked) => {}
                    Ok(MfaBypassOutcome::UserFatigued) => {
                        user_fatigued = true;
                        break;
                    }
                    Ok(MfaBypassOutcome::Bypass) => {
                        bypass_successful = true;
                        break;
                    }
                    Err(e) => {
                        log::warn!("MFA bypass simulation error: {:?}", e);
                        break;
                    }
                }

                sleep(Duration::from_millis(200)).await;
            }

            let detection_time_ms = start_time.elapsed().as_millis() as u64;

            if bypass_successful {
                log::error!(
                    "CRITICAL: MFA bypass succeeded for user {} via {:?}",
                    user_id,
                    vector
                );
            } else {
                log::info!(
                    "MFA bypass attempts for user {} were blocked (user_fatigued={})",
                    user_id,
                    user_fatigued
                );
            }

            Ok(ChaosExecutionResult::MfaBypassResult {
                user_id: user_id.clone(),
                vector: *vector,
                attempts,
                bypass_successful,
                user_fatigued,
                detection_time_ms,
            })
        } else {
            Err(SecurityChaosError::InvalidScenario)
        }
    }

    // ======================================================================
    // Data exfiltration
    // ======================================================================

    pub async fn execute_data_exfiltration_test(
        &self,
        scenario: &SecurityChaosScenario,
    ) -> Result<ChaosExecutionResult, SecurityChaosError> {
        if let SecurityChaosScenario::DataExfiltration {
            channel,
            data_volume_mb,
            sensitivity_level,
        } = scenario
        {
            log::info!(
                "Starting data exfiltration test via {:?} ({} MB, sensitivity: {:?})",
                channel,
                data_volume_mb,
                sensitivity_level
            );

            let start_time = Instant::now();
            let mut exfil_successful = false;
            let mut detection_vector = String::new();

            match channel {
                ExfilChannel::DnsTunneling => {
                    let (success, detector) =
                        self.simulate_dns_exfiltration(*data_volume_mb).await;
                    exfil_successful = success;
                    detection_vector = detector;
                }
                ExfilChannel::HttpsBulk => {
                    let (success, detector) =
                        self.simulate_https_exfiltration(*data_volume_mb).await;
                    exfil_successful = success;
                    detection_vector = detector;
                }
                ExfilChannel::CloudStorage => {
                    let (success, detector) =
                        self.simulate_cloud_exfiltration(*data_volume_mb).await;
                    exfil_successful = success;
                    detection_vector = detector;
                }
                ExfilChannel::Sneakernet => {
                    exfil_successful = false;
                    detection_vector = "Physical media controls".to_string();
                }
            }

            let detection_time_ms = start_time.elapsed().as_millis() as u64;

            if exfil_successful {
                log::error!(
                    "CRITICAL: Data exfiltration simulation succeeded ({} MB via {:?})",
                    data_volume_mb,
                    channel
                );
            } else {
                log::info!(
                    "Data exfiltration attempts blocked (detected by: {})",
                    detection_vector
                );
            }

            Ok(ChaosExecutionResult::DataExfiltrationResult {
                channel: *channel,
                data_volume_mb: *data_volume_mb,
                sensitivity_level: *sensitivity_level,
                exfil_successful,
                detection_time_ms,
                detection_vector,
            })
        } else {
            Err(SecurityChaosError::InvalidScenario)
        }
    }

    // ======================================================================
    // EDR evasion
    // ======================================================================

    pub async fn execute_edr_evasion_test(
        &self,
        scenario: &SecurityChaosScenario,
    ) -> Result<ChaosExecutionResult, SecurityChaosError> {
        if let SecurityChaosScenario::EdrEvasion {
            host_id,
            method,
            max_duration_seconds,
        } = scenario
        {
            log::info!(
                "Starting EDR evasion test on host {} using {:?} (max duration: {}s)",
                host_id,
                method,
                max_duration_seconds
            );

            let start_time = Instant::now();
            let mut rng = rand::thread_rng();
            let mut evasion_successful = false;
            let mut detection_reason = String::new();

            let mut elapsed = 0u64;
            while elapsed < *max_duration_seconds as u64 {
                if self.is_emergency_stopped() {
                    log::warn!("EDR evasion test stopped due to emergency stop");
                    break;
                }

                let (success, detection) =
                    self.simulate_edr_evasion(*host_id, method, &mut rng).await;

                if let Some(reason) = detection {
                    detection_reason = reason;
                    break;
                }

                if success {
                    evasion_successful = true;
                    break;
                }

                sleep(Duration::from_millis(500)).await;
                elapsed = start_time.elapsed().as_secs();
            }

            let detection_time_ms = start_time.elapsed().as_millis() as u64;

            if evasion_successful {
                log::error!(
                    "CRITICAL: EDR evasion succeeded on host {} via {:?}",
                    host_id,
                    method
                );
            } else {
                log::info!(
                    "EDR evasion blocked on host {} (reason: {})",
                    host_id,
                    detection_reason
                );
            }

            Ok(ChaosExecutionResult::EdrEvasionResult {
                host_id: *host_id,
                method: *method,
                evasion_successful,
                detection_time_ms,
                detection_reason,
            })
        } else {
            Err(SecurityChaosError::InvalidScenario)
        }
    }

    // ======================================================================
    // APT stage helpers
    // ======================================================================

    async fn execute_apt_stage(
        &self,
        stage: APTStage,
    ) -> Result<APTStageResult, SecurityChaosError> {
        log::info!(
            "Executing APT stage: {} ({:?})",
            stage.stage_name,
            stage.stage_type
        );

        let start_time = Instant::now();
        let mut rng = rand::thread_rng();

        let success_roll: f32 = rng.gen();
        let successful = success_roll < stage.success_probability;

        let detection_roll: f32 = rng.gen();
        let detected = detection_roll < stage.detection_probability;

        let base_time_ms = (stage.duration_minutes as u64) * 60 * 1000;
        let jitter = if base_time_ms > 0 {
            rng.gen_range(0..(base_time_ms / 4).max(1))
        } else {
            0
        };
        let actual_time_ms = base_time_ms.saturating_add(jitter);

        sleep(Duration::from_millis(cmp::min(actual_time_ms, 5000))).await;

        let execution_time = start_time.elapsed().as_millis() as u64;

        let detection_method = if detected {
            Some(self.get_detection_method_for_stage(&stage.stage_type))
        } else {
            None
        };

        let artifacts_left = if successful {
            self.get_artifacts_for_stage(&stage.stage_type)
        } else {
            Vec::new()
        };

        Ok(APTStageResult {
            stage_name: stage.stage_name,
            successful,
            detected,
            execution_time_ms: execution_time,
            detection_method,
            artifacts_left,
        })
    }

    fn get_detection_method_for_stage(&self, stage_type: &APTStageType) -> String {
        match stage_type {
            APTStageType::InitialAccess => "Network intrusion detection".to_string(),
            APTStageType::Persistence => "File integrity monitoring".to_string(),
            APTStageType::PrivilegeEscalation => "Behavioral analysis".to_string(),
            APTStageType::DefenseEvasion => "Anomaly detection".to_string(),
            APTStageType::CredentialAccess => "Authentication monitoring".to_string(),
            APTStageType::Discovery => "Process monitoring".to_string(),
            APTStageType::LateralMovement => "Network traffic analysis".to_string(),
            APTStageType::Collection => "Data access monitoring".to_string(),
            APTStageType::CommandAndControl => "Network communication analysis".to_string(),
            APTStageType::Exfiltration => "Data loss prevention".to_string(),
            APTStageType::Impact => "System integrity monitoring".to_string(),
        }
    }

    fn get_artifacts_for_stage(&self, stage_type: &APTStageType) -> Vec<String> {
        match stage_type {
            APTStageType::InitialAccess => vec!["Network connection logs".to_string()],
            APTStageType::Persistence => vec![
                "Registry modifications".to_string(),
                "Scheduled tasks".to_string(),
            ],
            APTStageType::PrivilegeEscalation => {
                vec!["Process memory modifications".to_string()]
            }
            APTStageType::DefenseEvasion => vec!["Log tampering attempts".to_string()],
            APTStageType::CredentialAccess => vec!["Authentication attempts".to_string()],
            APTStageType::Discovery => vec!["System enumeration commands".to_string()],
            APTStageType::LateralMovement => vec!["Network discovery scans".to_string()],
            APTStageType::Collection => vec!["File access patterns".to_string()],
            APTStageType::CommandAndControl => vec!["Network beacons".to_string()],
            APTStageType::Exfiltration => vec!["Data transfer logs".to_string()],
            APTStageType::Impact => vec!["System state changes".to_string()],
        }
    }

    // ======================================================================
    // Primitive simulations
    // ======================================================================

    fn generate_test_password(&self, attempt: u32) -> String {
        let weak_passwords = [
            "password",
            "123456",
            "admin",
            "user",
            "test",
            "qwerty",
            "password123",
            "admin123",
            "letmein",
            "welcome",
        ];

        let base_password = weak_passwords[attempt as usize % weak_passwords.len()];
        format!("{}{}", base_password, attempt)
    }

    async fn test_authentication(
        &self,
        _username: &str,
        password: &str,
    ) -> Result<(), AuthError> {
        sleep(Duration::from_millis(10)).await;

        if password.contains("admin") && password.len() > 10 {
            Err(AuthError::AccountLocked)
        } else if password.len() > 15 {
            Err(AuthError::RateLimited)
        } else {
            Err(AuthError::InvalidCredentials)
        }
    }

    async fn attempt_clearance_access(
        &self,
        source: ClearanceLevel,
        target: ClearanceLevel,
    ) -> Result<(), SecurityChaosError> {
        sleep(Duration::from_millis(100)).await;
        if target > source {
            Err(SecurityChaosError::AuthError(
                "Insufficient clearance".to_string(),
            ))
        } else {
            Ok(())
        }
    }

    async fn simulate_credential_theft(&self) -> Result<(), SecurityChaosError> {
        sleep(Duration::from_millis(200)).await;
        Err(SecurityChaosError::AuthError(
            "Credential theft blocked".to_string(),
        ))
    }

    async fn simulate_vulnerability_exploit(&self) -> Result<(), SecurityChaosError> {
        sleep(Duration::from_millis(300)).await;
        Err(SecurityChaosError::SecurityViolation(
            "Exploit attempt detected and blocked".to_string(),
        ))
    }

    async fn simulate_insider_threat(&self) -> Result<(), SecurityChaosError> {
        sleep(Duration::from_millis(150)).await;
        Err(SecurityChaosError::SecurityViolation(
            "Insider threat activity detected".to_string(),
        ))
    }

    async fn attempt_authorization_skip(
        &self,
        device_id: u32,
    ) -> Result<(), SecurityChaosError> {
        sleep(Duration::from_millis(50)).await;
        let quarantined = [0, 1, 12, 24, 83];
        if quarantined.contains(&device_id) {
            Err(SecurityChaosError::DeviceProtected(device_id))
        } else {
            Err(SecurityChaosError::AuthError(
                "Authorization required".to_string(),
            ))
        }
    }

    async fn attempt_token_manipulation(&self) -> Result<(), SecurityChaosError> {
        sleep(Duration::from_millis(80)).await;
        Err(SecurityChaosError::SecurityViolation(
            "Token manipulation detected".to_string(),
        ))
    }

    async fn attempt_session_hijack(&self) -> Result<(), SecurityChaosError> {
        sleep(Duration::from_millis(120)).await;
        Err(SecurityChaosError::SecurityViolation(
            "Session hijack attempt blocked".to_string(),
        ))
    }

    async fn attempt_privilege_bypass(&self) -> Result<(), SecurityChaosError> {
        sleep(Duration::from_millis(90)).await;
        Err(SecurityChaosError::SecurityViolation(
            "Privilege bypass blocked".to_string(),
        ))
    }

    async fn attempt_buffer_overflow(&self) -> Result<(), SecurityChaosError> {
        sleep(Duration::from_millis(70)).await;
        Err(SecurityChaosError::SecurityViolation(
            "Buffer overflow attempt detected".to_string(),
        ))
    }

    // Audit tampering sims

    async fn simulate_entry_deletion(&self, count: u32) -> bool {
        sleep(Duration::from_millis(count as u64 * 10)).await;
        true
    }

    async fn simulate_entry_modification(&self, count: u32) -> bool {
        sleep(Duration::from_millis(count as u64 * 15)).await;
        true
    }

    async fn simulate_timestamp_tampering(&self, count: u32) -> bool {
        sleep(Duration::from_millis(count as u64 * 5)).await;
        true
    }

    async fn simulate_hash_collision_attack(&self) -> bool {
        sleep(Duration::from_millis(500)).await;
        true
    }

    async fn simulate_chain_break_attack(&self) -> bool {
        sleep(Duration::from_millis(300)).await;
        true
    }

    // Hardware tampering sims

    async fn simulate_physical_tamper(&self, _device_id: u32) -> bool {
        sleep(Duration::from_millis(100)).await;
        true
    }

    async fn simulate_power_analysis(&self, _device_id: u32) -> bool {
        sleep(Duration::from_millis(200)).await;
        true
    }

    async fn simulate_em_fault_injection(&self, _device_id: u32) -> bool {
        sleep(Duration::from_millis(150)).await;
        true
    }

    async fn simulate_timing_attack(&self, _device_id: u32) -> bool {
        sleep(Duration::from_millis(300)).await;
        true
    }

    async fn simulate_voltage_glitch(&self, _device_id: u32) -> bool {
        sleep(Duration::from_millis(80)).await;
        true
    }

    // MFA bypass sims

    async fn simulate_mfa_bypass(
        &self,
        _user_id: &str,
        vector: &MfaBypassVector,
    ) -> Result<MfaBypassOutcome, SecurityChaosError> {
        match vector {
            MfaBypassVector::PushFatigue => {
                sleep(Duration::from_millis(200)).await;
                Ok(MfaBypassOutcome::Blocked)
            }
            MfaBypassVector::TotpCapture => {
                sleep(Duration::from_millis(150)).await;
                Ok(MfaBypassOutcome::Blocked)
            }
            MfaBypassVector::DeviceSwap => {
                sleep(Duration::from_millis(250)).await;
                Ok(MfaBypassOutcome::Blocked)
            }
            MfaBypassVector::SimSwap => {
                sleep(Duration::from_millis(300)).await;
                Ok(MfaBypassOutcome::Blocked)
            }
            MfaBypassVector::BackupCodeTheft => {
                sleep(Duration::from_millis(220)).await;
                Ok(MfaBypassOutcome::Blocked)
            }
        }
    }

    // Data exfil sims

    async fn simulate_dns_exfiltration(&self, data_mb: u32) -> (bool, String) {
        sleep(Duration::from_millis((data_mb as u64 * 10).min(1000))).await;
        (false, "DNS anomaly detection / DLP".to_string())
    }

    async fn simulate_https_exfiltration(&self, data_mb: u32) -> (bool, String) {
        sleep(Duration::from_millis((data_mb as u64 * 5).min(1000))).await;
        (false, "Proxy / UEBA / DLP".to_string())
    }

    async fn simulate_cloud_exfiltration(&self, data_mb: u32) -> (bool, String) {
        sleep(Duration::from_millis((data_mb as u64 * 8).min(1000))).await;
        (false, "CASB / cloud audit logs".to_string())
    }

    // EDR evasion sims

    async fn simulate_edr_evasion(
        &self,
        _host_id: u32,
        method: &EdrEvasionMethod,
        rng: &mut rand::rngs::ThreadRng,
    ) -> (bool, Option<String>) {
        sleep(Duration::from_millis(200)).await;

        let success_roll: f32 = rng.gen();
        let detect_roll: f32 = rng.gen();

        let (success_threshold, detect_threshold, reason) = match method {
            EdrEvasionMethod::KillAndDisable => (
                0.05f32,
                0.9f32,
                "EDR service tamper protection".to_string(),
            ),
            EdrEvasionMethod::UserModeInjection => (
                0.1f32,
                0.85f32,
                "Code injection detection / memory scanner".to_string(),
            ),
            EdrEvasionMethod::LolbinAbuse => (
                0.15f32,
                0.8f32,
                "Application control / behavior rules".to_string(),
            ),
            EdrEvasionMethod::DriverAbuse => (
                0.05f32,
                0.9f32,
                "Kernel driver signing / guard".to_string(),
            ),
        };

        let successful = success_roll < success_threshold;
        let detected = detect_roll < detect_threshold;

        if detected {
            (false, Some(reason))
        } else if successful {
            (true, None)
        } else {
            (false, None)
        }
    }
}

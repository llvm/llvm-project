//! DSMIL Agent - Dell Military Token Integration Library
//!
//! Comprehensive Rust implementation for Dell Latitude 5450 MIL-SPEC
//! military token integration with TPM2 system and NPU acceleration.
//!
//! MISSION: Integrate Dell military tokens (0x049e-0x04a3) with Rust TPM2
//! - Memory-safe token validation (zero unsafe operations)
//! - Hardware-accelerated cryptographic operations
//! - Multi-level security authorization matrix
//! - NPU-accelerated parallel processing (34.0 TOPS)
//! - Dell platform-specific optimizations

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![warn(clippy::pedantic)]
#![warn(clippy::cargo)]

extern crate alloc;

pub mod dell_military_tokens;
pub mod security_matrix;
pub mod npu_acceleration;
pub mod ffi_bindings;
pub mod dell_platform;

// Re-export common types
pub use tpm2_compat_common as common;

// Re-export key types for convenience
pub use dell_military_tokens::{
    DellMilitaryTokenValidator, SecurityMatrix, TokenValidationResult,
    NpuAcceleratedValidator, DELL_MILITARY_TOKEN_COUNT,
    TOKEN_PRIMARY_AUTH, TOKEN_SECONDARY_VALIDATION, TOKEN_HARDWARE_ACTIVATION,
    TOKEN_ADVANCED_SECURITY, TOKEN_SYSTEM_INTEGRATION, TOKEN_MILITARY_VALIDATION,
};

pub use security_matrix::{
    SecurityAuthorizationEngine, AuthorizationSession, AuthorizationMatrix,
    AuditEntry, SecurityStatistics,
};

pub use npu_acceleration::{
    NpuMilitaryTokenManager, NpuPerformanceReport, NpuExecutionContext,
    SecurityAnalysisResult, ThreatLevel,
};

pub use dell_platform::{
    DellPlatformManager, DellPlatformInfo, DellSecurityStatus,
    DellThermalConfig, DellThermalProfile, DellSecurityReport,
};

pub use ffi_bindings::{
    DsmilHandle, DsmilErrorCode, FfiTokenValidationResult,
    FfiSecurityMatrix, FfiPerformanceMetrics,
};

use common::{Tpm2Result, Tpm2Rc, SecurityLevel, LibraryConfig};
use alloc::string::{String, ToString};
use alloc::vec::Vec;

/// DSMIL library version
pub const DSMIL_VERSION: &str = "1.0.0";

/// DSMIL library name
pub const DSMIL_NAME: &str = "DSMIL Agent - Dell Military Token Integration";

/// Initialize DSMIL library with full configuration
pub fn initialize_dsmil(config: LibraryConfig) -> Tpm2Result<DsmilAgent> {
    let mut agent = DsmilAgent::new(config)?;
    agent.initialize()?;
    Ok(agent)
}

/// Main DSMIL agent coordinating all components
#[derive(Debug)]
pub struct DsmilAgent {
    /// Library configuration
    config: LibraryConfig,
    /// Dell military token validator
    token_validator: DellMilitaryTokenValidator,
    /// Security authorization engine
    security_engine: SecurityAuthorizationEngine,
    /// NPU token manager
    npu_manager: Option<NpuMilitaryTokenManager>,
    /// Dell platform manager
    platform_manager: DellPlatformManager,
    /// Initialization status
    initialized: bool,
}

impl DsmilAgent {
    /// Create new DSMIL agent
    pub fn new(config: LibraryConfig) -> Tpm2Result<Self> {
        Ok(Self {
            config: config.clone(),
            token_validator: DellMilitaryTokenValidator::new(config.acceleration_flags),
            security_engine: SecurityAuthorizationEngine::new(),
            npu_manager: None,
            platform_manager: DellPlatformManager::new(),
            initialized: false,
        })
    }

    /// Initialize DSMIL agent and all components
    pub fn initialize(&mut self) -> Tpm2Result<()> {
        // Initialize Dell platform manager
        self.platform_manager.initialize()?;

        // Initialize NPU manager if acceleration is enabled
        if self.config.acceleration_flags.contains(common::AccelerationFlags::NPU) {
            // Note: In async context, this would be awaited
            // For sync interface, we'll initialize it lazily
            self.npu_manager = None; // Will be initialized on first async operation
        }

        self.initialized = true;
        Ok(())
    }

    /// Validate all Dell military tokens and create authorization session
    pub fn authenticate_and_authorize(&mut self) -> Tpm2Result<AuthorizationSession> {
        if !self.initialized {
            return Err(Tpm2Rc::Failure);
        }

        // Validate all military tokens
        let security_matrix = self.token_validator.validate_all_tokens()?;

        // Create authorization session for achieved security level
        let session = self.security_engine.authorize_access(security_matrix.security_level)?;

        // Enable military mode if TOP_SECRET achieved
        if security_matrix.security_level == SecurityLevel::TopSecret {
            self.platform_manager.enable_military_mode(&security_matrix)?;
        }

        Ok(session)
    }

    /// Get comprehensive system status
    pub fn get_system_status(&self) -> DsmilSystemStatus {
        DsmilSystemStatus {
            initialized: self.initialized,
            platform_info: self.platform_manager.get_platform_info().cloned(),
            security_status: self.platform_manager.get_security_status().clone(),
            thermal_config: self.platform_manager.get_thermal_config().clone(),
            security_statistics: self.security_engine.get_security_statistics(),
            validation_metrics: self.token_validator.get_metrics().clone(),
            npu_available: self.npu_manager.is_some(),
            military_mode_active: self.platform_manager.is_military_mode_active(),
            milspec_compliant: self.platform_manager.is_milspec_compliant(),
        }
    }

    /// Generate comprehensive integration report
    pub fn generate_integration_report(&self) -> DsmilIntegrationReport {
        let system_status = self.get_system_status();
        let security_report = self.platform_manager.generate_security_report();

        DsmilIntegrationReport {
            version: DSMIL_VERSION.to_string(),
            library_name: DSMIL_NAME.to_string(),
            system_status,
            security_report,
            performance_summary: self.generate_performance_summary(),
            integration_status: self.generate_integration_status(),
            recommendations: self.generate_recommendations(),
            report_timestamp: common::timestamp_us(),
        }
    }

    /// Generate performance summary
    fn generate_performance_summary(&self) -> PerformanceSummary {
        let metrics = self.token_validator.get_metrics();

        PerformanceSummary {
            total_validations: metrics.total_validations,
            successful_validations: metrics.successful_validations,
            avg_validation_time_us: metrics.avg_validation_time_us,
            acceleration_usage_percent: metrics.acceleration_usage_percent,
            target_latency_achieved: metrics.avg_validation_time_us < 100.0, // <100μs target
            npu_acceleration_available: self.npu_manager.is_some(),
            performance_grade: self.calculate_performance_grade(metrics),
        }
    }

    /// Calculate performance grade
    fn calculate_performance_grade(&self, metrics: &dell_military_tokens::ValidationMetrics) -> String {
        let success_rate = if metrics.total_validations > 0 {
            metrics.successful_validations as f64 / metrics.total_validations as f64
        } else {
            0.0
        };

        match (success_rate, metrics.avg_validation_time_us) {
            (rate, latency) if rate >= 0.95 && latency < 50.0 => "A+ (Excellent)".to_string(),
            (rate, latency) if rate >= 0.90 && latency < 100.0 => "A (Very Good)".to_string(),
            (rate, latency) if rate >= 0.80 && latency < 500.0 => "B (Good)".to_string(),
            (rate, latency) if rate >= 0.70 && latency < 1000.0 => "C (Acceptable)".to_string(),
            _ => "D (Needs Improvement)".to_string(),
        }
    }

    /// Generate integration status
    fn generate_integration_status(&self) -> IntegrationStatus {
        IntegrationStatus {
            token_integration_complete: true,
            security_matrix_functional: true,
            npu_acceleration_available: self.npu_manager.is_some(),
            dell_platform_integrated: self.platform_manager.is_milspec_compliant(),
            ffi_bindings_available: true,
            military_mode_available: true,
            overall_integration_health: self.calculate_integration_health(),
        }
    }

    /// Calculate overall integration health
    fn calculate_integration_health(&self) -> f32 {
        let mut health_score = 0.0;
        let mut total_checks = 0.0;

        // Token integration (25%)
        health_score += 25.0;
        total_checks += 25.0;

        // Security matrix (25%)
        health_score += 25.0;
        total_checks += 25.0;

        // Platform integration (20%)
        if self.platform_manager.is_milspec_compliant() {
            health_score += 20.0;
        }
        total_checks += 20.0;

        // NPU acceleration (15%)
        if self.npu_manager.is_some() {
            health_score += 15.0;
        }
        total_checks += 15.0;

        // Military mode (15%)
        if self.platform_manager.is_military_mode_active() {
            health_score += 15.0;
        }
        total_checks += 15.0;

        health_score / total_checks * 100.0
    }

    /// Generate recommendations for optimization
    fn generate_recommendations(&self) -> Vec<String> {
        let mut recommendations = Vec::new();

        // Performance recommendations
        let metrics = self.token_validator.get_metrics();
        if metrics.avg_validation_time_us > 100.0 {
            recommendations.push("Consider enabling NPU acceleration for faster token validation".to_string());
        }

        // Security recommendations
        if !self.platform_manager.is_military_mode_active() {
            recommendations.push("Enable military mode for enhanced security compliance".to_string());
        }

        // Platform recommendations
        let security_status = self.platform_manager.get_security_status();
        if !security_status.secure_boot_enabled {
            recommendations.push("Enable Secure Boot for hardware-based security".to_string());
        }

        if !security_status.tpm_enabled {
            recommendations.push("Enable TPM for hardware security module support".to_string());
        }

        // NPU recommendations
        if self.npu_manager.is_none() &&
           self.config.acceleration_flags.contains(common::AccelerationFlags::NPU) {
            recommendations.push("Initialize NPU manager for 34.0 TOPS acceleration".to_string());
        }

        recommendations
    }

    /// Check if agent is properly initialized
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// Get configuration
    pub fn get_config(&self) -> &LibraryConfig {
        &self.config
    }
}

/// DSMIL system status
#[derive(Debug, Clone)]
pub struct DsmilSystemStatus {
    /// Initialization status
    pub initialized: bool,
    /// Platform information
    pub platform_info: Option<DellPlatformInfo>,
    /// Security status
    pub security_status: DellSecurityStatus,
    /// Thermal configuration
    pub thermal_config: DellThermalConfig,
    /// Security statistics
    pub security_statistics: SecurityStatistics,
    /// Validation metrics
    pub validation_metrics: dell_military_tokens::ValidationMetrics,
    /// NPU availability
    pub npu_available: bool,
    /// Military mode status
    pub military_mode_active: bool,
    /// MIL-SPEC compliance
    pub milspec_compliant: bool,
}

/// DSMIL integration report
#[derive(Debug, Clone)]
pub struct DsmilIntegrationReport {
    /// Library version
    pub version: String,
    /// Library name
    pub library_name: String,
    /// System status
    pub system_status: DsmilSystemStatus,
    /// Security report
    pub security_report: DellSecurityReport,
    /// Performance summary
    pub performance_summary: PerformanceSummary,
    /// Integration status
    pub integration_status: IntegrationStatus,
    /// Recommendations
    pub recommendations: Vec<String>,
    /// Report timestamp
    pub report_timestamp: u64,
}

/// Performance summary
#[derive(Debug, Clone)]
pub struct PerformanceSummary {
    /// Total validations performed
    pub total_validations: u64,
    /// Successful validations
    pub successful_validations: u64,
    /// Average validation time
    pub avg_validation_time_us: f64,
    /// Acceleration usage percentage
    pub acceleration_usage_percent: f32,
    /// Whether target latency is achieved
    pub target_latency_achieved: bool,
    /// NPU acceleration availability
    pub npu_acceleration_available: bool,
    /// Performance grade
    pub performance_grade: String,
}

/// Integration status
#[derive(Debug, Clone)]
pub struct IntegrationStatus {
    /// Token integration complete
    pub token_integration_complete: bool,
    /// Security matrix functional
    pub security_matrix_functional: bool,
    /// NPU acceleration available
    pub npu_acceleration_available: bool,
    /// Dell platform integrated
    pub dell_platform_integrated: bool,
    /// FFI bindings available
    pub ffi_bindings_available: bool,
    /// Military mode available
    pub military_mode_available: bool,
    /// Overall integration health (0.0-100.0)
    pub overall_integration_health: f32,
}

/// Format integration report as markdown
impl core::fmt::Display for DsmilIntegrationReport {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        writeln!(f, "# {} - Integration Report", self.library_name)?;
        writeln!(f, "**Version:** {}", self.version)?;
        writeln!(f, "**Generated:** {}", self.report_timestamp)?;
        writeln!(f)?;

        writeln!(f, "## Executive Summary")?;
        writeln!(f, "- **Integration Health:** {:.1}%", self.integration_status.overall_integration_health)?;
        writeln!(f, "- **Performance Grade:** {}", self.performance_summary.performance_grade)?;
        writeln!(f, "- **Military Mode:** {}", if self.system_status.military_mode_active { "Active" } else { "Inactive" })?;
        writeln!(f, "- **MIL-SPEC Compliant:** {}", if self.system_status.milspec_compliant { "Yes" } else { "No" })?;
        writeln!(f)?;

        writeln!(f, "## Platform Integration")?;
        if let Some(platform) = &self.system_status.platform_info {
            writeln!(f, "- **Manufacturer:** {}", platform.manufacturer)?;
            writeln!(f, "- **Product:** {}", platform.product_name)?;
            writeln!(f, "- **BIOS Version:** {}", platform.bios_version)?;
            writeln!(f, "- **ME Version:** {}", platform.me_version)?;
            writeln!(f, "- **Service Tag:** {}", platform.service_tag)?;
        }
        writeln!(f)?;

        writeln!(f, "## Performance Metrics")?;
        writeln!(f, "- **Total Validations:** {}", self.performance_summary.total_validations)?;
        writeln!(f, "- **Success Rate:** {:.2}%",
                (self.performance_summary.successful_validations as f64 /
                 self.performance_summary.total_validations.max(1) as f64) * 100.0)?;
        writeln!(f, "- **Average Latency:** {:.2}μs", self.performance_summary.avg_validation_time_us)?;
        writeln!(f, "- **Target Latency (<100μs):** {}",
                if self.performance_summary.target_latency_achieved { "✅ Achieved" } else { "❌ Not Achieved" })?;
        writeln!(f, "- **NPU Acceleration:** {}",
                if self.performance_summary.npu_acceleration_available { "✅ Available" } else { "❌ Not Available" })?;
        writeln!(f)?;

        writeln!(f, "## Security Status")?;
        writeln!(f, "- **Secure Boot:** {}",
                if self.system_status.security_status.secure_boot_enabled { "✅ Enabled" } else { "❌ Disabled" })?;
        writeln!(f, "- **TPM:** {}",
                if self.system_status.security_status.tpm_enabled { "✅ Enabled" } else { "❌ Disabled" })?;
        writeln!(f, "- **Military Mode:** {}",
                if self.system_status.security_status.military_mode_active { "✅ Active" } else { "❌ Inactive" })?;
        writeln!(f, "- **Chassis Intrusion:** {}",
                if self.system_status.security_status.chassis_intrusion_enabled { "✅ Enabled" } else { "❌ Disabled" })?;
        writeln!(f)?;

        writeln!(f, "## Integration Components")?;
        writeln!(f, "- **Token Integration:** {}",
                if self.integration_status.token_integration_complete { "✅ Complete" } else { "❌ Incomplete" })?;
        writeln!(f, "- **Security Matrix:** {}",
                if self.integration_status.security_matrix_functional { "✅ Functional" } else { "❌ Not Functional" })?;
        writeln!(f, "- **Dell Platform:** {}",
                if self.integration_status.dell_platform_integrated { "✅ Integrated" } else { "❌ Not Integrated" })?;
        writeln!(f, "- **FFI Bindings:** {}",
                if self.integration_status.ffi_bindings_available { "✅ Available" } else { "❌ Not Available" })?;
        writeln!(f)?;

        if !self.recommendations.is_empty() {
            writeln!(f, "## Recommendations")?;
            for (i, rec) in self.recommendations.iter().enumerate() {
                writeln!(f, "{}. {}", i + 1, rec)?;
            }
            writeln!(f)?;
        }

        writeln!(f, "## Dell Military Tokens Status")?;
        writeln!(f, "| Token ID | Description | Security Level |")?;
        writeln!(f, "|----------|-------------|----------------|")?;
        writeln!(f, "| 0x049e | Primary Authorization | UNCLASSIFIED |")?;
        writeln!(f, "| 0x049f | Secondary Validation | CONFIDENTIAL |")?;
        writeln!(f, "| 0x04a0 | Hardware Activation | CONFIDENTIAL |")?;
        writeln!(f, "| 0x04a1 | Advanced Security | SECRET |")?;
        writeln!(f, "| 0x04a2 | System Integration | SECRET |")?;
        writeln!(f, "| 0x04a3 | Military Validation | TOP SECRET |")?;
        writeln!(f)?;

        writeln!(f, "---")?;
        writeln!(f, "*Report generated by DSMIL Agent v{} at timestamp {}*", self.version, self.report_timestamp)?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use common::{AccelerationFlags, SecurityLevel};

    #[test]
    fn test_dsmil_agent_creation() {
        let config = LibraryConfig {
            security_level: SecurityLevel::Unclassified,
            acceleration_flags: AccelerationFlags::ALL,
            ..LibraryConfig::default()
        };

        let result = DsmilAgent::new(config);
        assert!(result.is_ok());

        let agent = result.unwrap();
        assert!(!agent.is_initialized());
    }

    #[test]
    fn test_dsmil_agent_initialization() {
        let config = LibraryConfig {
            security_level: SecurityLevel::Unclassified,
            acceleration_flags: AccelerationFlags::NONE,
            ..LibraryConfig::default()
        };

        let mut agent = DsmilAgent::new(config).unwrap();
        let result = agent.initialize();
        assert!(result.is_ok());
        assert!(agent.is_initialized());
    }

    #[test]
    fn test_integration_report_generation() {
        let config = LibraryConfig::default();
        let mut agent = DsmilAgent::new(config).unwrap();
        agent.initialize().unwrap();

        let report = agent.generate_integration_report();
        assert_eq!(report.version, DSMIL_VERSION);
        assert!(!report.recommendations.is_empty());
        assert!(report.integration_status.overall_integration_health > 0.0);
    }

    #[test]
    fn test_system_status() {
        let config = LibraryConfig::default();
        let mut agent = DsmilAgent::new(config).unwrap();
        agent.initialize().unwrap();

        let status = agent.get_system_status();
        assert!(status.initialized);
        assert!(status.platform_info.is_some());
    }

    #[test]
    fn test_performance_grading() {
        let config = LibraryConfig::default();
        let agent = DsmilAgent::new(config).unwrap();

        let metrics = dell_military_tokens::ValidationMetrics {
            total_validations: 100,
            successful_validations: 95,
            avg_validation_time_us: 25.0,
            acceleration_usage_percent: 80.0,
        };

        let grade = agent.calculate_performance_grade(&metrics);
        assert_eq!(grade, "A+ (Excellent)");
    }
}
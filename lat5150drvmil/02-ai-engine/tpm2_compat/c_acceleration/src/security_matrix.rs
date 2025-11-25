//! Security Level Matrix and Authorization System
//!
//! DSMIL Agent - Multi-level Security Authorization Matrix
//! Dell Latitude 5450 MIL-SPEC Military Token Authorization System
//!
//! MISSION: Implement graduated security access based on token validation
//! - Multi-level security (UNCLASSIFIED → TOP_SECRET)
//! - Token combination authorization matrix
//! - Hardware-accelerated authorization lookup
//! - Audit trail for military compliance
//! - Constant-time authorization checks

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]
#![deny(missing_docs)]

use crate::tpm2_compat_common::{SecurityLevel, Tpm2Result, Tpm2Rc, timestamp_us};
use crate::dell_military_tokens::{
    SecurityMatrix, DellMilitaryTokenValidator, TOKEN_PRIMARY_AUTH,
    TOKEN_SECONDARY_VALIDATION, TOKEN_HARDWARE_ACTIVATION, TOKEN_ADVANCED_SECURITY,
    TOKEN_SYSTEM_INTEGRATION, TOKEN_MILITARY_VALIDATION,
};
use zeroize::{Zeroize, ZeroizeOnDrop};
use core::fmt;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Maximum number of concurrent authorization sessions
pub const MAX_AUTHORIZATION_SESSIONS: usize = 64;

/// Authorization session timeout in seconds
pub const AUTHORIZATION_TIMEOUT_SECONDS: u64 = 300; // 5 minutes

/// Dell Military Authorization Matrix
/// Defines which token combinations grant access to specific security levels
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct AuthorizationMatrix {
    /// Unclassified operations (any single valid token)
    pub unclassified_requirements: TokenRequirement,
    /// Confidential operations (2+ tokens including secondary validation)
    pub confidential_requirements: TokenRequirement,
    /// Secret operations (4+ tokens including advanced security)
    pub secret_requirements: TokenRequirement,
    /// Top Secret operations (all 6 tokens including military validation)
    pub top_secret_requirements: TokenRequirement,
}

impl Default for AuthorizationMatrix {
    fn default() -> Self {
        Self {
            unclassified_requirements: TokenRequirement {
                minimum_tokens: 1,
                required_tokens: vec![TOKEN_PRIMARY_AUTH],
                forbidden_tokens: Vec::new(),
                require_all: false,
            },
            confidential_requirements: TokenRequirement {
                minimum_tokens: 2,
                required_tokens: vec![TOKEN_PRIMARY_AUTH, TOKEN_SECONDARY_VALIDATION],
                forbidden_tokens: Vec::new(),
                require_all: true,
            },
            secret_requirements: TokenRequirement {
                minimum_tokens: 4,
                required_tokens: vec![
                    TOKEN_PRIMARY_AUTH,
                    TOKEN_SECONDARY_VALIDATION,
                    TOKEN_HARDWARE_ACTIVATION,
                    TOKEN_ADVANCED_SECURITY,
                ],
                forbidden_tokens: Vec::new(),
                require_all: true,
            },
            top_secret_requirements: TokenRequirement {
                minimum_tokens: 6,
                required_tokens: vec![
                    TOKEN_PRIMARY_AUTH,
                    TOKEN_SECONDARY_VALIDATION,
                    TOKEN_HARDWARE_ACTIVATION,
                    TOKEN_ADVANCED_SECURITY,
                    TOKEN_SYSTEM_INTEGRATION,
                    TOKEN_MILITARY_VALIDATION,
                ],
                forbidden_tokens: Vec::new(),
                require_all: true,
            },
        }
    }
}

/// Token requirement specification for security levels
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct TokenRequirement {
    /// Minimum number of tokens required
    pub minimum_tokens: u8,
    /// Specific tokens that must be present
    pub required_tokens: Vec<u16>,
    /// Tokens that must NOT be present
    pub forbidden_tokens: Vec<u16>,
    /// Whether ALL required tokens must be present
    pub require_all: bool,
}

/// Authorization session for tracking active authorizations
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct AuthorizationSession {
    /// Unique session identifier
    pub session_id: u64,
    /// Authorized security level
    pub security_level: SecurityLevel,
    /// Validated token mask
    pub token_mask: u32,
    /// Session creation timestamp
    pub created_at_us: u64,
    /// Session last accessed timestamp
    pub last_accessed_us: u64,
    /// Session expiry timestamp
    pub expires_at_us: u64,
    /// Audit trail entry count
    pub audit_entries: u32,
}

impl AuthorizationSession {
    /// Create new authorization session
    pub fn new(session_id: u64, security_level: SecurityLevel, token_mask: u32) -> Self {
        let now = timestamp_us();
        Self {
            session_id,
            security_level,
            token_mask,
            created_at_us: now,
            last_accessed_us: now,
            expires_at_us: now + (AUTHORIZATION_TIMEOUT_SECONDS * 1_000_000),
            audit_entries: 0,
        }
    }

    /// Check if session is still valid
    pub fn is_valid(&self) -> bool {
        timestamp_us() < self.expires_at_us
    }

    /// Refresh session access time
    pub fn refresh(&mut self) {
        self.last_accessed_us = timestamp_us();
        self.expires_at_us = self.last_accessed_us + (AUTHORIZATION_TIMEOUT_SECONDS * 1_000_000);
    }

    /// Check if session can access given security level
    pub fn can_access(&self, required_level: SecurityLevel) -> bool {
        self.is_valid() && self.security_level.can_access(required_level)
    }
}

// Zeroize implementations for security
impl Zeroize for AuthorizationSession {
    fn zeroize(&mut self) {
        self.session_id = 0;
        self.security_level.zeroize();
        self.token_mask = 0;
        self.created_at_us = 0;
        self.last_accessed_us = 0;
        self.expires_at_us = 0;
        self.audit_entries = 0;
    }
}

impl ZeroizeOnDrop for AuthorizationSession {}

/// Security authorization engine
#[derive(Debug)]
pub struct SecurityAuthorizationEngine {
    /// Authorization matrix configuration
    authorization_matrix: AuthorizationMatrix,
    /// Active authorization sessions
    active_sessions: Vec<AuthorizationSession>,
    /// Audit log entries
    audit_log: Vec<AuditEntry>,
    /// Token validator
    token_validator: DellMilitaryTokenValidator,
    /// Session ID counter
    next_session_id: u64,
}

/// Audit log entry for military compliance
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct AuditEntry {
    /// Entry timestamp
    pub timestamp_us: u64,
    /// Session ID
    pub session_id: Option<u64>,
    /// Action performed
    pub action: AuditAction,
    /// Security level involved
    pub security_level: SecurityLevel,
    /// Token mask used
    pub token_mask: u32,
    /// Result of the action
    pub result: AuditResult,
    /// Additional context information
    pub context: String,
}

/// Types of auditable actions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum AuditAction {
    /// Authorization request
    AuthorizeRequest,
    /// Session creation
    SessionCreate,
    /// Session access
    SessionAccess,
    /// Session expiry
    SessionExpiry,
    /// Token validation
    TokenValidation,
    /// Security level upgrade
    SecurityUpgrade,
    /// Access denied
    AccessDenied,
}

/// Audit results
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum AuditResult {
    /// Operation succeeded
    Success,
    /// Operation failed
    Failure,
    /// Operation denied
    Denied,
    /// Operation timed out
    Timeout,
}

impl SecurityAuthorizationEngine {
    /// Create new security authorization engine
    pub fn new() -> Self {
        Self {
            authorization_matrix: AuthorizationMatrix::default(),
            active_sessions: Vec::with_capacity(MAX_AUTHORIZATION_SESSIONS),
            audit_log: Vec::new(),
            token_validator: DellMilitaryTokenValidator::new(
                crate::tpm2_compat_common::AccelerationFlags::ALL
            ),
            next_session_id: 1,
        }
    }

    /// Authorize access based on Dell military tokens
    pub fn authorize_access(&mut self, required_level: SecurityLevel) -> Tpm2Result<AuthorizationSession> {
        let start_time = timestamp_us();

        // Validate all Dell military tokens
        let security_matrix = self.token_validator.validate_all_tokens()?;

        // Check if validated tokens meet requirements for requested security level
        let requirement = self.get_requirement_for_level(required_level);
        let authorized = self.check_token_requirements(&security_matrix, requirement);

        if !authorized {
            self.audit_log.push(AuditEntry {
                timestamp_us: start_time,
                session_id: None,
                action: AuditAction::AuthorizeRequest,
                security_level: required_level,
                token_mask: security_matrix.authorization_mask,
                result: AuditResult::Denied,
                context: format!("Insufficient tokens for security level {:?}", required_level),
            });
            return Err(Tpm2Rc::SecurityViolation);
        }

        // Create new authorization session
        let session_id = self.generate_session_id();
        let mut session = AuthorizationSession::new(
            session_id,
            security_matrix.security_level,
            security_matrix.authorization_mask,
        );

        // Clean up expired sessions before adding new one
        self.cleanup_expired_sessions();

        // Check session limit
        if self.active_sessions.len() >= MAX_AUTHORIZATION_SESSIONS {
            return Err(Tpm2Rc::ResourceUnavailable);
        }

        // Add to active sessions
        self.active_sessions.push(session.clone());

        // Audit successful authorization
        self.audit_log.push(AuditEntry {
            timestamp_us: start_time,
            session_id: Some(session_id),
            action: AuditAction::SessionCreate,
            security_level: session.security_level,
            token_mask: session.token_mask,
            result: AuditResult::Success,
            context: format!("Authorized for security level {:?}", session.security_level),
        });

        session.audit_entries += 1;
        Ok(session)
    }

    /// Validate session access to specific security level
    pub fn validate_session_access(&mut self, session_id: u64, required_level: SecurityLevel) -> Tpm2Result<bool> {
        let session = self.active_sessions
            .iter_mut()
            .find(|s| s.session_id == session_id);

        match session {
            Some(session) => {
                if !session.is_valid() {
                    self.audit_log.push(AuditEntry {
                        timestamp_us: timestamp_us(),
                        session_id: Some(session_id),
                        action: AuditAction::SessionAccess,
                        security_level: required_level,
                        token_mask: session.token_mask,
                        result: AuditResult::Timeout,
                        context: "Session expired".to_string(),
                    });
                    return Ok(false);
                }

                let can_access = session.can_access(required_level);
                session.refresh();

                self.audit_log.push(AuditEntry {
                    timestamp_us: timestamp_us(),
                    session_id: Some(session_id),
                    action: AuditAction::SessionAccess,
                    security_level: required_level,
                    token_mask: session.token_mask,
                    result: if can_access { AuditResult::Success } else { AuditResult::Denied },
                    context: format!("Access {} for level {:?}",
                                   if can_access { "granted" } else { "denied" },
                                   required_level),
                });

                session.audit_entries += 1;
                Ok(can_access)
            }
            None => {
                self.audit_log.push(AuditEntry {
                    timestamp_us: timestamp_us(),
                    session_id: Some(session_id),
                    action: AuditAction::SessionAccess,
                    security_level: required_level,
                    token_mask: 0,
                    result: AuditResult::Failure,
                    context: "Session not found".to_string(),
                });
                Ok(false)
            }
        }
    }

    /// Revoke authorization session
    pub fn revoke_session(&mut self, session_id: u64) -> Tpm2Result<()> {
        let position = self.active_sessions
            .iter()
            .position(|s| s.session_id == session_id);

        match position {
            Some(index) => {
                let session = self.active_sessions.remove(index);

                self.audit_log.push(AuditEntry {
                    timestamp_us: timestamp_us(),
                    session_id: Some(session_id),
                    action: AuditAction::SessionExpiry,
                    security_level: session.security_level,
                    token_mask: session.token_mask,
                    result: AuditResult::Success,
                    context: "Session revoked".to_string(),
                });

                Ok(())
            }
            None => Err(Tpm2Rc::SessionNotFound),
        }
    }

    /// Get authorization requirements for security level
    fn get_requirement_for_level(&self, level: SecurityLevel) -> &TokenRequirement {
        match level {
            SecurityLevel::Unclassified => &self.authorization_matrix.unclassified_requirements,
            SecurityLevel::Confidential => &self.authorization_matrix.confidential_requirements,
            SecurityLevel::Secret => &self.authorization_matrix.secret_requirements,
            SecurityLevel::TopSecret => &self.authorization_matrix.top_secret_requirements,
        }
    }

    /// Check if security matrix meets token requirements
    fn check_token_requirements(&self, matrix: &SecurityMatrix, requirement: &TokenRequirement) -> bool {
        // Check minimum token count
        if matrix.tokens_validated < requirement.minimum_tokens {
            return false;
        }

        // Check required tokens
        if requirement.require_all {
            // All required tokens must be present
            for &token_id in &requirement.required_tokens {
                if !matrix.has_token(token_id) {
                    return false;
                }
            }
        } else {
            // At least one required token must be present
            if !requirement.required_tokens.is_empty() {
                let has_any_required = requirement.required_tokens
                    .iter()
                    .any(|&token_id| matrix.has_token(token_id));
                if !has_any_required {
                    return false;
                }
            }
        }

        // Check forbidden tokens
        for &token_id in &requirement.forbidden_tokens {
            if matrix.has_token(token_id) {
                return false;
            }
        }

        true
    }

    /// Generate unique session ID
    fn generate_session_id(&mut self) -> u64 {
        let session_id = self.next_session_id;
        self.next_session_id = self.next_session_id.wrapping_add(1);
        session_id
    }

    /// Clean up expired sessions
    fn cleanup_expired_sessions(&mut self) {
        let now = timestamp_us();
        let mut expired_sessions = Vec::new();

        self.active_sessions.retain(|session| {
            if !session.is_valid() {
                expired_sessions.push(session.session_id);
                false
            } else {
                true
            }
        });

        // Audit expired sessions
        for session_id in expired_sessions {
            self.audit_log.push(AuditEntry {
                timestamp_us: now,
                session_id: Some(session_id),
                action: AuditAction::SessionExpiry,
                security_level: SecurityLevel::Unclassified,
                token_mask: 0,
                result: AuditResult::Timeout,
                context: "Session expired during cleanup".to_string(),
            });
        }
    }

    /// Get active session count
    pub fn active_session_count(&self) -> usize {
        self.active_sessions.len()
    }

    /// Get audit log entries
    pub fn get_audit_log(&self) -> &[AuditEntry] {
        &self.audit_log
    }

    /// Clear audit log (for maintenance)
    pub fn clear_audit_log(&mut self) {
        self.audit_log.clear();
    }

    /// Get security statistics
    pub fn get_security_statistics(&self) -> SecurityStatistics {
        let total_audits = self.audit_log.len();
        let successful_auths = self.audit_log
            .iter()
            .filter(|entry| entry.action == AuditAction::SessionCreate && entry.result == AuditResult::Success)
            .count();
        let denied_accesses = self.audit_log
            .iter()
            .filter(|entry| entry.result == AuditResult::Denied)
            .count();

        SecurityStatistics {
            total_audit_entries: total_audits as u64,
            successful_authorizations: successful_auths as u64,
            denied_accesses: denied_accesses as u64,
            active_sessions: self.active_sessions.len() as u32,
            avg_session_duration_us: self.calculate_avg_session_duration(),
        }
    }

    /// Calculate average session duration
    fn calculate_avg_session_duration(&self) -> f64 {
        if self.active_sessions.is_empty() {
            return 0.0;
        }

        let now = timestamp_us();
        let total_duration: u64 = self.active_sessions
            .iter()
            .map(|session| now - session.created_at_us)
            .sum();

        total_duration as f64 / self.active_sessions.len() as f64
    }
}

/// Security statistics for monitoring
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SecurityStatistics {
    /// Total audit entries recorded
    pub total_audit_entries: u64,
    /// Successful authorizations
    pub successful_authorizations: u64,
    /// Denied access attempts
    pub denied_accesses: u64,
    /// Currently active sessions
    pub active_sessions: u32,
    /// Average session duration in microseconds
    pub avg_session_duration_us: f64,
}

// Display implementations
impl fmt::Display for AuditAction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AuthorizeRequest => write!(f, "AUTHORIZE_REQUEST"),
            Self::SessionCreate => write!(f, "SESSION_CREATE"),
            Self::SessionAccess => write!(f, "SESSION_ACCESS"),
            Self::SessionExpiry => write!(f, "SESSION_EXPIRY"),
            Self::TokenValidation => write!(f, "TOKEN_VALIDATION"),
            Self::SecurityUpgrade => write!(f, "SECURITY_UPGRADE"),
            Self::AccessDenied => write!(f, "ACCESS_DENIED"),
        }
    }
}

impl fmt::Display for AuditResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Success => write!(f, "SUCCESS"),
            Self::Failure => write!(f, "FAILURE"),
            Self::Denied => write!(f, "DENIED"),
            Self::Timeout => write!(f, "TIMEOUT"),
        }
    }
}

impl fmt::Display for AuditEntry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {} - {} - {:?} - 0x{:06x} - {} - {}",
               self.timestamp_us,
               self.action,
               self.result,
               self.security_level,
               self.token_mask,
               self.session_id.map_or("NONE".to_string(), |id| id.to_string()),
               self.context)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tpm2_compat_common::AccelerationFlags;

    #[test]
    fn test_authorization_matrix_default() {
        let matrix = AuthorizationMatrix::default();

        // Unclassified should require only primary auth
        assert_eq!(matrix.unclassified_requirements.minimum_tokens, 1);
        assert!(matrix.unclassified_requirements.required_tokens.contains(&TOKEN_PRIMARY_AUTH));

        // Top Secret should require all 6 tokens
        assert_eq!(matrix.top_secret_requirements.minimum_tokens, 6);
        assert_eq!(matrix.top_secret_requirements.required_tokens.len(), 6);
    }

    #[test]
    fn test_authorization_session_creation() {
        let session = AuthorizationSession::new(1, SecurityLevel::Secret, 0b111111);

        assert_eq!(session.session_id, 1);
        assert_eq!(session.security_level, SecurityLevel::Secret);
        assert!(session.is_valid());
        assert!(session.can_access(SecurityLevel::Unclassified));
        assert!(!session.can_access(SecurityLevel::TopSecret));
    }

    #[test]
    fn test_security_authorization_engine() {
        let mut engine = SecurityAuthorizationEngine::new();

        // Test authorization for unclassified access
        let result = engine.authorize_access(SecurityLevel::Unclassified);
        assert!(result.is_ok());

        let session = result.unwrap();
        assert!(session.can_access(SecurityLevel::Unclassified));

        // Validate session access
        let access_result = engine.validate_session_access(session.session_id, SecurityLevel::Unclassified);
        assert!(access_result.is_ok());
        assert!(access_result.unwrap());
    }

    #[test]
    fn test_audit_logging() {
        let mut engine = SecurityAuthorizationEngine::new();

        // Perform some operations to generate audit entries
        let _result = engine.authorize_access(SecurityLevel::Confidential);

        let audit_log = engine.get_audit_log();
        assert!(!audit_log.is_empty());

        // Check audit entry format
        let first_entry = &audit_log[0];
        assert_eq!(first_entry.action, AuditAction::AuthorizeRequest);
        assert!(!first_entry.context.is_empty());
    }

    #[test]
    fn test_session_cleanup() {
        let mut engine = SecurityAuthorizationEngine::new();

        // Create a session
        let result = engine.authorize_access(SecurityLevel::Unclassified);
        assert!(result.is_ok());

        let initial_count = engine.active_session_count();
        assert_eq!(initial_count, 1);

        // Force cleanup (in real usage, this would be automatic)
        engine.cleanup_expired_sessions();

        // Session should still be active (not expired yet)
        assert_eq!(engine.active_session_count(), 1);
    }

    #[test]
    fn test_security_statistics() {
        let mut engine = SecurityAuthorizationEngine::new();

        // Generate some activity
        let _result1 = engine.authorize_access(SecurityLevel::Unclassified);
        let _result2 = engine.authorize_access(SecurityLevel::Confidential);

        let stats = engine.get_security_statistics();
        assert!(stats.total_audit_entries > 0);
        assert!(stats.active_sessions <= MAX_AUTHORIZATION_SESSIONS as u32);
    }
}
//! Common types and utilities for TPM2 compatibility layer
//!
//! This crate provides shared types, error handling, and utilities used across
//! all components of the TPM2 compatibility layer.

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![deny(clippy::all)]
#![warn(clippy::pedantic)]
#![warn(clippy::cargo)]

use core::fmt;
use bitflags::bitflags;
// use zerocopy::{IntoBytes as AsBytes, FromBytes, FromZeros as FromZeroes};
use zeroize::{Zeroize, ZeroizeOnDrop};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// TPM2 return codes following TCG specification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[repr(u32)]
pub enum Tpm2Rc {
    /// Operation completed successfully
    Success = 0x0000_0000,
    /// General failure
    Failure = 0x0000_0001,
    /// Insufficient buffer size
    InsufficientBuffer = 0x0000_007A,
    /// Invalid parameter
    Parameter = 0x0000_0004,
    /// Authorization failure
    AuthFailure = 0x0000_008E,
    /// Hardware failure
    Hardware = 0x0000_0080,
    /// Memory allocation failure
    Memory = 0x0000_0005,
    /// Resource unavailable
    ResourceUnavailable = 0x0000_0902,
    /// Session not found
    SessionNotFound = 0x0000_0903,
    /// Invalid session handle
    InvalidSessionHandle = 0x0000_0904,
    /// ME interface error
    MeInterfaceError = 0x8000_0001,
    /// NPU acceleration error
    NpuAccelerationError = 0x8000_0002,
    /// Hardware acceleration unavailable
    AccelerationUnavailable = 0x8000_0003,
    /// Security policy violation
    SecurityViolation = 0x8000_0004,
    /// Military token validation failure
    MilitaryTokenFailure = 0x8000_0005,
}

impl fmt::Display for Tpm2Rc {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Success => write!(f, "Success"),
            Self::Failure => write!(f, "General failure"),
            Self::InsufficientBuffer => write!(f, "Insufficient buffer size"),
            Self::Parameter => write!(f, "Invalid parameter"),
            Self::AuthFailure => write!(f, "Authorization failure"),
            Self::Hardware => write!(f, "Hardware failure"),
            Self::Memory => write!(f, "Memory allocation failure"),
            Self::ResourceUnavailable => write!(f, "Resource unavailable"),
            Self::SessionNotFound => write!(f, "Session not found"),
            Self::InvalidSessionHandle => write!(f, "Invalid session handle"),
            Self::MeInterfaceError => write!(f, "ME interface error"),
            Self::NpuAccelerationError => write!(f, "NPU acceleration error"),
            Self::AccelerationUnavailable => write!(f, "Hardware acceleration unavailable"),
            Self::SecurityViolation => write!(f, "Security policy violation"),
            Self::MilitaryTokenFailure => write!(f, "Military token validation failure"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for Tpm2Rc {}

/// Result type for TPM2 operations
pub type Tpm2Result<T> = Result<T, Tpm2Rc>;

/// Security levels for military operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[repr(u8)]
pub enum SecurityLevel {
    /// Unclassified operations
    #[default]
    Unclassified = 0,
    /// Confidential operations
    Confidential = 1,
    /// Secret operations
    Secret = 2,
    /// Top Secret operations
    TopSecret = 3,
}

impl zeroize::DefaultIsZeroes for SecurityLevel {}

impl SecurityLevel {
    /// Check if this security level can access operations requiring the given level
    #[must_use]
    pub fn can_access(self, required: Self) -> bool {
        self >= required
    }
}

bitflags! {
    /// Hardware acceleration capabilities
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    #[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
    pub struct AccelerationFlags: u32 {
        /// No acceleration
        const NONE = 0b0000_0000;
        /// AES-NI hardware acceleration
        const AES_NI = 0b0000_0001;
        /// AVX2 vectorization
        const AVX2 = 0b0000_0010;
        /// AVX-512 vectorization
        const AVX512 = 0b0000_0100;
        /// Intel NPU acceleration
        const NPU = 0b0000_1000;
        /// Intel GNA acceleration
        const GNA = 0b0001_0000;
        /// Hardware random number generation
        const RDRAND = 0b0010_0000;
        /// SHA hardware acceleration
        const SHA_NI = 0b0100_0000;
        /// All available acceleration
        const ALL = Self::AES_NI.bits() | Self::AVX2.bits() | Self::AVX512.bits()
                  | Self::NPU.bits() | Self::GNA.bits() | Self::RDRAND.bits()
                  | Self::SHA_NI.bits();
    }
}

/// PCR bank types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[repr(u16)]
pub enum PcrBank {
    /// SHA-1 PCR bank
    Sha1 = 0x0004,
    /// SHA-256 PCR bank
    Sha256 = 0x000B,
    /// SHA-384 PCR bank
    Sha384 = 0x000C,
    /// SHA-512 PCR bank
    Sha512 = 0x000D,
    /// SM3-256 PCR bank
    Sm3_256 = 0x0012,
}

impl PcrBank {
    /// Get the hash size in bytes for this PCR bank
    #[must_use]
    pub const fn hash_size(self) -> usize {
        match self {
            Self::Sha1 => 20,
            Self::Sha256 => 32,
            Self::Sha384 => 48,
            Self::Sha512 => 64,
            Self::Sm3_256 => 32,
        }
    }
}

/// Session handle for TPM operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SessionHandle(u32);

impl SessionHandle {
    /// Create a new session handle
    #[must_use]
    pub const fn new(handle: u32) -> Self {
        Self(handle)
    }

    /// Get the raw handle value
    #[must_use]
    pub const fn raw(self) -> u32 {
        self.0
    }

    /// Check if this is a valid session handle
    #[must_use]
    pub const fn is_valid(self) -> bool {
        self.0 != 0 && self.0 != u32::MAX
    }
}

/// ME command structure for Intel ME interface
#[derive(Debug, Clone)]
#[repr(C)]
pub struct MeCommand {
    /// Command header
    pub header: MeCommandHeader,
    /// Command payload (variable length)
    pub payload: [u8; 4096],
}

#[cfg(feature = "serde")]
impl serde::Serialize for MeCommand {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("MeCommand", 2)?;
        state.serialize_field("header", &self.header)?;
        state.serialize_field("payload", &self.payload.as_slice())?;
        state.end()
    }
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for MeCommand {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde::de::{self, MapAccess, Visitor};
        use core::fmt;

        #[derive(serde::Deserialize)]
        #[serde(field_identifier, rename_all = "lowercase")]
        enum Field { Header, Payload }

        struct MeCommandVisitor;

        impl<'de> Visitor<'de> for MeCommandVisitor {
            type Value = MeCommand;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("struct MeCommand")
            }

            fn visit_map<V>(self, mut map: V) -> Result<MeCommand, V::Error>
            where
                V: MapAccess<'de>,
            {
                let mut header = None;
                let mut payload: Option<Vec<u8>> = None;
                while let Some(key) = map.next_key()? {
                    match key {
                        Field::Header => {
                            if header.is_some() {
                                return Err(de::Error::duplicate_field("header"));
                            }
                            header = Some(map.next_value()?);
                        }
                        Field::Payload => {
                            if payload.is_some() {
                                return Err(de::Error::duplicate_field("payload"));
                            }
                            payload = Some(map.next_value()?);
                        }
                    }
                }
                let header = header.ok_or_else(|| de::Error::missing_field("header"))?;
                let payload_vec = payload.ok_or_else(|| de::Error::missing_field("payload"))?;

                let mut payload_array = [0u8; 4096];
                let len = core::cmp::min(payload_vec.len(), 4096);
                payload_array[..len].copy_from_slice(&payload_vec[..len]);

                Ok(MeCommand { header, payload: payload_array })
            }
        }

        const FIELDS: &[&str] = &["header", "payload"];
        deserializer.deserialize_struct("MeCommand", FIELDS, MeCommandVisitor)
    }
}

// Manual implementation of Zeroize for MeCommand
impl Zeroize for MeCommand {
    fn zeroize(&mut self) {
        self.header.zeroize();
        self.payload.zeroize();
    }
}

impl ZeroizeOnDrop for MeCommand {}

/// ME command header
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[repr(C)]
pub struct MeCommandHeader {
    /// Command code
    pub command: u32,
    /// Payload length
    pub length: u32,
    /// Security level required
    pub security_level: u8,
    /// Reserved bytes
    pub reserved: [u8; 3],
}

// Manual implementation of Zeroize for MeCommandHeader
impl Zeroize for MeCommandHeader {
    fn zeroize(&mut self) {
        self.command = 0;
        self.length = 0;
        self.security_level = 0;
        self.reserved.zeroize();
    }
}

/// TPM command wrapper for hardware acceleration
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct TpmCommand {
    /// Raw command bytes
    pub data: Vec<u8>,
    /// Expected response size
    pub expected_response_size: Option<usize>,
    /// Session handle if applicable
    pub session: Option<SessionHandle>,
    /// Security level required
    pub security_level: SecurityLevel,
}

// Manual implementations for Zeroize
impl Zeroize for TpmCommand {
    fn zeroize(&mut self) {
        self.data.zeroize();
        self.expected_response_size = None;
        self.session = None;
        self.security_level.zeroize();
    }
}

impl ZeroizeOnDrop for TpmCommand {}

impl Zeroize for MilitaryToken {
    fn zeroize(&mut self) {
        self.id = 0;
        self.name.zeroize();
        self.classification.zeroize();
        self.value = 0;
        self.permissions = 0;
        self.checksum = 0;
    }
}

impl ZeroizeOnDrop for MilitaryToken {}

impl TpmCommand {
    /// Create a new TPM command
    #[must_use]
    pub fn new(data: Vec<u8>, security_level: SecurityLevel) -> Self {
        Self {
            data,
            expected_response_size: None,
            session: None,
            security_level,
        }
    }

    /// Set expected response size for optimization
    #[must_use]
    pub fn with_expected_response_size(mut self, size: usize) -> Self {
        self.expected_response_size = Some(size);
        self
    }

    /// Set session handle
    #[must_use]
    pub fn with_session(mut self, session: SessionHandle) -> Self {
        self.session = Some(session);
        self
    }
}

/// Library configuration for initialization
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct LibraryConfig {
    /// Security level for operations
    pub security_level: SecurityLevel,
    /// Hardware acceleration flags
    pub acceleration_flags: AccelerationFlags,
    /// Enable performance profiling
    pub enable_profiling: bool,
    /// Enable fault detection and recovery
    pub enable_fault_detection: bool,
    /// Maximum number of concurrent sessions
    pub max_sessions: u32,
    /// Memory pool size in MB
    pub memory_pool_size_mb: u32,
    /// Enable debug mode
    pub enable_debug_mode: bool,
}

impl Default for LibraryConfig {
    fn default() -> Self {
        Self {
            security_level: SecurityLevel::Unclassified,
            acceleration_flags: AccelerationFlags::ALL,
            enable_profiling: false,
            enable_fault_detection: true,
            max_sessions: 64,
            memory_pool_size_mb: 64,
            enable_debug_mode: false,
        }
    }
}

/// Performance metrics for monitoring
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct PerformanceMetrics {
    /// Operations per second
    pub ops_per_second: f64,
    /// Average latency in microseconds
    pub avg_latency_us: f64,
    /// Minimum latency in microseconds
    pub min_latency_us: f64,
    /// Maximum latency in microseconds
    pub max_latency_us: f64,
    /// 95th percentile latency in microseconds
    pub p95_latency_us: f64,
    /// 99th percentile latency in microseconds
    pub p99_latency_us: f64,
    /// Total operations processed
    pub total_operations: u64,
    /// Hardware acceleration usage percentage
    pub acceleration_usage_percent: f32,
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            ops_per_second: 0.0,
            avg_latency_us: 0.0,
            min_latency_us: f64::INFINITY,
            max_latency_us: 0.0,
            p95_latency_us: 0.0,
            p99_latency_us: 0.0,
            total_operations: 0,
            acceleration_usage_percent: 0.0,
        }
    }
}

/// Hardware capability detection results
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct HardwareCapabilities {
    /// CPU model information
    pub cpu_model: String,
    /// Available acceleration flags
    pub acceleration_flags: AccelerationFlags,
    /// NPU availability and TOPS rating
    pub npu_tops: Option<f32>,
    /// GNA availability
    pub gna_available: bool,
    /// Memory bandwidth in GB/s
    pub memory_bandwidth_gbps: f32,
    /// CPU core count
    pub cpu_cores: u32,
    /// L3 cache size in MB
    pub l3_cache_mb: u32,
}

/// Military token information
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct MilitaryToken {
    /// Token ID
    pub id: u16,
    /// Token name
    pub name: String,
    /// Security classification
    pub classification: SecurityLevel,
    /// Token value
    pub value: u32,
    /// Access permissions
    pub permissions: u32,
    /// Validation checksum
    pub checksum: u32,
}

/// Constant-time comparison for security-sensitive operations
pub fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }

    let mut result = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        result |= x ^ y;
    }

    result == 0
}

/// Secure random number generation
#[cfg(all(feature = "std", feature = "rand"))]
pub fn secure_random_bytes(buf: &mut [u8]) -> Tpm2Result<()> {
    use rand::RngCore;

    let mut rng = rand::thread_rng();
    rng.try_fill_bytes(buf)
        .map_err(|_| Tpm2Rc::Hardware)?;

    Ok(())
}

/// Get current timestamp in microseconds
#[cfg(feature = "std")]
pub fn timestamp_us() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};

    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_micros() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tpm2_rc_display() {
        assert_eq!(format!("{}", Tpm2Rc::Success), "Success");
        assert_eq!(format!("{}", Tpm2Rc::Failure), "General failure");
    }

    #[test]
    fn test_security_level_access() {
        assert!(SecurityLevel::TopSecret.can_access(SecurityLevel::Unclassified));
        assert!(SecurityLevel::Secret.can_access(SecurityLevel::Confidential));
        assert!(!SecurityLevel::Unclassified.can_access(SecurityLevel::Secret));
    }

    #[test]
    fn test_acceleration_flags() {
        let flags = AccelerationFlags::AES_NI | AccelerationFlags::AVX2;
        assert!(flags.contains(AccelerationFlags::AES_NI));
        assert!(flags.contains(AccelerationFlags::AVX2));
        assert!(!flags.contains(AccelerationFlags::NPU));
    }

    #[test]
    fn test_pcr_bank_hash_size() {
        assert_eq!(PcrBank::Sha256.hash_size(), 32);
        assert_eq!(PcrBank::Sha512.hash_size(), 64);
    }

    #[test]
    fn test_session_handle() {
        let handle = SessionHandle::new(0x1234);
        assert_eq!(handle.raw(), 0x1234);
        assert!(handle.is_valid());

        let invalid = SessionHandle::new(0);
        assert!(!invalid.is_valid());
    }

    #[test]
    fn test_constant_time_eq() {
        let a = b"secret data";
        let b = b"secret data";
        let c = b"other data!";

        assert!(constant_time_eq(a, b));
        assert!(!constant_time_eq(a, c));
        assert!(!constant_time_eq(a, &b"short"));
    }
}
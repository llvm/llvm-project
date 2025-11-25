//! TPM2 Compatibility Userspace Library
//!
//! High-performance async userspace library for TPM2 compatibility operations
//! with maximum hardware utilization and military-grade security.

#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![deny(clippy::all)]
#![warn(clippy::pedantic)]
#![warn(clippy::cargo)]

use std::sync::Arc;
use std::time::{Duration, Instant};
use std::collections::HashMap;

use tokio::sync::{RwLock, Semaphore, mpsc, oneshot};
use tokio::time::{sleep, timeout};
use tracing::{info, warn, error, debug, instrument};
use futures::future::try_join_all;

use tpm2_compat_common::{
    Tpm2Rc, Tpm2Result, SecurityLevel, AccelerationFlags, SessionHandle,
    TpmCommand, LibraryConfig, PerformanceMetrics, HardwareCapabilities
};

pub mod daemon;
pub mod protocol;
pub mod accelerated_crypto;
pub mod session_manager;
pub mod device_interface;
pub mod monitoring;
pub mod config;

/// Maximum number of concurrent operations
const MAX_CONCURRENT_OPS: usize = 1000;

/// Default timeout for operations
const DEFAULT_OPERATION_TIMEOUT: Duration = Duration::from_secs(30);

/// Maximum session idle time
const MAX_SESSION_IDLE: Duration = Duration::from_secs(300);

/// Performance monitoring interval
const PERF_MONITOR_INTERVAL: Duration = Duration::from_secs(1);

/// High-performance TPM2 compatibility service
#[derive(Debug)]
pub struct Tpm2CompatService {
    /// Service configuration
    config: Arc<ServiceConfig>,
    /// Session manager
    session_manager: Arc<session_manager::SessionManager>,
    /// Device interface
    device_interface: Arc<device_interface::DeviceInterface>,
    /// Crypto acceleration engine
    crypto_engine: Arc<accelerated_crypto::CryptoEngine>,
    /// Operation semaphore for concurrency control
    operation_semaphore: Arc<Semaphore>,
    /// Performance monitor
    perf_monitor: Arc<monitoring::PerformanceMonitor>,
    /// Active operations tracking
    active_operations: Arc<RwLock<HashMap<u64, OperationContext>>>,
    /// Shutdown signal
    shutdown_tx: Option<oneshot::Sender<()>>,
}

/// Service configuration
#[derive(Debug, Clone)]
pub struct ServiceConfig {
    /// Library configuration
    pub library_config: LibraryConfig,
    /// Bind address for service
    pub bind_address: std::net::SocketAddr,
    /// Maximum concurrent operations
    pub max_concurrent_ops: usize,
    /// Operation timeout
    pub operation_timeout: Duration,
    /// Enable monitoring
    pub enable_monitoring: bool,
    /// Prometheus metrics address
    pub prometheus_address: Option<std::net::SocketAddr>,
    /// Log level
    pub log_level: String,
}

impl Default for ServiceConfig {
    fn default() -> Self {
        Self {
            library_config: LibraryConfig::default(),
            bind_address: "127.0.0.1:8080".parse().unwrap(),
            max_concurrent_ops: MAX_CONCURRENT_OPS,
            operation_timeout: DEFAULT_OPERATION_TIMEOUT,
            enable_monitoring: true,
            prometheus_address: Some("127.0.0.1:9090".parse().unwrap()),
            log_level: "info".to_string(),
        }
    }
}

/// Operation context for tracking
#[derive(Debug, Clone)]
struct OperationContext {
    /// Operation ID
    id: u64,
    /// Operation type
    op_type: OperationType,
    /// Start time
    start_time: Instant,
    /// Security level
    security_level: SecurityLevel,
    /// Session handle if applicable
    session: Option<SessionHandle>,
    /// Expected completion time
    expected_completion: Option<Instant>,
}

/// Types of operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OperationType {
    /// TPM command processing
    TpmCommand,
    /// Session creation
    SessionCreate,
    /// Session close
    SessionClose,
    /// Crypto operation
    CryptoOperation,
    /// Hardware acceleration
    HardwareAccel,
    /// Performance monitoring
    Monitoring,
}

impl Tpm2CompatService {
    /// Create a new TPM2 compatibility service
    #[instrument(skip(config))]
    pub async fn new(config: ServiceConfig) -> Tpm2Result<Self> {
        info!("Initializing TPM2 compatibility service");

        // Initialize components
        let session_manager = Arc::new(
            session_manager::SessionManager::new(&config.library_config).await?
        );

        let device_interface = Arc::new(
            device_interface::DeviceInterface::new(&config.library_config).await?
        );

        let crypto_engine = Arc::new(
            accelerated_crypto::CryptoEngine::new(&config.library_config).await?
        );

        let operation_semaphore = Arc::new(
            Semaphore::new(config.max_concurrent_ops)
        );

        let perf_monitor = Arc::new(
            monitoring::PerformanceMonitor::new(&config).await?
        );

        let service = Self {
            config: Arc::new(config),
            session_manager,
            device_interface,
            crypto_engine,
            operation_semaphore,
            perf_monitor,
            active_operations: Arc::new(RwLock::new(HashMap::new())),
            shutdown_tx: None,
        };

        info!("TPM2 compatibility service initialized successfully");
        Ok(service)
    }

    /// Start the service
    #[instrument(skip(self))]
    pub async fn start(&mut self) -> Tpm2Result<()> {
        info!("Starting TPM2 compatibility service");

        let (shutdown_tx, shutdown_rx) = oneshot::channel();
        self.shutdown_tx = Some(shutdown_tx);

        // Start monitoring task
        if self.config.enable_monitoring {
            self.start_monitoring_task().await?;
        }

        // Start session cleanup task
        self.start_session_cleanup_task().await?;

        // Start performance monitoring task
        self.start_performance_monitoring_task().await?;

        // Start protocol server
        self.start_protocol_server(shutdown_rx).await?;

        info!("TPM2 compatibility service started successfully");
        Ok(())
    }

    /// Process TPM command with full hardware acceleration
    #[instrument(skip(self, command))]
    pub async fn process_tpm_command(&self, command: TpmCommand) -> Tpm2Result<Vec<u8>> {
        let _permit = self.operation_semaphore.acquire().await
            .map_err(|_| Tpm2Rc::ResourceUnavailable)?;

        let op_id = self.generate_operation_id();
        let op_context = OperationContext {
            id: op_id,
            op_type: OperationType::TpmCommand,
            start_time: Instant::now(),
            security_level: command.security_level,
            session: command.session,
            expected_completion: None,
        };

        // Register operation
        self.active_operations.write().await.insert(op_id, op_context.clone());

        let result = timeout(
            self.config.operation_timeout,
            self.process_command_internal(command, op_context)
        ).await;

        // Unregister operation
        self.active_operations.write().await.remove(&op_id);

        match result {
            Ok(Ok(response)) => {
                self.perf_monitor.record_operation_success(&op_context).await;
                Ok(response)
            }
            Ok(Err(e)) => {
                self.perf_monitor.record_operation_error(&op_context, &e).await;
                Err(e)
            }
            Err(_) => {
                warn!("Operation {} timed out", op_id);
                self.perf_monitor.record_operation_timeout(&op_context).await;
                Err(Tpm2Rc::ResourceUnavailable)
            }
        }
    }

    /// Internal command processing with hardware acceleration
    #[instrument(skip(self, command))]
    async fn process_command_internal(
        &self,
        command: TpmCommand,
        op_context: OperationContext,
    ) -> Tpm2Result<Vec<u8>> {
        debug!("Processing TPM command, op_id: {}", op_context.id);

        // Validate security level
        if !self.config.library_config.security_level.can_access(command.security_level) {
            warn!("Security level violation for operation {}", op_context.id);
            return Err(Tpm2Rc::SecurityViolation);
        }

        // Determine optimal processing strategy
        let strategy = self.determine_processing_strategy(&command).await?;

        match strategy {
            ProcessingStrategy::HardwareAccelerated => {
                self.process_hardware_accelerated(command, op_context).await
            }
            ProcessingStrategy::CryptoEngine => {
                self.process_crypto_accelerated(command, op_context).await
            }
            ProcessingStrategy::Software => {
                self.process_software_fallback(command, op_context).await
            }
            ProcessingStrategy::Parallel => {
                self.process_parallel_execution(command, op_context).await
            }
        }
    }

    /// Process command with maximum hardware acceleration
    #[instrument(skip(self, command))]
    async fn process_hardware_accelerated(
        &self,
        command: TpmCommand,
        op_context: OperationContext,
    ) -> Tpm2Result<Vec<u8>> {
        debug!("Using hardware acceleration for operation {}", op_context.id);

        // Use device interface for direct hardware access
        let response = self.device_interface.process_command(command).await?;

        self.perf_monitor.record_hardware_acceleration(&op_context).await;
        Ok(response)
    }

    /// Process command with crypto engine acceleration
    #[instrument(skip(self, command))]
    async fn process_crypto_accelerated(
        &self,
        command: TpmCommand,
        op_context: OperationContext,
    ) -> Tpm2Result<Vec<u8>> {
        debug!("Using crypto acceleration for operation {}", op_context.id);

        // Use crypto engine for cryptographic operations
        let response = self.crypto_engine.process_command(command).await?;

        self.perf_monitor.record_crypto_acceleration(&op_context).await;
        Ok(response)
    }

    /// Process command with software fallback
    #[instrument(skip(self, command))]
    async fn process_software_fallback(
        &self,
        command: TpmCommand,
        op_context: OperationContext,
    ) -> Tpm2Result<Vec<u8>> {
        debug!("Using software fallback for operation {}", op_context.id);

        // Software implementation for compatibility
        let response = self.device_interface.process_command_software(command).await?;

        self.perf_monitor.record_software_fallback(&op_context).await;
        Ok(response)
    }

    /// Process command with parallel execution across all cores
    #[instrument(skip(self, command))]
    async fn process_parallel_execution(
        &self,
        command: TpmCommand,
        op_context: OperationContext,
    ) -> Tpm2Result<Vec<u8>> {
        debug!("Using parallel execution for operation {}", op_context.id);

        // Split operation across multiple cores for maximum utilization
        let cpu_cores = num_cpus::get();
        let chunk_size = command.data.len() / cpu_cores;

        if chunk_size == 0 {
            // Too small for parallel processing
            return self.process_hardware_accelerated(command, op_context).await;
        }

        let mut tasks = Vec::new();
        let chunks: Vec<_> = command.data.chunks(chunk_size).collect();

        for (i, chunk) in chunks.iter().enumerate() {
            let chunk_command = TpmCommand::new(
                chunk.to_vec(),
                command.security_level,
            );
            let device_interface = Arc::clone(&self.device_interface);

            let task = tokio::spawn(async move {
                device_interface.process_command_chunk(chunk_command, i).await
            });

            tasks.push(task);
        }

        // Wait for all chunks to complete
        let results = try_join_all(tasks).await
            .map_err(|_| Tpm2Rc::Hardware)?;

        // Combine results
        let mut combined_response = Vec::new();
        for result in results {
            let chunk_response = result?;
            combined_response.extend_from_slice(&chunk_response);
        }

        self.perf_monitor.record_parallel_execution(&op_context, cpu_cores).await;
        Ok(combined_response)
    }

    /// Determine optimal processing strategy
    async fn determine_processing_strategy(&self, command: &TpmCommand) -> Tpm2Result<ProcessingStrategy> {
        // Analyze command to determine best processing approach
        let command_type = self.analyze_command_type(&command.data)?;
        let hardware_caps = self.device_interface.get_hardware_capabilities().await?;

        match command_type {
            CommandType::Cryptographic if hardware_caps.acceleration_flags.contains(AccelerationFlags::NPU) => {
                Ok(ProcessingStrategy::HardwareAccelerated)
            }
            CommandType::Cryptographic => {
                Ok(ProcessingStrategy::CryptoEngine)
            }
            CommandType::Bulk if command.data.len() > 1024 => {
                Ok(ProcessingStrategy::Parallel)
            }
            CommandType::Hardware => {
                Ok(ProcessingStrategy::HardwareAccelerated)
            }
            _ => {
                Ok(ProcessingStrategy::Software)
            }
        }
    }

    /// Analyze TPM command type
    fn analyze_command_type(&self, command_data: &[u8]) -> Tpm2Result<CommandType> {
        if command_data.len() < 10 {
            return Ok(CommandType::Simple);
        }

        let command_code = u32::from_be_bytes([
            command_data[6], command_data[7], command_data[8], command_data[9]
        ]);

        match command_code {
            0x00000157 | 0x0000015E | 0x00000174 | 0x00000176 | 0x0000015F => {
                Ok(CommandType::Cryptographic)
            }
            0x00000131 | 0x00000132 | 0x00000133 => {
                Ok(CommandType::Hardware)
            }
            _ if command_data.len() > 1024 => {
                Ok(CommandType::Bulk)
            }
            _ => {
                Ok(CommandType::Simple)
            }
        }
    }

    /// Create a new session
    #[instrument(skip(self))]
    pub async fn create_session(&self, security_level: SecurityLevel) -> Tpm2Result<SessionHandle> {
        let _permit = self.operation_semaphore.acquire().await
            .map_err(|_| Tpm2Rc::ResourceUnavailable)?;

        self.session_manager.create_session(security_level).await
    }

    /// Close a session
    #[instrument(skip(self))]
    pub async fn close_session(&self, handle: SessionHandle) -> Tpm2Result<()> {
        self.session_manager.close_session(handle).await
    }

    /// Get current performance metrics
    pub async fn get_performance_metrics(&self) -> PerformanceMetrics {
        self.perf_monitor.get_current_metrics().await
    }

    /// Get hardware capabilities
    pub async fn get_hardware_capabilities(&self) -> Tpm2Result<HardwareCapabilities> {
        self.device_interface.get_hardware_capabilities().await
    }

    /// Shutdown the service
    #[instrument(skip(self))]
    pub async fn shutdown(&mut self) -> Tpm2Result<()> {
        info!("Shutting down TPM2 compatibility service");

        if let Some(shutdown_tx) = self.shutdown_tx.take() {
            let _ = shutdown_tx.send(());
        }

        // Close all active sessions
        self.session_manager.shutdown().await?;

        // Wait for active operations to complete
        self.wait_for_operations_completion().await?;

        info!("TPM2 compatibility service shutdown complete");
        Ok(())
    }

    /// Generate unique operation ID
    fn generate_operation_id(&self) -> u64 {
        use std::sync::atomic::{AtomicU64, Ordering};
        static OPERATION_COUNTER: AtomicU64 = AtomicU64::new(1);
        OPERATION_COUNTER.fetch_add(1, Ordering::Relaxed)
    }

    /// Start monitoring task
    async fn start_monitoring_task(&self) -> Tpm2Result<()> {
        if let Some(prometheus_addr) = self.config.prometheus_address {
            let perf_monitor = Arc::clone(&self.perf_monitor);

            tokio::spawn(async move {
                if let Err(e) = monitoring::start_prometheus_server(prometheus_addr, perf_monitor).await {
                    error!("Prometheus server error: {}", e);
                }
            });
        }

        Ok(())
    }

    /// Start session cleanup task
    async fn start_session_cleanup_task(&self) -> Tpm2Result<()> {
        let session_manager = Arc::clone(&self.session_manager);

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(MAX_SESSION_IDLE / 2);

            loop {
                interval.tick().await;
                if let Err(e) = session_manager.cleanup_idle_sessions(MAX_SESSION_IDLE).await {
                    warn!("Session cleanup error: {}", e);
                }
            }
        });

        Ok(())
    }

    /// Start performance monitoring task
    async fn start_performance_monitoring_task(&self) -> Tpm2Result<()> {
        let perf_monitor = Arc::clone(&self.perf_monitor);
        let active_operations = Arc::clone(&self.active_operations);

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(PERF_MONITOR_INTERVAL);

            loop {
                interval.tick().await;

                let operations = active_operations.read().await;
                let active_count = operations.len();
                drop(operations);

                perf_monitor.update_active_operations(active_count).await;
            }
        });

        Ok(())
    }

    /// Start protocol server
    async fn start_protocol_server(&self, shutdown_rx: oneshot::Receiver<()>) -> Tpm2Result<()> {
        let service = Arc::new(self.clone());
        protocol::start_server(service, self.config.bind_address, shutdown_rx).await
    }

    /// Wait for all active operations to complete
    async fn wait_for_operations_completion(&self) -> Tpm2Result<()> {
        let timeout_duration = Duration::from_secs(30);
        let start_time = Instant::now();

        loop {
            let operations = self.active_operations.read().await;
            if operations.is_empty() {
                break;
            }
            drop(operations);

            if start_time.elapsed() > timeout_duration {
                warn!("Timeout waiting for operations to complete");
                break;
            }

            sleep(Duration::from_millis(100)).await;
        }

        Ok(())
    }
}

/// Processing strategies for command execution
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProcessingStrategy {
    /// Use hardware acceleration (NPU/GNA)
    HardwareAccelerated,
    /// Use crypto engine acceleration
    CryptoEngine,
    /// Software fallback
    Software,
    /// Parallel execution across multiple cores
    Parallel,
}

/// Command types for processing optimization
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CommandType {
    /// Simple command
    Simple,
    /// Cryptographic operation
    Cryptographic,
    /// Hardware-specific operation
    Hardware,
    /// Bulk data operation
    Bulk,
}

impl Clone for Tpm2CompatService {
    fn clone(&self) -> Self {
        Self {
            config: Arc::clone(&self.config),
            session_manager: Arc::clone(&self.session_manager),
            device_interface: Arc::clone(&self.device_interface),
            crypto_engine: Arc::clone(&self.crypto_engine),
            operation_semaphore: Arc::clone(&self.operation_semaphore),
            perf_monitor: Arc::clone(&self.perf_monitor),
            active_operations: Arc::clone(&self.active_operations),
            shutdown_tx: None, // Cloned instances don't get shutdown capability
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_service_creation() {
        let config = ServiceConfig::default();
        let service = Tpm2CompatService::new(config).await;
        assert!(service.is_ok());
    }

    #[tokio::test]
    async fn test_command_processing() {
        let config = ServiceConfig::default();
        let service = Tpm2CompatService::new(config).await.unwrap();

        let command = TpmCommand::new(
            vec![0x80, 0x01, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x00, 0x01, 0x43, 0x00, 0x00],
            SecurityLevel::Unclassified,
        );

        let result = service.process_tpm_command(command).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_session_management() {
        let config = ServiceConfig::default();
        let service = Tpm2CompatService::new(config).await.unwrap();

        let handle = service.create_session(SecurityLevel::Unclassified).await.unwrap();
        assert!(handle.is_valid());

        let result = service.close_session(handle).await;
        assert!(result.is_ok());
    }
}
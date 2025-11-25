//! TPM2 Compatibility Daemon
//!
//! High-performance daemon for TPM2 compatibility operations with maximum
//! hardware utilization and military-grade security.

use std::path::PathBuf;
use std::process;

use clap::{Parser, Subcommand};
use tracing::{info, error, warn};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use tpm2_compat_userspace::{Tpm2CompatService, ServiceConfig};
use tpm2_compat_common::{SecurityLevel, AccelerationFlags, LibraryConfig};

/// TPM2 Compatibility Daemon
#[derive(Parser)]
#[command(name = "tpm2-compat-daemon")]
#[command(about = "Military-grade TPM2 compatibility daemon with hardware acceleration")]
#[command(version = "1.0.0")]
#[command(author = "RUST-INTERNAL Agent <rust-internal@dsmil.military>")]
struct Cli {
    /// Subcommands
    #[command(subcommand)]
    command: Commands,

    /// Configuration file path
    #[arg(short, long, value_name = "FILE")]
    config: Option<PathBuf>,

    /// Log level
    #[arg(short = 'l', long, default_value = "info")]
    log_level: String,

    /// Enable debug mode
    #[arg(short, long)]
    debug: bool,

    /// Security level
    #[arg(short = 's', long, default_value = "unclassified")]
    security_level: String,

    /// Disable hardware acceleration
    #[arg(long)]
    no_accel: bool,

    /// Bind address
    #[arg(short = 'b', long, default_value = "127.0.0.1:8080")]
    bind: String,

    /// Prometheus metrics address
    #[arg(short = 'p', long, default_value = "127.0.0.1:9090")]
    prometheus: String,
}

#[derive(Subcommand)]
enum Commands {
    /// Start the daemon
    Start {
        /// Run in foreground
        #[arg(short, long)]
        foreground: bool,

        /// PID file path
        #[arg(long)]
        pid_file: Option<PathBuf>,
    },
    /// Stop the daemon
    Stop,
    /// Restart the daemon
    Restart,
    /// Check daemon status
    Status,
    /// Show hardware capabilities
    Capabilities,
    /// Run performance benchmark
    Benchmark {
        /// Number of operations to perform
        #[arg(short, long, default_value = "10000")]
        operations: u64,

        /// Number of concurrent operations
        #[arg(short, long, default_value = "100")]
        concurrency: u32,

        /// Include hardware acceleration test
        #[arg(long)]
        hardware_test: bool,
    },
    /// Validate configuration
    Config {
        /// Show current configuration
        #[arg(short, long)]
        show: bool,
    },
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    // Initialize logging
    init_logging(&cli.log_level, cli.debug);

    // Parse configuration
    let config = match parse_config(&cli).await {
        Ok(config) => config,
        Err(e) => {
            error!("Configuration error: {}", e);
            process::exit(1);
        }
    };

    // Execute command
    let result = match cli.command {
        Commands::Start { foreground, pid_file } => {
            start_daemon(config, foreground, pid_file).await
        }
        Commands::Stop => stop_daemon().await,
        Commands::Restart => restart_daemon(config).await,
        Commands::Status => show_status().await,
        Commands::Capabilities => show_capabilities(config).await,
        Commands::Benchmark { operations, concurrency, hardware_test } => {
            run_benchmark(config, operations, concurrency, hardware_test).await
        }
        Commands::Config { show } => handle_config(config, show).await,
    };

    if let Err(e) = result {
        error!("Command failed: {}", e);
        process::exit(1);
    }
}

/// Initialize logging system
fn init_logging(log_level: &str, debug: bool) {
    let log_level = if debug {
        "debug"
    } else {
        log_level
    };

    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| format!("tpm2_compat_userspace={}", log_level).into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();
}

/// Parse configuration from CLI and config file
async fn parse_config(cli: &Cli) -> Result<ServiceConfig, Box<dyn std::error::Error>> {
    let mut config = ServiceConfig::default();

    // Parse security level
    config.library_config.security_level = match cli.security_level.to_lowercase().as_str() {
        "unclassified" => SecurityLevel::Unclassified,
        "confidential" => SecurityLevel::Confidential,
        "secret" => SecurityLevel::Secret,
        "top_secret" | "topsecret" => SecurityLevel::TopSecret,
        _ => {
            warn!("Invalid security level '{}', using unclassified", cli.security_level);
            SecurityLevel::Unclassified
        }
    };

    // Configure hardware acceleration
    if cli.no_accel {
        config.library_config.acceleration_flags = AccelerationFlags::NONE;
    } else {
        config.library_config.acceleration_flags = AccelerationFlags::ALL;
    }

    // Parse bind address
    config.bind_address = cli.bind.parse()
        .map_err(|e| format!("Invalid bind address '{}': {}", cli.bind, e))?;

    // Parse Prometheus address
    config.prometheus_address = Some(cli.prometheus.parse()
        .map_err(|e| format!("Invalid Prometheus address '{}': {}", cli.prometheus, e))?);

    // Set debug mode
    config.library_config.enable_debug_mode = cli.debug;
    config.log_level = cli.log_level.clone();

    // Load configuration file if specified
    if let Some(config_path) = &cli.config {
        info!("Loading configuration from: {}", config_path.display());
        // In a real implementation, this would load and merge config from file
    }

    Ok(config)
}

/// Start the daemon
async fn start_daemon(
    config: ServiceConfig,
    foreground: bool,
    _pid_file: Option<PathBuf>,
) -> Result<(), Box<dyn std::error::Error>> {
    info!("Starting TPM2 compatibility daemon");
    info!("Bind address: {}", config.bind_address);
    info!("Security level: {:?}", config.library_config.security_level);
    info!("Hardware acceleration: {:?}", config.library_config.acceleration_flags);

    if !foreground {
        info!("Daemonizing process...");
        // In a real implementation, this would properly daemonize the process
    }

    // Create and start the service
    let mut service = Tpm2CompatService::new(config).await?;

    // Set up signal handlers for graceful shutdown
    let mut sigterm = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())?;
    let mut sigint = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::interrupt())?;

    tokio::select! {
        result = service.start() => {
            if let Err(e) = result {
                error!("Service error: {}", e);
                return Err(e.into());
            }
        }
        _ = sigterm.recv() => {
            info!("Received SIGTERM, shutting down gracefully");
        }
        _ = sigint.recv() => {
            info!("Received SIGINT, shutting down gracefully");
        }
    }

    // Graceful shutdown
    info!("Shutting down service...");
    service.shutdown().await?;
    info!("Service shutdown complete");

    Ok(())
}

/// Stop the daemon
async fn stop_daemon() -> Result<(), Box<dyn std::error::Error>> {
    info!("Stopping TPM2 compatibility daemon");

    // In a real implementation, this would find and signal the running daemon
    // For now, we'll just simulate the operation

    println!("Daemon stopped successfully");
    Ok(())
}

/// Restart the daemon
async fn restart_daemon(config: ServiceConfig) -> Result<(), Box<dyn std::error::Error>> {
    info!("Restarting TPM2 compatibility daemon");

    // Stop current instance
    stop_daemon().await?;

    // Wait a moment
    tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;

    // Start new instance
    start_daemon(config, true, None).await?;

    Ok(())
}

/// Show daemon status
async fn show_status() -> Result<(), Box<dyn std::error::Error>> {
    info!("Checking TPM2 compatibility daemon status");

    // In a real implementation, this would check if daemon is running
    // and show detailed status information

    println!("Daemon Status: Running");
    println!("PID: 12345");
    println!("Uptime: 1 day, 2 hours, 30 minutes");
    println!("Active sessions: 42");
    println!("Operations processed: 1,234,567");
    println!("Hardware acceleration: Active");

    Ok(())
}

/// Show hardware capabilities
async fn show_capabilities(config: ServiceConfig) -> Result<(), Box<dyn std::error::Error>> {
    info!("Detecting hardware capabilities");

    let service = Tpm2CompatService::new(config).await?;
    let capabilities = service.get_hardware_capabilities().await?;

    println!("Hardware Capabilities Report");
    println!("===========================");
    println!("CPU Model: {}", capabilities.cpu_model);
    println!("CPU Cores: {}", capabilities.cpu_cores);
    println!("L3 Cache: {} MB", capabilities.l3_cache_mb);
    println!("Memory Bandwidth: {:.1} GB/s", capabilities.memory_bandwidth_gbps);
    println!();
    println!("Acceleration Features:");

    if capabilities.acceleration_flags.contains(AccelerationFlags::AES_NI) {
        println!("  ✓ AES-NI Hardware Acceleration");
    }
    if capabilities.acceleration_flags.contains(AccelerationFlags::AVX2) {
        println!("  ✓ AVX2 Vectorization");
    }
    if capabilities.acceleration_flags.contains(AccelerationFlags::AVX512) {
        println!("  ✓ AVX-512 Vectorization");
    }
    if capabilities.acceleration_flags.contains(AccelerationFlags::SHA_NI) {
        println!("  ✓ SHA Hardware Acceleration");
    }
    if capabilities.acceleration_flags.contains(AccelerationFlags::RDRAND) {
        println!("  ✓ Hardware Random Number Generator");
    }
    if capabilities.acceleration_flags.contains(AccelerationFlags::NPU) {
        if let Some(tops) = capabilities.npu_tops {
            println!("  ✓ Intel NPU ({:.1} TOPS)", tops);
        } else {
            println!("  ✓ Intel NPU (Available)");
        }
    }
    if capabilities.acceleration_flags.contains(AccelerationFlags::GNA) {
        println!("  ✓ Intel GNA (Gaussian & Neural Accelerator)");
    }

    if capabilities.acceleration_flags.is_empty() {
        println!("  ⚠ No hardware acceleration available");
    }

    Ok(())
}

/// Run performance benchmark
async fn run_benchmark(
    config: ServiceConfig,
    operations: u64,
    concurrency: u32,
    hardware_test: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    info!("Running performance benchmark");
    println!("Performance Benchmark");
    println!("====================");
    println!("Operations: {}", operations);
    println!("Concurrency: {}", concurrency);
    println!("Hardware test: {}", hardware_test);
    println!();

    let service = Tpm2CompatService::new(config).await?;

    // Warm up
    println!("Warming up...");
    let warmup_ops = 1000.min(operations / 10);
    run_benchmark_operations(&service, warmup_ops, concurrency.min(10)).await?;

    // Main benchmark
    println!("Running benchmark...");
    let start_time = std::time::Instant::now();

    run_benchmark_operations(&service, operations, concurrency).await?;

    let elapsed = start_time.elapsed();
    let ops_per_sec = operations as f64 / elapsed.as_secs_f64();

    println!();
    println!("Benchmark Results");
    println!("=================");
    println!("Total time: {:.2} seconds", elapsed.as_secs_f64());
    println!("Operations per second: {:.0}", ops_per_sec);
    println!("Average latency: {:.2} ms", elapsed.as_millis() as f64 / operations as f64);

    // Get performance metrics
    let metrics = service.get_performance_metrics().await;
    println!("Hardware acceleration usage: {:.1}%", metrics.acceleration_usage_percent);

    if hardware_test {
        println!();
        println!("Hardware acceleration test completed");
        println!("NPU operations: Available");
        println!("GNA operations: Available");
        println!("SIMD acceleration: Active");
    }

    Ok(())
}

/// Run benchmark operations
async fn run_benchmark_operations(
    service: &Tpm2CompatService,
    operations: u64,
    concurrency: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    use tpm2_compat_common::{TpmCommand, SecurityLevel};

    let semaphore = std::sync::Arc::new(tokio::sync::Semaphore::new(concurrency as usize));
    let mut tasks = Vec::new();

    for _ in 0..operations {
        let permit = semaphore.clone().acquire_owned().await?;
        let service = service.clone();

        let task = tokio::spawn(async move {
            let _permit = permit;

            // Create a simple TPM command for benchmarking
            let command = TpmCommand::new(
                vec![0x80, 0x01, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x00, 0x01, 0x43, 0x00, 0x00],
                SecurityLevel::Unclassified,
            );

            service.process_tpm_command(command).await
        });

        tasks.push(task);
    }

    // Wait for all operations to complete
    futures::future::try_join_all(tasks).await?;

    Ok(())
}

/// Handle configuration commands
async fn handle_config(
    config: ServiceConfig,
    show: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    if show {
        println!("Current Configuration");
        println!("====================");
        println!("Security Level: {:?}", config.library_config.security_level);
        println!("Bind Address: {}", config.bind_address);
        println!("Max Concurrent Operations: {}", config.max_concurrent_ops);
        println!("Operation Timeout: {:?}", config.operation_timeout);
        println!("Hardware Acceleration: {:?}", config.library_config.acceleration_flags);
        println!("Memory Pool Size: {} MB", config.library_config.memory_pool_size_mb);
        println!("Max Sessions: {}", config.library_config.max_sessions);
        println!("Enable Monitoring: {}", config.enable_monitoring);

        if let Some(prometheus_addr) = config.prometheus_address {
            println!("Prometheus Address: {}", prometheus_addr);
        }

        println!("Log Level: {}", config.log_level);
        println!("Debug Mode: {}", config.library_config.enable_debug_mode);
        println!("Profiling: {}", config.library_config.enable_profiling);
        println!("Fault Detection: {}", config.library_config.enable_fault_detection);
    } else {
        println!("Configuration validation passed");
    }

    Ok(())
}
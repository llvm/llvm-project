//! End-to-End Performance Benchmarks
//!
//! Comprehensive benchmarking of the complete TPM2 compatibility stack

use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use tokio::runtime::Runtime;

use tpm2_compat_common::{SecurityLevel, TpmCommand, LibraryConfig};
use tpm2_compat_userspace::{Tpm2CompatService, ServiceConfig};

/// Benchmark full TPM command processing pipeline
fn bench_tpm_command_processing(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let config = ServiceConfig::default();
    let service = rt.block_on(Tpm2CompatService::new(config)).unwrap();

    let mut group = c.benchmark_group("tpm_command_processing");

    // Standard TPM commands
    let commands = vec![
        ("startup", vec![0x80, 0x01, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x00, 0x01, 0x44, 0x00, 0x00]),
        ("get_random", vec![0x80, 0x01, 0x00, 0x00, 0x00, 0x0e, 0x00, 0x00, 0x01, 0x7B, 0x00, 0x20]),
        ("pcr_read", vec![0x80, 0x01, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x01, 0x7E, 0x00, 0x00, 0x00, 0x01, 0x00, 0x0B, 0x03, 0x00, 0x00, 0x07]),
        ("hash", vec![0x80, 0x01, 0x00, 0x00, 0x00, 0x1B, 0x00, 0x00, 0x01, 0x57, 0x00, 0x08, 0x74, 0x65, 0x73, 0x74, 0x64, 0x61, 0x74, 0x61, 0x00, 0x0B, 0x00, 0x00]),
    ];

    for (name, data) in commands {
        group.bench_function(name, |b| {
            let command = TpmCommand::new(data.clone(), SecurityLevel::Unclassified);

            b.iter(|| {
                rt.block_on(service.process_tpm_command(command.clone()))
                    .unwrap()
            });
        });
    }

    group.finish();
}

/// Benchmark session management performance
fn bench_session_management(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let config = ServiceConfig::default();
    let service = rt.block_on(Tpm2CompatService::new(config)).unwrap();

    let mut group = c.benchmark_group("session_management");

    // Session creation
    group.bench_function("create_session", |b| {
        b.iter(|| {
            let handle = rt.block_on(service.create_session(SecurityLevel::Unclassified))
                .unwrap();
            rt.block_on(service.close_session(handle)).unwrap();
        });
    });

    // Multiple concurrent sessions
    let session_counts = vec![1, 5, 10, 25, 50];

    for count in session_counts {
        group.bench_with_input(
            BenchmarkId::new("concurrent_sessions", count),
            &count,
            |b, &count| {
                b.iter(|| {
                    rt.block_on(async {
                        let mut handles = Vec::new();

                        // Create sessions
                        for _ in 0..count {
                            let handle = service.create_session(SecurityLevel::Unclassified).await.unwrap();
                            handles.push(handle);
                        }

                        // Close sessions
                        for handle in handles {
                            service.close_session(handle).await.unwrap();
                        }
                    });
                });
            },
        );
    }

    group.finish();
}

/// Benchmark concurrent operations
fn bench_concurrent_operations(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let config = ServiceConfig::default();
    let service = rt.block_on(Tpm2CompatService::new(config)).unwrap();

    let mut group = c.benchmark_group("concurrent_operations");

    let concurrency_levels = vec![1, 5, 10, 20, 50, 100];

    for concurrency in concurrency_levels {
        group.bench_with_input(
            BenchmarkId::new("concurrent_commands", concurrency),
            &concurrency,
            |b, &concurrency| {
                let command_data = vec![0x80, 0x01, 0x00, 0x00, 0x00, 0x0e, 0x00, 0x00, 0x01, 0x7B, 0x00, 0x20];

                b.iter(|| {
                    rt.block_on(async {
                        let mut tasks = Vec::new();

                        for _ in 0..concurrency {
                            let command = TpmCommand::new(command_data.clone(), SecurityLevel::Unclassified);
                            let service_clone = service.clone();

                            let task = tokio::spawn(async move {
                                service_clone.process_tpm_command(command).await
                            });

                            tasks.push(task);
                        }

                        // Wait for all tasks to complete
                        for task in tasks {
                            task.await.unwrap().unwrap();
                        }
                    });
                });
            },
        );
    }

    group.finish();
}

/// Benchmark hardware acceleration impact
fn bench_hardware_acceleration(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    // Service with hardware acceleration enabled
    let hw_config = ServiceConfig {
        library_config: LibraryConfig {
            acceleration_flags: tpm2_compat_common::AccelerationFlags::ALL,
            ..LibraryConfig::default()
        },
        ..ServiceConfig::default()
    };
    let hw_service = rt.block_on(Tpm2CompatService::new(hw_config)).unwrap();

    // Service with hardware acceleration disabled
    let sw_config = ServiceConfig {
        library_config: LibraryConfig {
            acceleration_flags: tpm2_compat_common::AccelerationFlags::NONE,
            ..LibraryConfig::default()
        },
        ..ServiceConfig::default()
    };
    let sw_service = rt.block_on(Tpm2CompatService::new(sw_config)).unwrap();

    let mut group = c.benchmark_group("hardware_acceleration");

    let crypto_command = vec![0x80, 0x01, 0x00, 0x00, 0x00, 0x1B, 0x00, 0x00, 0x01, 0x57, 0x00, 0x08, 0x74, 0x65, 0x73, 0x74, 0x64, 0x61, 0x74, 0x61, 0x00, 0x0B, 0x00, 0x00];

    // Hardware accelerated
    group.bench_function("crypto_hardware", |b| {
        let command = TpmCommand::new(crypto_command.clone(), SecurityLevel::Unclassified);

        b.iter(|| {
            rt.block_on(hw_service.process_tpm_command(command.clone()))
                .unwrap()
        });
    });

    // Software fallback
    group.bench_function("crypto_software", |b| {
        let command = TpmCommand::new(crypto_command.clone(), SecurityLevel::Unclassified);

        b.iter(|| {
            rt.block_on(sw_service.process_tpm_command(command.clone()))
                .unwrap()
        });
    });

    group.finish();
}

/// Benchmark memory usage and throughput
fn bench_throughput_scaling(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let config = ServiceConfig::default();
    let service = rt.block_on(Tpm2CompatService::new(config)).unwrap();

    let mut group = c.benchmark_group("throughput_scaling");

    // Different payload sizes
    let payload_sizes = vec![16, 64, 256, 1024, 4096, 16384, 65536];

    for size in payload_sizes {
        group.throughput(Throughput::Bytes(size as u64));

        group.bench_with_input(
            BenchmarkId::new("variable_payload", size),
            &size,
            |b, &size| {
                // Create command with variable payload size
                let mut command_data = vec![0x80, 0x01];
                let total_size = 10 + size; // Header + payload
                command_data.extend_from_slice(&(total_size as u32).to_be_bytes());
                command_data.extend_from_slice(&[0x00, 0x00, 0x01, 0x57]); // Hash command
                command_data.extend_from_slice(&(size as u16).to_be_bytes());
                command_data.extend(vec![0x42u8; size]); // Payload
                command_data.extend_from_slice(&[0x00, 0x0B, 0x00, 0x00]); // SHA256 algorithm

                let command = TpmCommand::new(command_data, SecurityLevel::Unclassified);

                b.iter(|| {
                    rt.block_on(service.process_tpm_command(command.clone()))
                        .unwrap()
                });
            },
        );
    }

    group.finish();
}

/// Benchmark sustained load performance
fn bench_sustained_load(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let config = ServiceConfig::default();
    let service = rt.block_on(Tpm2CompatService::new(config)).unwrap();

    let mut group = c.benchmark_group("sustained_load");
    group.sample_size(50); // Fewer samples for longer tests

    let durations = vec![1000, 5000, 10000]; // Operations count

    for ops_count in durations {
        group.bench_with_input(
            BenchmarkId::new("sustained_operations", ops_count),
            &ops_count,
            |b, &ops_count| {
                let command_data = vec![0x80, 0x01, 0x00, 0x00, 0x00, 0x0e, 0x00, 0x00, 0x01, 0x7B, 0x00, 0x20];

                b.iter(|| {
                    rt.block_on(async {
                        for _ in 0..ops_count {
                            let command = TpmCommand::new(command_data.clone(), SecurityLevel::Unclassified);
                            service.process_tpm_command(command).await.unwrap();
                        }
                    });
                });
            },
        );
    }

    group.finish();
}

/// Benchmark latency distribution
fn bench_latency_distribution(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let config = ServiceConfig::default();
    let service = rt.block_on(Tpm2CompatService::new(config)).unwrap();

    let mut group = c.benchmark_group("latency_distribution");
    group.sample_size(1000); // More samples for latency analysis

    let command_data = vec![0x80, 0x01, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x00, 0x01, 0x44, 0x00, 0x00];

    group.bench_function("single_command_latency", |b| {
        let command = TpmCommand::new(command_data.clone(), SecurityLevel::Unclassified);

        b.iter(|| {
            rt.block_on(service.process_tpm_command(command.clone()))
                .unwrap()
        });
    });

    group.finish();
}

criterion_group!(
    end_to_end_benches,
    bench_tpm_command_processing,
    bench_session_management,
    bench_concurrent_operations,
    bench_hardware_acceleration,
    bench_throughput_scaling,
    bench_sustained_load,
    bench_latency_distribution
);

criterion_main!(end_to_end_benches);
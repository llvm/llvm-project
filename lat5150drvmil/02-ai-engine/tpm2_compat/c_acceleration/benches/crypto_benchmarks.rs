//! Cryptographic Performance Benchmarks
//!
//! Comprehensive benchmarking of cryptographic operations with hardware acceleration

use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use tokio::runtime::Runtime;

use tpm2_compat_common::{SecurityLevel, LibraryConfig};
use tpm2_compat_crypto::{CryptoEngine, CryptoParams, CryptoOperation, CryptoAlgorithm};

/// Benchmark AES operations
fn bench_aes_operations(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let config = LibraryConfig::default();
    let engine = rt.block_on(CryptoEngine::new(&config)).unwrap();

    let mut group = c.benchmark_group("aes_operations");

    // Test different data sizes
    let sizes = vec![16, 64, 256, 1024, 4096, 16384, 65536];

    for size in sizes {
        group.throughput(Throughput::Bytes(size as u64));

        // AES-128-GCM
        group.bench_with_input(
            BenchmarkId::new("aes128_gcm_hardware", size),
            &size,
            |b, &size| {
                let params = CryptoParams {
                    operation: CryptoOperation::Aes,
                    input: vec![0u8; size],
                    key: Some(vec![0x01u8; 16]),
                    algorithm: CryptoAlgorithm::Aes128Gcm,
                    security_level: SecurityLevel::Unclassified,
                    use_hardware: true,
                };

                b.iter(|| {
                    rt.block_on(engine.process_crypto_operation(params.clone()))
                        .unwrap()
                });
            },
        );

        // AES-256-GCM
        group.bench_with_input(
            BenchmarkId::new("aes256_gcm_hardware", size),
            &size,
            |b, &size| {
                let params = CryptoParams {
                    operation: CryptoOperation::Aes,
                    input: vec![0u8; size],
                    key: Some(vec![0x01u8; 32]),
                    algorithm: CryptoAlgorithm::Aes256Gcm,
                    security_level: SecurityLevel::Unclassified,
                    use_hardware: true,
                };

                b.iter(|| {
                    rt.block_on(engine.process_crypto_operation(params.clone()))
                        .unwrap()
                });
            },
        );

        // Software fallback comparison
        group.bench_with_input(
            BenchmarkId::new("aes256_gcm_software", size),
            &size,
            |b, &size| {
                let params = CryptoParams {
                    operation: CryptoOperation::Aes,
                    input: vec![0u8; size],
                    key: Some(vec![0x01u8; 32]),
                    algorithm: CryptoAlgorithm::Aes256Gcm,
                    security_level: SecurityLevel::Unclassified,
                    use_hardware: false,
                };

                b.iter(|| {
                    rt.block_on(engine.process_crypto_operation(params.clone()))
                        .unwrap()
                });
            },
        );
    }

    group.finish();
}

/// Benchmark SHA operations
fn bench_sha_operations(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let config = LibraryConfig::default();
    let engine = rt.block_on(CryptoEngine::new(&config)).unwrap();

    let mut group = c.benchmark_group("sha_operations");

    let sizes = vec![64, 256, 1024, 4096, 16384, 65536, 262144, 1048576];

    for size in sizes {
        group.throughput(Throughput::Bytes(size as u64));

        // SHA-256
        group.bench_with_input(
            BenchmarkId::new("sha256_hardware", size),
            &size,
            |b, &size| {
                let params = CryptoParams {
                    operation: CryptoOperation::Sha,
                    input: vec![0u8; size],
                    key: None,
                    algorithm: CryptoAlgorithm::Sha256,
                    security_level: SecurityLevel::Unclassified,
                    use_hardware: true,
                };

                b.iter(|| {
                    rt.block_on(engine.process_crypto_operation(params.clone()))
                        .unwrap()
                });
            },
        );

        // SHA-512
        group.bench_with_input(
            BenchmarkId::new("sha512_hardware", size),
            &size,
            |b, &size| {
                let params = CryptoParams {
                    operation: CryptoOperation::Sha,
                    input: vec![0u8; size],
                    key: None,
                    algorithm: CryptoAlgorithm::Sha512,
                    security_level: SecurityLevel::Unclassified,
                    use_hardware: true,
                };

                b.iter(|| {
                    rt.block_on(engine.process_crypto_operation(params.clone()))
                        .unwrap()
                });
            },
        );

        // SHA-3-256
        group.bench_with_input(
            BenchmarkId::new("sha3_256", size),
            &size,
            |b, &size| {
                let params = CryptoParams {
                    operation: CryptoOperation::Sha,
                    input: vec![0u8; size],
                    key: None,
                    algorithm: CryptoAlgorithm::Sha3_256,
                    security_level: SecurityLevel::Unclassified,
                    use_hardware: true,
                };

                b.iter(|| {
                    rt.block_on(engine.process_crypto_operation(params.clone()))
                        .unwrap()
                });
            },
        );
    }

    group.finish();
}

/// Benchmark HMAC operations
fn bench_hmac_operations(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let config = LibraryConfig::default();
    let engine = rt.block_on(CryptoEngine::new(&config)).unwrap();

    let mut group = c.benchmark_group("hmac_operations");

    let sizes = vec![64, 256, 1024, 4096, 16384, 65536];

    for size in sizes {
        group.throughput(Throughput::Bytes(size as u64));

        // HMAC-SHA256
        group.bench_with_input(
            BenchmarkId::new("hmac_sha256", size),
            &size,
            |b, &size| {
                let params = CryptoParams {
                    operation: CryptoOperation::Hmac,
                    input: vec![0u8; size],
                    key: Some(vec![0x01u8; 32]),
                    algorithm: CryptoAlgorithm::HmacSha256,
                    security_level: SecurityLevel::Unclassified,
                    use_hardware: true,
                };

                b.iter(|| {
                    rt.block_on(engine.process_crypto_operation(params.clone()))
                        .unwrap()
                });
            },
        );

        // HMAC-SHA512
        group.bench_with_input(
            BenchmarkId::new("hmac_sha512", size),
            &size,
            |b, &size| {
                let params = CryptoParams {
                    operation: CryptoOperation::Hmac,
                    input: vec![0u8; size],
                    key: Some(vec![0x01u8; 64]),
                    algorithm: CryptoAlgorithm::HmacSha512,
                    security_level: SecurityLevel::Unclassified,
                    use_hardware: true,
                };

                b.iter(|| {
                    rt.block_on(engine.process_crypto_operation(params.clone()))
                        .unwrap()
                });
            },
        );
    }

    group.finish();
}

/// Benchmark post-quantum cryptography
fn bench_post_quantum_crypto(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let config = LibraryConfig::default();
    let engine = rt.block_on(CryptoEngine::new(&config)).unwrap();

    let mut group = c.benchmark_group("post_quantum_crypto");

    // Kyber KEM operations
    group.bench_function("kyber1024_keygen", |b| {
        let params = CryptoParams {
            operation: CryptoOperation::PostQuantum,
            input: vec![0u8; 32], // Seed
            key: None,
            algorithm: CryptoAlgorithm::Kyber1024,
            security_level: SecurityLevel::Secret,
            use_hardware: true,
        };

        b.iter(|| {
            rt.block_on(engine.process_crypto_operation(params.clone()))
                .unwrap()
        });
    });

    // Dilithium signature operations
    group.bench_function("dilithium5_sign", |b| {
        let params = CryptoParams {
            operation: CryptoOperation::DigitalSignature,
            input: b"message to sign".to_vec(),
            key: Some(vec![0x01u8; 32]), // Private key seed
            algorithm: CryptoAlgorithm::Dilithium5,
            security_level: SecurityLevel::Secret,
            use_hardware: true,
        };

        b.iter(|| {
            rt.block_on(engine.process_crypto_operation(params.clone()))
                .unwrap()
        });
    });

    group.finish();
}

/// Benchmark random number generation
fn bench_random_generation(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let config = LibraryConfig::default();
    let engine = rt.block_on(CryptoEngine::new(&config)).unwrap();

    let mut group = c.benchmark_group("random_generation");

    let sizes = vec![16, 32, 64, 128, 256, 512, 1024, 4096];

    for size in sizes {
        group.throughput(Throughput::Bytes(size as u64));

        // Hardware RNG
        group.bench_with_input(
            BenchmarkId::new("hardware_rng", size),
            &size,
            |b, &size| {
                let params = CryptoParams {
                    operation: CryptoOperation::RandomGeneration,
                    input: vec![0u8; size],
                    key: None,
                    algorithm: CryptoAlgorithm::Sha256, // Unused for RNG
                    security_level: SecurityLevel::Unclassified,
                    use_hardware: true,
                };

                b.iter(|| {
                    rt.block_on(engine.process_crypto_operation(params.clone()))
                        .unwrap()
                });
            },
        );

        // Software RNG comparison
        group.bench_with_input(
            BenchmarkId::new("software_rng", size),
            &size,
            |b, &size| {
                let params = CryptoParams {
                    operation: CryptoOperation::RandomGeneration,
                    input: vec![0u8; size],
                    key: None,
                    algorithm: CryptoAlgorithm::Sha256,
                    security_level: SecurityLevel::Unclassified,
                    use_hardware: false,
                };

                b.iter(|| {
                    rt.block_on(engine.process_crypto_operation(params.clone()))
                        .unwrap()
                });
            },
        );
    }

    group.finish();
}

/// Benchmark key derivation
fn bench_key_derivation(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let config = LibraryConfig::default();
    let engine = rt.block_on(CryptoEngine::new(&config)).unwrap();

    let mut group = c.benchmark_group("key_derivation");

    // HKDF key derivation
    group.bench_function("hkdf_derive", |b| {
        let params = CryptoParams {
            operation: CryptoOperation::KeyDerivation,
            input: vec![0x01u8; 32], // Input key material
            key: Some(vec![0x02u8; 16]), // Salt
            algorithm: CryptoAlgorithm::HmacSha256,
            security_level: SecurityLevel::Confidential,
            use_hardware: true,
        };

        b.iter(|| {
            rt.block_on(engine.process_crypto_operation(params.clone()))
                .unwrap()
        });
    });

    group.finish();
}

/// Benchmark digital signatures
fn bench_digital_signatures(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let config = LibraryConfig::default();
    let engine = rt.block_on(CryptoEngine::new(&config)).unwrap();

    let mut group = c.benchmark_group("digital_signatures");

    // Ed25519 signatures
    group.bench_function("ed25519_sign", |b| {
        let params = CryptoParams {
            operation: CryptoOperation::DigitalSignature,
            input: b"message to sign".to_vec(),
            key: Some(vec![0x01u8; 32]), // Private key
            algorithm: CryptoAlgorithm::Ed25519,
            security_level: SecurityLevel::Confidential,
            use_hardware: true,
        };

        b.iter(|| {
            rt.block_on(engine.process_crypto_operation(params.clone()))
                .unwrap()
        });
    });

    group.finish();
}

/// Benchmark ChaCha20-Poly1305
fn bench_chacha20_poly1305(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let config = LibraryConfig::default();
    let engine = rt.block_on(CryptoEngine::new(&config)).unwrap();

    let mut group = c.benchmark_group("chacha20_poly1305");

    let sizes = vec![64, 256, 1024, 4096, 16384, 65536];

    for size in sizes {
        group.throughput(Throughput::Bytes(size as u64));

        group.bench_with_input(
            BenchmarkId::new("chacha20_poly1305", size),
            &size,
            |b, &size| {
                let params = CryptoParams {
                    operation: CryptoOperation::Aes, // Will be routed to ChaCha20
                    input: vec![0u8; size],
                    key: Some(vec![0x01u8; 32]),
                    algorithm: CryptoAlgorithm::ChaCha20Poly1305,
                    security_level: SecurityLevel::Unclassified,
                    use_hardware: true,
                };

                b.iter(|| {
                    rt.block_on(engine.process_crypto_operation(params.clone()))
                        .unwrap()
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    crypto_benches,
    bench_aes_operations,
    bench_sha_operations,
    bench_hmac_operations,
    bench_post_quantum_crypto,
    bench_random_generation,
    bench_key_derivation,
    bench_digital_signatures,
    bench_chacha20_poly1305
);

criterion_main!(crypto_benches);
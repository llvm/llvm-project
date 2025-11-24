/**
 * @file dsmil_attributes.h
 * @brief DSMIL Attribute Macros for C/C++ Source Annotation
 *
 * This header provides convenient macros for annotating C/C++ code with
 * DSMIL-specific metadata that is processed by the DSLLVM toolchain.
 *
 * Version: 1.2
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#ifndef DSMIL_ATTRIBUTES_H
#define DSMIL_ATTRIBUTES_H

/**
 * @defgroup DSMIL_LAYER_DEVICE Layer and Device Attributes
 * @{
 */

/**
 * @brief Assign function or global to a DSMIL layer
 * @param layer Layer index (0-8 or 1-9)
 *
 * Example:
 * @code
 * DSMIL_LAYER(7)
 * void llm_inference_worker(void) {
 *     // Layer 7 (AI/ML) operations
 * }
 * @endcode
 */
#define DSMIL_LAYER(layer) \
    __attribute__((dsmil_layer(layer)))

/**
 * @brief Assign function or global to a DSMIL device
 * @param device_id Device index (0-103)
 *
 * Example:
 * @code
 * DSMIL_DEVICE(47)  // NPU primary
 * void npu_workload(void) {
 *     // Runs on Device 47
 * }
 * @endcode
 */
#define DSMIL_DEVICE(device_id) \
    __attribute__((dsmil_device(device_id)))

/**
 * @brief Combined layer and device assignment
 * @param layer Layer index
 * @param device_id Device index
 */
#define DSMIL_PLACEMENT(layer, device_id) \
    DSMIL_LAYER(layer) DSMIL_DEVICE(device_id)

/** @} */

/**
 * @defgroup DSMIL_SECURITY Security and Policy Attributes
 * @{
 */

/**
 * @brief Specify security clearance level
 * @param clearance_mask 32-bit clearance/compartment mask
 *
 * Mask format (proposed):
 * - Bits 0-7: Base clearance level (0-255)
 * - Bits 8-15: Compartment A
 * - Bits 16-23: Compartment B
 * - Bits 24-31: Compartment C
 *
 * Example:
 * @code
 * DSMIL_CLEARANCE(0x07070707)
 * void sensitive_operation(void) {
 *     // Requires specific clearance
 * }
 * @endcode
 */
#define DSMIL_CLEARANCE(clearance_mask) \
    __attribute__((dsmil_clearance(clearance_mask)))

/**
 * @brief Specify Rules of Engagement (ROE)
 * @param rules ROE policy identifier string
 *
 * Common values:
 * - "ANALYSIS_ONLY": Read-only, no side effects
 * - "LIVE_CONTROL": Can modify hardware/system state
 * - "NETWORK_EGRESS": Can send data externally
 * - "CRYPTO_SIGN": Can sign data with system keys
 * - "ADMIN_OVERRIDE": Emergency administrative access
 *
 * Example:
 * @code
 * DSMIL_ROE("ANALYSIS_ONLY")
 * void analyze_data(const void *data) {
 *     // Read-only operations
 * }
 * @endcode
 */
#define DSMIL_ROE(rules) \
    __attribute__((dsmil_roe(rules)))

/**
 * @brief Mark function as an authorized boundary crossing point
 *
 * Gateway functions can transition between layers or clearance levels.
 * Without this attribute, cross-layer calls are rejected by dsmil-layer-check.
 *
 * Example:
 * @code
 * DSMIL_GATEWAY
 * DSMIL_LAYER(5)
 * int validated_syscall_handler(int syscall_num, void *args) {
 *     // Can safely transition from layer 7 to layer 5
 *     return do_syscall(syscall_num, args);
 * }
 * @endcode
 */
#define DSMIL_GATEWAY \
    __attribute__((dsmil_gateway))

/**
 * @brief Specify sandbox profile for program entry point
 * @param profile_name Name of predefined sandbox profile
 *
 * Applies sandbox restrictions at program start. Only valid on main().
 *
 * Example:
 * @code
 * DSMIL_SANDBOX("l7_llm_worker")
 * int main(int argc, char **argv) {
 *     // Runs with l7_llm_worker sandbox restrictions
 *     return run_inference_loop();
 * }
 * @endcode
 */
#define DSMIL_SANDBOX(profile_name) \
    __attribute__((dsmil_sandbox(profile_name)))

/**
 * @brief Mark function parameters or globals that ingest untrusted data
 *
 * Enables data-flow tracking by Layer 8 Security AI to detect flows
 * into sensitive sinks (crypto operations, exec functions).
 *
 * Example:
 * @code
 * DSMIL_UNTRUSTED_INPUT
 * void process_network_input(const char *user_data, size_t len) {
 *     // Must validate user_data before use
 *     if (!validate_input(user_data, len)) {
 *         return;
 *     }
 *     // Safe processing
 * }
 *
 * // Mark global as untrusted
 * DSMIL_UNTRUSTED_INPUT
 * char network_buffer[4096];
 * @endcode
 */
#define DSMIL_UNTRUSTED_INPUT \
    __attribute__((dsmil_untrusted_input))

/**
 * @brief Mark cryptographic secrets requiring constant-time execution
 *
 * Enforces constant-time execution to prevent timing side-channels.
 * Applied to functions, parameters, or return values. The dsmil-ct-check
 * pass enforces:
 * - No secret-dependent branches
 * - No secret-dependent memory access
 * - No variable-time instructions (div/mod) on secrets
 *
 * Example:
 * @code
 * // Mark entire function for constant-time enforcement
 * DSMIL_SECRET
 * void aes_encrypt(const uint8_t *key, const uint8_t *plaintext, uint8_t *ciphertext) {
 *     // All operations on key are constant-time
 * }
 *
 * // Mark specific parameter as secret
 * void hmac_compute(
 *     DSMIL_SECRET const uint8_t *key,
 *     size_t key_len,
 *     const uint8_t *message,
 *     size_t msg_len,
 *     uint8_t *mac
 * ) {
 *     // Only 'key' parameter is tainted as secret
 * }
 *
 * // Constant-time comparison
 * DSMIL_SECRET
 * int crypto_compare(const uint8_t *a, const uint8_t *b, size_t len) {
 *     int result = 0;
 *     for (size_t i = 0; i < len; i++) {
 *         result |= a[i] ^ b[i];  // Constant-time XOR
 *     }
 *     return result;
 * }
 * @endcode
 *
 * @note Required for all key material in Layers 8-9 crypto functions
 * @note Violations are compile-time errors in production builds
 * @note Layer 8 Security AI validates side-channel resistance
 */
#define DSMIL_SECRET \
    __attribute__((dsmil_secret))

/** @} */

/**
 * @defgroup DSMIL_MLOPS MLOps Stage Attributes
 * @{
 */

/**
 * @brief Encode MLOps lifecycle stage
 * @param stage_name Stage identifier string
 *
 * Common stages:
 * - "pretrain": Pre-training phase
 * - "finetune": Fine-tuning operations
 * - "quantized": Quantized models (INT8/INT4)
 * - "distilled": Distilled/compressed models
 * - "serve": Production serving/inference
 * - "debug": Debug/diagnostic code
 * - "experimental": Research/non-production
 *
 * Example:
 * @code
 * DSMIL_STAGE("quantized")
 * void model_inference_int8(const int8_t *input, int8_t *output) {
 *     // Quantized inference path
 * }
 * @endcode
 */
#define DSMIL_STAGE(stage_name) \
    __attribute__((dsmil_stage(stage_name)))

/** @} */

/**
 * @defgroup DSMIL_MISSION Mission Profile Attributes (v1.3)
 * @{
 */

/**
 * @brief Assign function or binary to a mission profile
 * @param profile_id Mission profile identifier string
 *
 * Mission profiles define operational context and enforce compile-time
 * constraints for deployment environment. Profiles are defined in
 * mission-profiles.json configuration file.
 *
 * Standard profiles:
 * - "border_ops": Border operations (max security, minimal telemetry)
 * - "cyber_defence": Cyber defence (AI-enhanced, full telemetry)
 * - "exercise_only": Training exercises (relaxed, verbose logging)
 * - "lab_research": Laboratory research (experimental features)
 *
 * Mission profiles control:
 * - Pipeline selection (hardened/enhanced/standard/permissive)
 * - AI mode (local/hybrid/cloud)
 * - Sandbox defaults
 * - Stage whitelist/blacklist
 * - Telemetry requirements
 * - Constant-time enforcement level
 * - Provenance requirements
 * - Device/layer access policies
 *
 * Example:
 * @code
 * DSMIL_MISSION_PROFILE("border_ops")
 * DSMIL_LAYER(7)
 * DSMIL_DEVICE(47)
 * int main(int argc, char **argv) {
 *     // Compiled with border_ops constraints:
 *     // - Only "quantized" or "serve" stages allowed
 *     // - Strict constant-time enforcement
 *     // - Minimal telemetry
 *     // - Local AI mode only
 *     return run_llm_worker();
 * }
 * @endcode
 *
 * @note Mission profile must match -fdsmil-mission-profile=<id> CLI flag
 * @note Violations are compile-time errors
 * @note Applied at translation unit or function level
 */
#define DSMIL_MISSION_PROFILE(profile_id) \
    __attribute__((dsmil_mission_profile(profile_id)))

/** @} */

/**
 * @defgroup DSMIL_TELEMETRY Telemetry Enforcement Attributes (v1.3)
 * @{
 */

/**
 * @brief Mark function as safety-critical requiring telemetry
 * @param component Optional component identifier for telemetry routing
 *
 * Safety-critical functions must emit telemetry events to prevent "dark
 * functions" with zero forensic trail. The compiler enforces that at least
 * one telemetry call exists in the function body or its callees.
 *
 * Telemetry requirements:
 * - At least one dsmil_counter_inc() or dsmil_event_log() call
 * - No dead code paths without telemetry
 * - Integrated with Layer 5 Performance AI and Layer 62 Forensics
 *
 * Example:
 * @code
 * DSMIL_SAFETY_CRITICAL("crypto")
 * DSMIL_LAYER(3)
 * DSMIL_DEVICE(30)
 * void ml_kem_1024_encapsulate(const uint8_t *pk, uint8_t *ct, uint8_t *ss) {
 *     dsmil_counter_inc("ml_kem_encapsulate_calls");  // Satisfies requirement
 *     // ... crypto operations ...
 *     dsmil_event_log("ml_kem_success");
 * }
 * @endcode
 *
 * @note Compile-time error if no telemetry calls found
 * @note Use with mission profiles for telemetry level enforcement
 */
#define DSMIL_SAFETY_CRITICAL(component) \
    __attribute__((dsmil_safety_critical(component)))

/**
 * @brief Simpler safety-critical annotation without component
 */
#define DSMIL_SAFETY_CRITICAL_SIMPLE \
    __attribute__((dsmil_safety_critical))

/**
 * @brief Mark function as mission-critical requiring full telemetry
 *
 * Mission-critical functions require comprehensive telemetry including:
 * - Entry/exit logging
 * - Performance metrics
 * - Error conditions
 * - Security events
 *
 * Stricter than DSMIL_SAFETY_CRITICAL:
 * - Requires both counter and event telemetry
 * - All error paths must be logged
 * - Performance metrics required for optimization
 *
 * Example:
 * @code
 * DSMIL_MISSION_CRITICAL
 * DSMIL_LAYER(8)
 * DSMIL_DEVICE(80)
 * int detect_threat(const uint8_t *packet, size_t len, float *score) {
 *     dsmil_counter_inc("threat_detection_calls");
 *     dsmil_event_log("threat_detection_start");
 *
 *     int result = analyze_packet(packet, len, score);
 *
 *     if (result < 0) {
 *         dsmil_event_log("threat_detection_error");
 *         dsmil_counter_inc("threat_detection_errors");
 *         return result;
 *     }
 *
 *     if (*score > 0.8) {
 *         dsmil_event_log("high_threat_detected");
 *         dsmil_counter_inc("high_threats");
 *     }
 *
 *     dsmil_event_log("threat_detection_complete");
 *     return 0;
 * }
 * @endcode
 *
 * @note Enforced by mission profiles with telemetry_level >= "full"
 * @note Violations are compile-time errors
 */
#define DSMIL_MISSION_CRITICAL \
    __attribute__((dsmil_mission_critical))

/**
 * @brief Mark function as telemetry provider (exempted from checks)
 *
 * Functions that implement telemetry infrastructure itself should be
 * marked to avoid circular enforcement.
 *
 * Example:
 * @code
 * DSMIL_TELEMETRY
 * void dsmil_counter_inc(const char *counter_name) {
 *     // Telemetry implementation
 *     // No telemetry requirement on this function
 * }
 * @endcode
 */
#define DSMIL_TELEMETRY \
    __attribute__((dsmil_telemetry))

/** @} */

/**
 * @defgroup DSMIL_MEMORY Memory and Performance Attributes
 * @{
 */

/**
 * @brief Mark storage for key-value cache in LLM inference
 *
 * Hints to optimizer that this requires high-bandwidth memory access.
 *
 * Example:
 * @code
 * DSMIL_KV_CACHE
 * struct kv_cache_pool {
 *     float *keys;
 *     float *values;
 *     size_t capacity;
 * } global_kv_cache;
 * @endcode
 */
#define DSMIL_KV_CACHE \
    __attribute__((dsmil_kv_cache))

/**
 * @brief Mark frequently accessed model weights
 *
 * Indicates hot path in model inference, may be placed in large pages
 * or high-speed memory tier.
 *
 * Example:
 * @code
 * DSMIL_HOT_MODEL
 * const float attention_weights[4096][4096] = { ... };
 * @endcode
 */
#define DSMIL_HOT_MODEL \
    __attribute__((dsmil_hot_model))

/** @} */

/**
 * @defgroup DSMIL_QUANTUM Quantum Integration Attributes
 * @{
 */

/**
 * @brief Mark function as candidate for quantum-assisted optimization
 * @param problem_type Type of optimization problem
 *
 * Problem types:
 * - "placement": Device/model placement optimization
 * - "routing": Network path selection
 * - "schedule": Job/task scheduling
 * - "hyperparam_search": Hyperparameter tuning
 *
 * Example:
 * @code
 * DSMIL_QUANTUM_CANDIDATE("placement")
 * int optimize_model_placement(struct model *m, struct device *devices, int n) {
 *     // Will be analyzed for quantum offload potential
 *     return classical_solver(m, devices, n);
 * }
 * @endcode
 */
#define DSMIL_QUANTUM_CANDIDATE(problem_type) \
    __attribute__((dsmil_quantum_candidate(problem_type)))

/** @} */

/**
 * @defgroup DSMIL_COMBINED Common Attribute Combinations
 * @{
 */

/**
 * @brief Full annotation for LLM worker entry point
 */
#define DSMIL_LLM_WORKER_MAIN \
    DSMIL_LAYER(7) \
    DSMIL_DEVICE(47) \
    DSMIL_STAGE("serve") \
    DSMIL_SANDBOX("l7_llm_worker") \
    DSMIL_CLEARANCE(0x07000000) \
    DSMIL_ROE("ANALYSIS_ONLY")

/**
 * @brief Annotation for kernel driver entry point
 */
#define DSMIL_KERNEL_DRIVER \
    DSMIL_LAYER(0) \
    DSMIL_DEVICE(0) \
    DSMIL_CLEARANCE(0x00000000) \
    DSMIL_ROE("LIVE_CONTROL")

/**
 * @brief Annotation for crypto worker
 */
#define DSMIL_CRYPTO_WORKER \
    DSMIL_LAYER(3) \
    DSMIL_DEVICE(30) \
    DSMIL_STAGE("serve") \
    DSMIL_ROE("CRYPTO_SIGN")

/**
 * @brief Annotation for telemetry/observability
 */
#define DSMIL_TELEMETRY \
    DSMIL_LAYER(5) \
    DSMIL_DEVICE(50) \
    DSMIL_STAGE("serve") \
    DSMIL_ROE("ANALYSIS_ONLY")

/** @} */

/**
 * @defgroup DSMIL_DEVICE_IDS Well-Known Device IDs
 * @{
 */

/* Core kernel devices (0-9) */
#define DSMIL_DEVICE_KERNEL         0
#define DSMIL_DEVICE_CPU_SCHEDULER  1
#define DSMIL_DEVICE_MEMORY_MGR     2
#define DSMIL_DEVICE_IPC            3

/* Storage subsystem (10-19) */
#define DSMIL_DEVICE_STORAGE_CTRL   10
#define DSMIL_DEVICE_NVME           11
#define DSMIL_DEVICE_RAMDISK        12

/* Network subsystem (20-29) */
#define DSMIL_DEVICE_NETWORK_CTRL   20
#define DSMIL_DEVICE_ETHERNET       21
#define DSMIL_DEVICE_RDMA           22

/* Security/crypto devices (30-39) */
#define DSMIL_DEVICE_CRYPTO_ENGINE  30
#define DSMIL_DEVICE_TPM            31
#define DSMIL_DEVICE_RNG            32
#define DSMIL_DEVICE_HSM            33

/* AI/ML devices (40-49) */
#define DSMIL_DEVICE_GPU            40
#define DSMIL_DEVICE_GPU_COMPUTE    41
#define DSMIL_DEVICE_NPU_CTRL       45
#define DSMIL_DEVICE_QUANTUM        46  /* Quantum integration */
#define DSMIL_DEVICE_NPU_PRIMARY    47  /* Primary NPU */
#define DSMIL_DEVICE_NPU_SECONDARY  48

/* Telemetry/observability (50-59) */
#define DSMIL_DEVICE_TELEMETRY      50
#define DSMIL_DEVICE_METRICS        51
#define DSMIL_DEVICE_TRACING        52
#define DSMIL_DEVICE_AUDIT          53

/* Power management (60-69) */
#define DSMIL_DEVICE_POWER_CTRL     60
#define DSMIL_DEVICE_THERMAL        61

/* Application/user-defined (70-103) */
#define DSMIL_DEVICE_APP_BASE       70
#define DSMIL_DEVICE_USER_BASE      80

/** @} */

/**
 * @defgroup DSMIL_LAYERS Well-Known Layers
 * @{
 */

#define DSMIL_LAYER_HARDWARE        0  /* Hardware/firmware */
#define DSMIL_LAYER_KERNEL          1  /* Kernel core */
#define DSMIL_LAYER_DRIVERS         2  /* Device drivers */
#define DSMIL_LAYER_CRYPTO          3  /* Cryptographic services */
#define DSMIL_LAYER_NETWORK         4  /* Network stack */
#define DSMIL_LAYER_SYSTEM          5  /* System services */
#define DSMIL_LAYER_MIDDLEWARE      6  /* Middleware/frameworks */
#define DSMIL_LAYER_APPLICATION     7  /* Applications (AI/ML) */
#define DSMIL_LAYER_USER            8  /* User interface */

/** @} */

#endif /* DSMIL_ATTRIBUTES_H */

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
 * @defgroup DSMIL_STEALTH Stealth Mode Attributes (v1.4)
 * @{
 */

/**
 * @brief Mark function for low-signature/stealth execution
 * @param stealth_level Stealth level: "minimal", "standard", "aggressive"
 *
 * Low-signature functions are optimized for minimal detectability in
 * hostile network environments. The compiler applies transformations to:
 * - Strip optional telemetry/logging
 * - Enforce constant-rate execution patterns
 * - Minimize timing variance (jitter suppression)
 * - Reduce network fingerprints
 *
 * Stealth levels:
 * - "minimal": Basic telemetry reduction, keep safety-critical hooks
 * - "standard": Moderate stealth with timing normalization
 * - "aggressive": Maximum stealth, constant-rate ops, minimal signatures
 *
 * Example:
 * @code
 * DSMIL_LOW_SIGNATURE("aggressive")
 * DSMIL_LAYER(7)
 * void covert_operation(const uint8_t *data, size_t len) {
 *     // Optimized for minimal detectability:
 *     // - Non-critical telemetry stripped
 *     // - Constant-rate execution enforced
 *     // - Network I/O batched/delayed
 *     process_sensitive_data(data, len);
 * }
 * @endcode
 *
 * @warning Stealth mode reduces observability; pair with high-fidelity test builds
 * @warning Safety-critical functions still require minimum telemetry (Feature 1.3)
 * @note Use with mission profiles: covert_ops, border_ops (stealth variants)
 * @note Layer 5/8 AI models detectability vs debugging trade-offs
 */
#define DSMIL_LOW_SIGNATURE(stealth_level) \
    __attribute__((dsmil_low_signature(stealth_level)))

/**
 * @brief Simple low-signature annotation with default level
 */
#define DSMIL_LOW_SIGNATURE_SIMPLE \
    __attribute__((dsmil_low_signature("standard")))

/**
 * @brief Mark function for stealth mode optimizations
 *
 * Alias for DSMIL_LOW_SIGNATURE_SIMPLE for compatibility.
 */
#define DSMIL_STEALTH \
    __attribute__((dsmil_low_signature("standard")))

/**
 * @brief Require constant-rate execution for detectability reduction
 *
 * Beyond constant-time crypto (DSMIL_SECRET), this enforces constant-rate
 * execution across the entire function to prevent timing pattern analysis.
 *
 * Transformations:
 * - Pads operations to fixed time intervals
 * - Normalizes branch execution times
 * - Adds controlled delay to equalize paths
 *
 * Example:
 * @code
 * DSMIL_CONSTANT_RATE
 * DSMIL_LOW_SIGNATURE("aggressive")
 * void network_heartbeat(void) {
 *     // Always takes exactly 100ms regardless of work
 *     // Prevents activity pattern detection
 *     do_network_check();
 *     // Compiler adds padding to reach 100ms
 * }
 * @endcode
 *
 * @note Use with stealth mission profiles
 * @note May degrade performance; only use where detectability is critical
 */
#define DSMIL_CONSTANT_RATE \
    __attribute__((dsmil_constant_rate))

/**
 * @brief Suppress timing jitter for predictable execution
 *
 * Minimizes timing variance by:
 * - Disabling dynamic frequency scaling hints
 * - Pinning to specific CPU cores
 * - Avoiding cache-timing variations
 *
 * Example:
 * @code
 * DSMIL_JITTER_SUPPRESS
 * DSMIL_STEALTH
 * void stealth_communication(void) {
 *     // Predictable timing, low variance
 *     send_covert_packet();
 * }
 * @endcode
 */
#define DSMIL_JITTER_SUPPRESS \
    __attribute__((dsmil_jitter_suppress))

/**
 * @brief Mark network I/O for fingerprint reduction
 *
 * Network I/O is transformed to reduce detectability:
 * - Batch operations to avoid patterns
 * - Add controlled delays to mask activity
 * - Normalize packet sizes/timing
 *
 * Example:
 * @code
 * DSMIL_NETWORK_STEALTH
 * void send_status_update(const char *msg) {
 *     // I/O batched and delayed to reduce fingerprint
 *     network_send(msg);
 * }
 * @endcode
 */
#define DSMIL_NETWORK_STEALTH \
    __attribute__((dsmil_network_stealth))

/** @} */

/**
 * @defgroup DSMIL_BLUE_RED Blue vs Red Testing Attributes (v1.4)
 * @{
 */

/**
 * @brief Mark function as red team test instrumentation point
 *
 * Red build functions include extra instrumentation to simulate adversarial
 * scenarios and test system defenses. Red builds are NEVER deployed to
 * production and must be confined to isolated test environments.
 *
 * The compiler automatically defines DSMIL_RED_BUILD macro when building
 * with -fdsmil-role=red flag.
 *
 * Example:
 * @code
 * DSMIL_RED_TEAM_HOOK("injection_point")
 * void process_user_input(const char *input) {
 *     #ifdef DSMIL_RED_BUILD
 *         // Red build: log potential attack vector
 *         dsmil_red_log("input_processing", "param=input");
 *
 *         // Simulate bypassing validation
 *         if (dsmil_red_scenario("bypass_validation")) {
 *             raw_process(input);  // Vulnerable path
 *             return;
 *         }
 *     #endif
 *
 *     // Normal path (blue build and red build)
 *     validate_and_process(input);
 * }
 * @endcode
 *
 * @warning RED BUILDS MUST NEVER BE DEPLOYED TO PRODUCTION
 * @warning Red builds signed with separate key, runtime rejects them
 * @note Use for adversarial testing and stress-testing only
 */
#define DSMIL_RED_TEAM_HOOK(hook_name) \
    __attribute__((dsmil_red_team_hook(hook_name)))

/**
 * @brief Mark function as attack surface (exposed to untrusted input)
 *
 * Attack surface functions are analyzed by Layer 8 Security AI in red builds
 * to identify potential vulnerabilities and blast radius.
 *
 * Example:
 * @code
 * DSMIL_ATTACK_SURFACE
 * void handle_network_packet(const uint8_t *packet, size_t len) {
 *     // Red build: map attack surface
 *     // Blue build: normal execution
 *     parse_packet(packet, len);
 * }
 * @endcode
 */
#define DSMIL_ATTACK_SURFACE \
    __attribute__((dsmil_attack_surface))

/**
 * @brief Mark vulnerability injection point for testing defenses
 * @param vuln_type Type of vulnerability to simulate
 *
 * Vulnerability injection points allow testing defense mechanisms against
 * specific attack classes. Only active in red builds.
 *
 * Common vulnerability types:
 * - "buffer_overflow": Buffer overflow simulation
 * - "use_after_free": Use-after-free simulation
 * - "race_condition": Race condition injection
 * - "injection": SQL/command injection point
 * - "auth_bypass": Authentication bypass simulation
 *
 * Example:
 * @code
 * DSMIL_VULN_INJECT("buffer_overflow")
 * void copy_user_data(char *dest, const char *src, size_t len) {
 *     #ifdef DSMIL_RED_BUILD
 *         if (dsmil_red_scenario("trigger_overflow")) {
 *             // Simulate overflow for testing
 *             memcpy(dest, src, len + 100);  // Intentional overflow
 *             return;
 *         }
 *     #endif
 *
 *     // Normal path: safe copy
 *     memcpy(dest, src, len);
 * }
 * @endcode
 *
 * @warning FOR TESTING ONLY - Never enable in production
 */
#define DSMIL_VULN_INJECT(vuln_type) \
    __attribute__((dsmil_vuln_inject(vuln_type)))

/**
 * @brief Mark function for blast radius analysis
 *
 * Functions marked for blast radius analysis are tracked in red builds
 * to determine impact of compromise. Layer 5/9 AI models campaign-level
 * effects of multi-binary compromise.
 *
 * Example:
 * @code
 * DSMIL_BLAST_RADIUS
 * DSMIL_LAYER(8)
 * void critical_security_function(void) {
 *     // If compromised, what's the blast radius?
 *     // L5/L9 AI analyzes cascading effects
 * }
 * @endcode
 */
#define DSMIL_BLAST_RADIUS \
    __attribute__((dsmil_blast_radius))

/**
 * @brief Specify build role (blue or red)
 * @param role Build role: "blue" (defender) or "red" (attacker)
 *
 * Applied at translation unit level to control build flavor.
 *
 * Example:
 * @code
 * DSMIL_BUILD_ROLE("blue")
 * int main(int argc, char **argv) {
 *     // Blue build: production configuration
 *     return run_production();
 * }
 * @endcode
 */
#define DSMIL_BUILD_ROLE(role) \
    __attribute__((dsmil_build_role(role)))

/** @} */

/**
 * @defgroup DSMIL_CLASSIFICATION Cross-Domain & Classification (v1.5)
 * @{
 */

/**
 * @brief Assign classification level to function or data
 * @param level Classification level: "U", "C", "S", "TS", "TS/SCI"
 *
 * Classification levels enforce cross-domain security policies. Functions
 * at different classification levels cannot call each other unless mediated
 * by an approved cross-domain gateway.
 *
 * Standard DoD classification levels:
 * - "U": UNCLASSIFIED
 * - "C": CONFIDENTIAL
 * - "S": SECRET (e.g., SIPRNET)
 * - "TS": TOP SECRET (e.g., JWICS)
 * - "TS/SCI": TOP SECRET / Sensitive Compartmented Information
 *
 * Example:
 * @code
 * DSMIL_CLASSIFICATION("S")
 * DSMIL_LAYER(7)
 * void process_secret_intel(const uint8_t *data, size_t len) {
 *     // SECRET classification
 *     // Cannot call CONFIDENTIAL or UNCLASS functions directly
 *     analyze_intelligence(data, len);
 * }
 * @endcode
 *
 * @warning Cross-domain calls require DSMIL_CROSS_DOMAIN_GATEWAY
 * @note Compile-time error if unsafe cross-domain call detected
 * @note Classification metadata embedded in provenance
 */
#define DSMIL_CLASSIFICATION(level) \
    __attribute__((dsmil_classification(level)))

/**
 * @brief Mark function as cross-domain gateway mediator
 * @param from_level Source classification level
 * @param to_level Destination classification level
 *
 * Cross-domain gateways mediate data flow between different classification
 * levels. Gateways must implement approved sanitization, filtering, or
 * manual review procedures.
 *
 * Common transitions:
 * - "S" → "C": SECRET to CONFIDENTIAL downgrade
 * - "C" → "U": CONFIDENTIAL to UNCLASSIFIED release
 * - "TS" → "S": TOP SECRET to SECRET downgrade
 *
 * Example:
 * @code
 * DSMIL_CROSS_DOMAIN_GATEWAY("S", "C")
 * DSMIL_GUARD_APPROVED
 * int sanitize_and_downgrade(const uint8_t *secret_data, size_t len,
 *                             uint8_t *confidential_output, size_t *out_len) {
 *     // Implement sanitization logic
 *     // Apply guard policy (manual review, automated filtering, etc.)
 *     return dsmil_cross_domain_guard(secret_data, len, "S", "C", "manual_review");
 * }
 * @endcode
 *
 * @warning Gateways must be approved by security authority
 * @warning All transitions logged to Layer 62 (Forensics)
 * @note Replaces simple DSMIL_GATEWAY for classification-aware systems
 */
#define DSMIL_CROSS_DOMAIN_GATEWAY(from_level, to_level) \
    __attribute__((dsmil_cross_domain_gateway(from_level, to_level)))

/**
 * @brief Mark function as approved cross-domain guard routine
 *
 * Guard routines implement sanitization, filtering, or review procedures
 * for cross-domain data transfers. Must be approved by security authority.
 *
 * Example:
 * @code
 * DSMIL_GUARD_APPROVED
 * DSMIL_LAYER(8)  // Security AI layer
 * int automated_sanitization_guard(const void *input, size_t len, void *output) {
 *     // AI-assisted sanitization and filtering
 *     // Layer 8 Security AI validates safety of downgrade
 *     return sanitize_for_lower_classification(input, len, output);
 * }
 * @endcode
 */
#define DSMIL_GUARD_APPROVED \
    __attribute__((dsmil_guard_approved))

/**
 * @brief Mark data as requiring cross-domain audit trail
 *
 * All accesses to this data are logged to Layer 62 (Forensics) for
 * cross-domain compliance auditing.
 *
 * Example:
 * @code
 * DSMIL_CROSS_DOMAIN_AUDIT
 * DSMIL_CLASSIFICATION("TS")
 * struct intelligence_report {
 *     char source[256];
 *     uint8_t data[4096];
 *     uint64_t timestamp;
 * } top_secret_report;
 * @endcode
 */
#define DSMIL_CROSS_DOMAIN_AUDIT \
    __attribute__((dsmil_cross_domain_audit))

/** @} */

/**
 * @defgroup DSMIL_JADC2 JADC2 & 5G/Edge Integration (v1.5)
 * @{
 */

/**
 * @brief Assign function to JADC2 operational profile
 * @param profile_name JADC2 profile identifier
 *
 * JADC2 (Joint All-Domain Command & Control) profiles define operational
 * context for multi-domain operations. Functions are optimized for 5G/MEC
 * deployment with low latency and high reliability.
 *
 * Standard JADC2 profiles:
 * - "sensor_fusion": Multi-sensor data aggregation
 * - "c2_processing": Command & control decision-making
 * - "targeting": Automated targeting coordination
 * - "situational_awareness": Real-time SA dashboard
 *
 * Example:
 * @code
 * DSMIL_JADC2_PROFILE("sensor_fusion")
 * DSMIL_LATENCY_BUDGET(5)  // 5ms JADC2 requirement
 * DSMIL_LAYER(7)
 * void fuse_sensor_data(const sensor_input_t *inputs, size_t count,
 *                        fusion_output_t *output) {
 *     // Optimized for 5G/MEC deployment
 *     // Low-latency sensor→C2→shooter pipeline
 *     aggregate_and_correlate(inputs, count, output);
 * }
 * @endcode
 *
 * @note Layer 5 AI optimizes for 5G latency/bandwidth constraints
 * @note Mission profile must enable JADC2 integration
 */
#define DSMIL_JADC2_PROFILE(profile_name) \
    __attribute__((dsmil_jadc2_profile(profile_name)))

/**
 * @brief Mark function for 5G Multi-Access Edge Computing (MEC) deployment
 *
 * 5G MEC functions are optimized for edge nodes with 99.999% reliability,
 * 5ms latency, and 10Gbps throughput. Compiler selects low-latency code
 * paths and power-efficient back-ends.
 *
 * Example:
 * @code
 * DSMIL_5G_EDGE
 * DSMIL_JADC2_PROFILE("c2_processing")
 * DSMIL_LATENCY_BUDGET(5)
 * void edge_decision_loop(void) {
 *     // Runs on 5G MEC node
 *     // Low-latency, high-reliability requirements
 *     process_sensor_data();
 *     make_c2_decision();
 *     send_shooter_command();
 * }
 * @endcode
 *
 * @note Layer 5/6 AI manages MEC node allocation
 * @note Automatic offload suggestions for latency-sensitive kernels
 */
#define DSMIL_5G_EDGE \
    __attribute__((dsmil_5g_edge))

/**
 * @brief Specify JADC2 data transport priority
 * @param priority Priority level (0-255, higher = more urgent)
 *
 * JADC2 transport layer prioritizes messages for sensor→C2→shooter pipeline.
 * High-priority messages (e.g., targeting data) bypass lower-priority traffic.
 *
 * Priority levels:
 * - 0-63: Routine (SA updates, status reports)
 * - 64-127: Priority (sensor fusion, C2 decisions)
 * - 128-191: Immediate (targeting, threat detection)
 * - 192-255: Flash (time-critical shooter commands)
 *
 * Example:
 * @code
 * DSMIL_JADC2_TRANSPORT(200)  // Flash priority for targeting
 * void send_targeting_solution(const target_t *target) {
 *     // High-priority JADC2 message
 *     dsmil_jadc2_send(target, sizeof(*target), 200, "air");
 * }
 * @endcode
 */
#define DSMIL_JADC2_TRANSPORT(priority) \
    __attribute__((dsmil_jadc2_transport(priority)))

/**
 * @brief Specify 5G latency budget in milliseconds
 * @param ms Latency budget in milliseconds
 *
 * Latency budgets enforce 5G JADC2 requirements (typically 5ms end-to-end).
 * Compiler performs static analysis; functions exceeding budget are rejected
 * or refactored by Layer 5 AI.
 *
 * Example:
 * @code
 * DSMIL_LATENCY_BUDGET(5)
 * DSMIL_5G_EDGE
 * void time_critical_function(void) {
 *     // Must complete in ≤5ms
 *     // Compiler optimizes for low latency
 *     fast_operation();
 * }
 * @endcode
 *
 * @warning Compile-time error if static analysis predicts budget violation
 * @note Layer 5 AI provides refactoring suggestions
 */
#define DSMIL_LATENCY_BUDGET(ms) \
    __attribute__((dsmil_latency_budget(ms)))

/**
 * @brief Specify bandwidth contract in Gbps
 * @param gbps Bandwidth limit in Gbps
 *
 * Bandwidth contracts enforce 5G throughput limits (typically 10Gbps).
 * Compiler estimates message sizes; violations trigger warnings.
 *
 * Example:
 * @code
 * DSMIL_BANDWIDTH_CONTRACT(10)
 * void stream_video_feed(const uint8_t *frames, size_t count) {
 *     // Must stay within 10Gbps bandwidth
 *     compress_and_send(frames, count);
 * }
 * @endcode
 */
#define DSMIL_BANDWIDTH_CONTRACT(gbps) \
    __attribute__((dsmil_bandwidth_contract(gbps)))

/**
 * @brief Mark function for Blue Force Tracker (BFT) integration
 * @param update_type Type of BFT update: "position", "status", "friendly"
 *
 * BFT integration automatically instruments position-reporting functions
 * with BFT API calls for real-time friendly force tracking.
 *
 * Update types:
 * - "position": GPS position updates
 * - "status": Unit status (fuel, ammo, readiness)
 * - "friendly": Friend/foe identification
 *
 * Example:
 * @code
 * DSMIL_BFT_HOOK("position")
 * DSMIL_BFT_AUTHORIZED
 * void report_position(double lat, double lon, double alt) {
 *     // Compiler inserts BFT API call
 *     dsmil_bft_send_position(lat, lon, alt, dsmil_timestamp_ns());
 * }
 * @endcode
 *
 * @note BFT data encrypted with AES-256
 * @note Layer 8 Security AI validates BFT authenticity
 */
#define DSMIL_BFT_HOOK(update_type) \
    __attribute__((dsmil_bft_hook(update_type)))

/**
 * @brief Mark function authorized to broadcast BFT data
 *
 * Only authorized functions can send BFT updates to prevent spoofing.
 * Authorization based on clearance and mission profile.
 *
 * Example:
 * @code
 * DSMIL_BFT_AUTHORIZED
 * DSMIL_CLASSIFICATION("S")
 * DSMIL_CLEARANCE(0x07000000)
 * void authorized_bft_sender(void) {
 *     // Can send BFT updates
 * }
 * @endcode
 */
#define DSMIL_BFT_AUTHORIZED \
    __attribute__((dsmil_bft_authorized))

/**
 * @brief Mark function for electromagnetic emission control (EMCON)
 * @param level EMCON level (1-4, higher = more restrictive)
 *
 * EMCON mode reduces RF emissions for operations in contested spectrum.
 * Compiler suppresses telemetry and minimizes transmissions.
 *
 * EMCON levels:
 * - 1: Normal operations
 * - 2: Reduced emissions (minimize non-essential transmissions)
 * - 3: Low signature (batch and delay all transmissions)
 * - 4: RF silent (no transmissions except emergency)
 *
 * Example:
 * @code
 * DSMIL_EMCON_MODE(3)
 * DSMIL_LOW_SIGNATURE("aggressive")
 * void covert_transmission(const uint8_t *data, size_t len) {
 *     // Low RF signature, batched transmission
 *     dsmil_emcon_send(data, len);
 * }
 * @endcode
 *
 * @note Integrates with v1.4 stealth modes
 * @note Layer 8 Security AI triggers EMCON escalation
 */
#define DSMIL_EMCON_MODE(level) \
    __attribute__((dsmil_emcon_mode(level)))

/**
 * @brief Specify BLOS (Beyond Line-of-Sight) fallback transports
 * @param primary Primary transport: "5g", "link16", "satcom", "muos"
 * @param secondary Fallback transport
 *
 * BLOS fallback enables resilient communications when primary link jammed.
 * Compiler generates alternate code paths for high-latency SATCOM links.
 *
 * Example:
 * @code
 * DSMIL_BLOS_FALLBACK("5g", "satcom")
 * void resilient_send(const uint8_t *msg, size_t len) {
 *     // Try 5G first, fallback to SATCOM if jammed
 *     if (!dsmil_5g_edge_available()) {
 *         dsmil_resilient_send(msg, len);  // Auto-fallback
 *     }
 * }
 * @endcode
 *
 * @note Layer 8 Security AI detects jamming
 * @note Latency compensation for SATCOM (100-500ms)
 */
#define DSMIL_BLOS_FALLBACK(primary, secondary) \
    __attribute__((dsmil_blos_fallback(primary, secondary)))

/**
 * @brief Specify tactical radio protocol
 * @param protocol Radio protocol: "link16", "satcom", "muos", "sincgars", "eplrs"
 *
 * Radio protocol specification generates appropriate framing, error correction,
 * and encryption for military tactical networks.
 *
 * Example:
 * @code
 * DSMIL_RADIO_PROFILE("link16")
 * void send_j_series_message(const link16_msg_t *msg) {
 *     // Compiler inserts Link-16 J-series framing
 *     send_tactical_message(msg);
 * }
 * @endcode
 */
#define DSMIL_RADIO_PROFILE(protocol) \
    __attribute__((dsmil_radio_profile(protocol)))

/**
 * @brief Mark function as multi-protocol radio bridge
 *
 * Bridge functions unify multiple tactical radio protocols (like TraX).
 * Compiler generates protocol-specific adapters.
 *
 * Example:
 * @code
 * DSMIL_RADIO_BRIDGE
 * int unified_send(const void *msg, size_t len, const char *protocol) {
 *     // Bridges Link-16, SATCOM, MUOS, etc.
 *     return protocol_specific_send(protocol, msg, len);
 * }
 * @endcode
 */
#define DSMIL_RADIO_BRIDGE \
    __attribute__((dsmil_radio_bridge))

/**
 * @brief Mark function for edge trusted execution zone
 *
 * Edge trusted zones run on hardened MEC nodes with enhanced security:
 * - Constant-time enforcement
 * - Memory safety instrumentation
 * - Tamper detection
 *
 * Example:
 * @code
 * DSMIL_EDGE_TRUSTED_ZONE
 * DSMIL_5G_EDGE
 * DSMIL_SECRET
 * void process_classified_data(const uint8_t *data, size_t len) {
 *     // Runs in secure edge enclave
 *     // Enhanced security checks
 * }
 * @endcode
 */
#define DSMIL_EDGE_TRUSTED_ZONE \
    __attribute__((dsmil_edge_trusted_zone))

/**
 * @brief Enable edge intrusion hardening
 *
 * Edge intrusion hardening instruments code with runtime monitors and
 * tamper-response routines for detecting physical/cyber intrusion.
 *
 * Example:
 * @code
 * DSMIL_EDGE_HARDEN
 * DSMIL_EDGE_TRUSTED_ZONE
 * void critical_edge_function(void) {
 *     // Runtime monitors active
 *     // Tamper detection enabled
 * }
 * @endcode
 */
#define DSMIL_EDGE_HARDEN \
    __attribute__((dsmil_edge_harden))

/**
 * @brief Mark function for sensor fusion aggregation
 *
 * Sensor fusion functions aggregate multi-sensor data (radar, EO/IR, SIGINT,
 * cyber) for JADC2 situational awareness.
 *
 * Example:
 * @code
 * DSMIL_SENSOR_FUSION
 * DSMIL_JADC2_PROFILE("sensor_fusion")
 * DSMIL_LATENCY_BUDGET(5)
 * void fuse_multi_sensor(const sensor_input_t *inputs, size_t count) {
 *     // Aggregate radar, EO/IR, SIGINT
 *     // Layer 9 Campaign AI coordinates fusion
 * }
 * @endcode
 *
 * @note Layer 9 Campaign AI manages sensor prioritization
 * @note All fusion decisions logged (Layer 62 Forensics)
 */
#define DSMIL_SENSOR_FUSION \
    __attribute__((dsmil_sensor_fusion))

/**
 * @brief Mark function as AI-assisted auto-targeting hook
 *
 * Auto-targeting functions coordinate sensor→C2→shooter pipeline for
 * automated target engagement. Must enforce ROE and human-in-loop.
 *
 * Example:
 * @code
 * DSMIL_AUTOTARGET
 * DSMIL_JADC2_TRANSPORT(200)  // Flash priority
 * DSMIL_ROE("LIVE_CONTROL")
 * void autotarget_engage(const target_t *target, float confidence) {
 *     // AI-assisted targeting
 *     // ROE compliance required
 *     // Human verification for lethal engagement
 *     if (confidence > 0.95 && roe_check(target)) {
 *         send_targeting_solution(target);
 *     }
 * }
 * @endcode
 *
 * @warning All targeting decisions logged to Layer 62 (Forensics)
 * @warning Human-in-loop verification required for lethal decisions
 */
#define DSMIL_AUTOTARGET \
    __attribute__((dsmil_autotarget))

/** @} */

/**
 * @defgroup DSMIL_MPE_NUCLEAR Mission Partner & Nuclear Surety (v1.6)
 * @{
 */

/**
 * @brief Mark code for Mission Partner Environment (MPE) release
 * @param partner_id Coalition partner identifier (e.g., "NATO", "FVEY", "AUS")
 *
 * MPE partner code is safe for release to allied networks. Must not call
 * U.S.-only functions without cross-domain gateway.
 *
 * Example:
 * @code
 * DSMIL_MPE_PARTNER("NATO")
 * DSMIL_RELEASABILITY("REL NATO")
 * void coalition_sharable_function(void) {
 *     // Safe for NATO partners
 * }
 * @endcode
 */
#define DSMIL_MPE_PARTNER(partner_id) \
    __attribute__((dsmil_mpe_partner(partner_id)))

/**
 * @brief Mark code as U.S.-only (not releasable to coalition)
 *
 * U.S.-only code cannot be called from MPE partner functions.
 *
 * Example:
 * @code
 * DSMIL_US_ONLY
 * DSMIL_CLASSIFICATION("TS")
 * void us_only_intelligence(void) {
 *     // Not releasable to coalition
 * }
 * @endcode
 */
#define DSMIL_US_ONLY \
    __attribute__((dsmil_us_only))

/**
 * @brief Specify releasability marking
 * @param marking Releasability (e.g., "REL NATO", "REL FVEY", "NOFORN")
 *
 * Example:
 * @code
 * DSMIL_RELEASABILITY("REL FVEY")
 * DSMIL_CLASSIFICATION("S")
 * void five_eyes_function(void) {
 *     // Releasable to Five Eyes partners
 * }
 * @endcode
 */
#define DSMIL_RELEASABILITY(marking) \
    __attribute__((dsmil_releasability(marking)))

/**
 * @brief Require two-person integrity control
 *
 * Two-person integrity (2PI) requires two independent approvals before
 * execution. Used for nuclear surety and critical operations.
 *
 * Example:
 * @code
 * DSMIL_TWO_PERSON
 * DSMIL_NC3_ISOLATED
 * DSMIL_APPROVAL_AUTHORITY("officer1")
 * DSMIL_APPROVAL_AUTHORITY("officer2")
 * void arm_weapon_system(void) {
 *     // Requires two ML-DSA-87 signatures
 *     // Nuclear surety compliance
 * }
 * @endcode
 *
 * @warning Compile-time error if 2PI function calls unauthorized code
 * @warning All executions logged to tamper-proof audit trail
 */
#define DSMIL_TWO_PERSON \
    __attribute__((dsmil_two_person))

/**
 * @brief Mark function for nuclear command & control (NC3) isolation
 *
 * NC3 functions cannot call network APIs or untrusted code. Enforced
 * at compile time for nuclear surety.
 *
 * Example:
 * @code
 * DSMIL_NC3_ISOLATED
 * DSMIL_TWO_PERSON
 * void nuclear_authorization_sequence(void) {
 *     // No network calls allowed
 *     // No untrusted code execution
 * }
 * @endcode
 */
#define DSMIL_NC3_ISOLATED \
    __attribute__((dsmil_nc3_isolated))

/**
 * @brief Specify approval authority for 2PI
 * @param key_id ML-DSA-87 key identifier
 *
 * Example:
 * @code
 * DSMIL_APPROVAL_AUTHORITY("launch_officer_1")
 * void authorize_with_key1(void) {
 *     // Provides one half of 2PI
 * }
 * @endcode
 */
#define DSMIL_APPROVAL_AUTHORITY(key_id) \
    __attribute__((dsmil_approval_authority(key_id)))

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

/**
 * @brief Mark function for telemetry export (v1.7+)
 * @param format Export format: "prometheus", "otel", "json"
 *
 * Functions marked with telemetry export automatically export metrics
 * to configured observability backends (Prometheus, OpenTelemetry, ELK).
 *
 * Example:
 * @code
 * DSMIL_TELEMETRY_EXPORT("prometheus")
 * DSMIL_MISSION_CRITICAL
 * void critical_function(void) {
 *     // Automatically exports:
 *     // - Function call count
 *     // - Execution time histogram
 *     // - Error rate
 *     // - Resource usage
 * }
 * @endcode
 *
 * @note Requires runtime telemetry collector (dsmil-telemetry-collector)
 * @note Integrates with Feature 1.3 telemetry enforcement
 */
#define DSMIL_TELEMETRY_EXPORT(format) \
    __attribute__((dsmil_telemetry_export(format)))

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
 * @brief Enhanced LLM worker with JADC2 integration
 */
#define DSMIL_LLM_WORKER_JADC2 \
    DSMIL_LLM_WORKER_MAIN \
    DSMIL_JADC2_PROFILE("c2_processing") \
    DSMIL_5G_EDGE \
    DSMIL_LATENCY_BUDGET(5)

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
#define DSMIL_TELEMETRY_WORKER \
    DSMIL_LAYER(5) \
    DSMIL_DEVICE(50) \
    DSMIL_STAGE("serve") \
    DSMIL_ROE("ANALYSIS_ONLY")

/**
 * @brief High-assurance crypto worker with nuclear surety
 */
#define DSMIL_CRYPTO_NUCLEAR \
    DSMIL_CRYPTO_WORKER \
    DSMIL_TWO_PERSON \
    DSMIL_NC3_ISOLATED \
    DSMIL_SECRET

/**
 * @brief Coalition-sharable function (MPE)
 */
#define DSMIL_MPE_SHARABLE(partner) \
    DSMIL_MPE_PARTNER(partner) \
    DSMIL_RELEASABILITY("REL " partner)

/**
 * @brief Edge-deployed function with security hardening
 */
#define DSMIL_EDGE_SECURE \
    DSMIL_5G_EDGE \
    DSMIL_EDGE_TRUSTED_ZONE \
    DSMIL_EDGE_HARDEN \
    DSMIL_SECRET

/**
 * @brief Covert operations function with maximum stealth
 */
#define DSMIL_COVERT_OPS \
    DSMIL_LOW_SIGNATURE("aggressive") \
    DSMIL_CONSTANT_RATE \
    DSMIL_JITTER_SUPPRESS \
    DSMIL_NETWORK_STEALTH \
    DSMIL_EMCON_MODE(3)

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

/**
 * @defgroup DSMIL_PATH_MACROS Path Resolution Macros
 * @{
 */

/**
 * @brief Macro to get DSMIL prefix at compile time (if available)
 *
 * Evaluates to DSMIL_PREFIX environment variable if set during compilation,
 * otherwise falls back to runtime resolution via dsmil_get_prefix().
 *
 * @note Include dsmil_paths.h for runtime path resolution
 */
#ifdef DSMIL_PREFIX
#define DSMIL_PREFIX_PATH DSMIL_PREFIX
#else
#define DSMIL_PREFIX_PATH NULL  /* Use runtime resolution */
#endif

/**
 * @brief Macro helper for path construction
 *
 * Example:
 * @code
 * #include <dsmil_paths.h>
 * char config_path[PATH_MAX];
 * snprintf(config_path, sizeof(config_path), "%s/mission-profiles.json", dsmil_get_config_dir());
 * @endcode
 */
#define DSMIL_CONFIG_PATH(filename) \
    (dsmil_get_config_dir() ? \
     (dsmil_get_config_dir() + "/" + filename) : \
     ("/etc/dsmil/" + filename))

/** @} */

#endif /* DSMIL_ATTRIBUTES_H */

/**
 * @file dsmil_layer8_security.h
 * @brief Layer 8 (ENHANCED_SEC) Security AI Runtime
 * 
 * Provides runtime support for Layer 8 Security AI operations:
 * - Adversarial ML defense (~188 TOPS INT8)
 * - Threat detection and anomaly analysis
 * - Security validation and risk scoring
 * - Side-channel vulnerability detection
 * - Zero-trust security enforcement
 * - PQC algorithm enforcement (via Device 255)
 * 
 * Layer 8 Devices: 53-62 (Security AI devices), Device 80 (Primary Security AI)
 * 
 * Version: 1.0.0
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#ifndef DSMIL_LAYER8_SECURITY_H
#define DSMIL_LAYER8_SECURITY_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup DSMIL_LAYER8 Layer 8 Security AI
 * @{
 */

/**
 * @brief Security threat types
 */
typedef enum {
    DSMIL_THREAT_MALWARE,
    DSMIL_THREAT_ADVERSARIAL_INPUT,
    DSMIL_THREAT_SIDE_CHANNEL,
    DSMIL_THREAT_DATA_EXFILTRATION,
    DSMIL_THREAT_UNAUTHORIZED_ACCESS,
    DSMIL_THREAT_CRYPTO_VIOLATION,
    DSMIL_THREAT_ANOMALY
} dsmil_threat_type_t;

/**
 * @brief Security risk score (0.0-1.0)
 */
typedef struct {
    float overall_risk;           // Overall risk score (0.0-1.0)
    float threat_probability;     // Probability of threat (0.0-1.0)
    float impact_score;          // Impact if threat materializes (0.0-1.0)
    dsmil_threat_type_t threat_type;
    uint32_t confidence;          // Confidence level (0-100)
    const char *threat_description;
} dsmil_security_risk_t;

/**
 * @brief Security AI context
 */
typedef struct {
    uint32_t device_id;           // Device 80 (Primary Security AI)
    uint8_t layer;                // 8
    uint64_t memory_budget_bytes; // 8 GB max
    uint64_t memory_used_bytes;
    float tops_capacity;          // 188 TOPS INT8
    float tops_utilization;      // Current utilization (0.0-1.0)
    uint64_t threats_detected;
    uint64_t anomalies_analyzed;
} dsmil_layer8_security_ctx_t;

/**
 * @brief Initialize Layer 8 Security AI runtime
 * 
 * @param ctx Output security context
 * @return 0 on success, negative on error
 */
int dsmil_layer8_security_init(dsmil_layer8_security_ctx_t *ctx);

/**
 * @brief Analyze binary for security vulnerabilities
 * 
 * Uses Security AI models to detect:
 * - Side-channel vulnerabilities
 * - Buffer overflows
 * - Use-after-free
 * - Cryptographic weaknesses
 * - Adversarial input vulnerabilities
 * 
 * @param binary_path Path to binary
 * @param risk Output risk score
 * @return 0 on success, negative on error
 */
int dsmil_layer8_analyze_binary(const char *binary_path, dsmil_security_risk_t *risk);

/**
 * @brief Detect adversarial inputs
 * 
 * Uses adversarial ML defense models to detect:
 * - Adversarial examples
 * - Input manipulation attacks
 * - Model evasion attempts
 * 
 * @param input_data Input data to analyze
 * @param input_size Input data size
 * @param model_id Target model ID
 * @param risk Output risk score
 * @return 0 on success, negative on error
 */
int dsmil_layer8_detect_adversarial(const void *input_data, size_t input_size,
                                    uint32_t model_id, dsmil_security_risk_t *risk);

/**
 * @brief Analyze function for side-channel vulnerabilities
 * 
 * Validates constant-time execution and detects:
 * - Timing side-channels
 * - Cache side-channels
 * - Power side-channels
 * - Branch prediction leaks
 * 
 * @param function_name Function name to analyze
 * @param binary_path Path to binary containing function
 * @param risk Output risk score
 * @return 0 on success, negative on error
 */
int dsmil_layer8_analyze_side_channel(const char *function_name,
                                     const char *binary_path,
                                     dsmil_security_risk_t *risk);

/**
 * @brief Detect anomalies in system behavior
 * 
 * Uses anomaly detection models to identify:
 * - Unusual network traffic
 * - Abnormal resource usage
 * - Suspicious process behavior
 * - Data exfiltration patterns
 * 
 * @param behavior_data Behavior metrics/data
 * @param data_size Data size
 * @param risk Output risk score
 * @return 0 on success, negative on error
 */
int dsmil_layer8_detect_anomaly(const void *behavior_data, size_t data_size,
                                dsmil_security_risk_t *risk);

/**
 * @brief Validate cryptographic implementation
 * 
 * Ensures:
 * - PQC-only mode (Layer 8 requirement)
 * - Constant-time execution
 * - Proper key management
 * - TPM attestation
 * 
 * @param crypto_function_name Cryptographic function name
 * @param binary_path Path to binary
 * @param risk Output risk score
 * @return 0 on success, negative on error
 */
int dsmil_layer8_validate_crypto(const char *crypto_function_name,
                                 const char *binary_path,
                                 dsmil_security_risk_t *risk);

/**
 * @brief Get security posture summary
 * 
 * @param ctx Security context
 * @param total_threats Output total threats detected
 * @param total_anomalies Output total anomalies analyzed
 * @param avg_risk_score Output average risk score
 * @return 0 on success, negative on error
 */
int dsmil_layer8_get_security_posture(const dsmil_layer8_security_ctx_t *ctx,
                                      uint64_t *total_threats,
                                      uint64_t *total_anomalies,
                                      float *avg_risk_score);

/**
 * @brief Enable zero-trust security mode
 * 
 * Enforces:
 * - All operations require verification
 * - No implicit trust
 * - Continuous validation
 * - PQC-only crypto
 * 
 * @param ctx Security context
 * @return 0 on success, negative on error
 */
int dsmil_layer8_enable_zero_trust(dsmil_layer8_security_ctx_t *ctx);

/** @} */

#ifdef __cplusplus
}
#endif

#endif /* DSMIL_LAYER8_SECURITY_H */

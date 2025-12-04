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
 * @brief Layer 8 device types (51-58)
 */
typedef enum {
    DSMIL_L8_DEVICE51_SECURITY_FRAMEWORK = 51,  // 15 TOPS - Anomaly detection, behavioral analytics
    DSMIL_L8_DEVICE52_ADVERSARIAL_DEFENSE = 52, // 30 TOPS - Adversarial training, robustness testing
    DSMIL_L8_DEVICE53_CYBERSECURITY_AI = 53,     // 25 TOPS - Threat intelligence, attack prediction
    DSMIL_L8_DEVICE54_THREAT_INTELLIGENCE = 54, // 25 TOPS - IOC extraction, attribution analysis
    DSMIL_L8_DEVICE55_AUTOMATED_RESPONSE = 55,  // 20 TOPS - Incident response automation
    DSMIL_L8_DEVICE56_POST_QUANTUM_CRYPTO = 56, // 20 TOPS - PQC algorithm optimization
    DSMIL_L8_DEVICE57_AUTONOMOUS_OPS = 57,      // 28 TOPS - Self-healing systems, adaptive defense
    DSMIL_L8_DEVICE58_SECURITY_ANALYTICS = 58   // 25 TOPS - Security event correlation, forensics
} dsmil_layer8_device_t;

/**
 * @brief Security AI context
 */
typedef struct {
    uint32_t device_id;           // Device 51-58 (8 devices)
    uint8_t layer;                // 8
    uint64_t memory_budget_bytes; // 8 GB max
    uint64_t memory_used_bytes;
    float tops_capacity;          // Device-specific TOPS (15-30 TOPS)
    float tops_total_capacity;   // 188 TOPS INT8 total for Layer 8
    float tops_utilization;      // Current utilization (0.0-1.0)
    uint64_t threats_detected;
    uint64_t anomalies_analyzed;
    uint32_t model_size_params;   // 50-300M parameters typical
    float detection_accuracy;    // >99% known threats, >95% zero-day
} dsmil_layer8_security_ctx_t;

/**
 * @brief Initialize Layer 8 Security AI runtime
 * 
 * @param device_id Device ID (51-58)
 * @param ctx Output security context
 * @return 0 on success, negative on error
 */
int dsmil_layer8_security_init(dsmil_layer8_device_t device_id,
                                dsmil_layer8_security_ctx_t *ctx);

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
 * Uses Device 52 (Adversarial ML Defense) for robustness testing.
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
 * @brief Extract threat intelligence indicators (IOC extraction)
 * 
 * Uses Device 54 (Threat Intelligence) for:
 * - IOC (Indicators of Compromise) extraction
 * - Attribution analysis using graph neural networks
 * - NLP-based threat intelligence processing
 * 
 * @param threat_data Raw threat data
 * @param data_size Data size
 * @param iocs Output extracted IOCs
 * @param ioc_count Output IOC count
 * @return 0 on success, negative on error
 */
int dsmil_layer8_extract_iocs(const void *threat_data, size_t data_size,
                              void *iocs, uint32_t *ioc_count);

/**
 * @brief Automated security incident response
 * 
 * Uses Device 55 (Automated Security Response) with RL-based automation:
 * - Incident classification
 * - Automated containment
 * - Response orchestration
 * 
 * @param incident_data Incident data
 * @param incident_size Incident data size
 * @param response_actions Output response actions
 * @param action_count Output action count
 * @return 0 on success, negative on error
 */
int dsmil_layer8_automated_response(const void *incident_data, size_t incident_size,
                                    void *response_actions, uint32_t *action_count);

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

/**
 * @brief Train adversarial defense model
 * 
 * Uses Device 52 (Adversarial ML Defense) with GANs for:
 * - Adversarial training
 * - Robustness testing
 * - Model hardening against adversarial attacks
 * 
 * @param model_path Path to model to harden
 * @param adversarial_samples Adversarial training samples
 * @param num_samples Number of adversarial samples
 * @param hardened_model_path Output path for hardened model
 * @return 0 on success, negative on error
 */
int dsmil_layer8_train_adversarial_defense(const char *model_path,
                                           const void *adversarial_samples,
                                           uint32_t num_samples,
                                           const char *hardened_model_path);

/**
 * @brief Correlate security events using Graph Neural Networks
 * 
 * Uses Device 58 (Security Analytics) with GNN for:
 * - Security event correlation
 * - Attack pattern detection
 * - Forensics analysis
 * 
 * @param events Security event data
 * @param num_events Number of events
 * @param correlation_graph Output correlation graph
 * @param graph_size Graph buffer size / actual length
 * @return 0 on success, negative on error
 */
int dsmil_layer8_correlate_security_events(const void *events, uint32_t num_events,
                                           void *correlation_graph, size_t *graph_size);

/**
 * @brief Predict zero-day attacks
 * 
 * Uses Device 53 (Cybersecurity AI) for:
 * - Zero-day attack prediction
 * - Attack pattern recognition
 * - Threat forecasting
 * 
 * @param threat_indicators Threat indicators
 * @param num_indicators Number of indicators
 * @param prediction Output attack prediction
 * @param confidence Output prediction confidence (0.0-1.0)
 * @return 0 on success, negative on error
 */
int dsmil_layer8_predict_zero_day(const void *threat_indicators, uint32_t num_indicators,
                                  void *prediction, float *confidence);

/**
 * @brief Analyze behavioral patterns using LSTM/GRU
 * 
 * Uses Device 51 (Enhanced Security Framework) for:
 * - Temporal pattern analysis
 * - Behavioral anomaly detection
 * - User/entity behavior analytics
 * 
 * @param behavior_data Time-series behavior data
 * @param data_size Data size
 * @param time_window Time window in seconds
 * @param anomaly_score Output anomaly score (0.0-1.0)
 * @return 0 on success, negative on error
 */
int dsmil_layer8_analyze_behavioral_patterns(const void *behavior_data, size_t data_size,
                                            uint32_t time_window, float *anomaly_score);

/**
 * @brief Optimize PQC algorithms using ML
 * 
 * Uses Device 56 (Post-Quantum Crypto) for:
 * - ML-KEM-1024 optimization
 * - ML-DSA-87 optimization
 * - PQC performance tuning
 * 
 * @param pqc_algorithm PQC algorithm ID (ML-KEM-1024, ML-DSA-87)
 * @param optimization_params Output optimization parameters
 * @param params_size Parameters buffer size / actual length
 * @return 0 on success, negative on error
 */
int dsmil_layer8_optimize_pqc(uint16_t pqc_algorithm,
                              void *optimization_params, size_t *params_size);

/** @} */

#ifdef __cplusplus
}
#endif

#endif /* DSMIL_LAYER8_SECURITY_H */

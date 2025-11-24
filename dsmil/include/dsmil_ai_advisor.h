/**
 * @file dsmil_ai_advisor.h
 * @brief DSMIL AI Advisor Runtime Interface
 *
 * Provides runtime support for AI-assisted compilation using DSMIL Layers 3-9.
 * Includes structures for advisor requests/responses and helper functions.
 *
 * Version: 1.0
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#ifndef DSMIL_AI_ADVISOR_H
#define DSMIL_AI_ADVISOR_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup DSMIL_AI_CONSTANTS Constants
 * @{
 */

/** Maximum string lengths */
#define DSMIL_AI_MAX_STRING          256
#define DSMIL_AI_MAX_FUNCTIONS       1024
#define DSMIL_AI_MAX_SUGGESTIONS     512
#define DSMIL_AI_MAX_WARNINGS        128

/** Schema versions */
#define DSMIL_AI_REQUEST_SCHEMA      "dsmilai-request-v1"
#define DSMIL_AI_RESPONSE_SCHEMA     "dsmilai-response-v1"

/** Default configuration */
#define DSMIL_AI_DEFAULT_TIMEOUT_MS  5000
#define DSMIL_AI_DEFAULT_CONFIDENCE  0.75
#define DSMIL_AI_MAX_RETRIES         2

/** @} */

/**
 * @defgroup DSMIL_AI_ENUMS Enumerations
 * @{
 */

/** AI integration modes */
typedef enum {
    DSMIL_AI_MODE_OFF = 0,      /**< No AI; deterministic only */
    DSMIL_AI_MODE_LOCAL = 1,    /**< Embedded ML models only */
    DSMIL_AI_MODE_ADVISOR = 2,  /**< External advisors + validation */
    DSMIL_AI_MODE_LAB = 3,      /**< Permissive; auto-apply suggestions */
} dsmil_ai_mode_t;

/** Advisor types */
typedef enum {
    DSMIL_ADVISOR_L7_LLM = 0,       /**< Layer 7 LLM for code analysis */
    DSMIL_ADVISOR_L8_SECURITY = 1,  /**< Layer 8 security AI */
    DSMIL_ADVISOR_L5_PERF = 2,      /**< Layer 5/6 performance forecasting */
} dsmil_advisor_type_t;

/** Request priority */
typedef enum {
    DSMIL_PRIORITY_LOW = 0,
    DSMIL_PRIORITY_NORMAL = 1,
    DSMIL_PRIORITY_HIGH = 2,
} dsmil_priority_t;

/** Suggestion verdict */
typedef enum {
    DSMIL_VERDICT_APPLIED = 0,      /**< Suggestion applied */
    DSMIL_VERDICT_REJECTED = 1,     /**< Failed validation */
    DSMIL_VERDICT_PENDING = 2,      /**< Awaiting verification */
    DSMIL_VERDICT_SKIPPED = 3,      /**< Low confidence */
} dsmil_verdict_t;

/** Result codes */
typedef enum {
    DSMIL_AI_OK = 0,
    DSMIL_AI_ERROR_NETWORK = 1,
    DSMIL_AI_ERROR_TIMEOUT = 2,
    DSMIL_AI_ERROR_INVALID_RESPONSE = 3,
    DSMIL_AI_ERROR_SERVICE_UNAVAILABLE = 4,
    DSMIL_AI_ERROR_QUOTA_EXCEEDED = 5,
    DSMIL_AI_ERROR_MODEL_LOAD_FAILED = 6,
} dsmil_ai_result_t;

/** @} */

/**
 * @defgroup DSMIL_AI_STRUCTS Data Structures
 * @{
 */

/** Build configuration */
typedef struct {
    dsmil_ai_mode_t mode;                   /**< AI integration mode */
    char policy[64];                        /**< Policy (production/development/lab) */
    char optimization_level[16];            /**< -O0, -O3, etc. */
} dsmil_build_config_t;

/** Build goals */
typedef struct {
    uint32_t latency_target_ms;             /**< Target latency in ms */
    uint32_t power_budget_w;                /**< Power budget in watts */
    char security_posture[32];              /**< low/medium/high */
    float accuracy_target;                  /**< 0.0-1.0 */
} dsmil_build_goals_t;

/** IR function summary */
typedef struct {
    char name[DSMIL_AI_MAX_STRING];         /**< Function name */
    char mangled_name[DSMIL_AI_MAX_STRING]; /**< Mangled name */
    char location[DSMIL_AI_MAX_STRING];     /**< Source location */
    uint32_t basic_blocks;                  /**< BB count */
    uint32_t instructions;                  /**< Instruction count */
    uint32_t loops;                         /**< Loop count */
    uint32_t max_loop_depth;                /**< Maximum nesting */
    uint32_t memory_loads;                  /**< Load count */
    uint32_t memory_stores;                 /**< Store count */
    uint64_t estimated_bytes;               /**< Memory footprint estimate */
    bool auto_vectorized;                   /**< Was vectorized */
    uint32_t vector_width;                  /**< Vector width in bits */
    uint32_t cyclomatic_complexity;         /**< Complexity metric */

    // Existing DSMIL metadata (may be null)
    int32_t dsmil_layer;                    /**< -1 if unset */
    int32_t dsmil_device;                   /**< -1 if unset */
    char dsmil_stage[64];                   /**< Empty if unset */
    uint32_t dsmil_clearance;               /**< 0 if unset */
} dsmil_ir_function_t;

/** Module summary */
typedef struct {
    char name[DSMIL_AI_MAX_STRING];         /**< Module name */
    char path[DSMIL_AI_MAX_STRING];         /**< Source path */
    uint8_t hash_sha384[48];                /**< SHA-384 hash */
    uint32_t source_lines;                  /**< Line count */
    uint32_t num_functions;                 /**< Function count */
    uint32_t num_globals;                   /**< Global count */

    dsmil_ir_function_t *functions;         /**< Function array */
    // globals, call_graph, data_flow omitted for brevity
} dsmil_module_summary_t;

/** AI advisor request */
typedef struct {
    char schema[64];                        /**< Schema version */
    char request_id[128];                   /**< UUID */
    dsmil_advisor_type_t advisor_type;      /**< Advisor type */
    dsmil_priority_t priority;              /**< Request priority */

    dsmil_build_config_t build_config;      /**< Build configuration */
    dsmil_build_goals_t goals;              /**< Optimization goals */
    dsmil_module_summary_t module;          /**< IR summary */

    char project_type[128];                 /**< Project context */
    char deployment_target[128];            /**< Deployment target */
} dsmil_ai_request_t;

/** Attribute suggestion */
typedef struct {
    char name[64];                          /**< Attribute name (e.g., "dsmil_layer") */
    char value_str[DSMIL_AI_MAX_STRING];    /**< String value */
    int64_t value_int;                      /**< Integer value */
    bool value_bool;                        /**< Boolean value */
    float confidence;                       /**< 0.0-1.0 */
    char rationale[512];                    /**< Explanation */
} dsmil_attribute_suggestion_t;

/** Function annotation suggestion */
typedef struct {
    char target[DSMIL_AI_MAX_STRING];       /**< Target function/global */
    dsmil_attribute_suggestion_t *attributes; /**< Attribute array */
    uint32_t num_attributes;                /**< Attribute count */
} dsmil_annotation_suggestion_t;

/** Security hint */
typedef struct {
    char target[DSMIL_AI_MAX_STRING];       /**< Target element */
    char severity[16];                      /**< low/medium/high/critical */
    float confidence;                       /**< 0.0-1.0 */
    char finding[512];                      /**< Issue description */
    char recommendation[512];               /**< Suggested fix */
    char cwe[32];                           /**< CWE identifier */
    float cvss_score;                       /**< CVSS 3.1 score */
} dsmil_security_hint_t;

/** Performance hint */
typedef struct {
    char target[DSMIL_AI_MAX_STRING];       /**< Target function */
    char hint_type[64];                     /**< device_offload/vectorize/inline */
    float confidence;                       /**< 0.0-1.0 */
    char description[512];                  /**< Explanation */
    float expected_speedup;                 /**< Predicted speedup multiplier */
    float power_impact_w;                   /**< Power impact in watts */
} dsmil_performance_hint_t;

/** AI advisor response */
typedef struct {
    char schema[64];                        /**< Schema version */
    char request_id[128];                   /**< Matching request UUID */
    dsmil_advisor_type_t advisor_type;      /**< Advisor type */
    char model_name[128];                   /**< Model used */
    char model_version[64];                 /**< Model version */
    uint32_t device;                        /**< DSMIL device used */
    uint32_t layer;                         /**< DSMIL layer */

    uint32_t processing_duration_ms;        /**< Processing time */
    float inference_cost_tops;              /**< Compute cost in TOPS */

    // Suggestions
    dsmil_annotation_suggestion_t *annotations; /**< Annotation suggestions */
    uint32_t num_annotations;

    dsmil_security_hint_t *security_hints;  /**< Security findings */
    uint32_t num_security_hints;

    dsmil_performance_hint_t *perf_hints;   /**< Performance hints */
    uint32_t num_perf_hints;

    // Diagnostics
    char **warnings;                        /**< Warning messages */
    uint32_t num_warnings;
    char **info;                            /**< Info messages */
    uint32_t num_info;

    // Metadata
    uint8_t model_hash_sha384[48];          /**< Model hash */
    bool fallback_used;                     /**< Used fallback heuristics */
    bool cached_response;                   /**< Response from cache */
} dsmil_ai_response_t;

/** AI advisor configuration */
typedef struct {
    dsmil_ai_mode_t mode;                   /**< Integration mode */

    // Service endpoints
    char l7_llm_url[DSMIL_AI_MAX_STRING];   /**< L7 LLM service URL */
    char l8_security_url[DSMIL_AI_MAX_STRING]; /**< L8 security service URL */
    char l5_perf_url[DSMIL_AI_MAX_STRING];  /**< L5 perf service URL */

    // Local models
    char cost_model_path[DSMIL_AI_MAX_STRING]; /**< Path to ONNX cost model */
    char security_model_path[DSMIL_AI_MAX_STRING]; /**< Path to security model */

    // Thresholds
    float confidence_threshold;             /**< Min confidence (default 0.75) */
    uint32_t timeout_ms;                    /**< Request timeout */
    uint32_t max_retries;                   /**< Retry attempts */

    // Rate limiting
    uint32_t max_requests_per_build;        /**< Max requests */
    uint32_t max_requests_per_second;       /**< Rate limit */

    // Logging
    char audit_log_path[DSMIL_AI_MAX_STRING]; /**< Audit log file */
    bool verbose;                           /**< Verbose logging */
} dsmil_ai_config_t;

/** @} */

/**
 * @defgroup DSMIL_AI_API API Functions
 * @{
 */

/**
 * @brief Initialize AI advisor system
 *
 * @param[in] config Configuration (or NULL for defaults)
 * @return Result code
 */
dsmil_ai_result_t dsmil_ai_init(const dsmil_ai_config_t *config);

/**
 * @brief Shutdown AI advisor system
 */
void dsmil_ai_shutdown(void);

/**
 * @brief Get current configuration
 *
 * @param[out] config Output configuration
 * @return Result code
 */
dsmil_ai_result_t dsmil_ai_get_config(dsmil_ai_config_t *config);

/**
 * @brief Submit advisor request
 *
 * @param[in] request Request structure
 * @param[out] response Response structure (caller must free)
 * @return Result code
 */
dsmil_ai_result_t dsmil_ai_submit_request(
    const dsmil_ai_request_t *request,
    dsmil_ai_response_t **response);

/**
 * @brief Submit request asynchronously
 *
 * @param[in] request Request structure
 * @param[out] request_id Output request ID
 * @return Result code
 */
dsmil_ai_result_t dsmil_ai_submit_async(
    const dsmil_ai_request_t *request,
    char *request_id);

/**
 * @brief Poll for async response
 *
 * @param[in] request_id Request ID
 * @param[out] response Response structure (NULL if not ready)
 * @return Result code
 */
dsmil_ai_result_t dsmil_ai_poll_response(
    const char *request_id,
    dsmil_ai_response_t **response);

/**
 * @brief Free response structure
 *
 * @param[in] response Response to free
 */
void dsmil_ai_free_response(dsmil_ai_response_t *response);

/**
 * @brief Export request to JSON file
 *
 * @param[in] request Request structure
 * @param[in] json_path Output file path
 * @return Result code
 */
dsmil_ai_result_t dsmil_ai_export_request_json(
    const dsmil_ai_request_t *request,
    const char *json_path);

/**
 * @brief Import response from JSON file
 *
 * @param[in] json_path Input file path
 * @param[out] response Parsed response (caller must free)
 * @return Result code
 */
dsmil_ai_result_t dsmil_ai_import_response_json(
    const char *json_path,
    dsmil_ai_response_t **response);

/**
 * @brief Validate suggestion against DSMIL constraints
 *
 * @param[in] suggestion Attribute suggestion
 * @param[in] context Module/function context
 * @param[out] verdict Validation verdict
 * @return Result code
 */
dsmil_ai_result_t dsmil_ai_validate_suggestion(
    const dsmil_attribute_suggestion_t *suggestion,
    const void *context,
    dsmil_verdict_t *verdict);

/**
 * @brief Convert result code to string
 *
 * @param[in] result Result code
 * @return Human-readable string
 */
const char *dsmil_ai_result_str(dsmil_ai_result_t result);

/** @} */

/**
 * @defgroup DSMIL_AI_COSTMODEL Cost Model API
 * @{
 */

/** Cost model handle (opaque) */
typedef struct dsmil_cost_model dsmil_cost_model_t;

/**
 * @brief Load ONNX cost model
 *
 * @param[in] onnx_path Path to ONNX file
 * @param[out] model Output model handle
 * @return Result code
 */
dsmil_ai_result_t dsmil_ai_load_cost_model(
    const char *onnx_path,
    dsmil_cost_model_t **model);

/**
 * @brief Unload cost model
 *
 * @param[in] model Model handle
 */
void dsmil_ai_unload_cost_model(dsmil_cost_model_t *model);

/**
 * @brief Run cost model inference
 *
 * @param[in] model Model handle
 * @param[in] features Input feature vector (256 floats)
 * @param[out] predictions Output predictions (N floats)
 * @param[in] num_predictions Size of predictions array
 * @return Result code
 */
dsmil_ai_result_t dsmil_ai_cost_model_infer(
    dsmil_cost_model_t *model,
    const float *features,
    float *predictions,
    uint32_t num_predictions);

/**
 * @brief Get model metadata
 *
 * @param[in] model Model handle
 * @param[out] name Output model name
 * @param[out] version Output model version
 * @param[out] hash_sha384 Output model hash
 * @return Result code
 */
dsmil_ai_result_t dsmil_ai_cost_model_metadata(
    dsmil_cost_model_t *model,
    char *name,
    char *version,
    uint8_t hash_sha384[48]);

/** @} */

/**
 * @defgroup DSMIL_AI_UTIL Utility Functions
 * @{
 */

/**
 * @brief Get AI integration mode from environment
 *
 * Checks DSMIL_AI_MODE environment variable.
 *
 * @param[in] default_mode Default if not set
 * @return AI mode
 */
dsmil_ai_mode_t dsmil_ai_get_mode_from_env(dsmil_ai_mode_t default_mode);

/**
 * @brief Load configuration from file
 *
 * @param[in] config_path Path to config file (TOML)
 * @param[out] config Output configuration
 * @return Result code
 */
dsmil_ai_result_t dsmil_ai_load_config_file(
    const char *config_path,
    dsmil_ai_config_t *config);

/**
 * @brief Generate unique request ID
 *
 * @param[out] request_id Output buffer (min 128 bytes)
 */
void dsmil_ai_generate_request_id(char *request_id);

/**
 * @brief Log audit event
 *
 * @param[in] request_id Request ID
 * @param[in] event_type Event type string
 * @param[in] details JSON details
 * @return Result code
 */
dsmil_ai_result_t dsmil_ai_log_audit(
    const char *request_id,
    const char *event_type,
    const char *details);

/**
 * @brief Check if advisor service is available
 *
 * @param[in] advisor_type Advisor type
 * @param[in] timeout_ms Timeout
 * @return true if available, false otherwise
 */
bool dsmil_ai_service_available(
    dsmil_advisor_type_t advisor_type,
    uint32_t timeout_ms);

/** @} */

/**
 * @defgroup DSMIL_AI_MACROS Convenience Macros
 * @{
 */

/**
 * @brief Check if AI mode enables external advisors
 */
#define DSMIL_AI_USES_EXTERNAL(mode) \
    ((mode) == DSMIL_AI_MODE_ADVISOR || (mode) == DSMIL_AI_MODE_LAB)

/**
 * @brief Check if AI mode uses embedded models
 */
#define DSMIL_AI_USES_LOCAL(mode) \
    ((mode) != DSMIL_AI_MODE_OFF)

/**
 * @brief Check if suggestion meets confidence threshold
 */
#define DSMIL_AI_MEETS_THRESHOLD(suggestion, config) \
    ((suggestion)->confidence >= (config)->confidence_threshold)

/** @} */

#ifdef __cplusplus
}
#endif

#endif /* DSMIL_AI_ADVISOR_H */

/**
 * @file dsmil_layer9_executive.h
 * @brief Layer 9 (EXECUTIVE) Strategic Command AI Runtime
 * 
 * Provides runtime support for Layer 9 Executive Command operations:
 * - Strategic planning and decision support (~330 TOPS INT8)
 * - Nuclear Command & Control (NC3) integration
 * - Coalition fusion and interoperability
 * - Executive-level intelligence synthesis
 * - Campaign-level mission planning
 * - Global resource orchestration
 * 
 * Layer 9 Devices: 59-60 (Executive Command), Device 90 (Strategic AI)
 * 
 * Version: 1.0.0
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#ifndef DSMIL_LAYER9_EXECUTIVE_H
#define DSMIL_LAYER9_EXECUTIVE_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup DSMIL_LAYER9 Layer 9 Executive Command
 * @{
 */

/**
 * @brief Mission priority levels
 */
typedef enum {
    DSMIL_PRIORITY_ROUTINE = 1,
    DSMIL_PRIORITY_IMPORTANT = 2,
    DSMIL_PRIORITY_URGENT = 3,
    DSMIL_PRIORITY_CRITICAL = 4,
    DSMIL_PRIORITY_NC3 = 5  // Nuclear Command & Control
} dsmil_mission_priority_t;

/**
 * @brief Coalition partner types
 */
typedef enum {
    DSMIL_COALITION_NATO,
    DSMIL_COALITION_FVEY,
    DSMIL_COALITION_BILATERAL,
    DSMIL_COALITION_UNILATERAL
} dsmil_coalition_type_t;

/**
 * @brief Strategic decision context
 */
typedef struct {
    uint32_t decision_id;
    dsmil_mission_priority_t priority;
    const char *mission_type;
    dsmil_coalition_type_t coalition_context;
    uint32_t clearance_level;  // Classification level
    bool nc3_critical;         // NC3-critical decision
    uint64_t timestamp_ns;
} dsmil_strategic_decision_t;

/**
 * @brief Layer 9 device types (59-62)
 */
typedef enum {
    DSMIL_L9_DEVICE59_EXECUTIVE_CMD = 59,  // 85 TOPS - Strategic decision support, crisis management
    DSMIL_L9_DEVICE60_COALITION_FUSION = 60, // 85 TOPS - Multi-national intelligence fusion
    DSMIL_L9_DEVICE61_NUCLEAR_CC = 61,     // 80 TOPS - NC3 analysis, strategic stability (ROE-governed)
    DSMIL_L9_DEVICE62_STRATEGIC_INTEL = 62  // 80 TOPS - Global threat assessment, strategic planning
} dsmil_layer9_device_t;

/**
 * @brief Executive AI context
 */
typedef struct {
    uint32_t device_id;           // Device 59-62 (4 devices)
    uint8_t layer;                // 9
    uint64_t memory_budget_bytes; // 12 GB max
    uint64_t memory_used_bytes;
    float tops_capacity;          // Device-specific TOPS (80-85 TOPS)
    float tops_total_capacity;   // 330 TOPS INT8 total for Layer 9
    float tops_utilization;      // Current utilization (0.0-1.0)
    uint64_t decisions_made;
    uint64_t campaigns_planned;
    bool nc3_enabled;             // NC3 enabled (Device 61 only)
    uint32_t model_size_params;   // 1B-7B parameters typical
    uint32_t context_window_tokens; // Up to 32K tokens
} dsmil_layer9_executive_ctx_t;

/**
 * @brief Initialize Layer 9 Executive Command runtime
 * 
 * @param device_id Device ID (59-62)
 * @param ctx Output executive context
 * @return 0 on success, negative on error
 */
int dsmil_layer9_executive_init(dsmil_layer9_device_t device_id,
                                 dsmil_layer9_executive_ctx_t *ctx);

/**
 * @brief Synthesize intelligence from lower layers
 * 
 * Aggregates intelligence from Layers 3-8 into strategic-level insights.
 * 
 * @param ctx Executive context
 * @param intelligence_summary Output intelligence summary
 * @param summary_size Summary buffer size / actual length
 * @return 0 on success, negative on error
 */
int dsmil_layer9_synthesize_intelligence(const dsmil_layer9_executive_ctx_t *ctx,
                                         void *intelligence_summary,
                                         size_t *summary_size);

/**
 * @brief Generate strategic decision recommendation
 * 
 * Uses Strategic AI models to recommend executive-level decisions.
 * 
 * @param ctx Executive context
 * @param decision_context Decision context
 * @param recommendation Output recommendation
 * @param rec_size Recommendation buffer size / actual length
 * @return 0 on success, negative on error
 */
int dsmil_layer9_generate_recommendation(const dsmil_layer9_executive_ctx_t *ctx,
                                         const dsmil_strategic_decision_t *decision_context,
                                         void *recommendation, size_t *rec_size);

/**
 * @brief Plan campaign-level mission
 * 
 * Creates comprehensive campaign plan integrating:
 * - Resource allocation
 * - Timeline and phases
 * - Coalition coordination
 * - Risk assessment
 * 
 * @param ctx Executive context
 * @param campaign_id Campaign identifier
 * @param mission_objectives Mission objectives
 * @param campaign_plan Output campaign plan
 * @param plan_size Plan buffer size / actual length
 * @return 0 on success, negative on error
 */
int dsmil_layer9_plan_campaign(const dsmil_layer9_executive_ctx_t *ctx,
                               const char *campaign_id,
                               const char *mission_objectives,
                               void *campaign_plan, size_t *plan_size);

/**
 * @brief Coordinate coalition operations
 * 
 * Manages coalition interoperability:
 * - Releasability markings (REL NATO, REL FVEY, NOFORN)
 * - Information sharing policies
 * - Joint operations coordination
 * 
 * @param ctx Executive context
 * @param coalition_type Coalition type
 * @param operation_id Operation identifier
 * @param coordination_data Output coordination data
 * @param data_size Coordination data buffer size / actual length
 * @return 0 on success, negative on error
 */
int dsmil_layer9_coordinate_coalition(const dsmil_layer9_executive_ctx_t *ctx,
                                      dsmil_coalition_type_t coalition_type,
                                      const char *operation_id,
                                      void *coordination_data, size_t *data_size);

/**
 * @brief Validate NC3 (Nuclear Command & Control) decision
 * 
 * Uses Device 61 (Nuclear C&C Integration) - ROE-governed per Rescindment 220330R NOV 25.
 * Ensures NC3-critical decisions meet:
 * - Two-person integrity requirements
 * - Proper authorization chain
 * - TPM attestation
 * - Audit trail
 * - Section 4.1c compliance (ANALYSIS ONLY, NO kinetic control)
 * 
 * @param ctx Executive context (must be Device 61)
 * @param decision_context Decision context
 * @param validation_result Output validation result
 * @return 0 if valid, negative if invalid
 */
int dsmil_layer9_validate_nc3(const dsmil_layer9_executive_ctx_t *ctx,
                              const dsmil_strategic_decision_t *decision_context,
                              bool *validation_result);

/**
 * @brief Global threat assessment
 * 
 * Uses Device 62 (Strategic Intelligence) for:
 * - Geopolitical modeling
 * - Risk forecasting
 * - Global threat analysis
 * 
 * @param ctx Executive context
 * @param threat_assessment Output threat assessment
 * @param assessment_size Assessment buffer size / actual length
 * @return 0 on success, negative on error
 */
int dsmil_layer9_assess_global_threats(const dsmil_layer9_executive_ctx_t *ctx,
                                       void *threat_assessment, size_t *assessment_size);

/**
 * @brief Get executive resource utilization
 * 
 * @param ctx Executive context
 * @param memory_used Output memory used (bytes)
 * @param tops_utilization Output TOPS utilization (0.0-1.0)
 * @param active_campaigns Output active campaigns count
 * @return 0 on success, negative on error
 */
int dsmil_layer9_get_utilization(const dsmil_layer9_executive_ctx_t *ctx,
                                 uint64_t *memory_used,
                                 float *tops_utilization,
                                 uint32_t *active_campaigns);

/** @} */

#ifdef __cplusplus
}
#endif

#endif /* DSMIL_LAYER9_EXECUTIVE_H */

/**
 * @file dsmil_intelligence_flow.h
 * @brief Cross-Layer Intelligence Flow & Orchestration
 * 
 * Implements upward intelligence flow pattern:
 * - Lower layers push intelligence upward
 * - Higher layers subscribe with clearance verification
 * - Event-driven architecture
 * - Security boundary enforcement
 * 
 * Version: 1.0.0
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#ifndef DSMIL_INTELLIGENCE_FLOW_H
#define DSMIL_INTELLIGENCE_FLOW_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup DSMIL_INTELLIGENCE Cross-Layer Intelligence Flow
 * @{
 */

/**
 * @brief Intelligence event types
 */
typedef enum {
    DSMIL_INTEL_RAW_DATA,        // Layer 3: Raw sensor/data feeds
    DSMIL_INTEL_DOMAIN_ANALYTICS, // Layer 3: Domain analytics
    DSMIL_INTEL_MISSION_PLANNING, // Layer 4: Mission planning
    DSMIL_INTEL_PREDICTIVE,      // Layer 5: Predictive analytics
    DSMIL_INTEL_NUCLEAR,         // Layer 6: Nuclear intelligence
    DSMIL_INTEL_AI_SYNTHESIS,    // Layer 7: AI synthesis (Device 47)
    DSMIL_INTEL_SECURITY,        // Layer 8: Security overlay
    DSMIL_INTEL_EXECUTIVE        // Layer 9: Executive command
} dsmil_intelligence_type_t;

/**
 * @brief Intelligence event structure
 */
typedef struct {
    uint8_t source_layer;
    uint32_t source_device;
    uint8_t target_layer;
    uint32_t target_device;
    dsmil_intelligence_type_t intel_type;
    uint32_t clearance_mask;
    void *payload;
    size_t payload_size;
    uint64_t timestamp_ns;
} dsmil_intelligence_event_t;

/**
 * @brief Event callback function type
 */
typedef void (*dsmil_intelligence_callback_t)(const dsmil_intelligence_event_t *event);

/**
 * @brief Initialize intelligence flow system
 * 
 * @return 0 on success, negative on error
 */
int dsmil_intelligence_flow_init(void);

/**
 * @brief Publish intelligence event (upward flow)
 * 
 * @param event Intelligence event
 * @return 0 on success, negative on error
 */
int dsmil_intelligence_publish(const dsmil_intelligence_event_t *event);

/**
 * @brief Subscribe to intelligence events (higher layers)
 * 
 * @param layer Target layer
 * @param device Target device
 * @param intel_type Intelligence type filter
 * @param callback Event callback function
 * @return 0 on success, negative on error
 */
int dsmil_intelligence_subscribe(uint8_t layer, uint32_t device,
                                  dsmil_intelligence_type_t intel_type,
                                  dsmil_intelligence_callback_t callback);

/**
 * @brief Verify clearance for cross-layer intelligence flow
 * 
 * @param source_layer Source layer
 * @param target_layer Target layer
 * @param clearance_mask Required clearance
 * @return true if authorized, false otherwise
 */
bool dsmil_intelligence_verify_clearance(uint8_t source_layer, uint8_t target_layer,
                                         uint32_t clearance_mask);

/**
 * @brief Shutdown intelligence flow system
 * 
 * @return 0 on success, negative on error
 */
int dsmil_intelligence_flow_shutdown(void);

/** @} */

#ifdef __cplusplus
}
#endif

#endif /* DSMIL_INTELLIGENCE_FLOW_H */

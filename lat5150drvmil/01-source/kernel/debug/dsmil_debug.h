/*
 * Dell MIL-SPEC Enhanced DSMIL Kernel Debug Interface Header
 * 
 * Copyright (C) 2025 JRTC1 Educational Development
 * 
 * This header provides the interface for comprehensive kernel-level
 * debugging and logging capabilities for the DSMIL system.
 */

#ifndef _DSMIL_DEBUG_H
#define _DSMIL_DEBUG_H

#include <linux/types.h>
#include <linux/time.h>
#include "dsmil_hal.h"

/* Debug system version */
#define DSMIL_DEBUG_VERSION_MAJOR    1
#define DSMIL_DEBUG_VERSION_MINOR    0
#define DSMIL_DEBUG_VERSION_PATCH    0
#define DSMIL_DEBUG_VERSION_STRING   "1.0.0"

/* Debug Log Levels */
enum dsmil_debug_level {
    DSMIL_DEBUG_LEVEL_NONE = 0,
    DSMIL_DEBUG_LEVEL_ERROR,
    DSMIL_DEBUG_LEVEL_WARN,
    DSMIL_DEBUG_LEVEL_INFO,
    DSMIL_DEBUG_LEVEL_DEBUG,
    DSMIL_DEBUG_LEVEL_TRACE
};

/* Debug Categories */
enum dsmil_debug_category {
    DSMIL_DEBUG_CAT_GENERAL = 0,
    DSMIL_DEBUG_CAT_HAL,
    DSMIL_DEBUG_CAT_SAFETY,
    DSMIL_DEBUG_CAT_ACCESS_CONTROL,
    DSMIL_DEBUG_CAT_RUST,
    DSMIL_DEBUG_CAT_DEVICE_OPS,
    DSMIL_DEBUG_CAT_QUARANTINE,
    DSMIL_DEBUG_CAT_EMERGENCY
};

#ifdef __cplusplus
extern "C" {
#endif

/* Core Debug Functions */
int dsmil_debug_init(void);
void dsmil_debug_cleanup(void);

/* Logging Functions */
void dsmil_debug_log(enum dsmil_debug_level level, enum dsmil_debug_category category,
                    u32 device_id, const char *format, ...);

/* Real-time Monitoring */
void dsmil_debug_log_device_operation(u32 device_id, enum dsmil_hal_access_type access_type,
                                     bool success, u32 value_or_error, 
                                     const char *operation);

/* System State Dumping */
int dsmil_debug_dump_system_state(char *buffer, size_t buffer_size);

/* Convenience Macros */
#define DSMIL_DEBUG_ERROR(dev_id, fmt, ...) \
    dsmil_debug_log(DSMIL_DEBUG_LEVEL_ERROR, DSMIL_DEBUG_CAT_GENERAL, \
                   dev_id, fmt, ##__VA_ARGS__)

#define DSMIL_DEBUG_WARN(dev_id, fmt, ...) \
    dsmil_debug_log(DSMIL_DEBUG_LEVEL_WARN, DSMIL_DEBUG_CAT_GENERAL, \
                   dev_id, fmt, ##__VA_ARGS__)

#define DSMIL_DEBUG_INFO(dev_id, fmt, ...) \
    dsmil_debug_log(DSMIL_DEBUG_LEVEL_INFO, DSMIL_DEBUG_CAT_GENERAL, \
                   dev_id, fmt, ##__VA_ARGS__)

#define DSMIL_DEBUG_HAL(dev_id, fmt, ...) \
    dsmil_debug_log(DSMIL_DEBUG_LEVEL_DEBUG, DSMIL_DEBUG_CAT_HAL, \
                   dev_id, fmt, ##__VA_ARGS__)

#define DSMIL_DEBUG_SAFETY(dev_id, fmt, ...) \
    dsmil_debug_log(DSMIL_DEBUG_LEVEL_DEBUG, DSMIL_DEBUG_CAT_SAFETY, \
                   dev_id, fmt, ##__VA_ARGS__)

#define DSMIL_DEBUG_QUARANTINE(dev_id, fmt, ...) \
    dsmil_debug_log(DSMIL_DEBUG_LEVEL_ERROR, DSMIL_DEBUG_CAT_QUARANTINE, \
                   dev_id, fmt, ##__VA_ARGS__)

#define DSMIL_DEBUG_EMERGENCY(dev_id, fmt, ...) \
    dsmil_debug_log(DSMIL_DEBUG_LEVEL_ERROR, DSMIL_DEBUG_CAT_EMERGENCY, \
                   dev_id, fmt, ##__VA_ARGS__)

#ifdef __cplusplus
}
#endif

#endif /* _DSMIL_DEBUG_H */
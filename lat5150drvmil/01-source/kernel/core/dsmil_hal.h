/*
 * Dell MIL-SPEC Enhanced DSMIL Hardware Abstraction Layer
 * 
 * Copyright (C) 2025 JRTC1 Educational Development
 * 
 * This header provides a comprehensive hardware abstraction layer
 * for safe access to 84 DSMIL devices with quarantine protection.
 * 
 * CRITICAL SAFETY: This HAL enforces absolute quarantine protection
 * for 5 critical devices that must NEVER be written to.
 */

#ifndef _DSMIL_HAL_H
#define _DSMIL_HAL_H

#include <linux/types.h>
#include <linux/mutex.h>
#include <linux/io.h>
#include <linux/device.h>
#include <linux/thermal.h>

/* HAL Version Information */
#define DSMIL_HAL_VERSION_MAJOR    1
#define DSMIL_HAL_VERSION_MINOR    0
#define DSMIL_HAL_VERSION_PATCH    0
#define DSMIL_HAL_VERSION_STRING   "1.0.0"

/* Forward declaration from enhanced kernel module */
struct dsmil_device_enhanced;
struct dsmil_group_enhanced;
struct dsmil_enhanced_state;

/* HAL Operation Result Codes */
enum dsmil_hal_result {
    DSMIL_HAL_SUCCESS = 0,
    DSMIL_HAL_ERROR_INVALID_DEVICE = -1,
    DSMIL_HAL_ERROR_QUARANTINE_VIOLATION = -2,
    DSMIL_HAL_ERROR_SAFETY_VIOLATION = -3,
    DSMIL_HAL_ERROR_PERMISSION_DENIED = -4,
    DSMIL_HAL_ERROR_DEVICE_BUSY = -5,
    DSMIL_HAL_ERROR_DEVICE_OFFLINE = -6,
    DSMIL_HAL_ERROR_TIMEOUT = -7,
    DSMIL_HAL_ERROR_IO_ERROR = -8,
    DSMIL_HAL_ERROR_CRC_MISMATCH = -9,
    DSMIL_HAL_ERROR_THERMAL_VIOLATION = -10,
    DSMIL_HAL_ERROR_EMERGENCY_STOP = -11,
    DSMIL_HAL_ERROR_RUST_SAFETY = -12,
    DSMIL_HAL_ERROR_MEMORY_MAP = -13,
    DSMIL_HAL_ERROR_AUTH_REQUIRED = -14,
    DSMIL_HAL_ERROR_UNKNOWN = -99
};

/* HAL Device Access Types */
enum dsmil_hal_access_type {
    DSMIL_HAL_ACCESS_READ_ONLY = 0,
    DSMIL_HAL_ACCESS_WRITE_SAFE,        /* Write to non-critical registers */
    DSMIL_HAL_ACCESS_WRITE_CONTROLLED,  /* Write with authorization */
    DSMIL_HAL_ACCESS_WRITE_CRITICAL,    /* Critical write (requires elevated auth) */
    DSMIL_HAL_ACCESS_DIAGNOSTIC,        /* Diagnostic access */
    DSMIL_HAL_ACCESS_MAINTENANCE        /* Maintenance mode access */
};

/* HAL Register Types */
enum dsmil_hal_register_type {
    DSMIL_HAL_REG_CONTROL = 0,     /* Primary control register */
    DSMIL_HAL_REG_STATUS,          /* Status register */
    DSMIL_HAL_REG_CONFIG,          /* Configuration register */
    DSMIL_HAL_REG_DATA,            /* Data register */
    DSMIL_HAL_REG_INTERRUPT,       /* Interrupt control */
    DSMIL_HAL_REG_ERROR,           /* Error status */
    DSMIL_HAL_REG_DEBUG,           /* Debug information */
    DSMIL_HAL_REG_SAFETY,          /* Safety validation */
    DSMIL_HAL_REG_THERMAL,         /* Thermal status */
    DSMIL_HAL_REG_POWER            /* Power management */
};

/* HAL Operation Context */
struct dsmil_hal_context {
    u32 operation_id;                   /* Unique operation identifier */
    enum dsmil_hal_access_type access_type; /* Type of access requested */
    u32 device_id;                      /* Target device ID (0-83) */
    u32 timeout_ms;                     /* Operation timeout */
    bool bypass_safety;                 /* DANGEROUS: bypass safety (never set) */
    bool requires_auth;                 /* Requires authorization */
    char auth_token[64];                /* Authorization token */
    void *user_context;                 /* User-defined context */
    ktime_t start_time;                 /* Operation start time */
};

/* HAL Device Information Structure */
struct dsmil_hal_device_info {
    u32 device_id;                      /* Device ID (0-83) */
    u32 group_id;                       /* Group ID (0-6) */
    u32 device_index;                   /* Index within group (0-11) */
    char function_name[64];             /* Device function */
    char safety_reason[128];            /* Safety classification reason */
    
    enum dsmil_safety_level safety_level; /* Safety classification */
    bool is_quarantined;                /* Quarantine status */
    bool read_enabled;                  /* Read operations allowed */
    bool write_enabled;                 /* Write operations allowed */
    bool requires_auth;                 /* Requires authorization */
    
    u32 access_count_read;              /* Read operation counter */
    u32 access_count_write;             /* Write operation counter */
    u32 safety_violations;              /* Safety violation counter */
    
    void __iomem *mmio_base;           /* Memory-mapped I/O base */
    resource_size_t mmio_size;          /* MMIO region size */
    
    u32 supported_registers;            /* Bitmask of supported registers */
    u32 register_offsets[10];           /* Register offset table */
    
    ktime_t last_access;                /* Last access timestamp */
    enum dsmil_device_state current_state; /* Current device state */
};

/* HAL Statistics Structure */
struct dsmil_hal_statistics {
    u64 total_operations;               /* Total HAL operations */
    u64 read_operations;                /* Total read operations */
    u64 write_operations;               /* Total write operations */
    u64 safety_violations;              /* Total safety violations */
    u64 quarantine_violations;          /* Quarantine violation attempts */
    u64 auth_failures;                  /* Authentication failures */
    u64 timeout_errors;                 /* Timeout errors */
    u64 crc_errors;                     /* CRC validation errors */
    u64 thermal_violations;             /* Thermal violations */
    u64 emergency_stops;                /* Emergency stop events */
    
    ktime_t last_reset;                 /* Last statistics reset */
    u32 active_operations;              /* Currently active operations */
    u32 peak_concurrent_ops;            /* Peak concurrent operations */
};

/* HAL Thermal Management */
struct dsmil_hal_thermal_info {
    s32 current_temp;                   /* Current temperature (milli-Celsius) */
    s32 critical_temp;                  /* Critical temperature threshold */
    s32 warning_temp;                   /* Warning temperature threshold */
    bool thermal_protection_active;     /* Thermal protection status */
    u32 thermal_events;                 /* Number of thermal events */
    ktime_t last_thermal_event;         /* Last thermal event timestamp */
};

/* HAL Configuration Structure */
struct dsmil_hal_config {
    bool quarantine_enforcement;        /* Enable quarantine enforcement */
    bool safety_monitoring;             /* Enable safety monitoring */
    bool thermal_protection;            /* Enable thermal protection */
    bool crc_validation;                /* Enable CRC validation */
    bool access_logging;                /* Enable access logging */
    bool debug_mode;                    /* Enable debug mode */
    
    u32 default_timeout_ms;             /* Default operation timeout */
    u32 max_concurrent_ops;             /* Maximum concurrent operations */
    u32 safety_check_interval;          /* Safety check interval (ms) */
    u32 thermal_check_interval;         /* Thermal check interval (ms) */
    
    s32 thermal_critical_temp;          /* Critical temperature (milli-C) */
    s32 thermal_warning_temp;           /* Warning temperature (milli-C) */
};

/* Safety Statistics Structure (for safety validation system) */
struct dsmil_safety_statistics {
    u64 total_checks;                   /* Total safety checks performed */
    u64 violations_detected;            /* Total violations detected */
    u64 quarantine_blocks;              /* Quarantine violation blocks */
    u64 auto_resolutions;               /* Automatic resolutions */
    u32 current_safety_level;           /* Current safety level (1-4) */
    bool emergency_mode;                /* Emergency mode active */
    bool monitoring_active;             /* Safety monitoring active */
    u32 violation_count;                /* Number of recorded violations */
    u64 uptime_seconds;                 /* Safety system uptime */
};

/* Function Prototypes */

/* HAL Initialization and Cleanup */
int dsmil_hal_init(struct dsmil_enhanced_state *state);
void dsmil_hal_cleanup(void);
int dsmil_hal_reset(void);

/* Device Information and Management */
int dsmil_hal_get_device_info(u32 device_id, struct dsmil_hal_device_info *info);
int dsmil_hal_get_device_count(void);
int dsmil_hal_get_group_count(void);
bool dsmil_hal_is_device_valid(u32 device_id);
bool dsmil_hal_is_device_quarantined(u32 device_id);
int dsmil_hal_get_thermal_info(struct dsmil_hal_thermal_info *info);

/* Safe Device Access Functions */
int dsmil_hal_read_register(u32 device_id, enum dsmil_hal_register_type reg_type, 
                           u32 *value, struct dsmil_hal_context *ctx);
int dsmil_hal_write_register(u32 device_id, enum dsmil_hal_register_type reg_type, 
                            u32 value, struct dsmil_hal_context *ctx);
int dsmil_hal_read_memory(u32 device_id, u32 offset, void *buffer, 
                         size_t size, struct dsmil_hal_context *ctx);
int dsmil_hal_write_memory(u32 device_id, u32 offset, const void *buffer, 
                          size_t size, struct dsmil_hal_context *ctx);

/* Bulk Operations */
int dsmil_hal_read_multiple_registers(u32 device_id, 
                                     enum dsmil_hal_register_type *reg_types,
                                     u32 *values, u32 count, 
                                     struct dsmil_hal_context *ctx);
int dsmil_hal_write_multiple_registers(u32 device_id, 
                                      enum dsmil_hal_register_type *reg_types,
                                      const u32 *values, u32 count, 
                                      struct dsmil_hal_context *ctx);

/* Safety and Validation Functions */
int dsmil_hal_validate_operation(u32 device_id, enum dsmil_hal_access_type access_type);
int dsmil_hal_check_quarantine_status(u32 device_id);
int dsmil_hal_perform_safety_check(u32 device_id);
int dsmil_hal_validate_crc(u32 device_id, u32 *calculated_crc);
int dsmil_hal_get_safety_level(u32 device_id, enum dsmil_safety_level *level);

/* Authorization and Authentication */
int dsmil_hal_check_authorization(struct dsmil_hal_context *ctx);
int dsmil_hal_generate_auth_token(u32 device_id, enum dsmil_hal_access_type access_type,
                                 char *token, size_t token_size);
int dsmil_hal_validate_auth_token(const char *token, u32 device_id, 
                                 enum dsmil_hal_access_type access_type);

/* Context Management */
int dsmil_hal_create_context(struct dsmil_hal_context *ctx, u32 device_id, 
                            enum dsmil_hal_access_type access_type);
void dsmil_hal_destroy_context(struct dsmil_hal_context *ctx);
int dsmil_hal_set_context_timeout(struct dsmil_hal_context *ctx, u32 timeout_ms);

/* Emergency and Error Handling */
int dsmil_hal_emergency_stop(u32 device_id);
int dsmil_hal_emergency_stop_all(void);
int dsmil_hal_reset_device(u32 device_id);
int dsmil_hal_get_last_error(u32 device_id);
const char *dsmil_hal_error_string(enum dsmil_hal_result result);

/* Statistics and Monitoring */
int dsmil_hal_get_statistics(struct dsmil_hal_statistics *stats);
int dsmil_hal_reset_statistics(void);
int dsmil_hal_get_thermal_info(struct dsmil_hal_thermal_info *thermal_info);
int dsmil_hal_get_device_statistics(u32 device_id, struct dsmil_hal_statistics *stats);

/* Configuration Management */
int dsmil_hal_get_config(struct dsmil_hal_config *config);
int dsmil_hal_set_config(const struct dsmil_hal_config *config);
int dsmil_hal_reset_config_to_defaults(void);

/* Debug and Diagnostics */
int dsmil_hal_dump_device_state(u32 device_id, char *buffer, size_t buffer_size);
int dsmil_hal_run_device_diagnostics(u32 device_id);
int dsmil_hal_get_debug_info(char *buffer, size_t buffer_size);

/* Rust Integration Functions (if enabled) */
#ifdef CONFIG_DSMIL_RUST_INTEGRATION
int dsmil_hal_rust_validate_memory(u32 device_id, void *ptr, size_t size);
int dsmil_hal_rust_safe_read(u32 device_id, u32 offset, u32 *value);
int dsmil_hal_rust_safe_write(u32 device_id, u32 offset, u32 value);
#endif

/* Utility Functions */
static inline bool dsmil_hal_is_valid_device_id(u32 device_id)
{
    return device_id < 84; /* 84 total devices (0-83) */
}

static inline bool dsmil_hal_is_valid_group_id(u32 group_id)
{
    return group_id < 7; /* 7 total groups (0-6) */
}

static inline u32 dsmil_hal_device_to_group(u32 device_id)
{
    if (!dsmil_hal_is_valid_device_id(device_id))
        return 0xFFFFFFFF; /* Invalid */
    return device_id / 12; /* 12 devices per group */
}

static inline u32 dsmil_hal_device_index_in_group(u32 device_id)
{
    if (!dsmil_hal_is_valid_device_id(device_id))
        return 0xFFFFFFFF; /* Invalid */
    return device_id % 12; /* Index within group */
}

static inline u32 dsmil_hal_group_device_to_global(u32 group_id, u32 device_index)
{
    if (!dsmil_hal_is_valid_group_id(group_id) || device_index >= 12)
        return 0xFFFFFFFF; /* Invalid */
    return (group_id * 12) + device_index;
}

/* HAL Constants */
#define DSMIL_HAL_MAX_DEVICES           84
#define DSMIL_HAL_MAX_GROUPS            7
#define DSMIL_HAL_DEVICES_PER_GROUP     12
#define DSMIL_HAL_QUARANTINE_COUNT      5
#define DSMIL_HAL_DEFAULT_TIMEOUT_MS    5000
#define DSMIL_HAL_MAX_CONCURRENT_OPS    16
#define DSMIL_HAL_AUTH_TOKEN_SIZE       64
#define DSMIL_HAL_MAX_BUFFER_SIZE       4096

/* HAL Magic Numbers */
#define DSMIL_HAL_MAGIC_INIT            0x48414C30  /* "HAL0" */
#define DSMIL_HAL_MAGIC_CONTEXT         0x43545830  /* "CTX0" */
#define DSMIL_HAL_MAGIC_DEVICE          0x44455630  /* "DEV0" */

/* Error Checking Macros */
#define DSMIL_HAL_CHECK_DEVICE_ID(dev_id) \
    do { \
        if (!dsmil_hal_is_valid_device_id(dev_id)) { \
            pr_err("DSMIL HAL: Invalid device ID %u\n", dev_id); \
            return DSMIL_HAL_ERROR_INVALID_DEVICE; \
        } \
    } while(0)

#define DSMIL_HAL_CHECK_QUARANTINE(dev_id) \
    do { \
        if (dsmil_hal_is_device_quarantined(dev_id)) { \
            pr_err("DSMIL HAL: Device %u is QUARANTINED\n", dev_id); \
            return DSMIL_HAL_ERROR_QUARANTINE_VIOLATION; \
        } \
    } while(0)

#define DSMIL_HAL_CHECK_NULL_PTR(ptr, name) \
    do { \
        if (!(ptr)) { \
            pr_err("DSMIL HAL: NULL pointer for %s\n", name); \
            return DSMIL_HAL_ERROR_INVALID_DEVICE; \
        } \
    } while(0)

#endif /* _DSMIL_HAL_H */

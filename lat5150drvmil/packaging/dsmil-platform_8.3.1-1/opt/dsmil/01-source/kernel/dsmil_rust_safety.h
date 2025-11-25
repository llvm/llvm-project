/*
 * Dell MIL-SPEC Enhanced DSMIL Rust Safety Layer Integration
 * 
 * Copyright (C) 2025 JRTC1 Educational Development
 * 
 * This header defines the C interface to the enhanced Rust safety layer
 * for memory protection and safe device access in the DSMIL system.
 * 
 * CRITICAL SAFETY: The Rust layer provides additional memory safety
 * and bounds checking beyond the C implementation to prevent corruption
 * and ensure quarantine enforcement.
 */

#ifndef _DSMIL_RUST_SAFETY_H
#define _DSMIL_RUST_SAFETY_H

#include <linux/types.h>
#include <linux/device.h>

/* Rust Safety Layer Version */
#define DSMIL_RUST_SAFETY_VERSION_MAJOR    2
#define DSMIL_RUST_SAFETY_VERSION_MINOR    0
#define DSMIL_RUST_SAFETY_VERSION_PATCH    0
#define DSMIL_RUST_SAFETY_VERSION_STRING   "2.0.0"

/* Rust Safety Configuration */
#define RUST_SAFETY_MAX_DEVICES     84
#define RUST_SAFETY_MAX_MEMORY      (420UL * 1024 * 1024)  /* 420MB for 84 devices */
#define RUST_SAFETY_QUARANTINE_COUNT 5

/* Rust Safety Result Codes */
enum rust_safety_result {
    RUST_SAFETY_SUCCESS = 0,
    RUST_SAFETY_ERROR_INVALID_DEVICE = -1,
    RUST_SAFETY_ERROR_QUARANTINE_VIOLATION = -2,
    RUST_SAFETY_ERROR_BOUNDS_CHECK_FAILED = -3,
    RUST_SAFETY_ERROR_MEMORY_CORRUPTION = -4,
    RUST_SAFETY_ERROR_NULL_POINTER = -5,
    RUST_SAFETY_ERROR_BUFFER_OVERFLOW = -6,
    RUST_SAFETY_ERROR_UNINITIALIZED = -7,
    RUST_SAFETY_ERROR_PANIC = -8,
    RUST_SAFETY_ERROR_RUST_UNAVAILABLE = -9
};

/* Rust Memory Protection Levels */
enum rust_memory_protection_level {
    RUST_PROTECTION_NONE = 0,
    RUST_PROTECTION_BASIC,
    RUST_PROTECTION_STANDARD,
    RUST_PROTECTION_ENHANCED,
    RUST_PROTECTION_MAXIMUM
};

/* Rust Safety Context */
struct rust_safety_context {
    u32 context_id;
    u32 device_id;
    void *memory_base;
    size_t memory_size;
    enum rust_memory_protection_level protection_level;
    bool quarantine_check_enabled;
    bool bounds_check_enabled;
    bool corruption_check_enabled;
    u32 magic_number;
    ktime_t created_time;
};

/* Rust Device Safety Information */
struct rust_device_safety_info {
    u32 device_id;
    bool is_quarantined;
    bool rust_protection_active;
    enum rust_memory_protection_level protection_level;
    u32 bounds_violations;
    u32 quarantine_violations;
    u32 memory_corruptions;
    u32 total_checks;
    ktime_t last_check;
    char safety_status[64];
};

/* Rust Safety Statistics */
struct rust_safety_statistics {
    u64 total_memory_checks;
    u64 bounds_violations_detected;
    u64 quarantine_violations_blocked;
    u64 memory_corruptions_prevented;
    u64 panics_handled;
    u64 successful_operations;
    u32 active_contexts;
    ktime_t system_uptime;
};

#ifdef __cplusplus
extern "C" {
#endif

/* Core Rust Safety Functions */
int rust_safety_init(void);
void rust_safety_cleanup(void);
bool rust_safety_is_available(void);

/* Context Management */
int rust_safety_create_context(struct rust_safety_context *ctx, u32 device_id);
void rust_safety_destroy_context(struct rust_safety_context *ctx);
int rust_safety_validate_context(const struct rust_safety_context *ctx);

/* Memory Safety Functions */
int rust_safety_validate_memory_access(u32 device_id, void *ptr, size_t size, bool is_write);
int rust_safety_check_buffer_bounds(const void *buffer, size_t buffer_size, 
                                   size_t access_offset, size_t access_size);
int rust_safety_validate_pointer(const void *ptr, size_t expected_size);
int rust_safety_check_memory_corruption(u32 device_id, void *memory_region, size_t size);

/* Device Safety Functions */
int rust_safety_validate_device_access(u32 device_id, bool is_write_access);
int rust_safety_check_quarantine_status(u32 device_id);
int rust_safety_verify_device_safety(u32 device_id, struct rust_device_safety_info *info);

/* Safe Memory Operations */
int rust_safety_safe_read(u32 device_id, void *dest, const void *src, size_t size);
int rust_safety_safe_write(u32 device_id, void *dest, const void *src, size_t size);
int rust_safety_safe_memset(u32 device_id, void *ptr, int value, size_t size);
int rust_safety_safe_memcpy(u32 device_id, void *dest, const void *src, size_t size);

/* Register Access Safety */
int rust_safety_safe_read_register(u32 device_id, void __iomem *reg_addr, u32 *value);
int rust_safety_safe_write_register(u32 device_id, void __iomem *reg_addr, u32 value);
int rust_safety_safe_read_memory_region(u32 device_id, void __iomem *base, 
                                       size_t offset, void *buffer, size_t size);
int rust_safety_safe_write_memory_region(u32 device_id, void __iomem *base, 
                                        size_t offset, const void *buffer, size_t size);

/* Error Handling and Recovery */
int rust_safety_handle_panic(u32 device_id, const char *panic_message);
int rust_safety_recover_from_error(u32 device_id, enum rust_safety_result error);
void rust_safety_emergency_stop(u32 device_id);
void rust_safety_global_emergency_stop(void);

/* Statistics and Monitoring */
int rust_safety_get_statistics(struct rust_safety_statistics *stats);
int rust_safety_get_device_info(u32 device_id, struct rust_device_safety_info *info);
int rust_safety_reset_statistics(void);

/* Configuration and Control */
int rust_safety_set_protection_level(u32 device_id, enum rust_memory_protection_level level);
int rust_safety_enable_quarantine_checking(bool enable);
int rust_safety_enable_bounds_checking(bool enable);
int rust_safety_enable_corruption_checking(bool enable);

/* Debug and Diagnostics */
int rust_safety_dump_context(const struct rust_safety_context *ctx, char *buffer, size_t buffer_size);
int rust_safety_run_self_test(void);
int rust_safety_validate_system_integrity(void);
const char *rust_safety_error_string(enum rust_safety_result result);

/* Utility Functions */
static inline bool rust_safety_is_valid_device_id(u32 device_id)
{
    return device_id < RUST_SAFETY_MAX_DEVICES;
}

static inline bool rust_safety_is_critical_device(u32 device_id)
{
    /* Critical quarantined devices: 0, 1, 12, 24, 83 */
    return (device_id == 0 || device_id == 1 || device_id == 12 || 
            device_id == 24 || device_id == 83);
}

static inline size_t rust_safety_align_size(size_t size, size_t alignment)
{
    return (size + alignment - 1) & ~(alignment - 1);
}

static inline bool rust_safety_is_aligned_ptr(const void *ptr, size_t alignment)
{
    return ((uintptr_t)ptr & (alignment - 1)) == 0;
}

/* Safety Macros */
#define RUST_SAFETY_CHECK_DEVICE_ID(dev_id) \
    do { \
        if (!rust_safety_is_valid_device_id(dev_id)) { \
            pr_err("Rust Safety: Invalid device ID %u\n", dev_id); \
            return RUST_SAFETY_ERROR_INVALID_DEVICE; \
        } \
    } while(0)

#define RUST_SAFETY_CHECK_QUARANTINE(dev_id) \
    do { \
        if (rust_safety_is_critical_device(dev_id)) { \
            pr_err("Rust Safety: Device %u is QUARANTINED\n", dev_id); \
            return RUST_SAFETY_ERROR_QUARANTINE_VIOLATION; \
        } \
    } while(0)

#define RUST_SAFETY_CHECK_NULL_PTR(ptr, name) \
    do { \
        if (!(ptr)) { \
            pr_err("Rust Safety: NULL pointer for %s\n", name); \
            return RUST_SAFETY_ERROR_NULL_POINTER; \
        } \
    } while(0)

#define RUST_SAFETY_CHECK_BOUNDS(ptr, size, max_size) \
    do { \
        if ((size) > (max_size) || !(ptr)) { \
            pr_err("Rust Safety: Bounds check failed - size %zu > max %zu\n", \
                   (size_t)(size), (size_t)(max_size)); \
            return RUST_SAFETY_ERROR_BOUNDS_CHECK_FAILED; \
        } \
    } while(0)

/* Constants */
#define RUST_SAFETY_MAGIC_CONTEXT      0x52535443  /* "RSTC" */
#define RUST_SAFETY_MAX_CONTEXTS       32
#define RUST_SAFETY_DEFAULT_ALIGNMENT  8
#define RUST_SAFETY_MAX_REGISTER_SIZE   4
#define RUST_SAFETY_MAX_BUFFER_SIZE     4096

#ifdef __cplusplus
}
#endif

#endif /* _DSMIL_RUST_SAFETY_H */
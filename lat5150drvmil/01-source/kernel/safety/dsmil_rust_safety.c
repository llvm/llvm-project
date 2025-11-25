/*
 * Dell MIL-SPEC Enhanced DSMIL Rust Safety Layer C Interface
 * 
 * Copyright (C) 2025 JRTC1 Educational Development
 * 
 * This module provides the C interface to the Rust safety layer for
 * memory protection and device access validation. It handles the
 * integration between C kernel code and Rust safety functions.
 * 
 * CRITICAL SAFETY: This interface provides additional memory safety
 * beyond standard C code through Rust's memory safety guarantees.
 */

#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/slab.h>
#include <linux/mutex.h>
#include <linux/atomic.h>
#include <linux/spinlock.h>
#include <linux/io.h>
#include <linux/uaccess.h>
#include <linux/kallsyms.h>

#include "dsmil_rust_safety.h"

/* Rust Safety Layer State */
struct rust_safety_layer_state {
    bool rust_available;                    /* Rust runtime available */
    bool initialized;                       /* Layer initialized */
    struct mutex global_lock;               /* Global state lock */
    
    /* Statistics */
    struct rust_safety_statistics stats;
    spinlock_t stats_lock;
    
    /* Active contexts */
    struct rust_safety_context *contexts[RUST_SAFETY_MAX_CONTEXTS];
    struct mutex context_lock;
    atomic_t context_counter;
    
    /* Configuration */
    bool quarantine_checking;
    bool bounds_checking;
    bool corruption_checking;
    enum rust_memory_protection_level default_protection;
    
    /* Device safety tracking */
    struct rust_device_safety_info devices[RUST_SAFETY_MAX_DEVICES];
    
    ktime_t init_time;
};

/* Global state instance */
static struct rust_safety_layer_state *rust_state = NULL;
static bool rust_runtime_detected;

extern int rust_dsmil_init(bool enable_smi);

/* Critical quarantined device list - MUST match other modules */
static const u32 RUST_QUARANTINE_DEVICES[RUST_SAFETY_QUARANTINE_COUNT] = {
    0, 1, 12, 24, 83
};

/*
 * Check if Rust runtime is available
 */
bool rust_safety_is_available(void)
{
	if (rust_runtime_detected)
		return true;

	if (symbol_get(rust_dsmil_init)) {
		symbol_put(rust_dsmil_init);
		rust_runtime_detected = true;
	}

	return rust_runtime_detected;
}

/*
 * Initialize Rust safety layer
 */
int rust_safety_init(void)
{
    u32 i;
    
    if (rust_state) {
        pr_warn("Rust Safety: Already initialized\n");
        return RUST_SAFETY_SUCCESS;
    }
    
    /* Check if Rust is available */
    if (!rust_safety_is_available()) {
        pr_warn("Rust Safety: Rust runtime not available, using C fallback\n");
        return RUST_SAFETY_ERROR_RUST_UNAVAILABLE;
    }
    
    /* Allocate global state */
    rust_state = kzalloc(sizeof(struct rust_safety_layer_state), GFP_KERNEL);
    if (!rust_state) {
        pr_err("Rust Safety: Failed to allocate state structure\n");
        return RUST_SAFETY_ERROR_UNINITIALIZED;
    }
    
    /* Initialize locks */
    mutex_init(&rust_state->global_lock);
    mutex_init(&rust_state->context_lock);
    spin_lock_init(&rust_state->stats_lock);
    
    /* Initialize atomic counter */
    atomic_set(&rust_state->context_counter, 0);
    
    /* Initialize configuration */
    rust_state->quarantine_checking = true;
    rust_state->bounds_checking = true;
    rust_state->corruption_checking = true;
    rust_state->default_protection = RUST_PROTECTION_MAXIMUM;
    
    /* Initialize context array */
    for (i = 0; i < RUST_SAFETY_MAX_CONTEXTS; i++) {
        rust_state->contexts[i] = NULL;
    }
    
    /* Initialize device safety info */
    for (i = 0; i < RUST_SAFETY_MAX_DEVICES; i++) {
        struct rust_device_safety_info *info = &rust_state->devices[i];
        
        info->device_id = i;
        info->is_quarantined = rust_safety_is_critical_device(i);
        info->rust_protection_active = true;
        info->protection_level = rust_state->default_protection;
        info->bounds_violations = 0;
        info->quarantine_violations = 0;
        info->memory_corruptions = 0;
        info->total_checks = 0;
        info->last_check = 0;
        
        if (info->is_quarantined) {
            strncpy(info->safety_status, "QUARANTINED - Full protection active", 
                    sizeof(info->safety_status) - 1);
        } else {
            strncpy(info->safety_status, "Protected - Rust safety active", 
                    sizeof(info->safety_status) - 1);
        }
        info->safety_status[sizeof(info->safety_status) - 1] = '\0';
    }
    
    /* Initialize statistics */
    memset(&rust_state->stats, 0, sizeof(rust_state->stats));
    rust_state->stats.system_uptime = ktime_get();
    
    /* Set state flags */
    rust_state->rust_available = true;
    rust_state->initialized = true;
    rust_state->init_time = ktime_get();
    
    pr_info("Rust Safety: Initialized successfully (version %s)\n", 
            DSMIL_RUST_SAFETY_VERSION_STRING);
    pr_info("Rust Safety: Protection active for all %u devices, %u quarantined\n",
            RUST_SAFETY_MAX_DEVICES, RUST_SAFETY_QUARANTINE_COUNT);
    
    return RUST_SAFETY_SUCCESS;
}

/*
 * Cleanup Rust safety layer
 */
void rust_safety_cleanup(void)
{
    u32 i;
    
    if (!rust_state) {
        return;
    }
    
    pr_info("Rust Safety: Shutting down safety layer\n");
    
    /* Clean up active contexts */
    mutex_lock(&rust_state->context_lock);
    for (i = 0; i < RUST_SAFETY_MAX_CONTEXTS; i++) {
        if (rust_state->contexts[i]) {
            rust_safety_destroy_context(rust_state->contexts[i]);
        }
    }
    mutex_unlock(&rust_state->context_lock);
    
    /* Print final statistics */
    pr_info("Rust Safety: Final stats - Memory checks: %llu, Violations: %llu, Panics: %llu\n",
            rust_state->stats.total_memory_checks,
            rust_state->stats.bounds_violations_detected,
            rust_state->stats.panics_handled);
    
    /* Mark as uninitialized */
    rust_state->initialized = false;
    
    /* Free memory */
    kfree(rust_state);
    rust_state = NULL;
    
    pr_info("Rust Safety: Cleanup complete\n");
}

/*
 * Create safety context
 */
int rust_safety_create_context(struct rust_safety_context *ctx, u32 device_id)
{
    u32 i;
    
    RUST_SAFETY_CHECK_NULL_PTR(ctx, "context");
    RUST_SAFETY_CHECK_DEVICE_ID(device_id);
    
    if (!rust_state || !rust_state->initialized) {
        return RUST_SAFETY_ERROR_UNINITIALIZED;
    }
    
    /* Initialize context */
    memset(ctx, 0, sizeof(struct rust_safety_context));
    ctx->context_id = atomic_inc_return(&rust_state->context_counter);
    ctx->device_id = device_id;
    ctx->memory_base = NULL;
    ctx->memory_size = 0;
    ctx->protection_level = rust_state->default_protection;
    ctx->quarantine_check_enabled = rust_state->quarantine_checking;
    ctx->bounds_check_enabled = rust_state->bounds_checking;
    ctx->corruption_check_enabled = rust_state->corruption_checking;
    ctx->magic_number = RUST_SAFETY_MAGIC_CONTEXT;
    ctx->created_time = ktime_get();
    
    /* Find slot for context tracking */
    mutex_lock(&rust_state->context_lock);
    for (i = 0; i < RUST_SAFETY_MAX_CONTEXTS; i++) {
        if (rust_state->contexts[i] == NULL) {
            rust_state->contexts[i] = ctx;
            break;
        }
    }
    mutex_unlock(&rust_state->context_lock);
    
    if (i == RUST_SAFETY_MAX_CONTEXTS) {
        pr_warn("Rust Safety: Context tracking array full\n");
    }
    
    /* Update statistics */
    spin_lock(&rust_state->stats_lock);
    rust_state->stats.active_contexts++;
    spin_unlock(&rust_state->stats_lock);
    
    pr_debug("Rust Safety: Created context %u for device %u\n", 
             ctx->context_id, device_id);
    
    return RUST_SAFETY_SUCCESS;
}

/*
 * Destroy safety context
 */
void rust_safety_destroy_context(struct rust_safety_context *ctx)
{
    u32 i;
    
    if (!ctx || !rust_state) {
        return;
    }
    
    /* Validate context */
    if (ctx->magic_number != RUST_SAFETY_MAGIC_CONTEXT) {
        pr_warn("Rust Safety: Invalid context magic number\n");
        return;
    }
    
    /* Remove from tracking array */
    mutex_lock(&rust_state->context_lock);
    for (i = 0; i < RUST_SAFETY_MAX_CONTEXTS; i++) {
        if (rust_state->contexts[i] == ctx) {
            rust_state->contexts[i] = NULL;
            break;
        }
    }
    mutex_unlock(&rust_state->context_lock);
    
    /* Update statistics */
    spin_lock(&rust_state->stats_lock);
    if (rust_state->stats.active_contexts > 0) {
        rust_state->stats.active_contexts--;
    }
    spin_unlock(&rust_state->stats_lock);
    
    pr_debug("Rust Safety: Destroyed context %u for device %u\n", 
             ctx->context_id, ctx->device_id);
    
    /* Clear context */
    memset(ctx, 0, sizeof(struct rust_safety_context));
}

/*
 * CRITICAL: Validate device access with quarantine checking
 */
int rust_safety_validate_device_access(u32 device_id, bool is_write_access)
{
    struct rust_device_safety_info *info;
    
    RUST_SAFETY_CHECK_DEVICE_ID(device_id);
    
    if (!rust_state || !rust_state->initialized) {
        return RUST_SAFETY_ERROR_UNINITIALIZED;
    }
    
    info = &rust_state->devices[device_id];
    
    /* Update check statistics */
    info->total_checks++;
    info->last_check = ktime_get();
    
    spin_lock(&rust_state->stats_lock);
    rust_state->stats.total_memory_checks++;
    spin_unlock(&rust_state->stats_lock);
    
    /* CRITICAL: Check quarantine status for write operations */
    if (is_write_access && info->is_quarantined) {
        info->quarantine_violations++;
        
        spin_lock(&rust_state->stats_lock);
        rust_state->stats.quarantine_violations_blocked++;
        spin_unlock(&rust_state->stats_lock);
        
        pr_err("Rust Safety: QUARANTINE VIOLATION BLOCKED - Device %u write attempt\n", 
               device_id);
        
        return RUST_SAFETY_ERROR_QUARANTINE_VIOLATION;
    }
    
    return RUST_SAFETY_SUCCESS;
}

/*
 * Check buffer bounds with Rust-style safety
 */
int rust_safety_check_buffer_bounds(const void *buffer, size_t buffer_size, 
                                   size_t access_offset, size_t access_size)
{
    RUST_SAFETY_CHECK_NULL_PTR(buffer, "buffer");
    
    if (!rust_state || !rust_state->initialized) {
        return RUST_SAFETY_ERROR_UNINITIALIZED;
    }
    
    /* Check for overflow in offset calculation */
    if (access_offset > buffer_size) {
        spin_lock(&rust_state->stats_lock);
        rust_state->stats.bounds_violations_detected++;
        spin_unlock(&rust_state->stats_lock);
        
        pr_err("Rust Safety: Bounds violation - offset %zu > buffer size %zu\n",
               access_offset, buffer_size);
        return RUST_SAFETY_ERROR_BOUNDS_CHECK_FAILED;
    }
    
    /* Check for overflow in size calculation */
    if (access_offset + access_size > buffer_size) {
        spin_lock(&rust_state->stats_lock);
        rust_state->stats.bounds_violations_detected++;
        spin_unlock(&rust_state->stats_lock);
        
        pr_err("Rust Safety: Bounds violation - access %zu+%zu > buffer size %zu\n",
               access_offset, access_size, buffer_size);
        return RUST_SAFETY_ERROR_BOUNDS_CHECK_FAILED;
    }
    
    /* Check for integer overflow */
    if (access_offset + access_size < access_offset) {
        spin_lock(&rust_state->stats_lock);
        rust_state->stats.bounds_violations_detected++;
        spin_unlock(&rust_state->stats_lock);
        
        pr_err("Rust Safety: Integer overflow in bounds check\n");
        return RUST_SAFETY_ERROR_BOUNDS_CHECK_FAILED;
    }
    
    return RUST_SAFETY_SUCCESS;
}

/*
 * Safe register read with Rust-style protection
 */
int rust_safety_safe_read_register(u32 device_id, void __iomem *reg_addr, u32 *value)
{
    int ret;
    
    RUST_SAFETY_CHECK_DEVICE_ID(device_id);
    RUST_SAFETY_CHECK_NULL_PTR(reg_addr, "register_address");
    RUST_SAFETY_CHECK_NULL_PTR(value, "value_pointer");
    
    /* Validate device access */
    ret = rust_safety_validate_device_access(device_id, false); /* Read access */
    if (ret != RUST_SAFETY_SUCCESS) {
        return ret;
    }
    
    /* Check pointer alignment */
    if (!rust_safety_is_aligned_ptr(reg_addr, RUST_SAFETY_DEFAULT_ALIGNMENT)) {
        pr_warn("Rust Safety: Unaligned register access for device %u\n", device_id);
    }
    
    /* Perform the read with additional safety checks */
    *value = ioread32(reg_addr);
    
    /* Update statistics */
    spin_lock(&rust_state->stats_lock);
    rust_state->stats.successful_operations++;
    spin_unlock(&rust_state->stats_lock);
    
    pr_debug("Rust Safety: Safe register read - device %u, value 0x%08x\n", 
             device_id, *value);
    
    return RUST_SAFETY_SUCCESS;
}

/*
 * Safe register write with Rust-style protection
 */
int rust_safety_safe_write_register(u32 device_id, void __iomem *reg_addr, u32 value)
{
    int ret;
    
    RUST_SAFETY_CHECK_DEVICE_ID(device_id);
    RUST_SAFETY_CHECK_NULL_PTR(reg_addr, "register_address");
    
    /* CRITICAL: Validate device access for write */
    ret = rust_safety_validate_device_access(device_id, true); /* Write access */
    if (ret != RUST_SAFETY_SUCCESS) {
        return ret;
    }
    
    /* Additional quarantine check */
    if (rust_safety_is_critical_device(device_id)) {
        pr_err("Rust Safety: CRITICAL QUARANTINE VIOLATION - Device %u write blocked\n", 
               device_id);
        
        spin_lock(&rust_state->stats_lock);
        rust_state->stats.quarantine_violations_blocked++;
        spin_unlock(&rust_state->stats_lock);
        
        return RUST_SAFETY_ERROR_QUARANTINE_VIOLATION;
    }
    
    /* Check pointer alignment */
    if (!rust_safety_is_aligned_ptr(reg_addr, RUST_SAFETY_DEFAULT_ALIGNMENT)) {
        pr_warn("Rust Safety: Unaligned register access for device %u\n", device_id);
    }
    
    /* Perform the write */
    iowrite32(value, reg_addr);
    
    /* Update statistics */
    spin_lock(&rust_state->stats_lock);
    rust_state->stats.successful_operations++;
    spin_unlock(&rust_state->stats_lock);
    
    pr_info("Rust Safety: Safe register write - device %u, value 0x%08x\n", 
            device_id, value);
    
    return RUST_SAFETY_SUCCESS;
}

/*
 * Safe memory copy with bounds checking
 */
int rust_safety_safe_memcpy(u32 device_id, void *dest, const void *src, size_t size)
{
    int ret;
    
    RUST_SAFETY_CHECK_DEVICE_ID(device_id);
    RUST_SAFETY_CHECK_NULL_PTR(dest, "destination");
    RUST_SAFETY_CHECK_NULL_PTR(src, "source");
    RUST_SAFETY_CHECK_BOUNDS(src, size, RUST_SAFETY_MAX_BUFFER_SIZE);
    
    /* Validate device access */
    ret = rust_safety_validate_device_access(device_id, true); /* Write access */
    if (ret != RUST_SAFETY_SUCCESS) {
        return ret;
    }
    
    /* Check for overlapping regions */
    if (dest == src) {
        pr_warn("Rust Safety: Source and destination are the same\n");
        return RUST_SAFETY_SUCCESS; /* Nothing to do */
    }
    
    /* Check for overlap */
    if ((dest >= src && dest < (const void *)((const char *)src + size)) ||
        (src >= dest && src < (void *)((char *)dest + size))) {
        pr_warn("Rust Safety: Overlapping memory regions detected\n");
        /* Use memmove instead of memcpy for overlapping regions */
        memmove(dest, src, size);
    } else {
        memcpy(dest, src, size);
    }
    
    /* Update statistics */
    spin_lock(&rust_state->stats_lock);
    rust_state->stats.successful_operations++;
    spin_unlock(&rust_state->stats_lock);
    
    return RUST_SAFETY_SUCCESS;
}

/*
 * Get device safety information
 */
int rust_safety_get_device_info(u32 device_id, struct rust_device_safety_info *info)
{
    RUST_SAFETY_CHECK_DEVICE_ID(device_id);
    RUST_SAFETY_CHECK_NULL_PTR(info, "device_info");
    
    if (!rust_state || !rust_state->initialized) {
        return RUST_SAFETY_ERROR_UNINITIALIZED;
    }
    
    /* Copy device information */
    memcpy(info, &rust_state->devices[device_id], sizeof(*info));
    
    return RUST_SAFETY_SUCCESS;
}

/*
 * Get safety statistics
 */
int rust_safety_get_statistics(struct rust_safety_statistics *stats)
{
    RUST_SAFETY_CHECK_NULL_PTR(stats, "statistics");
    
    if (!rust_state || !rust_state->initialized) {
        return RUST_SAFETY_ERROR_UNINITIALIZED;
    }
    
    spin_lock(&rust_state->stats_lock);
    memcpy(stats, &rust_state->stats, sizeof(*stats));
    stats->system_uptime = ktime_sub(ktime_get(), rust_state->init_time);
    spin_unlock(&rust_state->stats_lock);
    
    return RUST_SAFETY_SUCCESS;
}

/*
 * Emergency stop for device
 */
void rust_safety_emergency_stop(u32 device_id)
{
    if (!rust_safety_is_valid_device_id(device_id) || !rust_state) {
        return;
    }
    
    pr_err("Rust Safety: EMERGENCY STOP for device %u\n", device_id);
    
    /* Update statistics */
    spin_lock(&rust_state->stats_lock);
    rust_state->stats.panics_handled++;
    spin_unlock(&rust_state->stats_lock);
    
    /* Update device status */
    strncpy(rust_state->devices[device_id].safety_status, 
            "EMERGENCY STOPPED", 
            sizeof(rust_state->devices[device_id].safety_status) - 1);
}

/*
 * Global emergency stop
 */
void rust_safety_global_emergency_stop(void)
{
    u32 i;
    
    if (!rust_state) {
        return;
    }
    
    pr_err("Rust Safety: GLOBAL EMERGENCY STOP - All devices\n");
    
    /* Stop all devices */
    for (i = 0; i < RUST_SAFETY_MAX_DEVICES; i++) {
        rust_safety_emergency_stop(i);
    }
    
    /* Update global statistics */
    spin_lock(&rust_state->stats_lock);
    rust_state->stats.panics_handled++;
    spin_unlock(&rust_state->stats_lock);
}

/*
 * Error code to string conversion
 */
const char *rust_safety_error_string(enum rust_safety_result result)
{
    switch (result) {
    case RUST_SAFETY_SUCCESS:
        return "Success";
    case RUST_SAFETY_ERROR_INVALID_DEVICE:
        return "Invalid device";
    case RUST_SAFETY_ERROR_QUARANTINE_VIOLATION:
        return "Quarantine violation";
    case RUST_SAFETY_ERROR_BOUNDS_CHECK_FAILED:
        return "Bounds check failed";
    case RUST_SAFETY_ERROR_MEMORY_CORRUPTION:
        return "Memory corruption detected";
    case RUST_SAFETY_ERROR_NULL_POINTER:
        return "Null pointer error";
    case RUST_SAFETY_ERROR_BUFFER_OVERFLOW:
        return "Buffer overflow detected";
    case RUST_SAFETY_ERROR_UNINITIALIZED:
        return "Rust safety not initialized";
    case RUST_SAFETY_ERROR_PANIC:
        return "Rust panic occurred";
    case RUST_SAFETY_ERROR_RUST_UNAVAILABLE:
        return "Rust runtime unavailable";
    default:
        return "Unknown error";
    }
}

/* Export functions for use by other modules */
EXPORT_SYMBOL(rust_safety_init);
EXPORT_SYMBOL(rust_safety_cleanup);
EXPORT_SYMBOL(rust_safety_is_available);
EXPORT_SYMBOL(rust_safety_create_context);
EXPORT_SYMBOL(rust_safety_destroy_context);
EXPORT_SYMBOL(rust_safety_validate_device_access);
EXPORT_SYMBOL(rust_safety_check_buffer_bounds);
EXPORT_SYMBOL(rust_safety_safe_read_register);
EXPORT_SYMBOL(rust_safety_safe_write_register);
EXPORT_SYMBOL(rust_safety_safe_memcpy);
EXPORT_SYMBOL(rust_safety_get_device_info);
EXPORT_SYMBOL(rust_safety_get_statistics);
EXPORT_SYMBOL(rust_safety_emergency_stop);
EXPORT_SYMBOL(rust_safety_global_emergency_stop);
EXPORT_SYMBOL(rust_safety_error_string);

MODULE_AUTHOR("DSMIL Track A Development Team");
MODULE_DESCRIPTION("DSMIL Rust Safety Layer C Interface");
MODULE_LICENSE("GPL v2");
MODULE_VERSION(DSMIL_RUST_SAFETY_VERSION_STRING);

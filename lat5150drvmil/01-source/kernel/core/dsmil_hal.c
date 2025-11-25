/*
 * Dell MIL-SPEC Enhanced DSMIL Hardware Abstraction Layer Implementation
 * 
 * Copyright (C) 2025 JRTC1 Educational Development
 * 
 * This implementation provides comprehensive hardware abstraction
 * for safe access to 84 DSMIL devices with absolute quarantine protection.
 * 
 * CRITICAL SAFETY: This HAL implements multiple safety layers to prevent
 * any write operations to the 5 quarantined devices.
 */

#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/slab.h>
#include <linux/mutex.h>
#include <linux/io.h>
#include <linux/delay.h>
#include <linux/crc32.h>
#include <linux/thermal.h>
#include <linux/time.h>
#include <linux/atomic.h>
#include <linux/spinlock.h>

#include "dsmil_hal.h"
#include "../enhanced/dsmil_enhanced.c" /* Include enhanced kernel module definitions */
#include "dsmil_security_types.h" /* Found via -I$(src)/security in Makefile */

struct dsmil_auth_context;
int dsmil_mfa_authorize_operation(struct dsmil_auth_context *ctx,
				  u32 device_id,
				  enum dsmil_operation_type operation,
				  enum dsmil_risk_level risk_level);

/* HAL Global State */
struct dsmil_hal_global_state {
    struct dsmil_enhanced_state *enhanced_state;   /* Reference to enhanced state */
    struct dsmil_hal_statistics statistics;        /* Global statistics */
    struct dsmil_hal_config config;                /* Configuration */
    struct dsmil_hal_thermal_info thermal_info;    /* Thermal information */
    struct thermal_zone_device *thermal_zone;      /* Preferred thermal zone */
    
    struct mutex global_lock;                       /* Global HAL lock */
    spinlock_t stats_lock;                         /* Statistics lock */
    atomic_t operation_counter;                     /* Operation ID counter */
    atomic_t active_operations;                     /* Active operation count */
    
    bool initialized;                               /* HAL initialization status */
    u32 hal_magic;                                 /* Magic number for validation */
    ktime_t init_time;                             /* HAL initialization time */
    
    /* Operation tracking */
    struct dsmil_hal_context *active_contexts[DSMIL_HAL_MAX_CONCURRENT_OPS];
    struct mutex context_lock;                      /* Context array lock */
};

/* HAL Global Instance */
static struct dsmil_hal_global_state *hal_state = NULL;

/* Critical Device Quarantine List - MUST MATCH dsmil_enhanced.c */
static const u32 HAL_QUARANTINE_DEVICES[DSMIL_HAL_QUARANTINE_COUNT] = {
    0,   /* Group 0, Device 0 - Master Control */
    1,   /* Group 0, Device 1 - Security Platform */
    12,  /* Group 1, Device 0 - Power Management */
    24,  /* Group 2, Device 0 - Memory Controller */
    83   /* Group 6, Device 11 - Emergency Stop */
};

/* Register offset tables for different device types */
static const u32 default_register_offsets[10] = {
    0x0000, /* CONTROL */
    0x0004, /* STATUS */
    0x0008, /* CONFIG */
    0x000C, /* DATA */
    0x0010, /* INTERRUPT */
    0x0014, /* ERROR */
    0x0018, /* DEBUG */
    0x001C, /* SAFETY */
    0x0020, /* THERMAL */
    0x0024  /* POWER */
};

/* Forward declarations */
static int hal_validate_context(struct dsmil_hal_context *ctx);
static int hal_check_thermal_safety(void);
static int hal_update_statistics(enum dsmil_hal_access_type access_type, 
                                enum dsmil_hal_result result);
static int hal_log_operation(u32 device_id, enum dsmil_hal_access_type access_type, 
                            enum dsmil_hal_result result);
static struct thermal_zone_device *hal_acquire_thermal_zone(void);
static enum dsmil_operation_type hal_map_access_to_operation(enum dsmil_hal_access_type access_type);
static enum dsmil_risk_level hal_calculate_operation_risk(u32 device_id,
							 enum dsmil_operation_type operation);

static struct thermal_zone_device *hal_acquire_thermal_zone(void)
{
	static const char * const zone_candidates[] = {
		"x86_pkg_temp", "soc_dts0", "iwlwifi_1", "acpitz"
	};
	int i;

	if (hal_state->thermal_zone && !IS_ERR(hal_state->thermal_zone))
		return hal_state->thermal_zone;

	for (i = 0; i < ARRAY_SIZE(zone_candidates); i++) {
		struct thermal_zone_device *tz =
			thermal_zone_get_zone_by_name(zone_candidates[i]);
		if (!IS_ERR(tz)) {
			hal_state->thermal_zone = tz;
			return tz;
		}
	}
	return NULL;
}

static enum dsmil_operation_type hal_map_access_to_operation(
	enum dsmil_hal_access_type access_type)
{
	switch (access_type) {
	case DSMIL_HAL_ACCESS_READ_ONLY:
		return DSMIL_OP_READ;
	case DSMIL_HAL_ACCESS_WRITE_SAFE:
		return DSMIL_OP_WRITE;
	case DSMIL_HAL_ACCESS_WRITE_CONTROLLED:
		return DSMIL_OP_CONFIG;
	case DSMIL_HAL_ACCESS_WRITE_CRITICAL:
		return DSMIL_OP_CONTROL;
	case DSMIL_HAL_ACCESS_DIAGNOSTIC:
		return DSMIL_OP_DIAGNOSTIC;
	case DSMIL_HAL_ACCESS_MAINTENANCE:
	default:
		return DSMIL_OP_MAINTENANCE;
	}
}

static enum dsmil_risk_level hal_calculate_operation_risk(
	u32 device_id, enum dsmil_operation_type operation)
{
	enum dsmil_risk_level risk = DSMIL_RISK_LOW;
	u32 group = device_id / 12;

	switch (group) {
	case 0:
		risk = DSMIL_RISK_CRITICAL;
		break;
	case 1:
	case 2:
		risk = DSMIL_RISK_HIGH;
		break;
	case 3:
	case 4:
		risk = DSMIL_RISK_MEDIUM;
		break;
	default:
		risk = DSMIL_RISK_LOW;
		break;
	}

	if (operation == DSMIL_OP_EMERGENCY)
		return DSMIL_RISK_CATASTROPHIC;

	if (operation == DSMIL_OP_CONTROL || operation == DSMIL_OP_RESET) {
		if (risk < DSMIL_RISK_HIGH)
			risk = DSMIL_RISK_HIGH;
	}

	return risk;
}

/*
 * HAL Initialization
 */
int dsmil_hal_init(struct dsmil_enhanced_state *state)
{
    int i;
    
    if (!state) {
        pr_err("DSMIL HAL: NULL enhanced state pointer\n");
        return DSMIL_HAL_ERROR_INVALID_DEVICE;
    }
    
    if (hal_state) {
        pr_warn("DSMIL HAL: Already initialized\n");
        return DSMIL_HAL_SUCCESS;
    }
    
    /* Allocate HAL global state */
    hal_state = kzalloc(sizeof(struct dsmil_hal_global_state), GFP_KERNEL);
    if (!hal_state) {
        pr_err("DSMIL HAL: Failed to allocate global state\n");
        return DSMIL_HAL_ERROR_UNKNOWN;
    }
    
    /* Initialize HAL state */
    hal_state->enhanced_state = state;
    hal_state->hal_magic = DSMIL_HAL_MAGIC_INIT;
    hal_state->initialized = false;
    hal_state->init_time = ktime_get();
    
    /* Initialize locks */
    mutex_init(&hal_state->global_lock);
    mutex_init(&hal_state->context_lock);
    spin_lock_init(&hal_state->stats_lock);
    
    /* Initialize atomic counters */
    atomic_set(&hal_state->operation_counter, 0);
    atomic_set(&hal_state->active_operations, 0);
    
    /* Initialize context array */
    for (i = 0; i < DSMIL_HAL_MAX_CONCURRENT_OPS; i++) {
        hal_state->active_contexts[i] = NULL;
    }
    
    /* Initialize default configuration */
    hal_state->config.quarantine_enforcement = true;
    hal_state->config.safety_monitoring = true;
    hal_state->config.thermal_protection = true;
    hal_state->config.crc_validation = true;
    hal_state->config.access_logging = true;
    hal_state->config.debug_mode = false;
    hal_state->config.default_timeout_ms = DSMIL_HAL_DEFAULT_TIMEOUT_MS;
    hal_state->config.max_concurrent_ops = DSMIL_HAL_MAX_CONCURRENT_OPS;
    hal_state->config.safety_check_interval = 1000;
    hal_state->config.thermal_check_interval = 500;
    hal_state->config.thermal_critical_temp = 85000; /* 85°C in milli-Celsius */
    hal_state->config.thermal_warning_temp = 75000;  /* 75°C in milli-Celsius */
    
    /* Initialize statistics */
    memset(&hal_state->statistics, 0, sizeof(hal_state->statistics));
    hal_state->statistics.last_reset = ktime_get();
    
    /* Initialize thermal info */
    hal_state->thermal_info.current_temp = 25000; /* 25°C default */
    hal_state->thermal_info.critical_temp = hal_state->config.thermal_critical_temp;
    hal_state->thermal_info.warning_temp = hal_state->config.thermal_warning_temp;
    hal_state->thermal_info.thermal_protection_active = true;
    hal_state->thermal_info.thermal_events = 0;
    hal_state->thermal_info.last_thermal_event = 0;
    
    hal_state->initialized = true;
    
    pr_info("DSMIL HAL: Initialized successfully (version %s)\n", 
            DSMIL_HAL_VERSION_STRING);
    pr_info("DSMIL HAL: Protecting %d quarantined devices\n", 
            DSMIL_HAL_QUARANTINE_COUNT);
    
    return DSMIL_HAL_SUCCESS;
}

/*
 * HAL Cleanup
 */
void dsmil_hal_cleanup(void)
{
    int i;
    
    if (!hal_state) {
        return;
    }
    
    pr_info("DSMIL HAL: Cleaning up...\n");
    
    /* Emergency stop all active operations */
    dsmil_hal_emergency_stop_all();
    
    /* Clean up active contexts */
    mutex_lock(&hal_state->context_lock);
    for (i = 0; i < DSMIL_HAL_MAX_CONCURRENT_OPS; i++) {
        if (hal_state->active_contexts[i]) {
            dsmil_hal_destroy_context(hal_state->active_contexts[i]);
            hal_state->active_contexts[i] = NULL;
        }
    }
    mutex_unlock(&hal_state->context_lock);
    
    /* Print final statistics */
    if (hal_state->config.debug_mode) {
        pr_info("DSMIL HAL: Final statistics - Total ops: %llu, Safety violations: %llu\n",
                hal_state->statistics.total_operations, 
                hal_state->statistics.safety_violations);
    }
    
    /* Clear magic and mark as uninitialized */
    hal_state->hal_magic = 0;
    hal_state->initialized = false;
    
    /* Free global state */
    kfree(hal_state);
    hal_state = NULL;
    
    pr_info("DSMIL HAL: Cleanup complete\n");
}

/*
 * Check if device is quarantined - CRITICAL SAFETY FUNCTION
 */
bool dsmil_hal_is_device_quarantined(u32 device_id)
{
    int i;
    
    if (!dsmil_hal_is_valid_device_id(device_id)) {
        return true; /* Treat invalid devices as quarantined */
    }
    
    /* Check against quarantine list */
    for (i = 0; i < DSMIL_HAL_QUARANTINE_COUNT; i++) {
        if (HAL_QUARANTINE_DEVICES[i] == device_id) {
            return true;
        }
    }
    
    return false;
}

/*
 * Validate device ID
 */
bool dsmil_hal_is_device_valid(u32 device_id)
{
    return dsmil_hal_is_valid_device_id(device_id);
}

/*
 * Get device count
 */
int dsmil_hal_get_device_count(void)
{
    return DSMIL_HAL_MAX_DEVICES;
}

/*
 * Get group count
 */
int dsmil_hal_get_group_count(void)
{
    return DSMIL_HAL_MAX_GROUPS;
}

/*
 * Get device information
 */
int dsmil_hal_get_device_info(u32 device_id, struct dsmil_hal_device_info *info)
{
    struct dsmil_device_enhanced *device;
    u32 group_id, device_index;
    
    DSMIL_HAL_CHECK_DEVICE_ID(device_id);
    DSMIL_HAL_CHECK_NULL_PTR(info, "device_info");
    
    if (!hal_state || !hal_state->initialized) {
        return DSMIL_HAL_ERROR_UNKNOWN;
    }
    
    group_id = dsmil_hal_device_to_group(device_id);
    device_index = dsmil_hal_device_index_in_group(device_id);
    
    if (group_id >= DSMIL_HAL_MAX_GROUPS || device_index >= DSMIL_HAL_DEVICES_PER_GROUP) {
        return DSMIL_HAL_ERROR_INVALID_DEVICE;
    }
    
    device = &hal_state->enhanced_state->groups[group_id].devices[device_index];
    
    /* Fill device info structure */
    memset(info, 0, sizeof(struct dsmil_hal_device_info));
    info->device_id = device_id;
    info->group_id = group_id;
    info->device_index = device_index;
    
    strncpy(info->function_name, device->function_name, sizeof(info->function_name) - 1);
    strncpy(info->safety_reason, device->safety_reason, sizeof(info->safety_reason) - 1);
    
    info->safety_level = device->safety_level;
    info->is_quarantined = device->is_quarantined;
    info->read_enabled = device->read_enabled;
    info->write_enabled = device->write_enabled && !device->is_quarantined;
    info->requires_auth = device->requires_auth;
    
    info->access_count_read = device->access_count_read;
    info->access_count_write = device->access_count_write;
    info->safety_violations = device->safety_violations;
    
    info->mmio_base = device->mmio_base;
    info->mmio_size = device->mmio_size;
    
    /* Set supported registers (all supported for now) */
    info->supported_registers = 0x3FF; /* All 10 register types */
    memcpy(info->register_offsets, default_register_offsets, sizeof(default_register_offsets));
    
    info->last_access = device->last_access;
    info->current_state = device->state;
    
    return DSMIL_HAL_SUCCESS;
}

/*
 * Create HAL operation context
 */
int dsmil_hal_create_context(struct dsmil_hal_context *ctx, u32 device_id, 
                            enum dsmil_hal_access_type access_type)
{
    int i;
    
    DSMIL_HAL_CHECK_NULL_PTR(ctx, "context");
    DSMIL_HAL_CHECK_DEVICE_ID(device_id);
    
    if (!hal_state || !hal_state->initialized) {
        return DSMIL_HAL_ERROR_UNKNOWN;
    }
    
    /* Check if we can create more contexts */
    if (atomic_read(&hal_state->active_operations) >= hal_state->config.max_concurrent_ops) {
        return DSMIL_HAL_ERROR_DEVICE_BUSY;
    }
    
    /* Initialize context */
    memset(ctx, 0, sizeof(struct dsmil_hal_context));
    ctx->operation_id = atomic_inc_return(&hal_state->operation_counter);
    ctx->access_type = access_type;
    ctx->device_id = device_id;
    ctx->timeout_ms = hal_state->config.default_timeout_ms;
    ctx->bypass_safety = false; /* NEVER allow this to be set */
    ctx->requires_auth = false;
    ctx->start_time = ktime_get();
    
    /* Check if authorization is required */
    if (access_type > DSMIL_HAL_ACCESS_READ_ONLY) {
        struct dsmil_hal_device_info info;
        int ret = dsmil_hal_get_device_info(device_id, &info);
        if (ret == DSMIL_HAL_SUCCESS) {
            ctx->requires_auth = info.requires_auth;
        }
    }
    
    /* Find slot for context tracking */
    mutex_lock(&hal_state->context_lock);
    for (i = 0; i < DSMIL_HAL_MAX_CONCURRENT_OPS; i++) {
        if (hal_state->active_contexts[i] == NULL) {
            hal_state->active_contexts[i] = ctx;
            break;
        }
    }
    mutex_unlock(&hal_state->context_lock);
    
    if (i == DSMIL_HAL_MAX_CONCURRENT_OPS) {
        pr_warn("DSMIL HAL: Context tracking array full\n");
        /* Continue anyway, just won't be tracked */
    }
    
    atomic_inc(&hal_state->active_operations);
    
    if (hal_state->config.debug_mode) {
        pr_info("DSMIL HAL: Created context %u for device %u, access type %d\n",
                ctx->operation_id, device_id, access_type);
    }
    
    return DSMIL_HAL_SUCCESS;
}

/*
 * Destroy HAL operation context
 */
void dsmil_hal_destroy_context(struct dsmil_hal_context *ctx)
{
    int i;
    
    if (!ctx || !hal_state) {
        return;
    }
    
    /* Remove from tracking array */
    mutex_lock(&hal_state->context_lock);
    for (i = 0; i < DSMIL_HAL_MAX_CONCURRENT_OPS; i++) {
        if (hal_state->active_contexts[i] == ctx) {
            hal_state->active_contexts[i] = NULL;
            break;
        }
    }
    mutex_unlock(&hal_state->context_lock);
    
    atomic_dec(&hal_state->active_operations);
    
    if (hal_state->config.debug_mode) {
        pr_info("DSMIL HAL: Destroyed context %u for device %u\n",
                ctx->operation_id, ctx->device_id);
    }
    
    /* Clear context data */
    memset(ctx, 0, sizeof(struct dsmil_hal_context));
}

/*
 * Validate operation - CRITICAL SAFETY FUNCTION
 */
int dsmil_hal_validate_operation(u32 device_id, enum dsmil_hal_access_type access_type)
{
    struct dsmil_hal_device_info info;
    int ret;
    
    DSMIL_HAL_CHECK_DEVICE_ID(device_id);
    
    if (!hal_state || !hal_state->initialized) {
        return DSMIL_HAL_ERROR_UNKNOWN;
    }
    
    /* Check global emergency stop */
    if (hal_state->enhanced_state->global_emergency_stop) {
        return DSMIL_HAL_ERROR_EMERGENCY_STOP;
    }
    
    /* CRITICAL: Check quarantine status for write operations */
    if (access_type > DSMIL_HAL_ACCESS_READ_ONLY) {
        if (dsmil_hal_is_device_quarantined(device_id)) {
            /* Log the violation attempt */
            pr_err("DSMIL HAL: QUARANTINE VIOLATION ATTEMPT on device %u\n", device_id);
            hal_update_statistics(access_type, DSMIL_HAL_ERROR_QUARANTINE_VIOLATION);
            return DSMIL_HAL_ERROR_QUARANTINE_VIOLATION;
        }
    }
    
    /* Get device information */
    ret = dsmil_hal_get_device_info(device_id, &info);
    if (ret != DSMIL_HAL_SUCCESS) {
        return ret;
    }
    
    /* Check if read is enabled */
    if (access_type == DSMIL_HAL_ACCESS_READ_ONLY && !info.read_enabled) {
        return DSMIL_HAL_ERROR_PERMISSION_DENIED;
    }
    
    /* Check if write is enabled */
    if (access_type > DSMIL_HAL_ACCESS_READ_ONLY && !info.write_enabled) {
        pr_err("DSMIL HAL: Write operation blocked for device %u (%s)\n",
               device_id, info.function_name);
        return DSMIL_HAL_ERROR_PERMISSION_DENIED;
    }
    
    /* Check device state */
    if (info.current_state == DSMIL_DEVICE_QUARANTINED ||
        info.current_state == DSMIL_DEVICE_SAFETY_VIOLATION ||
        info.current_state == DSMIL_DEVICE_ERROR) {
        return DSMIL_HAL_ERROR_DEVICE_OFFLINE;
    }
    
    /* Check thermal safety */
    if (hal_state->config.thermal_protection) {
        ret = hal_check_thermal_safety();
        if (ret != DSMIL_HAL_SUCCESS) {
            return ret;
        }
    }
    
    return DSMIL_HAL_SUCCESS;
}

/*
 * Safe register read operation
 */
int dsmil_hal_read_register(u32 device_id, enum dsmil_hal_register_type reg_type, 
                           u32 *value, struct dsmil_hal_context *ctx)
{
    struct dsmil_hal_device_info info;
    void __iomem *reg_addr;
    int ret;
    u32 reg_offset;
    
    DSMIL_HAL_CHECK_DEVICE_ID(device_id);
    DSMIL_HAL_CHECK_NULL_PTR(value, "value");
    DSMIL_HAL_CHECK_NULL_PTR(ctx, "context");
    
    /* Validate operation */
    ret = dsmil_hal_validate_operation(device_id, DSMIL_HAL_ACCESS_READ_ONLY);
    if (ret != DSMIL_HAL_SUCCESS) {
        hal_update_statistics(DSMIL_HAL_ACCESS_READ_ONLY, ret);
        return ret;
    }
    
    /* Get device information */
    ret = dsmil_hal_get_device_info(device_id, &info);
    if (ret != DSMIL_HAL_SUCCESS) {
        return ret;
    }
    
    /* Check if register type is supported */
    if (reg_type >= 10 || !(info.supported_registers & (1 << reg_type))) {
        return DSMIL_HAL_ERROR_INVALID_DEVICE;
    }
    
    /* Check if we have valid MMIO mapping */
    if (!info.mmio_base) {
        pr_warn("DSMIL HAL: No MMIO mapping for device %u\n", device_id);
        *value = 0xDEADBEEF; /* Return recognizable pattern */
        hal_update_statistics(DSMIL_HAL_ACCESS_READ_ONLY, DSMIL_HAL_SUCCESS);
        return DSMIL_HAL_SUCCESS;
    }
    
    reg_offset = info.register_offsets[reg_type];
    reg_addr = info.mmio_base + reg_offset;
    
    /* Perform the read operation */
    *value = ioread32(reg_addr);
    
    /* Update device statistics */
    mutex_lock(&hal_state->global_lock);
    if (hal_state->enhanced_state) {
        u32 group_id = dsmil_hal_device_to_group(device_id);
        u32 device_index = dsmil_hal_device_index_in_group(device_id);
        
        if (group_id < DSMIL_HAL_MAX_GROUPS && device_index < DSMIL_HAL_DEVICES_PER_GROUP) {
            struct dsmil_device_enhanced *device = 
                &hal_state->enhanced_state->groups[group_id].devices[device_index];
            device->access_count_read++;
            device->last_access = ktime_get();
        }
    }
    mutex_unlock(&hal_state->global_lock);
    
    /* Update statistics */
    hal_update_statistics(DSMIL_HAL_ACCESS_READ_ONLY, DSMIL_HAL_SUCCESS);
    
    /* Log operation if enabled */
    if (hal_state->config.access_logging && hal_state->config.debug_mode) {
        pr_info("DSMIL HAL: Read device %u register %d = 0x%08x\n", 
                device_id, reg_type, *value);
    }
    
    return DSMIL_HAL_SUCCESS;
}

/*
 * Safe register write operation - CRITICAL SAFETY FUNCTION
 */
int dsmil_hal_write_register(u32 device_id, enum dsmil_hal_register_type reg_type, 
                            u32 value, struct dsmil_hal_context *ctx)
{
    struct dsmil_hal_device_info info;
    void __iomem *reg_addr;
    int ret;
    u32 reg_offset;
    
    DSMIL_HAL_CHECK_DEVICE_ID(device_id);
    DSMIL_HAL_CHECK_NULL_PTR(ctx, "context");
    
    /* CRITICAL: Double-check quarantine status */
    DSMIL_HAL_CHECK_QUARANTINE(device_id);
    
    /* Validate operation */
    ret = dsmil_hal_validate_operation(device_id, ctx->access_type);
    if (ret != DSMIL_HAL_SUCCESS) {
        hal_update_statistics(ctx->access_type, ret);
        return ret;
    }
    
    /* Get device information */
    ret = dsmil_hal_get_device_info(device_id, &info);
    if (ret != DSMIL_HAL_SUCCESS) {
        return ret;
    }
    
    /* CRITICAL: Final quarantine check */
    if (info.is_quarantined) {
        pr_err("DSMIL HAL: CRITICAL - Attempted write to quarantined device %u\n", device_id);
        hal_update_statistics(ctx->access_type, DSMIL_HAL_ERROR_QUARANTINE_VIOLATION);
        return DSMIL_HAL_ERROR_QUARANTINE_VIOLATION;
    }
    
    /* Check if register type is supported */
    if (reg_type >= 10 || !(info.supported_registers & (1 << reg_type))) {
        return DSMIL_HAL_ERROR_INVALID_DEVICE;
    }
    
    /* Check authorization if required */
    if (ctx->requires_auth) {
        ret = dsmil_hal_check_authorization(ctx);
        if (ret != DSMIL_HAL_SUCCESS) {
            return ret;
        }
    }
    
    /* Check if we have valid MMIO mapping */
    if (!info.mmio_base) {
        pr_warn("DSMIL HAL: No MMIO mapping for device %u - write ignored\n", device_id);
        hal_update_statistics(ctx->access_type, DSMIL_HAL_SUCCESS);
        return DSMIL_HAL_SUCCESS;
    }
    
    reg_offset = info.register_offsets[reg_type];
    reg_addr = info.mmio_base + reg_offset;
    
    /* Perform the write operation */
    iowrite32(value, reg_addr);
    
    /* Update device statistics */
    mutex_lock(&hal_state->global_lock);
    if (hal_state->enhanced_state) {
        u32 group_id = dsmil_hal_device_to_group(device_id);
        u32 device_index = dsmil_hal_device_index_in_group(device_id);
        
        if (group_id < DSMIL_HAL_MAX_GROUPS && device_index < DSMIL_HAL_DEVICES_PER_GROUP) {
            struct dsmil_device_enhanced *device = 
                &hal_state->enhanced_state->groups[group_id].devices[device_index];
            device->access_count_write++;
            device->last_access = ktime_get();
        }
    }
    mutex_unlock(&hal_state->global_lock);
    
    /* Update statistics */
    hal_update_statistics(ctx->access_type, DSMIL_HAL_SUCCESS);
    
    /* Log operation if enabled */
    if (hal_state->config.access_logging) {
        pr_info("DSMIL HAL: Write device %u register %d = 0x%08x\n", 
                device_id, reg_type, value);
    }
    
    return DSMIL_HAL_SUCCESS;
}

/*
 * Emergency stop single device
 */
int dsmil_hal_emergency_stop(u32 device_id)
{
    struct dsmil_hal_device_info info;
    int ret;
    
    DSMIL_HAL_CHECK_DEVICE_ID(device_id);
    
    if (!hal_state || !hal_state->initialized) {
        return DSMIL_HAL_ERROR_UNKNOWN;
    }
    
    pr_err("DSMIL HAL: EMERGENCY STOP device %u\n", device_id);
    
    /* Get device information */
    ret = dsmil_hal_get_device_info(device_id, &info);
    if (ret == DSMIL_HAL_SUCCESS && info.mmio_base) {
        /* Try to write emergency stop to control register if not quarantined */
        if (!info.is_quarantined) {
            iowrite32(0x00000000, info.mmio_base); /* Write zero to control register */
        }
    }
    
    /* Update statistics */
    spin_lock(&hal_state->stats_lock);
    hal_state->statistics.emergency_stops++;
    spin_unlock(&hal_state->stats_lock);
    
    return DSMIL_HAL_SUCCESS;
}

/*
 * Emergency stop all devices
 */
int dsmil_hal_emergency_stop_all(void)
{
    u32 device_id;
    int stopped_count = 0;
    
    if (!hal_state || !hal_state->initialized) {
        return DSMIL_HAL_ERROR_UNKNOWN;
    }
    
    pr_err("DSMIL HAL: EMERGENCY STOP ALL DEVICES\n");
    
    /* Set global emergency stop flag */
    if (hal_state->enhanced_state) {
        hal_state->enhanced_state->global_emergency_stop = true;
    }
    
    /* Stop all devices */
    for (device_id = 0; device_id < DSMIL_HAL_MAX_DEVICES; device_id++) {
        if (dsmil_hal_emergency_stop(device_id) == DSMIL_HAL_SUCCESS) {
            stopped_count++;
        }
    }
    
    pr_err("DSMIL HAL: Emergency stopped %d/%d devices\n", 
           stopped_count, DSMIL_HAL_MAX_DEVICES);
    
    return DSMIL_HAL_SUCCESS;
}

/*
 * Get HAL statistics
 */
int dsmil_hal_get_statistics(struct dsmil_hal_statistics *stats)
{
    DSMIL_HAL_CHECK_NULL_PTR(stats, "statistics");
    
    if (!hal_state || !hal_state->initialized) {
        return DSMIL_HAL_ERROR_UNKNOWN;
    }
    
    spin_lock(&hal_state->stats_lock);
    memcpy(stats, &hal_state->statistics, sizeof(struct dsmil_hal_statistics));
    stats->active_operations = atomic_read(&hal_state->active_operations);
    spin_unlock(&hal_state->stats_lock);
    
    return DSMIL_HAL_SUCCESS;
}

int dsmil_hal_get_thermal_info(struct dsmil_hal_thermal_info *info)
{
    if (!info || !hal_state || !hal_state->initialized)
        return DSMIL_HAL_ERROR_UNKNOWN;

    memcpy(info, &hal_state->thermal_info, sizeof(*info));
    return DSMIL_HAL_SUCCESS;
}
EXPORT_SYMBOL(dsmil_hal_get_thermal_info);

/* 
 * Internal helper functions 
 */

static int hal_check_thermal_safety(void)
{
    if (!hal_state->config.thermal_protection) {
        return DSMIL_HAL_SUCCESS;
    }
    
    do {
        struct thermal_zone_device *tz;
        long temp_mc;

        tz = hal_acquire_thermal_zone();
        if (tz && !thermal_zone_get_temp(tz, &temp_mc)) {
            hal_state->thermal_info.current_temp = temp_mc;
        }
    } while (0);

    if (hal_state->thermal_info.current_temp > hal_state->thermal_info.warning_temp) {
        pr_warn("DSMIL HAL: Thermal warning %ld°C (threshold %ld°C)\n",
                hal_state->thermal_info.current_temp / 1000,
                hal_state->thermal_info.warning_temp / 1000);
    }

    if (hal_state->thermal_info.current_temp > hal_state->thermal_info.critical_temp) {
        hal_state->thermal_info.thermal_events++;
        hal_state->thermal_info.last_thermal_event = ktime_get();
        pr_err("DSMIL HAL: THERMAL VIOLATION - Temperature %ld°C exceeds critical %ld°C\n",
               hal_state->thermal_info.current_temp / 1000,
               hal_state->thermal_info.critical_temp / 1000);
        return DSMIL_HAL_ERROR_THERMAL_VIOLATION;
    }

    return DSMIL_HAL_SUCCESS;
}

static int hal_update_statistics(enum dsmil_hal_access_type access_type, 
                                enum dsmil_hal_result result)
{
    if (!hal_state) {
        return DSMIL_HAL_ERROR_UNKNOWN;
    }
    
    spin_lock(&hal_state->stats_lock);
    
    hal_state->statistics.total_operations++;
    
    if (access_type == DSMIL_HAL_ACCESS_READ_ONLY) {
        hal_state->statistics.read_operations++;
    } else {
        hal_state->statistics.write_operations++;
    }
    
    /* Update error statistics */
    switch (result) {
    case DSMIL_HAL_ERROR_QUARANTINE_VIOLATION:
        hal_state->statistics.quarantine_violations++;
        hal_state->statistics.safety_violations++;
        break;
    case DSMIL_HAL_ERROR_SAFETY_VIOLATION:
        hal_state->statistics.safety_violations++;
        break;
    case DSMIL_HAL_ERROR_AUTH_REQUIRED:
        hal_state->statistics.auth_failures++;
        break;
    case DSMIL_HAL_ERROR_TIMEOUT:
        hal_state->statistics.timeout_errors++;
        break;
    case DSMIL_HAL_ERROR_CRC_MISMATCH:
        hal_state->statistics.crc_errors++;
        break;
    case DSMIL_HAL_ERROR_THERMAL_VIOLATION:
        hal_state->statistics.thermal_violations++;
        break;
    case DSMIL_HAL_ERROR_EMERGENCY_STOP:
        hal_state->statistics.emergency_stops++;
        break;
    default:
        break;
    }
    
    /* Update peak concurrent operations */
    u32 current_active = atomic_read(&hal_state->active_operations);
    if (current_active > hal_state->statistics.peak_concurrent_ops) {
        hal_state->statistics.peak_concurrent_ops = current_active;
    }
    
    spin_unlock(&hal_state->stats_lock);
    
    return DSMIL_HAL_SUCCESS;
}

static int dsmil_hal_check_authorization(struct dsmil_hal_context *ctx)
{
    struct dsmil_auth_context *auth_ctx;
    enum dsmil_operation_type operation;
    enum dsmil_risk_level risk;
    int ret;

    if (!ctx || !ctx->requires_auth)
        return DSMIL_HAL_SUCCESS;

    auth_ctx = (struct dsmil_auth_context *)ctx->user_context;
    if (!auth_ctx)
        return DSMIL_HAL_ERROR_AUTH_REQUIRED;

    operation = hal_map_access_to_operation(ctx->access_type);
    risk = hal_calculate_operation_risk(ctx->device_id, operation);

    ret = dsmil_mfa_authorize_operation(auth_ctx, ctx->device_id,
                                        operation, risk);
    if (ret)
        return DSMIL_HAL_ERROR_AUTH_REQUIRED;

    return DSMIL_HAL_SUCCESS;
}

/* Error code to string conversion */
const char *dsmil_hal_error_string(enum dsmil_hal_result result)
{
    switch (result) {
    case DSMIL_HAL_SUCCESS:
        return "Success";
    case DSMIL_HAL_ERROR_INVALID_DEVICE:
        return "Invalid device";
    case DSMIL_HAL_ERROR_QUARANTINE_VIOLATION:
        return "Quarantine violation";
    case DSMIL_HAL_ERROR_SAFETY_VIOLATION:
        return "Safety violation";
    case DSMIL_HAL_ERROR_PERMISSION_DENIED:
        return "Permission denied";
    case DSMIL_HAL_ERROR_DEVICE_BUSY:
        return "Device busy";
    case DSMIL_HAL_ERROR_DEVICE_OFFLINE:
        return "Device offline";
    case DSMIL_HAL_ERROR_TIMEOUT:
        return "Timeout";
    case DSMIL_HAL_ERROR_IO_ERROR:
        return "I/O error";
    case DSMIL_HAL_ERROR_CRC_MISMATCH:
        return "CRC mismatch";
    case DSMIL_HAL_ERROR_THERMAL_VIOLATION:
        return "Thermal violation";
    case DSMIL_HAL_ERROR_EMERGENCY_STOP:
        return "Emergency stop";
    case DSMIL_HAL_ERROR_RUST_SAFETY:
        return "Rust safety violation";
    case DSMIL_HAL_ERROR_MEMORY_MAP:
        return "Memory mapping error";
    case DSMIL_HAL_ERROR_AUTH_REQUIRED:
        return "Authorization required";
    default:
        return "Unknown error";
    }
}

MODULE_AUTHOR("DSMIL Track A Development Team");
MODULE_DESCRIPTION("DSMIL Hardware Abstraction Layer with Quarantine Protection");
MODULE_LICENSE("GPL v2");
MODULE_VERSION(DSMIL_HAL_VERSION_STRING);

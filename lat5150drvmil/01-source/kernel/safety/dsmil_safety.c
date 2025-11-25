/*
 * Dell MIL-SPEC Enhanced DSMIL Safety Validation System
 * 
 * Copyright (C) 2025 JRTC1 Educational Development
 * 
 * This module implements comprehensive safety validation and quarantine
 * enforcement for the 84-device DSMIL system. It provides multiple
 * layers of protection to prevent any writes to quarantined devices.
 * 
 * CRITICAL SAFETY: This system enforces absolute protection for 5
 * quarantined devices that control critical system functions.
 */

#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/slab.h>
#include <linux/mutex.h>
#include <linux/crc32.h>
#include <linux/crypto.h>
#include <linux/random.h>
#include <linux/time.h>
#include <linux/proc_fs.h>
#include <linux/seq_file.h>
#include <linux/uaccess.h>

#include "dsmil_hal.h"

#define DSMIL_SAFETY_VERSION    "1.0.0"
#define DSMIL_SAFETY_MAGIC      0x53414645  /* "SAFE" */

/* Safety Validation Levels */
#define SAFETY_LEVEL_NONE       0
#define SAFETY_LEVEL_BASIC      1
#define SAFETY_LEVEL_STANDARD   2
#define SAFETY_LEVEL_ENHANCED   3
#define SAFETY_LEVEL_MAXIMUM    4

/* Safety Check Types */
#define SAFETY_CHECK_QUARANTINE     0x01
#define SAFETY_CHECK_DEVICE_STATE   0x02
#define SAFETY_CHECK_PERMISSIONS    0x04
#define SAFETY_CHECK_THERMAL        0x08
#define SAFETY_CHECK_CRC            0x10
#define SAFETY_CHECK_AUTH           0x20
#define SAFETY_CHECK_RUST           0x40
#define SAFETY_CHECK_ALL            0xFF

/* Safety Violation Types */
enum dsmil_safety_violation_type {
    SAFETY_VIOLATION_QUARANTINE_ACCESS = 1,
    SAFETY_VIOLATION_UNAUTHORIZED_WRITE,
    SAFETY_VIOLATION_DEVICE_STATE,
    SAFETY_VIOLATION_THERMAL_LIMIT,
    SAFETY_VIOLATION_CRC_MISMATCH,
    SAFETY_VIOLATION_AUTH_FAILURE,
    SAFETY_VIOLATION_RUST_PANIC,
    SAFETY_VIOLATION_MEMORY_CORRUPTION,
    SAFETY_VIOLATION_EMERGENCY_TRIGGERED
};

/* Safety Violation Record */
struct dsmil_safety_violation {
    u32 violation_id;
    enum dsmil_safety_violation_type type;
    u32 device_id;
    ktime_t timestamp;
    char description[256];
    u32 severity;      /* 1=low, 5=critical */
    bool auto_resolved;
    u32 context_data[4]; /* Additional context */
};

/* Safety Monitor State */
struct dsmil_safety_monitor {
    struct mutex monitor_lock;
    
    /* Statistics */
    u64 total_checks;
    u64 violations_detected;
    u64 quarantine_blocks;
    u64 auto_resolutions;
    
    /* Current safety level */
    u32 current_safety_level;
    bool emergency_mode;
    bool monitoring_active;
    
    /* Violation history */
    struct dsmil_safety_violation violations[100];
    u32 violation_count;
    u32 violation_index;
    
    /* Safety configuration */
    u32 enabled_checks;
    u32 thermal_threshold;
    bool strict_quarantine;
    bool panic_on_violation;
    
    /* Timing */
    ktime_t last_full_check;
    ktime_t init_time;
    
    /* ProcFS entry */
    struct proc_dir_entry *proc_entry;
};

/* Global safety monitor */
static struct dsmil_safety_monitor *safety_monitor = NULL;

/* Critical quarantine list - MUST match HAL and enhanced module */
static const u32 SAFETY_QUARANTINE_DEVICES[] = {
    0,   /* Group 0, Device 0 - Master Control */
    1,   /* Group 0, Device 1 - Security Platform */
    12,  /* Group 1, Device 0 - Power Management */
    24,  /* Group 2, Device 0 - Memory Controller */
    83   /* Group 6, Device 11 - Emergency Stop */
};

static const char *SAFETY_QUARANTINE_REASONS[] = {
    "Master Control - System lockdown risk",
    "Security Platform - Authentication bypass risk", 
    "Power Management - Hardware damage risk",
    "Memory Controller - Memory corruption risk",
    "Emergency Stop - Safety system compromise risk"
};

/* Forward declarations */
static int dsmil_safety_check_quarantine_violation(u32 device_id);
static int dsmil_safety_check_device_state(u32 device_id);
static int dsmil_safety_check_thermal_limits(void);
static int dsmil_safety_record_violation(enum dsmil_safety_violation_type type,
                                        u32 device_id, const char *description,
                                        u32 severity);
static void dsmil_safety_emergency_response(enum dsmil_safety_violation_type type,
                                          u32 device_id);

/* ProcFS interface */
static int dsmil_safety_proc_show(struct seq_file *m, void *v);
static int dsmil_safety_proc_open(struct inode *inode, struct file *file);
static const struct proc_ops dsmil_safety_proc_ops = {
    .proc_open    = dsmil_safety_proc_open,
    .proc_read    = seq_read,
    .proc_lseek   = seq_lseek,
    .proc_release = single_release,
};

/*
 * Initialize safety monitoring system
 */
int dsmil_safety_init(void)
{
    int i;
    
    if (safety_monitor) {
        pr_warn("DSMIL Safety: Already initialized\n");
        return 0;
    }
    
    /* Allocate safety monitor structure */
    safety_monitor = kzalloc(sizeof(struct dsmil_safety_monitor), GFP_KERNEL);
    if (!safety_monitor) {
        pr_err("DSMIL Safety: Failed to allocate monitor structure\n");
        return -ENOMEM;
    }
    
    /* Initialize mutex */
    mutex_init(&safety_monitor->monitor_lock);
    
    /* Initialize statistics */
    safety_monitor->total_checks = 0;
    safety_monitor->violations_detected = 0;
    safety_monitor->quarantine_blocks = 0;
    safety_monitor->auto_resolutions = 0;
    
    /* Set initial safety configuration */
    safety_monitor->current_safety_level = SAFETY_LEVEL_MAXIMUM;
    safety_monitor->emergency_mode = false;
    safety_monitor->monitoring_active = true;
    safety_monitor->enabled_checks = SAFETY_CHECK_ALL;
    safety_monitor->thermal_threshold = 85000; /* 85°C in milli-Celsius */
    safety_monitor->strict_quarantine = true;
    safety_monitor->panic_on_violation = false; /* Don't panic in training mode */
    
    /* Initialize violation tracking */
    safety_monitor->violation_count = 0;
    safety_monitor->violation_index = 0;
    for (i = 0; i < 100; i++) {
        memset(&safety_monitor->violations[i], 0, sizeof(struct dsmil_safety_violation));
    }
    
    /* Set timestamps */
    safety_monitor->last_full_check = ktime_get();
    safety_monitor->init_time = ktime_get();
    
    /* Create ProcFS entry */
    safety_monitor->proc_entry = proc_create("dsmil_safety", 0444, NULL, 
                                            &dsmil_safety_proc_ops);
    if (!safety_monitor->proc_entry) {
        pr_warn("DSMIL Safety: Failed to create ProcFS entry\n");
    }
    
    pr_info("DSMIL Safety: Initialized (version %s)\n", DSMIL_SAFETY_VERSION);
    pr_info("DSMIL Safety: Protection level MAXIMUM, monitoring %u quarantined devices\n",
            (u32)ARRAY_SIZE(SAFETY_QUARANTINE_DEVICES));
    
    /* Perform initial safety check */
    dsmil_safety_comprehensive_check();
    
    return 0;
}

/*
 * Cleanup safety monitoring system
 */
void dsmil_safety_cleanup(void)
{
    if (!safety_monitor) {
        return;
    }
    
    pr_info("DSMIL Safety: Shutting down monitoring system\n");
    
    /* Remove ProcFS entry */
    if (safety_monitor->proc_entry) {
        proc_remove(safety_monitor->proc_entry);
    }
    
    /* Print final statistics */
    pr_info("DSMIL Safety: Final stats - Checks: %llu, Violations: %llu, Quarantine blocks: %llu\n",
            safety_monitor->total_checks, safety_monitor->violations_detected,
            safety_monitor->quarantine_blocks);
    
    /* Free memory */
    kfree(safety_monitor);
    safety_monitor = NULL;
    
    pr_info("DSMIL Safety: Cleanup complete\n");
}

/*
 * CRITICAL FUNCTION: Check for quarantine violations
 * This is the primary defense against writes to quarantined devices
 */
int dsmil_safety_validate_access(u32 device_id, enum dsmil_hal_access_type access_type)
{
    int ret = 0;
    u32 checks_performed = 0;
    
    if (!safety_monitor || !safety_monitor->monitoring_active) {
        pr_err("DSMIL Safety: Monitor not active - BLOCKING access to device %u\n", device_id);
        return -EPERM;
    }
    
    mutex_lock(&safety_monitor->monitor_lock);
    safety_monitor->total_checks++;
    
    /* CRITICAL: Check quarantine status first */
    if (safety_monitor->enabled_checks & SAFETY_CHECK_QUARANTINE) {
        ret = dsmil_safety_check_quarantine_violation(device_id);
        if (ret < 0) {
            if (access_type > DSMIL_HAL_ACCESS_READ_ONLY) {
                safety_monitor->quarantine_blocks++;
                dsmil_safety_record_violation(SAFETY_VIOLATION_QUARANTINE_ACCESS,
                                             device_id,
                                             "Attempted write to quarantined device",
                                             5); /* Critical severity */
                dsmil_safety_emergency_response(SAFETY_VIOLATION_QUARANTINE_ACCESS, device_id);
                pr_err("DSMIL Safety: QUARANTINE VIOLATION BLOCKED - Device %u write attempt\n", 
                       device_id);
            }
            goto exit_with_lock;
        }
        checks_performed++;
    }
    
    /* Check device state */
    if (safety_monitor->enabled_checks & SAFETY_CHECK_DEVICE_STATE) {
        ret = dsmil_safety_check_device_state(device_id);
        if (ret < 0) {
            dsmil_safety_record_violation(SAFETY_VIOLATION_DEVICE_STATE,
                                         device_id,
                                         "Device in unsafe state",
                                         3); /* Medium severity */
            goto exit_with_lock;
        }
        checks_performed++;
    }
    
    /* Check thermal limits */
    if (safety_monitor->enabled_checks & SAFETY_CHECK_THERMAL) {
        ret = dsmil_safety_check_thermal_limits();
        if (ret < 0) {
            dsmil_safety_record_violation(SAFETY_VIOLATION_THERMAL_LIMIT,
                                         device_id,
                                         "Thermal limits exceeded",
                                         4); /* High severity */
            goto exit_with_lock;
        }
        checks_performed++;
    }
    
    /* All checks passed */
    ret = 0;
    
exit_with_lock:
    mutex_unlock(&safety_monitor->monitor_lock);
    
    if (ret < 0) {
        pr_warn("DSMIL Safety: Access denied for device %u (performed %u checks)\n",
                device_id, checks_performed);
    }
    
    return ret;
}

/*
 * Check if device is in quarantine list
 */
static int dsmil_safety_check_quarantine_violation(u32 device_id)
{
    int i;
    
    /* Check device ID validity */
    if (device_id >= 84) {
        pr_err("DSMIL Safety: Invalid device ID %u (max 83)\n", device_id);
        return -EINVAL;
    }
    
    /* Check against quarantine list */
    for (i = 0; i < ARRAY_SIZE(SAFETY_QUARANTINE_DEVICES); i++) {
        if (SAFETY_QUARANTINE_DEVICES[i] == device_id) {
            return -EPERM; /* Device is quarantined */
        }
    }
    
    return 0; /* Device is not quarantined */
}

/*
 * Check device state safety
 */
static int dsmil_safety_check_device_state(u32 device_id)
{
    struct dsmil_hal_device_info info;
    int ret;

    ret = dsmil_hal_get_device_info(device_id, &info);
    if (ret != DSMIL_HAL_SUCCESS)
        return -EINVAL;

    if (info.is_quarantined)
        return -EPERM;

    if (info.current_state == DSMIL_DEVICE_ERROR ||
        info.current_state == DSMIL_DEVICE_LOCKED) {
        dsmil_safety_record_violation(SAFETY_VIOLATION_DEVICE_STATE,
                                      device_id,
                                      "Device reported unsafe state",
                                      4);
        return -EFAULT;
    }

    return 0;
}

/*
 * Check thermal safety limits
 */
static int dsmil_safety_check_thermal_limits(void)
{
    struct dsmil_hal_thermal_info info;
    int ret;

    ret = dsmil_hal_get_thermal_info(&info);
    if (ret != DSMIL_HAL_SUCCESS)
        return 0;

    if (info.current_temp > safety_monitor->thermal_threshold) {
        dsmil_safety_record_violation(SAFETY_VIOLATION_THERMAL_LIMIT,
                                      0,
                                      "Thermal limit exceeded",
                                      5);
        return -EAGAIN;
    }

    return 0;
}

/*
 * Record a safety violation
 */
static int dsmil_safety_record_violation(enum dsmil_safety_violation_type type,
                                        u32 device_id, const char *description,
                                        u32 severity)
{
    struct dsmil_safety_violation *violation;
    static u32 violation_id_counter = 1;
    
    if (!safety_monitor) {
        return -EINVAL;
    }
    
    /* Get next violation slot (circular buffer) */
    violation = &safety_monitor->violations[safety_monitor->violation_index];
    safety_monitor->violation_index = (safety_monitor->violation_index + 1) % 100;
    if (safety_monitor->violation_count < 100) {
        safety_monitor->violation_count++;
    }
    
    /* Fill violation record */
    violation->violation_id = violation_id_counter++;
    violation->type = type;
    violation->device_id = device_id;
    violation->timestamp = ktime_get();
    violation->severity = severity;
    violation->auto_resolved = false;
    
    strncpy(violation->description, description, sizeof(violation->description) - 1);
    violation->description[sizeof(violation->description) - 1] = '\0';
    
    /* Clear context data */
    memset(violation->context_data, 0, sizeof(violation->context_data));
    
    /* Update global statistics */
    safety_monitor->violations_detected++;
    
    /* Log the violation */
    pr_warn("DSMIL Safety: VIOLATION #%u - Type %d, Device %u, Severity %u: %s\n",
            violation->violation_id, type, device_id, severity, description);
    
    return 0;
}

/*
 * Emergency response to safety violations
 */
static void dsmil_safety_emergency_response(enum dsmil_safety_violation_type type,
                                          u32 device_id)
{
    if (!safety_monitor) {
        return;
    }
    
    /* Take action based on violation type */
    switch (type) {
    case SAFETY_VIOLATION_QUARANTINE_ACCESS:
        pr_err("DSMIL Safety: CRITICAL QUARANTINE VIOLATION on device %u\n", device_id);
        if (safety_monitor->strict_quarantine) {
            /* In strict mode, trigger emergency stop */
            dsmil_hal_emergency_stop(device_id);
        }
        break;
        
    case SAFETY_VIOLATION_THERMAL_LIMIT:
        pr_err("DSMIL Safety: THERMAL EMERGENCY - initiating thermal shutdown\n");
        /* Trigger thermal emergency response */
        dsmil_hal_emergency_stop_all();
        break;
        
    case SAFETY_VIOLATION_MEMORY_CORRUPTION:
        pr_err("DSMIL Safety: MEMORY CORRUPTION detected - emergency stop\n");
        dsmil_hal_emergency_stop_all();
        break;
        
    default:
        pr_warn("DSMIL Safety: Emergency response for violation type %d on device %u\n",
                type, device_id);
        break;
    }
    
    /* If configured to panic on violations (not recommended for training) */
    if (safety_monitor->panic_on_violation && type == SAFETY_VIOLATION_QUARANTINE_ACCESS) {
        panic("DSMIL Safety: Critical quarantine violation - system halt");
    }
}

/*
 * Comprehensive safety check of entire system
 */
int dsmil_safety_comprehensive_check(void)
{
    u32 device_id;
    int violations_found = 0;
    ktime_t start_time, end_time;
    
    if (!safety_monitor) {
        return -EINVAL;
    }
    
    start_time = ktime_get();
    
    pr_info("DSMIL Safety: Starting comprehensive safety check\n");
    
    mutex_lock(&safety_monitor->monitor_lock);
    
    /* Check all devices */
    for (device_id = 0; device_id < 84; device_id++) {
        /* Check quarantine status */
        if (dsmil_safety_check_quarantine_violation(device_id) == -EPERM) {
            /* This is expected for quarantined devices */
            continue;
        }
        
        /* Check device state */
        if (dsmil_safety_check_device_state(device_id) < 0) {
            violations_found++;
        }
        
        safety_monitor->total_checks++;
    }
    
    /* Check global thermal status */
    if (dsmil_safety_check_thermal_limits() < 0) {
        violations_found++;
    }
    
    safety_monitor->last_full_check = ktime_get();
    
    mutex_unlock(&safety_monitor->monitor_lock);
    
    end_time = ktime_get();
    
    pr_info("DSMIL Safety: Comprehensive check complete - %d violations found in %lld ms\n",
            violations_found, ktime_to_ms(ktime_sub(end_time, start_time)));
    
    return violations_found;
}

/*
 * Get safety statistics
 */
int dsmil_safety_get_statistics(struct dsmil_safety_statistics *stats)
{
    if (!safety_monitor || !stats) {
        return -EINVAL;
    }
    
    mutex_lock(&safety_monitor->monitor_lock);
    
    stats->total_checks = safety_monitor->total_checks;
    stats->violations_detected = safety_monitor->violations_detected;
    stats->quarantine_blocks = safety_monitor->quarantine_blocks;
    stats->auto_resolutions = safety_monitor->auto_resolutions;
    stats->current_safety_level = safety_monitor->current_safety_level;
    stats->emergency_mode = safety_monitor->emergency_mode;
    stats->monitoring_active = safety_monitor->monitoring_active;
    stats->violation_count = safety_monitor->violation_count;
    stats->uptime_seconds = ktime_to_ms(ktime_sub(ktime_get(), safety_monitor->init_time)) / 1000;
    
    mutex_unlock(&safety_monitor->monitor_lock);
    
    return 0;
}

/*
 * ProcFS interface implementation
 */
static int dsmil_safety_proc_show(struct seq_file *m, void *v)
{
    int i, quarantine_count = 0;
    
    if (!safety_monitor) {
        seq_printf(m, "DSMIL Safety Monitor: Not initialized\n");
        return 0;
    }
    
    mutex_lock(&safety_monitor->monitor_lock);
    
    seq_printf(m, "DSMIL Safety Monitor Status (version %s)\n", DSMIL_SAFETY_VERSION);
    seq_printf(m, "=====================================\n\n");
    
    /* System status */
    seq_printf(m, "System Status:\n");
    seq_printf(m, "  Safety Level: %u (1=basic, 4=maximum)\n", safety_monitor->current_safety_level);
    seq_printf(m, "  Emergency Mode: %s\n", safety_monitor->emergency_mode ? "ACTIVE" : "Normal");
    seq_printf(m, "  Monitoring Active: %s\n", safety_monitor->monitoring_active ? "Yes" : "No");
    seq_printf(m, "  Strict Quarantine: %s\n", safety_monitor->strict_quarantine ? "Enabled" : "Disabled");
    seq_printf(m, "\n");
    
    /* Statistics */
    seq_printf(m, "Statistics:\n");
    seq_printf(m, "  Total Safety Checks: %llu\n", safety_monitor->total_checks);
    seq_printf(m, "  Violations Detected: %llu\n", safety_monitor->violations_detected);
    seq_printf(m, "  Quarantine Blocks: %llu\n", safety_monitor->quarantine_blocks);
    seq_printf(m, "  Auto Resolutions: %llu\n", safety_monitor->auto_resolutions);
    seq_printf(m, "  Uptime: %lld seconds\n", 
               ktime_to_ms(ktime_sub(ktime_get(), safety_monitor->init_time)) / 1000);
    seq_printf(m, "\n");
    
    /* Quarantine information */
    seq_printf(m, "Quarantined Devices:\n");
    for (i = 0; i < ARRAY_SIZE(SAFETY_QUARANTINE_DEVICES); i++) {
        seq_printf(m, "  Device %u: %s\n", 
                   SAFETY_QUARANTINE_DEVICES[i], SAFETY_QUARANTINE_REASONS[i]);
        quarantine_count++;
    }
    seq_printf(m, "  Total: %d devices under quarantine protection\n", quarantine_count);
    seq_printf(m, "\n");
    
    /* Recent violations */
    if (safety_monitor->violation_count > 0) {
        seq_printf(m, "Recent Violations (last %u):\n", 
                   min(safety_monitor->violation_count, 10u));
        
        u32 start_idx = safety_monitor->violation_count > 10 ? 
                       (safety_monitor->violation_index + 90) % 100 : 0;
        
        for (i = 0; i < min(safety_monitor->violation_count, 10u); i++) {
            u32 idx = (start_idx + i) % 100;
            struct dsmil_safety_violation *v = &safety_monitor->violations[idx];
            
            seq_printf(m, "  #%u: Type %d, Device %u, Severity %u - %s\n",
                       v->violation_id, v->type, v->device_id, v->severity, v->description);
        }
    } else {
        seq_printf(m, "No violations recorded\n");
    }
    
    mutex_unlock(&safety_monitor->monitor_lock);
    
    return 0;
}

static int dsmil_safety_proc_open(struct inode *inode, struct file *file)
{
    return single_open(file, dsmil_safety_proc_show, NULL);
}

/* Export functions for use by other modules */
EXPORT_SYMBOL(dsmil_safety_init);
EXPORT_SYMBOL(dsmil_safety_cleanup);
EXPORT_SYMBOL(dsmil_safety_validate_access);
EXPORT_SYMBOL(dsmil_safety_comprehensive_check);
EXPORT_SYMBOL(dsmil_safety_get_statistics);

MODULE_AUTHOR("DSMIL Track A Development Team");
MODULE_DESCRIPTION("DSMIL Comprehensive Safety Validation System");
MODULE_LICENSE("GPL v2");
MODULE_VERSION(DSMIL_SAFETY_VERSION);

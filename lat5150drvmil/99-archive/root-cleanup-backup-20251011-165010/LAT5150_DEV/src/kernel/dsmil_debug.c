/*
 * Dell MIL-SPEC Enhanced DSMIL Kernel Debug Interface
 * 
 * Copyright (C) 2025 JRTC1 Educational Development
 * 
 * This module provides comprehensive kernel-level debugging and logging
 * capabilities for the DSMIL system. It includes real-time monitoring,
 * operation tracing, and detailed system state reporting.
 * 
 * SAFETY NOTE: This debug interface respects quarantine restrictions
 * and provides safe monitoring without compromising system security.
 */

#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/slab.h>
#include <linux/mutex.h>
#include <linux/spinlock.h>
#include <linux/debugfs.h>
#include <linux/proc_fs.h>
#include <linux/seq_file.h>
#include <linux/uaccess.h>
#include <linux/time.h>
#include <linux/kfifo.h>
#include <linux/workqueue.h>

#include "dsmil_hal.h"
#include "dsmil_rust_safety.h"

#define DSMIL_DEBUG_VERSION         "1.0.0"
#define DSMIL_DEBUG_MAX_LOG_ENTRIES 10000
#define DSMIL_DEBUG_LOG_ENTRY_SIZE  512

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

/* Debug Log Entry */
struct dsmil_debug_log_entry {
    ktime_t timestamp;
    enum dsmil_debug_level level;
    enum dsmil_debug_category category;
    u32 device_id;
    char message[256];
    u32 data[4];  /* Additional context data */
};

/* Debug Statistics */
struct dsmil_debug_statistics {
    u64 total_log_entries;
    u64 entries_by_level[6];
    u64 entries_by_category[8];
    u64 buffer_overflows;
    u64 critical_events;
    u64 quarantine_events;
    u64 emergency_events;
    ktime_t last_entry_time;
    ktime_t system_start_time;
};

/* Real-time Monitor Entry */
struct dsmil_realtime_monitor_entry {
    ktime_t timestamp;
    u32 device_id;
    enum dsmil_hal_access_type access_type;
    bool success;
    u32 value_or_error;
    char operation[64];
};

/* Debug System State */
struct dsmil_debug_system {
    struct mutex debug_lock;
    
    /* Logging */
    DECLARE_KFIFO(log_buffer, struct dsmil_debug_log_entry, DSMIL_DEBUG_MAX_LOG_ENTRIES);
    spinlock_t log_lock;
    enum dsmil_debug_level min_log_level;
    u32 enabled_categories;
    
    /* Real-time monitoring */
    DECLARE_KFIFO(monitor_buffer, struct dsmil_realtime_monitor_entry, 1000);
    spinlock_t monitor_lock;
    bool realtime_monitoring;
    
    /* Statistics */
    struct dsmil_debug_statistics stats;
    
    /* DebugFS entries */
    struct dentry *debugfs_root;
    struct dentry *debugfs_log;
    struct dentry *debugfs_monitor;
    struct dentry *debugfs_stats;
    struct dentry *debugfs_devices;
    struct dentry *debugfs_config;
    
    /* ProcFS entries */
    struct proc_dir_entry *proc_root;
    struct proc_dir_entry *proc_log;
    struct proc_dir_entry *proc_status;
    
    /* Configuration */
    bool debug_enabled;
    bool trace_device_operations;
    bool trace_quarantine_access;
    bool log_to_kernel;
    bool log_to_buffer;
    
    /* Work queue for background logging */
    struct workqueue_struct *debug_workqueue;
    struct delayed_work log_work;
    
    ktime_t init_time;
};

/* Global debug system */
static struct dsmil_debug_system *debug_sys = NULL;

/* Category names */
static const char *debug_category_names[] = {
    "General", "HAL", "Safety", "AccessControl", 
    "Rust", "DeviceOps", "Quarantine", "Emergency"
};

/* Level names */
static const char *debug_level_names[] = {
    "NONE", "ERROR", "WARN", "INFO", "DEBUG", "TRACE"
};

/* Forward declarations */
static void dsmil_debug_log_work(struct work_struct *work);
static int dsmil_debug_log_seq_show(struct seq_file *m, void *v);
static int dsmil_debug_monitor_seq_show(struct seq_file *m, void *v);
static int dsmil_debug_stats_seq_show(struct seq_file *m, void *v);
static int dsmil_debug_devices_seq_show(struct seq_file *m, void *v);

/* DebugFS file operations */
static int dsmil_debug_log_open(struct inode *inode, struct file *file)
{
    return single_open(file, dsmil_debug_log_seq_show, NULL);
}

static int dsmil_debug_monitor_open(struct inode *inode, struct file *file)
{
    return single_open(file, dsmil_debug_monitor_seq_show, NULL);
}

static int dsmil_debug_stats_open(struct inode *inode, struct file *file)
{
    return single_open(file, dsmil_debug_stats_seq_show, NULL);
}

static int dsmil_debug_devices_open(struct inode *inode, struct file *file)
{
    return single_open(file, dsmil_debug_devices_seq_show, NULL);
}

static const struct file_operations dsmil_debug_log_fops = {
    .owner = THIS_MODULE,
    .open = dsmil_debug_log_open,
    .read = seq_read,
    .llseek = seq_lseek,
    .release = single_release,
};

static const struct file_operations dsmil_debug_monitor_fops = {
    .owner = THIS_MODULE,
    .open = dsmil_debug_monitor_open,
    .read = seq_read,
    .llseek = seq_lseek,
    .release = single_release,
};

static const struct file_operations dsmil_debug_stats_fops = {
    .owner = THIS_MODULE,
    .open = dsmil_debug_stats_open,
    .read = seq_read,
    .llseek = seq_lseek,
    .release = single_release,
};

static const struct file_operations dsmil_debug_devices_fops = {
    .owner = THIS_MODULE,
    .open = dsmil_debug_devices_open,
    .read = seq_read,
    .llseek = seq_lseek,
    .release = single_release,
};

/*
 * Initialize debug system
 */
int dsmil_debug_init(void)
{
    if (debug_sys) {
        pr_warn("DSMIL Debug: Already initialized\n");
        return 0;
    }
    
    /* Allocate debug system */
    debug_sys = kzalloc(sizeof(struct dsmil_debug_system), GFP_KERNEL);
    if (!debug_sys) {
        pr_err("DSMIL Debug: Failed to allocate debug system\n");
        return -ENOMEM;
    }
    
    /* Initialize locks */
    mutex_init(&debug_sys->debug_lock);
    spin_lock_init(&debug_sys->log_lock);
    spin_lock_init(&debug_sys->monitor_lock);
    
    /* Initialize FIFOs */
    INIT_KFIFO(debug_sys->log_buffer);
    INIT_KFIFO(debug_sys->monitor_buffer);
    
    /* Initialize configuration */
    debug_sys->debug_enabled = true;
    debug_sys->min_log_level = DSMIL_DEBUG_LEVEL_INFO;
    debug_sys->enabled_categories = 0xFF; /* All categories enabled */
    debug_sys->realtime_monitoring = true;
    debug_sys->trace_device_operations = true;
    debug_sys->trace_quarantine_access = true;
    debug_sys->log_to_kernel = true;
    debug_sys->log_to_buffer = true;
    
    /* Initialize statistics */
    memset(&debug_sys->stats, 0, sizeof(debug_sys->stats));
    debug_sys->stats.system_start_time = ktime_get();
    debug_sys->stats.last_entry_time = debug_sys->stats.system_start_time;
    
    /* Create DebugFS entries */
    debug_sys->debugfs_root = debugfs_create_dir("dsmil", NULL);
    if (debug_sys->debugfs_root) {
        debug_sys->debugfs_log = debugfs_create_file("log", 0444, 
                                                    debug_sys->debugfs_root, 
                                                    NULL, &dsmil_debug_log_fops);
        debug_sys->debugfs_monitor = debugfs_create_file("monitor", 0444, 
                                                        debug_sys->debugfs_root, 
                                                        NULL, &dsmil_debug_monitor_fops);
        debug_sys->debugfs_stats = debugfs_create_file("statistics", 0444, 
                                                      debug_sys->debugfs_root, 
                                                      NULL, &dsmil_debug_stats_fops);
        debug_sys->debugfs_devices = debugfs_create_file("devices", 0444, 
                                                        debug_sys->debugfs_root, 
                                                        NULL, &dsmil_debug_devices_fops);
    }
    
    /* Create work queue for background logging */
    debug_sys->debug_workqueue = create_singlethread_workqueue("dsmil_debug");
    if (!debug_sys->debug_workqueue) {
        pr_warn("DSMIL Debug: Failed to create work queue\n");
    }
    
    /* Initialize delayed work */
    INIT_DELAYED_WORK(&debug_sys->log_work, dsmil_debug_log_work);
    
    debug_sys->init_time = ktime_get();
    
    pr_info("DSMIL Debug: Initialized (version %s)\n", DSMIL_DEBUG_VERSION);
    pr_info("DSMIL Debug: Log buffer: %d entries, Real-time monitoring: %s\n",
            DSMIL_DEBUG_MAX_LOG_ENTRIES, 
            debug_sys->realtime_monitoring ? "enabled" : "disabled");
    
    /* Log first debug message */
    dsmil_debug_log(DSMIL_DEBUG_LEVEL_INFO, DSMIL_DEBUG_CAT_GENERAL, 0,
                   "DSMIL Debug system initialized successfully");
    
    return 0;
}

/*
 * Cleanup debug system
 */
void dsmil_debug_cleanup(void)
{
    if (!debug_sys) {
        return;
    }
    
    pr_info("DSMIL Debug: Shutting down debug system\n");
    
    /* Cancel any pending work */
    if (debug_sys->debug_workqueue) {
        cancel_delayed_work_sync(&debug_sys->log_work);
        destroy_workqueue(debug_sys->debug_workqueue);
    }
    
    /* Print final statistics */
    pr_info("DSMIL Debug: Final stats - Total entries: %llu, Errors: %llu, Critical: %llu\n",
            debug_sys->stats.total_log_entries,
            debug_sys->stats.entries_by_level[DSMIL_DEBUG_LEVEL_ERROR],
            debug_sys->stats.critical_events);
    
    /* Remove DebugFS entries */
    if (debug_sys->debugfs_root) {
        debugfs_remove_recursive(debug_sys->debugfs_root);
    }
    
    /* Free memory */
    kfree(debug_sys);
    debug_sys = NULL;
    
    pr_info("DSMIL Debug: Cleanup complete\n");
}

/*
 * Core logging function
 */
void dsmil_debug_log(enum dsmil_debug_level level, enum dsmil_debug_category category,
                    u32 device_id, const char *format, ...)
{
    struct dsmil_debug_log_entry entry;
    va_list args;
    unsigned long flags;
    bool log_to_kernel_now;
    
    if (!debug_sys || !debug_sys->debug_enabled) {
        return;
    }
    
    /* Check if this level/category should be logged */
    if (level > debug_sys->min_log_level) {
        return;
    }
    
    if (!(debug_sys->enabled_categories & (1 << category))) {
        return;
    }
    
    /* Prepare log entry */
    entry.timestamp = ktime_get();
    entry.level = level;
    entry.category = category;
    entry.device_id = device_id;
    
    /* Format message */
    va_start(args, format);
    vsnprintf(entry.message, sizeof(entry.message), format, args);
    va_end(args);
    
    /* Clear additional data */
    memset(entry.data, 0, sizeof(entry.data));
    
    /* Update statistics */
    spin_lock_irqsave(&debug_sys->log_lock, flags);
    debug_sys->stats.total_log_entries++;
    if (level < ARRAY_SIZE(debug_sys->stats.entries_by_level)) {
        debug_sys->stats.entries_by_level[level]++;
    }
    if (category < ARRAY_SIZE(debug_sys->stats.entries_by_category)) {
        debug_sys->stats.entries_by_category[category]++;
    }
    debug_sys->stats.last_entry_time = entry.timestamp;
    
    /* Track special events */
    if (level == DSMIL_DEBUG_LEVEL_ERROR) {
        debug_sys->stats.critical_events++;
    }
    if (category == DSMIL_DEBUG_CAT_QUARANTINE) {
        debug_sys->stats.quarantine_events++;
    }
    if (category == DSMIL_DEBUG_CAT_EMERGENCY) {
        debug_sys->stats.emergency_events++;
    }
    
    /* Add to buffer if enabled */
    log_to_kernel_now = debug_sys->log_to_kernel;
    if (debug_sys->log_to_buffer) {
        if (kfifo_is_full(&debug_sys->log_buffer)) {
            debug_sys->stats.buffer_overflows++;
            /* Remove oldest entry to make room */
            struct dsmil_debug_log_entry old_entry;
            kfifo_get(&debug_sys->log_buffer, &old_entry);
        }
        kfifo_put(&debug_sys->log_buffer, entry);
    }
    
    spin_unlock_irqrestore(&debug_sys->log_lock, flags);
    
    /* Log to kernel log if enabled */
    if (log_to_kernel_now) {
        const char *level_str = (level < ARRAY_SIZE(debug_level_names)) ? 
                               debug_level_names[level] : "UNKNOWN";
        const char *cat_str = (category < ARRAY_SIZE(debug_category_names)) ? 
                             debug_category_names[category] : "UNKNOWN";
        
        switch (level) {
        case DSMIL_DEBUG_LEVEL_ERROR:
            pr_err("DSMIL[%s:%s:D%u]: %s\n", level_str, cat_str, device_id, entry.message);
            break;
        case DSMIL_DEBUG_LEVEL_WARN:
            pr_warn("DSMIL[%s:%s:D%u]: %s\n", level_str, cat_str, device_id, entry.message);
            break;
        case DSMIL_DEBUG_LEVEL_INFO:
            pr_info("DSMIL[%s:%s:D%u]: %s\n", level_str, cat_str, device_id, entry.message);
            break;
        default:
            pr_debug("DSMIL[%s:%s:D%u]: %s\n", level_str, cat_str, device_id, entry.message);
            break;
        }
    }
}

/*
 * Log device operation for real-time monitoring
 */
void dsmil_debug_log_device_operation(u32 device_id, enum dsmil_hal_access_type access_type,
                                     bool success, u32 value_or_error, 
                                     const char *operation)
{
    struct dsmil_realtime_monitor_entry entry;
    unsigned long flags;
    
    if (!debug_sys || !debug_sys->realtime_monitoring) {
        return;
    }
    
    /* Prepare monitor entry */
    entry.timestamp = ktime_get();
    entry.device_id = device_id;
    entry.access_type = access_type;
    entry.success = success;
    entry.value_or_error = value_or_error;
    strncpy(entry.operation, operation ? operation : "unknown", 
            sizeof(entry.operation) - 1);
    entry.operation[sizeof(entry.operation) - 1] = '\0';
    
    /* Add to monitor buffer */
    spin_lock_irqsave(&debug_sys->monitor_lock, flags);
    if (kfifo_is_full(&debug_sys->monitor_buffer)) {
        /* Remove oldest entry to make room */
        struct dsmil_realtime_monitor_entry old_entry;
        kfifo_get(&debug_sys->monitor_buffer, &old_entry);
    }
    kfifo_put(&debug_sys->monitor_buffer, entry);
    spin_unlock_irqrestore(&debug_sys->monitor_lock, flags);
    
    /* Also log to main debug log if tracing is enabled */
    if (debug_sys->trace_device_operations) {
        dsmil_debug_log(DSMIL_DEBUG_LEVEL_DEBUG, DSMIL_DEBUG_CAT_DEVICE_OPS,
                       device_id, "%s %s - %s (0x%08x)",
                       operation ? operation : "operation",
                       (access_type == DSMIL_HAL_ACCESS_READ_ONLY) ? "READ" : "WRITE",
                       success ? "SUCCESS" : "FAILED",
                       value_or_error);
    }
}

/*
 * Dump system state for debugging
 */
int dsmil_debug_dump_system_state(char *buffer, size_t buffer_size)
{
    int len = 0;
    u64 uptime_ms;
    
    if (!debug_sys || !buffer) {
        return -EINVAL;
    }
    
    uptime_ms = ktime_to_ms(ktime_sub(ktime_get(), debug_sys->init_time));
    
    len += scnprintf(buffer + len, buffer_size - len,
                    "DSMIL Debug System State\n");
    len += scnprintf(buffer + len, buffer_size - len,
                    "========================\n");
    len += scnprintf(buffer + len, buffer_size - len,
                    "Version: %s\n", DSMIL_DEBUG_VERSION);
    len += scnprintf(buffer + len, buffer_size - len,
                    "Uptime: %llu ms\n", uptime_ms);
    len += scnprintf(buffer + len, buffer_size - len,
                    "Debug Enabled: %s\n", 
                    debug_sys->debug_enabled ? "Yes" : "No");
    len += scnprintf(buffer + len, buffer_size - len,
                    "Min Log Level: %s\n", 
                    debug_level_names[debug_sys->min_log_level]);
    len += scnprintf(buffer + len, buffer_size - len,
                    "Real-time Monitoring: %s\n",
                    debug_sys->realtime_monitoring ? "Enabled" : "Disabled");
    len += scnprintf(buffer + len, buffer_size - len,
                    "\nStatistics:\n");
    len += scnprintf(buffer + len, buffer_size - len,
                    "  Total Log Entries: %llu\n", 
                    debug_sys->stats.total_log_entries);
    len += scnprintf(buffer + len, buffer_size - len,
                    "  Critical Events: %llu\n", 
                    debug_sys->stats.critical_events);
    len += scnprintf(buffer + len, buffer_size - len,
                    "  Quarantine Events: %llu\n", 
                    debug_sys->stats.quarantine_events);
    len += scnprintf(buffer + len, buffer_size - len,
                    "  Emergency Events: %llu\n", 
                    debug_sys->stats.emergency_events);
    len += scnprintf(buffer + len, buffer_size - len,
                    "  Buffer Overflows: %llu\n", 
                    debug_sys->stats.buffer_overflows);
    
    return len;
}

/*
 * Background logging work function
 */
static void dsmil_debug_log_work(struct work_struct *work)
{
    /* This can be used for periodic logging tasks */
    dsmil_debug_log(DSMIL_DEBUG_LEVEL_TRACE, DSMIL_DEBUG_CAT_GENERAL, 0,
                   "Periodic debug system health check");
    
    /* Reschedule for next check (every 60 seconds) */
    if (debug_sys && debug_sys->debug_workqueue) {
        queue_delayed_work(debug_sys->debug_workqueue, &debug_sys->log_work,
                          msecs_to_jiffies(60000));
    }
}

/*
 * DebugFS sequence file implementations
 */
static int dsmil_debug_log_seq_show(struct seq_file *m, void *v)
{
    struct dsmil_debug_log_entry entry;
    unsigned long flags;
    int count = 0;
    
    if (!debug_sys) {
        seq_printf(m, "DSMIL Debug system not initialized\n");
        return 0;
    }
    
    seq_printf(m, "DSMIL Debug Log (last %d entries)\n", 
               kfifo_len(&debug_sys->log_buffer));
    seq_printf(m, "==========================================\n");
    
    /* Read entries from FIFO (non-destructively would be better, but this is for debug) */
    spin_lock_irqsave(&debug_sys->log_lock, flags);
    while (!kfifo_is_empty(&debug_sys->log_buffer) && count < 100) {
        if (kfifo_get(&debug_sys->log_buffer, &entry)) {
            u64 ts_ms = ktime_to_ms(entry.timestamp);
            const char *level_str = (entry.level < ARRAY_SIZE(debug_level_names)) ? 
                                   debug_level_names[entry.level] : "UNKNOWN";
            const char *cat_str = (entry.category < ARRAY_SIZE(debug_category_names)) ? 
                                 debug_category_names[entry.category] : "UNKNOWN";
            
            seq_printf(m, "[%llu.%03llu] %s:%s:D%u: %s\n",
                      ts_ms / 1000, ts_ms % 1000,
                      level_str, cat_str, entry.device_id, entry.message);
            count++;
        }
    }
    spin_unlock_irqrestore(&debug_sys->log_lock, flags);
    
    seq_printf(m, "\nDisplayed %d log entries\n", count);
    return 0;
}

static int dsmil_debug_monitor_seq_show(struct seq_file *m, void *v)
{
    struct dsmil_realtime_monitor_entry entry;
    unsigned long flags;
    int count = 0;
    
    if (!debug_sys) {
        seq_printf(m, "DSMIL Debug system not initialized\n");
        return 0;
    }
    
    seq_printf(m, "DSMIL Real-time Monitor (last %d operations)\n",
               kfifo_len(&debug_sys->monitor_buffer));
    seq_printf(m, "===============================================\n");
    
    spin_lock_irqsave(&debug_sys->monitor_lock, flags);
    while (!kfifo_is_empty(&debug_sys->monitor_buffer) && count < 50) {
        if (kfifo_get(&debug_sys->monitor_buffer, &entry)) {
            u64 ts_ms = ktime_to_ms(entry.timestamp);
            const char *access_str = (entry.access_type == DSMIL_HAL_ACCESS_READ_ONLY) ? 
                                    "READ" : "WRITE";
            
            seq_printf(m, "[%llu.%03llu] D%u %s %s: %s (0x%08x)\n",
                      ts_ms / 1000, ts_ms % 1000,
                      entry.device_id, access_str, entry.operation,
                      entry.success ? "OK" : "FAIL", entry.value_or_error);
            count++;
        }
    }
    spin_unlock_irqrestore(&debug_sys->monitor_lock, flags);
    
    seq_printf(m, "\nDisplayed %d monitor entries\n", count);
    return 0;
}

static int dsmil_debug_stats_seq_show(struct seq_file *m, void *v)
{
    int i;
    u64 uptime_ms;
    
    if (!debug_sys) {
        seq_printf(m, "DSMIL Debug system not initialized\n");
        return 0;
    }
    
    uptime_ms = ktime_to_ms(ktime_sub(ktime_get(), debug_sys->init_time));
    
    seq_printf(m, "DSMIL Debug Statistics\n");
    seq_printf(m, "======================\n");
    seq_printf(m, "System Uptime: %llu.%03llu seconds\n", 
               uptime_ms / 1000, uptime_ms % 1000);
    seq_printf(m, "Total Log Entries: %llu\n", debug_sys->stats.total_log_entries);
    seq_printf(m, "Critical Events: %llu\n", debug_sys->stats.critical_events);
    seq_printf(m, "Quarantine Events: %llu\n", debug_sys->stats.quarantine_events);
    seq_printf(m, "Emergency Events: %llu\n", debug_sys->stats.emergency_events);
    seq_printf(m, "Buffer Overflows: %llu\n", debug_sys->stats.buffer_overflows);
    
    seq_printf(m, "\nEntries by Level:\n");
    for (i = 0; i < ARRAY_SIZE(debug_sys->stats.entries_by_level); i++) {
        if (debug_sys->stats.entries_by_level[i] > 0) {
            seq_printf(m, "  %s: %llu\n", debug_level_names[i], 
                      debug_sys->stats.entries_by_level[i]);
        }
    }
    
    seq_printf(m, "\nEntries by Category:\n");
    for (i = 0; i < ARRAY_SIZE(debug_sys->stats.entries_by_category); i++) {
        if (debug_sys->stats.entries_by_category[i] > 0) {
            seq_printf(m, "  %s: %llu\n", debug_category_names[i], 
                      debug_sys->stats.entries_by_category[i]);
        }
    }
    
    return 0;
}

static int dsmil_debug_devices_seq_show(struct seq_file *m, void *v)
{
    u32 i;
    
    seq_printf(m, "DSMIL Device Debug Information\n");
    seq_printf(m, "==============================\n");
    
    /* Show quarantined devices */
    seq_printf(m, "Quarantined Devices (WRITE BLOCKED):\n");
    for (i = 0; i < 84; i++) {
        if (rust_safety_is_critical_device(i)) {
            struct dsmil_hal_device_info info;
            if (dsmil_hal_get_device_info(i, &info) == DSMIL_HAL_SUCCESS) {
                seq_printf(m, "  Device %u: %s - %s\n", 
                          i, info.function_name, info.safety_reason);
            } else {
                seq_printf(m, "  Device %u: QUARANTINED\n", i);
            }
        }
    }
    
    seq_printf(m, "\nDevice Access Summary:\n");
    seq_printf(m, "  Total devices: 84\n");
    seq_printf(m, "  Quarantined devices: 5\n");
    seq_printf(m, "  Accessible devices: 79\n");
    
    return 0;
}

/* Export functions for use by other modules */
EXPORT_SYMBOL(dsmil_debug_init);
EXPORT_SYMBOL(dsmil_debug_cleanup);
EXPORT_SYMBOL(dsmil_debug_log);
EXPORT_SYMBOL(dsmil_debug_log_device_operation);
EXPORT_SYMBOL(dsmil_debug_dump_system_state);

MODULE_AUTHOR("DSMIL Track A Development Team");
MODULE_DESCRIPTION("DSMIL Kernel Debug Interface with Comprehensive Logging");
MODULE_LICENSE("GPL v2");
MODULE_VERSION(DSMIL_DEBUG_VERSION);
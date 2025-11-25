/*
 * Dell MIL-SPEC Enhanced DSMIL Access Control System
 * 
 * Copyright (C) 2025 JRTC1 Educational Development
 * 
 * This module implements comprehensive access control for write operations
 * to the 84-device DSMIL system. It provides controlled write capability
 * ONLY for non-critical devices, with absolute prohibition for quarantined devices.
 * 
 * CRITICAL SAFETY: This system NEVER allows writes to the 5 quarantined devices.
 * Multiple independent verification layers ensure quarantine enforcement.
 */

#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/slab.h>
#include <linux/mutex.h>
#include <linux/crypto.h>
#include <linux/random.h>
#include <linux/time.h>
#include <linux/crc32.h>
#include <linux/atomic.h>
#include <linux/spinlock.h>

#include "dsmil_hal.h"

#define DSMIL_ACCESS_CONTROL_VERSION    "1.0.0"
#define DSMIL_ACCESS_MAGIC              0x41434354  /* "ACCT" */

/* Access Control Levels */
enum dsmil_access_control_level {
    ACCESS_LEVEL_BLOCKED = 0,       /* Access completely blocked */
    ACCESS_LEVEL_READ_ONLY,         /* Read-only access */
    ACCESS_LEVEL_RESTRICTED_WRITE,  /* Limited write operations */
    ACCESS_LEVEL_CONTROLLED_WRITE,  /* Standard controlled writes */
    ACCESS_LEVEL_FULL_ACCESS        /* Full access (for general devices) */
};

/* Write Operation Types */
enum dsmil_write_operation_type {
    WRITE_OP_REGISTER = 0,          /* Register write */
    WRITE_OP_MEMORY,                /* Memory write */
    WRITE_OP_CONFIG,                /* Configuration write */
    WRITE_OP_CONTROL,               /* Control operation */
    WRITE_OP_DATA,                  /* Data write */
    WRITE_OP_BULK,                  /* Bulk operation */
    WRITE_OP_DIAGNOSTIC,            /* Diagnostic write */
    WRITE_OP_MAINTENANCE            /* Maintenance operation */
};

/* Access Control Entry */
struct dsmil_access_control_entry {
    u32 device_id;                          /* Device ID (0-83) */
    enum dsmil_access_control_level level;  /* Access control level */
    u32 allowed_operations;                 /* Bitmask of allowed operations */
    u32 write_restrictions;                 /* Write restriction flags */
    u32 max_write_size;                     /* Maximum write size in bytes */
    u32 auth_requirements;                  /* Authentication requirements */
    
    /* Statistics */
    u64 read_attempts;
    u64 read_successes;
    u64 write_attempts;
    u64 write_successes;
    u64 access_denials;
    u64 violations;
    
    /* Timing */
    ktime_t last_access;
    ktime_t last_write;
    
    /* Lock for per-device statistics */
    spinlock_t stats_lock;
    
    char reason[128];                       /* Access control reason */
};

/* Global Access Control State */
struct dsmil_access_control_state {
    struct mutex global_lock;
    
    /* Access control table for all 84 devices */
    struct dsmil_access_control_entry devices[84];
    
    /* Global settings */
    bool strict_quarantine_mode;            /* Ultra-strict quarantine enforcement */
    bool audit_all_accesses;                /* Audit all access attempts */
    bool block_unknown_operations;          /* Block undefined operations */
    u32 global_auth_level;                  /* Global authentication level */
    
    /* Global statistics */
    atomic64_t total_read_attempts;
    atomic64_t total_write_attempts;
    atomic64_t total_access_denials;
    atomic64_t quarantine_violations;
    
    /* Runtime state */
    bool initialized;
    ktime_t init_time;
    u32 access_control_magic;
};

/* Global access control instance */
static struct dsmil_access_control_state *access_control = NULL;

/* Critical device access levels - ABSOLUTE QUARANTINE */
static const enum dsmil_access_control_level DEVICE_ACCESS_LEVELS[84] = {
    /* Group 0 - Critical Control (2 quarantined) */
    ACCESS_LEVEL_BLOCKED,           /* Device 0 - Master Control - QUARANTINED */
    ACCESS_LEVEL_BLOCKED,           /* Device 1 - Security Platform - QUARANTINED */
    ACCESS_LEVEL_READ_ONLY,         /* Device 2 - Authentication */
    ACCESS_LEVEL_READ_ONLY,         /* Device 3 - Access Control */
    ACCESS_LEVEL_READ_ONLY,         /* Device 4 - Encryption Engine */
    ACCESS_LEVEL_READ_ONLY,         /* Device 5 - Key Management */
    ACCESS_LEVEL_READ_ONLY,         /* Device 6 - Certificate Store */
    ACCESS_LEVEL_READ_ONLY,         /* Device 7 - Secure Boot */
    ACCESS_LEVEL_READ_ONLY,         /* Device 8 - Trusted Platform */
    ACCESS_LEVEL_RESTRICTED_WRITE,  /* Device 9 - Attestation */
    ACCESS_LEVEL_RESTRICTED_WRITE,  /* Device 10 - Hardware Security */
    ACCESS_LEVEL_CONTROLLED_WRITE,  /* Device 11 - Random Generator */
    
    /* Group 1 - Power Management (1 quarantined) */
    ACCESS_LEVEL_BLOCKED,           /* Device 12 - Power Controller - QUARANTINED */
    ACCESS_LEVEL_RESTRICTED_WRITE,  /* Device 13 - Voltage Regulator */
    ACCESS_LEVEL_RESTRICTED_WRITE,  /* Device 14 - Clock Generator */
    ACCESS_LEVEL_RESTRICTED_WRITE,  /* Device 15 - Power Sequencer */
    ACCESS_LEVEL_CONTROLLED_WRITE,  /* Device 16 - Battery Management */
    ACCESS_LEVEL_CONTROLLED_WRITE,  /* Device 17 - Thermal Monitor */
    ACCESS_LEVEL_CONTROLLED_WRITE,  /* Device 18 - Fan Controller */
    ACCESS_LEVEL_CONTROLLED_WRITE,  /* Device 19 - Frequency Scaler */
    ACCESS_LEVEL_CONTROLLED_WRITE,  /* Device 20 - Sleep Controller */
    ACCESS_LEVEL_CONTROLLED_WRITE,  /* Device 21 - Wake Engine */
    ACCESS_LEVEL_FULL_ACCESS,       /* Device 22 - Power Analyzer */
    ACCESS_LEVEL_FULL_ACCESS,       /* Device 23 - Energy Meter */
    
    /* Group 2 - Memory Management (1 quarantined) */
    ACCESS_LEVEL_BLOCKED,           /* Device 24 - Memory Controller - QUARANTINED */
    ACCESS_LEVEL_RESTRICTED_WRITE,  /* Device 25 - Cache Controller */
    ACCESS_LEVEL_RESTRICTED_WRITE,  /* Device 26 - DMA Engine */
    ACCESS_LEVEL_RESTRICTED_WRITE,  /* Device 27 - Memory Encryption */
    ACCESS_LEVEL_CONTROLLED_WRITE,  /* Device 28 - ECC Controller */
    ACCESS_LEVEL_CONTROLLED_WRITE,  /* Device 29 - Memory Tester */
    ACCESS_LEVEL_CONTROLLED_WRITE,  /* Device 30 - Buffer Manager */
    ACCESS_LEVEL_CONTROLLED_WRITE,  /* Device 31 - Memory Mapper */
    ACCESS_LEVEL_CONTROLLED_WRITE,  /* Device 32 - Address Translator */
    ACCESS_LEVEL_FULL_ACCESS,       /* Device 33 - Memory Monitor */
    ACCESS_LEVEL_FULL_ACCESS,       /* Device 34 - Coherency Engine */
    ACCESS_LEVEL_FULL_ACCESS,       /* Device 35 - Memory Optimizer */
    
    /* Group 3 - I/O and Communication */
    ACCESS_LEVEL_CONTROLLED_WRITE,  /* Device 36 - I/O Controller */
    ACCESS_LEVEL_CONTROLLED_WRITE,  /* Device 37 - Serial Interface */
    ACCESS_LEVEL_CONTROLLED_WRITE,  /* Device 38 - Parallel Interface */
    ACCESS_LEVEL_CONTROLLED_WRITE,  /* Device 39 - Network Interface */
    ACCESS_LEVEL_CONTROLLED_WRITE,  /* Device 40 - USB Controller */
    ACCESS_LEVEL_CONTROLLED_WRITE,  /* Device 41 - PCIe Controller */
    ACCESS_LEVEL_FULL_ACCESS,       /* Device 42 - SPI Controller */
    ACCESS_LEVEL_FULL_ACCESS,       /* Device 43 - I2C Controller */
    ACCESS_LEVEL_FULL_ACCESS,       /* Device 44 - UART Controller */
    ACCESS_LEVEL_FULL_ACCESS,       /* Device 45 - GPIO Controller */
    ACCESS_LEVEL_FULL_ACCESS,       /* Device 46 - Interrupt Controller */
    ACCESS_LEVEL_FULL_ACCESS,       /* Device 47 - Timer Controller */
    
    /* Group 4 - Processing and Acceleration */
    ACCESS_LEVEL_RESTRICTED_WRITE,  /* Device 48 - Crypto Accelerator */
    ACCESS_LEVEL_CONTROLLED_WRITE,  /* Device 49 - DSP Engine */
    ACCESS_LEVEL_CONTROLLED_WRITE,  /* Device 50 - Vector Processor */
    ACCESS_LEVEL_CONTROLLED_WRITE,  /* Device 51 - Neural Processor */
    ACCESS_LEVEL_CONTROLLED_WRITE,  /* Device 52 - Graphics Engine */
    ACCESS_LEVEL_CONTROLLED_WRITE,  /* Device 53 - Video Decoder */
    ACCESS_LEVEL_FULL_ACCESS,       /* Device 54 - Audio Processor */
    ACCESS_LEVEL_FULL_ACCESS,       /* Device 55 - Signal Processor */
    ACCESS_LEVEL_FULL_ACCESS,       /* Device 56 - Math Coprocessor */
    ACCESS_LEVEL_FULL_ACCESS,       /* Device 57 - Compression Engine */
    ACCESS_LEVEL_FULL_ACCESS,       /* Device 58 - Decompression Engine */
    ACCESS_LEVEL_FULL_ACCESS,       /* Device 59 - Hash Engine */
    
    /* Group 5 - Monitoring and Diagnostics */
    ACCESS_LEVEL_FULL_ACCESS,       /* Device 60 - System Monitor */
    ACCESS_LEVEL_FULL_ACCESS,       /* Device 61 - Health Monitor */
    ACCESS_LEVEL_FULL_ACCESS,       /* Device 62 - Performance Monitor */
    ACCESS_LEVEL_FULL_ACCESS,       /* Device 63 - Security Monitor */
    ACCESS_LEVEL_FULL_ACCESS,       /* Device 64 - Temperature Sensor */
    ACCESS_LEVEL_FULL_ACCESS,       /* Device 65 - Voltage Sensor */
    ACCESS_LEVEL_FULL_ACCESS,       /* Device 66 - Current Sensor */
    ACCESS_LEVEL_FULL_ACCESS,       /* Device 67 - Vibration Sensor */
    ACCESS_LEVEL_FULL_ACCESS,       /* Device 68 - Tamper Detector */
    ACCESS_LEVEL_FULL_ACCESS,       /* Device 69 - Error Logger */
    ACCESS_LEVEL_FULL_ACCESS,       /* Device 70 - Event Recorder */
    ACCESS_LEVEL_FULL_ACCESS,       /* Device 71 - Diagnostic Engine */
    
    /* Group 6 - System Control and Safety (1 quarantined) */
    ACCESS_LEVEL_RESTRICTED_WRITE,  /* Device 72 - System Controller */
    ACCESS_LEVEL_RESTRICTED_WRITE,  /* Device 73 - Reset Controller */
    ACCESS_LEVEL_RESTRICTED_WRITE,  /* Device 74 - Configuration Manager */
    ACCESS_LEVEL_RESTRICTED_WRITE,  /* Device 75 - Policy Engine */
    ACCESS_LEVEL_CONTROLLED_WRITE,  /* Device 76 - Compliance Monitor */
    ACCESS_LEVEL_CONTROLLED_WRITE,  /* Device 77 - Audit Logger */
    ACCESS_LEVEL_CONTROLLED_WRITE,  /* Device 78 - Safety Controller */
    ACCESS_LEVEL_CONTROLLED_WRITE,  /* Device 79 - Watchdog Timer */
    ACCESS_LEVEL_CONTROLLED_WRITE,  /* Device 80 - Failsafe Controller */
    ACCESS_LEVEL_CONTROLLED_WRITE,  /* Device 81 - Recovery Engine */
    ACCESS_LEVEL_CONTROLLED_WRITE,  /* Device 82 - Backup Controller */
    ACCESS_LEVEL_BLOCKED            /* Device 83 - Emergency Stop - QUARANTINED */
};

/* Access control reasons for each device */
static const char *DEVICE_ACCESS_REASONS[84] = {
    "Master Control - QUARANTINED for system safety",
    "Security Platform - QUARANTINED for security integrity",
    "Authentication - Read-only to prevent bypass",
    "Access Control - Read-only to prevent escalation",
    "Encryption Engine - Read-only to prevent key exposure", 
    "Key Management - Read-only to prevent key compromise",
    "Certificate Store - Read-only to prevent cert manipulation",
    "Secure Boot - Read-only to prevent boot compromise",
    "Trusted Platform - Read-only to prevent trust violation",
    "Attestation - Limited writes for attestation functions",
    "Hardware Security - Limited writes for security ops",
    "Random Generator - Controlled access for entropy",
    
    "Power Controller - QUARANTINED for hardware protection",
    "Voltage Regulator - Limited writes to prevent damage",
    "Clock Generator - Limited writes to prevent instability",
    "Power Sequencer - Limited writes to prevent damage",
    "Battery Management - Controlled access for safety",
    "Thermal Monitor - Controlled access for monitoring", 
    "Fan Controller - Controlled access for cooling",
    "Frequency Scaler - Controlled access for performance",
    "Sleep Controller - Controlled access for power mgmt",
    "Wake Engine - Controlled access for wake functions",
    "Power Analyzer - Full access for analysis",
    "Energy Meter - Full access for measurement",
    
    "Memory Controller - QUARANTINED for memory safety",
    "Cache Controller - Limited writes to prevent corruption",
    "DMA Engine - Limited writes to prevent memory issues",
    "Memory Encryption - Limited writes to prevent bypass",
    "ECC Controller - Controlled access for error correction",
    "Memory Tester - Controlled access for testing",
    "Buffer Manager - Controlled access for buffering",
    "Memory Mapper - Controlled access for mapping",
    "Address Translator - Controlled access for translation",
    "Memory Monitor - Full access for monitoring",
    "Coherency Engine - Full access for coherency",
    "Memory Optimizer - Full access for optimization",
    
    "I/O Controller - Controlled access for I/O operations",
    "Serial Interface - Controlled access for serial comm",
    "Parallel Interface - Controlled access for parallel comm",
    "Network Interface - Controlled access for networking",
    "USB Controller - Controlled access for USB operations",
    "PCIe Controller - Controlled access for PCIe operations",
    "SPI Controller - Full access for SPI operations",
    "I2C Controller - Full access for I2C operations",
    "UART Controller - Full access for UART operations",
    "GPIO Controller - Full access for GPIO operations",
    "Interrupt Controller - Full access for interrupt handling",
    "Timer Controller - Full access for timing operations",
    
    "Crypto Accelerator - Limited writes for security",
    "DSP Engine - Controlled access for signal processing",
    "Vector Processor - Controlled access for vector ops",
    "Neural Processor - Controlled access for AI operations",
    "Graphics Engine - Controlled access for graphics",
    "Video Decoder - Controlled access for video processing",
    "Audio Processor - Full access for audio processing",
    "Signal Processor - Full access for signal processing",
    "Math Coprocessor - Full access for math operations",
    "Compression Engine - Full access for compression",
    "Decompression Engine - Full access for decompression",
    "Hash Engine - Full access for hashing operations",
    
    "System Monitor - Full access for monitoring",
    "Health Monitor - Full access for health monitoring",
    "Performance Monitor - Full access for performance monitoring",
    "Security Monitor - Full access for security monitoring",
    "Temperature Sensor - Full access for temperature monitoring",
    "Voltage Sensor - Full access for voltage monitoring",
    "Current Sensor - Full access for current monitoring",
    "Vibration Sensor - Full access for vibration monitoring",
    "Tamper Detector - Full access for tamper detection",
    "Error Logger - Full access for error logging",
    "Event Recorder - Full access for event recording",
    "Diagnostic Engine - Full access for diagnostics",
    
    "System Controller - Limited writes for system control",
    "Reset Controller - Limited writes for reset operations",
    "Configuration Manager - Limited writes for config mgmt",
    "Policy Engine - Limited writes for policy enforcement",
    "Compliance Monitor - Controlled access for compliance",
    "Audit Logger - Controlled access for audit logging",
    "Safety Controller - Controlled access for safety operations",
    "Watchdog Timer - Controlled access for watchdog functions",
    "Failsafe Controller - Controlled access for failsafe ops",
    "Recovery Engine - Controlled access for recovery ops",
    "Backup Controller - Controlled access for backup ops",
    "Emergency Stop - QUARANTINED for safety system integrity"
};

/* Forward declarations */
static int dsmil_access_control_validate_write(u32 device_id, 
                                              enum dsmil_write_operation_type op_type,
                                              size_t size);
static int dsmil_access_control_check_quarantine(u32 device_id);
static void dsmil_access_control_update_stats(u32 device_id, bool is_write, bool success);

/*
 * Initialize access control system
 */
int dsmil_access_control_init(void)
{
    u32 i;
    u32 quarantined_count = 0;
    
    if (access_control) {
        pr_warn("DSMIL Access Control: Already initialized\n");
        return 0;
    }
    
    /* Allocate global state */
    access_control = kzalloc(sizeof(struct dsmil_access_control_state), GFP_KERNEL);
    if (!access_control) {
        pr_err("DSMIL Access Control: Failed to allocate state structure\n");
        return -ENOMEM;
    }
    
    /* Initialize mutex */
    mutex_init(&access_control->global_lock);
    
    /* Initialize global settings */
    access_control->strict_quarantine_mode = true;
    access_control->audit_all_accesses = true;
    access_control->block_unknown_operations = true;
    access_control->global_auth_level = 3; /* High security */
    
    /* Initialize atomic counters */
    atomic64_set(&access_control->total_read_attempts, 0);
    atomic64_set(&access_control->total_write_attempts, 0);
    atomic64_set(&access_control->total_access_denials, 0);
    atomic64_set(&access_control->quarantine_violations, 0);
    
    /* Initialize all device access control entries */
    for (i = 0; i < 84; i++) {
        struct dsmil_access_control_entry *entry = &access_control->devices[i];
        
        entry->device_id = i;
        entry->level = DEVICE_ACCESS_LEVELS[i];
        
        /* Set allowed operations based on access level */
        switch (entry->level) {
        case ACCESS_LEVEL_BLOCKED:
            entry->allowed_operations = 0; /* Nothing allowed */
            entry->write_restrictions = 0xFFFFFFFF; /* All writes blocked */
            entry->max_write_size = 0;
            entry->auth_requirements = 0xFFFFFFFF; /* Infinite auth required */
            quarantined_count++;
            break;
            
        case ACCESS_LEVEL_READ_ONLY:
            entry->allowed_operations = 0x01; /* Only read */
            entry->write_restrictions = 0xFFFFFFFF; /* All writes blocked */
            entry->max_write_size = 0;
            entry->auth_requirements = 0x01; /* Basic auth for read */
            break;
            
        case ACCESS_LEVEL_RESTRICTED_WRITE:
            entry->allowed_operations = 0x03; /* Read and limited write */
            entry->write_restrictions = 0xF0; /* Most writes blocked */
            entry->max_write_size = 64; /* 64 bytes max */
            entry->auth_requirements = 0x07; /* High auth for write */
            break;
            
        case ACCESS_LEVEL_CONTROLLED_WRITE:
            entry->allowed_operations = 0x0F; /* Read and controlled writes */
            entry->write_restrictions = 0x30; /* Some writes blocked */
            entry->max_write_size = 512; /* 512 bytes max */
            entry->auth_requirements = 0x03; /* Medium auth */
            break;
            
        case ACCESS_LEVEL_FULL_ACCESS:
            entry->allowed_operations = 0xFF; /* All operations */
            entry->write_restrictions = 0x00; /* No write restrictions */
            entry->max_write_size = 4096; /* 4KB max */
            entry->auth_requirements = 0x01; /* Basic auth */
            break;
        }
        
        /* Initialize statistics */
        entry->read_attempts = 0;
        entry->read_successes = 0;
        entry->write_attempts = 0;
        entry->write_successes = 0;
        entry->access_denials = 0;
        entry->violations = 0;
        
        /* Initialize timing */
        entry->last_access = 0;
        entry->last_write = 0;
        
        /* Initialize spinlock */
        spin_lock_init(&entry->stats_lock);
        
        /* Set reason */
        strncpy(entry->reason, DEVICE_ACCESS_REASONS[i], sizeof(entry->reason) - 1);
        entry->reason[sizeof(entry->reason) - 1] = '\0';
    }
    
    /* Set runtime state */
    access_control->initialized = true;
    access_control->init_time = ktime_get();
    access_control->access_control_magic = DSMIL_ACCESS_MAGIC;
    
    pr_info("DSMIL Access Control: Initialized (version %s)\n", DSMIL_ACCESS_CONTROL_VERSION);
    pr_info("DSMIL Access Control: %u devices quarantined (BLOCKED access)\n", quarantined_count);
    pr_info("DSMIL Access Control: Strict quarantine mode %s\n", 
            access_control->strict_quarantine_mode ? "ENABLED" : "disabled");
    
    return 0;
}

/*
 * Cleanup access control system
 */
void dsmil_access_control_cleanup(void)
{
    u64 total_reads, total_writes, total_denials, quarantine_violations;
    
    if (!access_control) {
        return;
    }
    
    /* Get final statistics */
    total_reads = atomic64_read(&access_control->total_read_attempts);
    total_writes = atomic64_read(&access_control->total_write_attempts);
    total_denials = atomic64_read(&access_control->total_access_denials);
    quarantine_violations = atomic64_read(&access_control->quarantine_violations);
    
    pr_info("DSMIL Access Control: Shutting down\n");
    pr_info("DSMIL Access Control: Final stats - Reads: %llu, Writes: %llu, Denials: %llu, Quarantine violations: %llu\n",
            total_reads, total_writes, total_denials, quarantine_violations);
    
    /* Clear magic and mark as uninitialized */
    access_control->access_control_magic = 0;
    access_control->initialized = false;
    
    /* Free memory */
    kfree(access_control);
    access_control = NULL;
    
    pr_info("DSMIL Access Control: Cleanup complete\n");
}

/*
 * CRITICAL FUNCTION: Validate write access to device
 * This is the primary gatekeeper for all write operations
 */
int dsmil_access_control_validate_write_access(u32 device_id, 
                                              enum dsmil_write_operation_type op_type,
                                              size_t write_size)
{
    struct dsmil_access_control_entry *entry;
    int ret;
    
    if (!access_control || !access_control->initialized) {
        pr_err("DSMIL Access Control: System not initialized - BLOCKING write\n");
        return -EPERM;
    }
    
    if (device_id >= 84) {
        pr_err("DSMIL Access Control: Invalid device ID %u\n", device_id);
        atomic64_inc(&access_control->total_access_denials);
        return -EINVAL;
    }
    
    entry = &access_control->devices[device_id];
    
    atomic64_inc(&access_control->total_write_attempts);
    dsmil_access_control_update_stats(device_id, true, false); /* Mark as attempted */
    
    /* CRITICAL: Check for quarantine violation */
    ret = dsmil_access_control_check_quarantine(device_id);
    if (ret < 0) {
        atomic64_inc(&access_control->quarantine_violations);
        pr_err("DSMIL Access Control: QUARANTINE VIOLATION - Write blocked for device %u\n", 
               device_id);
        return ret;
    }
    
    /* Check access level */
    if (entry->level <= ACCESS_LEVEL_READ_ONLY) {
        pr_warn("DSMIL Access Control: Write denied - Device %u is read-only (%s)\n",
                device_id, entry->reason);
        atomic64_inc(&access_control->total_access_denials);
        return -EPERM;
    }
    
    /* Check write size limits */
    if (write_size > entry->max_write_size) {
        pr_warn("DSMIL Access Control: Write denied - Size %zu exceeds limit %u for device %u\n",
                write_size, entry->max_write_size, device_id);
        atomic64_inc(&access_control->total_access_denials);
        return -E2BIG;
    }
    
    /* Check operation type restrictions */
    if (entry->write_restrictions & (1 << op_type)) {
        pr_warn("DSMIL Access Control: Write denied - Operation type %d restricted for device %u\n",
                op_type, device_id);
        atomic64_inc(&access_control->total_access_denials);
        return -EPERM;
    }
    
    /* Additional validation based on access level */
    ret = dsmil_access_control_validate_write(device_id, op_type, write_size);
    if (ret < 0) {
        atomic64_inc(&access_control->total_access_denials);
        return ret;
    }
    
    /* All checks passed - allow write */
    dsmil_access_control_update_stats(device_id, true, true); /* Mark as successful */
    
    if (access_control->audit_all_accesses) {
        pr_info("DSMIL Access Control: Write ALLOWED - Device %u, op %d, size %zu\n",
                device_id, op_type, write_size);
    }
    
    return 0;
}

/*
 * Validate read access to device
 */
int dsmil_access_control_validate_read_access(u32 device_id, size_t read_size)
{
    struct dsmil_access_control_entry *entry;
    
    if (!access_control || !access_control->initialized) {
        pr_warn("DSMIL Access Control: System not initialized - allowing read\n");
        return 0; /* Allow reads when system not initialized */
    }
    
    if (device_id >= 84) {
        pr_err("DSMIL Access Control: Invalid device ID %u\n", device_id);
        atomic64_inc(&access_control->total_access_denials);
        return -EINVAL;
    }
    
    entry = &access_control->devices[device_id];
    
    atomic64_inc(&access_control->total_read_attempts);
    dsmil_access_control_update_stats(device_id, false, false); /* Mark as attempted */
    
    /* Check if read operations are allowed */
    if (!(entry->allowed_operations & 0x01)) {
        pr_warn("DSMIL Access Control: Read denied - Device %u blocked completely\n", device_id);
        atomic64_inc(&access_control->total_access_denials);
        return -EPERM;
    }
    
    /* All devices allow read (even quarantined ones for status) */
    dsmil_access_control_update_stats(device_id, false, true); /* Mark as successful */
    
    return 0;
}

/*
 * Get device access information
 */
int dsmil_access_control_get_device_info(u32 device_id, 
                                        struct dsmil_access_control_entry *info)
{
    if (!access_control || !access_control->initialized) {
        return -EINVAL;
    }
    
    if (device_id >= 84 || !info) {
        return -EINVAL;
    }
    
    mutex_lock(&access_control->global_lock);
    
    /* Copy device info (stats protected by spinlock) */
    memcpy(info, &access_control->devices[device_id], sizeof(*info));
    
    mutex_unlock(&access_control->global_lock);
    
    return 0;
}

/*
 * CRITICAL: Check if device is quarantined
 * Multiple independent checks for maximum safety
 */
static int dsmil_access_control_check_quarantine(u32 device_id)
{
    /* Critical devices that must NEVER be written to */
    static const u32 QUARANTINE_LIST[] = { 0, 1, 12, 24, 83 };
    int i;
    
    /* Check 1: Direct device ID check */
    for (i = 0; i < ARRAY_SIZE(QUARANTINE_LIST); i++) {
        if (QUARANTINE_LIST[i] == device_id) {
            return -EPERM; /* Device is quarantined */
        }
    }
    
    /* Check 2: Access level check */
    if (DEVICE_ACCESS_LEVELS[device_id] == ACCESS_LEVEL_BLOCKED) {
        return -EPERM; /* Device is blocked */
    }
    
    /* Check 3: Additional safety check for critical ranges */
    if (device_id == 0 || device_id == 1) {
        /* Group 0 critical devices */
        return -EPERM;
    }
    if (device_id == 12) {
        /* Group 1 power controller */
        return -EPERM;
    }
    if (device_id == 24) {
        /* Group 2 memory controller */
        return -EPERM;
    }
    if (device_id == 83) {
        /* Group 6 emergency stop */
        return -EPERM;
    }
    
    return 0; /* Device is not quarantined */
}

/*
 * Additional write validation based on operation type
 */
static int dsmil_access_control_validate_write(u32 device_id, 
                                              enum dsmil_write_operation_type op_type,
                                              size_t size)
{
    struct dsmil_access_control_entry *entry = &access_control->devices[device_id];
    
    /* Additional checks based on access level */
    switch (entry->level) {
    case ACCESS_LEVEL_RESTRICTED_WRITE:
        /* Only allow basic register writes */
        if (op_type != WRITE_OP_REGISTER && op_type != WRITE_OP_DATA) {
            return -EPERM;
        }
        /* Small writes only */
        if (size > 32) {
            return -E2BIG;
        }
        break;
        
    case ACCESS_LEVEL_CONTROLLED_WRITE:
        /* Allow most writes except control operations */
        if (op_type == WRITE_OP_CONTROL) {
            return -EPERM;
        }
        break;
        
    case ACCESS_LEVEL_FULL_ACCESS:
        /* All writes allowed */
        break;
        
    default:
        return -EPERM;
    }
    
    return 0;
}

/*
 * Update device statistics
 */
static void dsmil_access_control_update_stats(u32 device_id, bool is_write, bool success)
{
    struct dsmil_access_control_entry *entry;
    
    if (device_id >= 84) {
        return;
    }
    
    entry = &access_control->devices[device_id];
    
    spin_lock(&entry->stats_lock);
    
    if (is_write) {
        entry->write_attempts++;
        if (success) {
            entry->write_successes++;
            entry->last_write = ktime_get();
        } else {
            entry->access_denials++;
        }
    } else {
        entry->read_attempts++;
        if (success) {
            entry->read_successes++;
        } else {
            entry->access_denials++;
        }
    }
    
    entry->last_access = ktime_get();
    
    spin_unlock(&entry->stats_lock);
}

/*
 * Get global access control statistics
 */
int dsmil_access_control_get_statistics(u64 *total_reads, u64 *total_writes, 
                                       u64 *total_denials, u64 *quarantine_violations)
{
    if (!access_control) {
        return -EINVAL;
    }
    
    if (total_reads) {
        *total_reads = atomic64_read(&access_control->total_read_attempts);
    }
    if (total_writes) {
        *total_writes = atomic64_read(&access_control->total_write_attempts);
    }
    if (total_denials) {
        *total_denials = atomic64_read(&access_control->total_access_denials);
    }
    if (quarantine_violations) {
        *quarantine_violations = atomic64_read(&access_control->quarantine_violations);
    }
    
    return 0;
}

/* Export functions for use by other modules */
EXPORT_SYMBOL(dsmil_access_control_init);
EXPORT_SYMBOL(dsmil_access_control_cleanup);
EXPORT_SYMBOL(dsmil_access_control_validate_write_access);
EXPORT_SYMBOL(dsmil_access_control_validate_read_access);
EXPORT_SYMBOL(dsmil_access_control_get_device_info);
EXPORT_SYMBOL(dsmil_access_control_get_statistics);

MODULE_AUTHOR("DSMIL Track A Development Team");
MODULE_DESCRIPTION("DSMIL Access Control System with Quarantine Protection");
MODULE_LICENSE("GPL v2");
MODULE_VERSION(DSMIL_ACCESS_CONTROL_VERSION);
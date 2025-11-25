/*
 * TPM2 Early Boot Acceleration - Header File
 *
 * Public API definitions and structures for the TPM2 early boot acceleration
 * kernel module.
 *
 * Copyright (C) 2025 Military TPM2 Acceleration Project
 * Classification: UNCLASSIFIED // FOR OFFICIAL USE ONLY
 */

#ifndef _TPM2_ACCEL_EARLY_H
#define _TPM2_ACCEL_EARLY_H

#include <linux/types.h>
#include <linux/ioctl.h>

#ifdef __KERNEL__
#include <linux/device.h>
#include <linux/pci.h>
#include <linux/completion.h>
#include <linux/atomic.h>
#include <linux/spinlock.h>
#include <linux/wait.h>
#include <linux/timer.h>
#include <linux/workqueue.h>
#endif

/*
 * Driver Information
 */
#define TPM2_ACCEL_DRIVER_NAME    "tpm2_accel_early"
#define TPM2_ACCEL_DRIVER_VERSION "1.0.0"
#define TPM2_ACCEL_DRIVER_DESCRIPTION "Early Boot TPM2 Hardware Acceleration"

/*
 * Hardware Constants
 */
#define TPM2_ACCEL_INTEL_VENDOR_ID         0x8086
#define TPM2_ACCEL_INTEL_NPU_DEVICE_ID     0x7d1d  // Core Ultra NPU
#define TPM2_ACCEL_INTEL_GNA_DEVICE_ID     0x7d1e  // GNA 3.5
#define TPM2_ACCEL_INTEL_ME_DEVICE_ID      0x7d1f  // Management Engine

#define TPM2_ACCEL_DELL_SMI_PORT_COMMAND   0x164E
#define TPM2_ACCEL_DELL_SMI_PORT_DATA      0x164F
#define TPM2_ACCEL_DELL_MILITARY_TOKEN_START 0x049e
#define TPM2_ACCEL_DELL_MILITARY_TOKEN_END   0x04a3

/*
 * Configuration Limits
 */
#define TPM2_ACCEL_MAX_BATCH_SIZE       128
#define TPM2_ACCEL_SHARED_MEM_SIZE      (4 * 1024 * 1024)  // 4MB
#define TPM2_ACCEL_CMD_RING_SIZE        4096
#define TPM2_ACCEL_RESP_RING_SIZE       4096
#define TPM2_ACCEL_MAX_CONCURRENT_OPS   256
#define TPM2_ACCEL_DEFAULT_TIMEOUT_MS   30000
#define TPM2_ACCEL_MAX_CMD_SIZE         8192
#define TPM2_ACCEL_MAX_RESP_SIZE        8192

/*
 * Security Levels
 */
enum tpm2_accel_security_level {
    TPM2_ACCEL_SEC_UNCLASSIFIED = 0,
    TPM2_ACCEL_SEC_CONFIDENTIAL = 1,
    TPM2_ACCEL_SEC_SECRET = 2,
    TPM2_ACCEL_SEC_TOP_SECRET = 3,
    TPM2_ACCEL_SEC_MAX = 4
};

/*
 * Hardware Status Flags
 */
#define TPM2_ACCEL_HW_NPU_PRESENT       (1 << 0)
#define TPM2_ACCEL_HW_GNA_PRESENT       (1 << 1)
#define TPM2_ACCEL_HW_ME_PRESENT        (1 << 2)
#define TPM2_ACCEL_HW_TPM_PRESENT       (1 << 3)
#define TPM2_ACCEL_HW_DELL_SMBIOS_PRESENT (1 << 4)

/*
 * Acceleration Flags
 */
#define TPM2_ACCEL_FLAG_NPU_ACCELERATION   (1 << 0)
#define TPM2_ACCEL_FLAG_GNA_MONITORING     (1 << 1)
#define TPM2_ACCEL_FLAG_ME_INTEGRATION     (1 << 2)
#define TPM2_ACCEL_FLAG_BATCH_PROCESSING   (1 << 3)
#define TPM2_ACCEL_FLAG_PRIORITY_HIGH      (1 << 4)
#define TPM2_ACCEL_FLAG_SECURE_MEMORY      (1 << 5)
#define TPM2_ACCEL_FLAG_AUDIT_LOGGING      (1 << 6)
#define TPM2_ACCEL_FLAG_ALL                0xFFFFFFFF

/*
 * Performance Modes
 */
enum tpm2_accel_performance_mode {
    TPM2_ACCEL_PERF_BALANCED = 0,
    TPM2_ACCEL_PERF_PERFORMANCE = 1,
    TPM2_ACCEL_PERF_POWER_SAVE = 2,
};

/*
 * Event Types for Userspace Notification
 */
enum tpm2_accel_event_type {
    TPM2_ACCEL_EVENT_HARDWARE_FAILURE = 1,
    TPM2_ACCEL_EVENT_SECURITY_VIOLATION = 2,
    TPM2_ACCEL_EVENT_EMERGENCY_STOP = 3,
    TPM2_ACCEL_EVENT_TPM_FAILURE = 4,
    TPM2_ACCEL_EVENT_NPU_FAILURE = 5,
    TPM2_ACCEL_EVENT_GNA_ALERT = 6,
    TPM2_ACCEL_EVENT_DELL_TOKEN_VIOLATION = 7,
};

/*
 * Command Structure for IOCTL Communication
 */
struct tpm2_accel_cmd {
    __u32 cmd_id;              /* Command identifier */
    __u32 security_level;      /* Security level (enum tpm2_accel_security_level) */
    __u32 flags;               /* Acceleration flags */
    __u32 input_len;           /* Input data length */
    __u32 output_len;          /* Output buffer length */
    __u64 input_ptr;           /* User space input pointer */
    __u64 output_ptr;          /* User space output pointer */
    __u32 timeout_ms;          /* Operation timeout in milliseconds */
    __u32 dell_token;          /* Dell military token for authorization */
    __u32 session_id;          /* Session identifier */
    __u32 reserved[3];         /* Reserved for future use */
};

/*
 * Status Structure
 */
struct tpm2_accel_status {
    __u32 hardware_status;     /* Bitmask of available hardware */
    __u32 npu_utilization;     /* NPU utilization percentage (0-100) */
    __u32 gna_status;          /* GNA security status */
    __u32 me_status;           /* Intel ME status */
    __u32 tpm_status;          /* TPM 2.0 status */
    __u32 performance_ops_sec; /* Operations per second */
    __u64 total_operations;    /* Total operations processed */
    __u64 total_errors;        /* Total errors encountered */
    __u64 bytes_processed;     /* Total bytes processed */
    __u32 security_violations; /* Security violations detected */
    __u32 hardware_errors;     /* Hardware errors detected */
    __u32 current_sessions;    /* Current active sessions */
    __u32 max_sessions;        /* Maximum allowed sessions */
    __u64 uptime_seconds;      /* Module uptime in seconds */
    __u32 reserved[4];         /* Reserved for future use */
};

/*
 * Configuration Structure
 */
struct tmp2_accel_config {
    __u32 max_concurrent_ops;  /* Maximum concurrent operations */
    __u32 npu_batch_size;      /* NPU batch processing size */
    __u32 timeout_default_ms;  /* Default operation timeout */
    __u32 security_level;      /* Default security level */
    __u32 debug_level;         /* Debug output level (0-7) */
    __u32 performance_mode;    /* Performance mode (enum tpm2_accel_performance_mode) */
    __u32 memory_pool_size_mb; /* Memory pool size in MB */
    __u32 enable_monitoring;   /* Enable performance monitoring */
    __u32 enable_audit;        /* Enable audit logging */
    __u32 gna_sensitivity;     /* GNA security sensitivity (0-100) */
    __u32 reserved[6];         /* Reserved for future use */
};

/*
 * Hardware Information Structure
 */
struct tpm2_accel_hardware_info {
    struct {
        __u8 present;
        __u32 tops_capacity;    /* NPU capacity in milli-TOPS */
        __u32 version;
        __u32 features;
    } npu;

    struct {
        __u8 present;
        __u8 version;           /* GNA version (e.g., 35 for GNA 3.5) */
        __u32 features;
        __u32 max_inference_rate;
    } gna;

    struct {
        __u8 present;
        __u32 version;
        __u32 features;
        __u32 status;
    } me;

    struct {
        __u8 present;
        __u32 vendor_id;
        __u32 device_id;
        __u32 version;
        __u32 features;
    } tpm;

    struct {
        __u8 present;
        __u32 token_count;      /* Number of available tokens */
        __u32 authorized_tokens; /* Number of authorized tokens */
        __u64 token_bitmap;     /* Bitmap of authorized tokens */
    } dell_smbios;

    __u32 total_memory_mb;      /* Total available memory */
    __u32 cpu_cores;            /* Number of CPU cores */
    __u32 cpu_frequency_mhz;    /* CPU frequency */
    __u32 reserved[8];          /* Reserved for future use */
};

/*
 * Event Notification Structure
 */
struct tpm2_accel_event {
    __u32 event_type;          /* Event type (enum tpm2_accel_event_type) */
    __u32 severity;            /* Severity level (0=info, 1=warning, 2=error, 3=critical) */
    __u64 timestamp;           /* Event timestamp (nanoseconds since boot) */
    __u32 source;              /* Event source (hardware component) */
    __u32 error_code;          /* Error code if applicable */
    __u64 data1;               /* Event-specific data 1 */
    __u64 data2;               /* Event-specific data 2 */
    char message[128];         /* Human-readable message */
    __u32 reserved[4];         /* Reserved for future use */
};

/*
 * Performance Metrics Structure
 */
struct tpm2_accel_metrics {
    __u64 operations_per_second;
    __u64 average_latency_us;   /* Average latency in microseconds */
    __u64 max_latency_us;       /* Maximum latency in microseconds */
    __u64 min_latency_us;       /* Minimum latency in microseconds */
    __u32 npu_utilization_percent;
    __u32 gna_utilization_percent;
    __u32 cpu_utilization_percent;
    __u32 memory_utilization_percent;
    __u64 cache_hit_rate_percent;
    __u64 error_rate_percent;
    __u32 queue_depth_current;
    __u32 queue_depth_max;
    __u64 throughput_bytes_per_second;
    __u32 reserved[8];         /* Reserved for future use */
};

/*
 * Debug Information Structure
 */
struct tpm2_accel_debug_info {
    __u32 module_state;        /* Current module state */
    __u32 initialization_stage; /* Initialization stage */
    __u32 error_count;         /* Total error count */
    __u32 warning_count;       /* Total warning count */
    __u64 last_error_timestamp;
    __u32 last_error_code;
    char last_error_message[128];
    __u32 hardware_failures;   /* Hardware failure count */
    __u32 security_failures;   /* Security failure count */
    __u32 communication_failures; /* Communication failure count */
    __u32 reserved[16];        /* Reserved for future use */
};

/*
 * IOCTL Command Definitions
 */
#define TPM2_ACCEL_IOC_MAGIC    'T'

/* Basic Operations */
#define TPM2_ACCEL_IOC_INIT     _IO(TPM2_ACCEL_IOC_MAGIC, 1)
#define TPM2_ACCEL_IOC_PROCESS  _IOWR(TPM2_ACCEL_IOC_MAGIC, 2, struct tpm2_accel_cmd)
#define TPM2_ACCEL_IOC_STATUS   _IOR(TPM2_ACCEL_IOC_MAGIC, 3, struct tpm2_accel_status)
#define TPM2_ACCEL_IOC_CONFIG   _IOW(TPM2_ACCEL_IOC_MAGIC, 4, struct tpm2_accel_config)

/* Information Queries */
#define TPM2_ACCEL_IOC_GET_HARDWARE_INFO _IOR(TPM2_ACCEL_IOC_MAGIC, 5, struct tpm2_accel_hardware_info)
#define TPM2_ACCEL_IOC_GET_METRICS       _IOR(TPM2_ACCEL_IOC_MAGIC, 6, struct tpm2_accel_metrics)
#define TPM2_ACCEL_IOC_GET_DEBUG_INFO    _IOR(TPM2_ACCEL_IOC_MAGIC, 7, struct tpm2_accel_debug_info)

/* Event Management */
#define TPM2_ACCEL_IOC_WAIT_EVENT        _IOR(TPM2_ACCEL_IOC_MAGIC, 8, struct tpm2_accel_event)
#define TPM2_ACCEL_IOC_ACK_EVENT         _IOW(TPM2_ACCEL_IOC_MAGIC, 9, __u32)

/* Session Management */
#define TPM2_ACCEL_IOC_CREATE_SESSION    _IOWR(TPM2_ACCEL_IOC_MAGIC, 10, __u32)
#define TPM2_ACCEL_IOC_DESTROY_SESSION   _IOW(TPM2_ACCEL_IOC_MAGIC, 11, __u32)

/* Advanced Operations */
#define TPM2_ACCEL_IOC_BATCH_PROCESS     _IOWR(TPM2_ACCEL_IOC_MAGIC, 12, struct tpm2_accel_cmd)
#define TPM2_ACCEL_IOC_EMERGENCY_STOP    _IO(TPM2_ACCEL_IOC_MAGIC, 13)
#define TPM2_ACCEL_IOC_RESET             _IO(TPM2_ACCEL_IOC_MAGIC, 14)

/* Security Operations */
#define TPM2_ACCEL_IOC_VALIDATE_TOKEN    _IOW(TPM2_ACCEL_IOC_MAGIC, 15, __u32)
#define TPM2_ACCEL_IOC_SET_SECURITY_LEVEL _IOW(TPM2_ACCEL_IOC_MAGIC, 16, __u32)

#define TPM2_ACCEL_IOC_MAXNR 16

/*
 * Sysfs Attribute Names
 */
#define TPM2_ACCEL_SYSFS_HARDWARE_STATUS    "hardware_status"
#define TPM2_ACCEL_SYSFS_NPU_UTILIZATION    "npu_utilization"
#define TPM2_ACCEL_SYSFS_GNA_STATUS         "gna_status"
#define TPM2_ACCEL_SYSFS_PERFORMANCE_STATS  "performance_stats"
#define TPM2_ACCEL_SYSFS_SECURITY_LEVEL     "security_level"
#define TPM2_ACCEL_SYSFS_DELL_TOKENS        "dell_tokens"
#define TPM2_ACCEL_SYSFS_DEBUG_LEVEL        "debug_level"
#define TPM2_ACCEL_SYSFS_CONFIG             "config"

/*
 * Error Codes
 */
#define TPM2_ACCEL_ERROR_NONE               0
#define TPM2_ACCEL_ERROR_INVALID_COMMAND    1
#define TPM2_ACCEL_ERROR_INVALID_PARAMETER  2
#define TPM2_ACCEL_ERROR_UNAUTHORIZED       3
#define TPM2_ACCEL_ERROR_HARDWARE_FAILURE   4
#define TPM2_ACCEL_ERROR_TIMEOUT            5
#define TPM2_ACCEL_ERROR_OUT_OF_MEMORY      6
#define TPM2_ACCEL_ERROR_BUSY               7
#define TPM2_ACCEL_ERROR_NOT_SUPPORTED      8
#define TPM2_ACCEL_ERROR_SECURITY_VIOLATION 9
#define TPM2_ACCEL_ERROR_DELL_TOKEN_INVALID 10
#define TPM2_ACCEL_ERROR_NPU_FAILURE        11
#define TPM2_ACCEL_ERROR_GNA_FAILURE        12
#define TPM2_ACCEL_ERROR_ME_FAILURE         13
#define TPM2_ACCEL_ERROR_TPM_FAILURE        14

#ifdef __KERNEL__

/*
 * Kernel-Internal Structures
 */

/* Forward declarations for kernel structures */
struct tpm2_accel_device;
struct tpm2_accel_session;
struct tpm2_accel_operation;

/*
 * Hardware Abstraction Layer
 */
struct tpm2_accel_hw_ops {
    int (*detect)(struct tpm2_accel_device *dev);
    int (*init)(struct tpm2_accel_device *dev);
    int (*cleanup)(struct tpm2_accel_device *dev);
    int (*process)(struct tmp2_accel_device *dev, struct tpm2_accel_operation *op);
    int (*get_status)(struct tpm2_accel_device *dev, struct tpm2_accel_status *status);
    int (*emergency_stop)(struct tpm2_accel_device *dev);
};

/*
 * Device Structure
 */
struct tpm2_accel_device {
    struct device *dev;
    struct cdev cdev;
    dev_t devt;

    /* Hardware information */
    struct tpm2_accel_hardware_info hw_info;

    /* Hardware operations */
    struct tpm2_accel_hw_ops *hw_ops;

    /* Configuration */
    struct tpm2_accel_config config;

    /* State */
    atomic_t ref_count;
    bool initialized;
    bool emergency_stop;

    /* Synchronization */
    struct mutex lock;
    wait_queue_head_t wait_queue;

    /* Performance monitoring */
    struct tpm2_accel_metrics metrics;

    /* Security */
    u64 authorized_tokens;
    u32 current_security_level;

    /* Memory management */
    void *shared_memory;
    dma_addr_t shared_memory_dma;
    size_t shared_memory_size;

    /* Private data */
    void *private_data;
};

/*
 * Session Structure
 */
struct tpm2_accel_session {
    u32 session_id;
    u32 security_level;
    u32 dell_token;
    struct tpm2_accel_device *device;
    struct list_head session_list;
    atomic_t ref_count;
    bool active;
    ktime_t created_time;
    ktime_t last_access_time;
    u64 operations_count;
    u64 bytes_processed;
};

/*
 * Operation Structure
 */
struct tpm2_accel_operation {
    u32 op_id;
    struct tpm2_accel_cmd cmd;
    struct tpm2_accel_session *session;
    struct completion completion;
    int result;
    ktime_t start_time;
    ktime_t end_time;
    void *input_buffer;
    void *output_buffer;
    size_t actual_output_len;
    struct list_head op_list;
};

/*
 * Kernel API Functions
 */
int tpm2_accel_register_device(struct tpm2_accel_device *dev);
void tpm2_accel_unregister_device(struct tpm2_accel_device *dev);
int tpm2_accel_process_command(struct tpm2_accel_device *dev,
                              struct tpm2_accel_cmd *cmd,
                              void *input_data,
                              void *output_data,
                              size_t *output_len);
int tpm2_accel_notify_event(struct tpm2_accel_device *dev,
                           struct tpm2_accel_event *event);

/* Hardware-specific registration functions */
int tpm2_accel_register_npu_ops(struct tpm2_accel_hw_ops *ops);
int tpm2_accel_register_gna_ops(struct tpm2_accel_hw_ops *ops);
int tpm2_accel_register_me_ops(struct tpm2_accel_hw_ops *ops);
int tpm2_accel_register_tpm_ops(struct tpm2_accel_hw_ops *ops);

/* Security functions */
int tpm2_accel_validate_dell_token(u32 token);
int tpm2_accel_check_security_level(u32 required_level);
int tpm2_accel_authorize_operation(struct tpm2_accel_session *session,
                                  struct tpm2_accel_cmd *cmd);

/* Utility functions */
void tpm2_accel_update_metrics(struct tpm2_accel_device *dev,
                              struct tpm2_accel_operation *op);
int tpm2_accel_emergency_stop_all(void);

#endif /* __KERNEL__ */

#endif /* _TPM2_ACCEL_EARLY_H */
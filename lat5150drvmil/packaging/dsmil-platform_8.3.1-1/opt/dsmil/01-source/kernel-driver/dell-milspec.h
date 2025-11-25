/* SPDX-License-Identifier: GPL-2.0 WITH Linux-syscall-note */
/*
 * Dell Military Specification Subsystem Interface
 * Userspace API definitions
 */

#ifndef _UAPI_LINUX_DELL_MILSPEC_H
#define _UAPI_LINUX_DELL_MILSPEC_H

#include <linux/types.h>
#include <linux/ioctl.h>

/* Version for compatibility checking */
#define MILSPEC_API_VERSION 0x020000 /* 2.0.0 */

/* Mode 5 Security Levels */
enum milspec_mode5_level {
    MODE5_DISABLED = 0,
    MODE5_STANDARD = 1,      /* VM migration allowed */
    MODE5_ENHANCED = 2,      /* VMs locked to hardware */
    MODE5_PARANOID = 3,      /* Secure wipe on intrusion */
    MODE5_PARANOID_PLUS = 4, /* Maximum security, irreversible */
};

/* DSMIL Subsystem Modes */
enum milspec_dsmil_mode {
    DSMIL_OFF = 0,
    DSMIL_BASIC = 1,         /* Basic military features */
    DSMIL_ENHANCED = 2,      /* Full tactical capabilities */
    DSMIL_CLASSIFIED = 3,    /* Restricted mode */
};

/* Event Types for Logging */
enum milspec_event_type {
    MILSPEC_EVENT_BOOT = 1,
    MILSPEC_EVENT_ACTIVATION = 2,
    MILSPEC_EVENT_MODE_CHANGE = 3,
    MILSPEC_EVENT_ERROR = 4,
    MILSPEC_EVENT_USER_REQUEST = 5,
    MILSPEC_EVENT_SECURITY = 6,
    MILSPEC_EVENT_POWER = 7,
    MILSPEC_EVENT_FIRMWARE = 8,
    MILSPEC_EVENT_INTRUSION = 9,
    MILSPEC_EVENT_CRYPTO = 10,
};

/* Status Structure */
struct milspec_status {
    __u32 api_version;
    __u8 mode5_enabled;
    __u8 mode5_level;
    __u8 dsmil_active[12];
    __u8 dsmil_mode;
    __u8 service_mode;
    __u8 boot_progress;
    __u8 tpm_measured;
    __u8 crypto_present;
    __u8 intrusion_detected;
    __u32 activation_count;
    __u32 error_count;
    __u64 activation_time_ns;
    __u64 uptime_ns;
    __u32 event_count;
    __u8 reserved[28];
};

/* Event Structure */
struct milspec_event {
    __u64 timestamp_ns;
    __u32 event_type;
    __u32 data1;
    __u32 data2;
    char message[64];
};

/* Events Buffer */
struct milspec_events {
    __u32 count;
    __u32 lost;
    struct milspec_event events[64];
};

/* Security Status */
struct milspec_security {
    __u8 intrusion_detected;
    __u8 tamper_detected;
    __u8 secure_boot_enabled;
    __u8 measured_boot;
    __u8 encrypted_memory;
    __u8 dma_protection;
    __u16 security_flags;
    __u8 tpm_pcr[32];
    __u8 crypto_serial[16];
};

/* Firmware Update Types */
#define MILSPEC_FW_TYPE_MICROCODE  0x01
#define MILSPEC_FW_TYPE_DSMIL      0x02
#define MILSPEC_FW_TYPE_CRYPTO     0x03
#define MILSPEC_FW_TYPE_BIOS       0x04
#define MILSPEC_FW_TYPE_EC         0x05

/* Firmware Update Structure */
struct milspec_fw_update {
    __u32 type;
    __u32 version;
    __u64 size;
    __u64 data_ptr;
    __u8 signature[256];
};

/* IOCTL Commands */
#define MILSPEC_IOC_MAGIC 'M'

#define MILSPEC_IOC_GET_STATUS    _IOR(MILSPEC_IOC_MAGIC, 1, struct milspec_status)
#define MILSPEC_IOC_SET_MODE5     _IOW(MILSPEC_IOC_MAGIC, 2, __u32)
#define MILSPEC_IOC_ACTIVATE_DSMIL _IOW(MILSPEC_IOC_MAGIC, 3, __u32)
#define MILSPEC_IOC_GET_EVENTS    _IOR(MILSPEC_IOC_MAGIC, 4, struct milspec_events)
#define MILSPEC_IOC_FORCE_ACTIVATE _IO(MILSPEC_IOC_MAGIC, 5)
#define MILSPEC_IOC_GET_SECURITY  _IOR(MILSPEC_IOC_MAGIC, 6, struct milspec_security)
#define MILSPEC_IOC_EMERGENCY_WIPE _IOW(MILSPEC_IOC_MAGIC, 7, __u32)
#define MILSPEC_IOC_TPM_MEASURE   _IO(MILSPEC_IOC_MAGIC, 8)
#define MILSPEC_IOC_UPDATE_FW     _IOW(MILSPEC_IOC_MAGIC, 9, struct milspec_fw_update)

/* Emergency wipe confirmation code */
#define MILSPEC_WIPE_CONFIRM    0xDEADBEEF

/* sysfs paths */
#define MILSPEC_SYSFS_PATH "/sys/devices/platform/dell-milspec/"
#define MILSPEC_DEBUGFS_PATH "/sys/kernel/debug/dell-milspec/"

#endif /* _UAPI_LINUX_DELL_MILSPEC_H */

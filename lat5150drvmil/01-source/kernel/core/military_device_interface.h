/*
 * Military Device Interface Header
 * Dell Latitude 5450 MIL-SPEC DSMIL Device Interface
 * 
 * PHASE 1: Safe Foundation Interface for 0x8000-0x806B range
 * - READ-ONLY operations only
 * - Hardcoded quarantine enforcement
 * - Thermal safety monitoring
 * - JRTC1 training mode compatibility
 * 
 * Copyright (C) 2025 JRTC1 Educational Development
 * Security Level: READ-ONLY SAFE INTERFACE
 */

#ifndef MILITARY_DEVICE_INTERFACE_H
#define MILITARY_DEVICE_INTERFACE_H

#include <stdint.h>
#include <stdbool.h>
#include <sys/ioctl.h>

/* Version Information */
#define MILDEV_VERSION_MAJOR    1
#define MILDEV_VERSION_MINOR    0
#define MILDEV_VERSION_PATCH    0
#ifndef MILDEV_VERSION_STRING
#define MILDEV_VERSION_STRING   "1.0.0-Phase1"
#endif

/* Critical Safety Constants */
#define MILDEV_MAX_THERMAL_C    100     /* Maximum safe operating temperature */
#define MILDEV_DEVICE_PATH      "/dev/dsmil-72dev"
#define MILDEV_MAX_RETRIES      3       /* Maximum operation retries */
#define MILDEV_TIMEOUT_MS       5000    /* Operation timeout in milliseconds */

/* DSMIL Device Range Definitions - Phase 1 Target: 0x8000-0x806B */
#define MILDEV_BASE_ADDR        0x8000  /* Start of military device range */
#define MILDEV_END_ADDR         0x806B  /* End of Phase 1 target range */
#define MILDEV_RANGE_SIZE       (MILDEV_END_ADDR - MILDEV_BASE_ADDR + 1)  /* 108 devices */

/* CRITICAL QUARANTINE LIST - HARDCODED SAFETY */
#define MILDEV_QUARANTINE_COUNT 5
static const uint16_t MILDEV_QUARANTINE_LIST[MILDEV_QUARANTINE_COUNT] = {
    0x8009,  /* Critical security token */
    0x800A,  /* Master control token */
    0x800B,  /* System state token */
    0x8019,  /* Hardware control token */
    0x8029   /* Emergency override token */
};

/* Device State Definitions */
typedef enum {
    MILDEV_STATE_UNKNOWN = 0,
    MILDEV_STATE_OFFLINE,
    MILDEV_STATE_SAFE,
    MILDEV_STATE_QUARANTINED,
    MILDEV_STATE_ERROR,
    MILDEV_STATE_THERMAL_LIMIT
} mildev_state_t;

/* Device Access Level */
typedef enum {
    MILDEV_ACCESS_NONE = 0,
    MILDEV_ACCESS_READ,
    MILDEV_ACCESS_RESERVED  /* Future use - no write access in Phase 1 */
} mildev_access_level_t;

/* Device Information Structure */
typedef struct {
    uint16_t device_id;             /* Device ID (0x8000-0x806B) */
    mildev_state_t state;           /* Current device state */
    mildev_access_level_t access;   /* Access level granted */
    bool is_quarantined;            /* True if device is in quarantine list */
    uint32_t last_response;         /* Last response from device */
    uint64_t timestamp;             /* Last access timestamp */
    int thermal_celsius;            /* Current thermal reading */
} mildev_device_info_t;

/* System Status Structure */
typedef struct {
    bool kernel_module_loaded;      /* DSMIL kernel module status */
    bool thermal_safe;              /* Thermal conditions safe */
    int current_temp_celsius;       /* Current system temperature */
    uint32_t safe_device_count;     /* Number of safe devices found */
    uint32_t quarantined_count;     /* Number of quarantined devices */
    uint64_t last_scan_timestamp;   /* Last device scan timestamp */
} mildev_system_status_t;

/* Device Discovery Results */
typedef struct {
    uint32_t total_devices_found;
    uint32_t safe_devices_found;
    uint32_t quarantined_devices_found;
    uint64_t last_scan_timestamp;   /* Last scan timestamp */
    mildev_device_info_t devices[MILDEV_RANGE_SIZE];
} mildev_discovery_result_t;

/* IOCTL Command Definitions */
#define MILDEV_IOC_MAGIC        'M'
#define MILDEV_IOC_GET_VERSION  _IOR(MILDEV_IOC_MAGIC, 1, uint32_t)
#define MILDEV_IOC_GET_STATUS   _IOR(MILDEV_IOC_MAGIC, 2, mildev_system_status_t)
#define MILDEV_IOC_SCAN_DEVICES _IOR(MILDEV_IOC_MAGIC, 3, mildev_discovery_result_t)
#define MILDEV_IOC_READ_DEVICE  _IOWR(MILDEV_IOC_MAGIC, 4, mildev_device_info_t)
#define MILDEV_IOC_GET_THERMAL  _IOR(MILDEV_IOC_MAGIC, 5, int)

/* Error Codes - Military Device Specific */
typedef enum {
    MILDEV_SUCCESS = 0,
    MILDEV_ERROR_INVALID_DEVICE = -1000,
    MILDEV_ERROR_QUARANTINED = -1001,
    MILDEV_ERROR_THERMAL_LIMIT = -1002,
    MILDEV_ERROR_KERNEL_MODULE = -1003,
    MILDEV_ERROR_ACCESS_DENIED = -1004,
    MILDEV_ERROR_TIMEOUT = -1005,
    MILDEV_ERROR_HARDWARE_FAULT = -1006,
    MILDEV_ERROR_INVALID_RANGE = -1007,
    MILDEV_ERROR_SYSTEM_UNSAFE = -1008
} mildev_error_t;

/* Safety Macros */
#define MILDEV_IS_QUARANTINED(device_id) mildev_check_quarantine(device_id)
#define MILDEV_IS_SAFE_RANGE(device_id) ((device_id) >= MILDEV_BASE_ADDR && (device_id) <= MILDEV_END_ADDR)
#define MILDEV_IS_THERMAL_SAFE(temp) ((temp) > 0 && (temp) < MILDEV_MAX_THERMAL_C)

/* Validation Macros */
#define MILDEV_VALIDATE_DEVICE_ID(device_id) \
    do { \
        if (!MILDEV_IS_SAFE_RANGE(device_id)) { \
            return MILDEV_ERROR_INVALID_RANGE; \
        } \
        if (MILDEV_IS_QUARANTINED(device_id)) { \
            return MILDEV_ERROR_QUARANTINED; \
        } \
    } while(0)

#define MILDEV_VALIDATE_THERMAL() \
    do { \
        int current_temp = mildev_get_thermal_celsius(); \
        if (!MILDEV_IS_THERMAL_SAFE(current_temp)) { \
            return MILDEV_ERROR_THERMAL_LIMIT; \
        } \
    } while(0)

/* Function Declarations */

/* Initialization and Cleanup */
int mildev_init(void);
void mildev_cleanup(void);

/* Device Discovery and Information */
int mildev_scan_devices(mildev_discovery_result_t *result);
int mildev_get_device_info(uint16_t device_id, mildev_device_info_t *info);
int mildev_get_system_status(mildev_system_status_t *status);

/* Safe Read Operations (READ-ONLY) */
int mildev_read_device_safe(uint16_t device_id, uint32_t *response);
int mildev_read_device_with_retry(uint16_t device_id, uint32_t *response, int max_retries);

/* Safety and Validation Functions */
bool mildev_check_quarantine(uint16_t device_id);
int mildev_get_thermal_celsius(void);
bool mildev_is_system_safe(void);

/* Utility Functions */
const char* mildev_error_string(mildev_error_t error);
const char* mildev_state_string(mildev_state_t state);
void mildev_print_device_info(const mildev_device_info_t *info);
void mildev_print_discovery_summary(const mildev_discovery_result_t *result);

/* Memory-Mapped I/O Functions (READ-ONLY) */
int mildev_mmap_init(void);
void mildev_mmap_cleanup(void);
int mildev_mmap_read_device(uint16_t device_id, uint32_t *value);

/* Logging and Debug Functions */
void mildev_log_info(const char *format, ...);
void mildev_log_warning(const char *format, ...);
void mildev_log_error(const char *format, ...);
void mildev_emergency_stop(const char *reason);

/* Version Information */
void mildev_print_version(void);
uint32_t mildev_get_version_code(void);

#endif /* MILITARY_DEVICE_INTERFACE_H */
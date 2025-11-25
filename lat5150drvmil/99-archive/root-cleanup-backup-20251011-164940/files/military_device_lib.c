/*
 * Military Device Library Implementation
 * Dell Latitude 5450 MIL-SPEC DSMIL Device Interface
 * 
 * PHASE 1: Safe Foundation Library for 0x8000-0x806B range
 * - READ-ONLY operations only
 * - Hardcoded quarantine enforcement  
 * - Thermal safety monitoring
 * - Integration with existing DSMIL kernel module
 * 
 * Copyright (C) 2025 JRTC1 Educational Development
 * Security Level: READ-ONLY SAFE INTERFACE
 */

#include "military_device_interface.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <stdarg.h>
#include <time.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <pthread.h>

/* Global state */
static int g_dsmil_fd = -1;
static void *g_mmap_base = NULL;
static bool g_initialized = false;
static pthread_mutex_t g_mildev_mutex = PTHREAD_MUTEX_INITIALIZER;

/* Internal constants */
#define MILDEV_LOG_BUFFER_SIZE  1024
#define MILDEV_MMAP_SIZE       0x1000  /* 4KB mapping size */
#define MILDEV_DEVICE_BASE_PHYS 0x52000000  /* Physical base from DSMIL */

/* Internal function declarations */
static int mildev_open_kernel_device(void);
static int mildev_read_thermal_sensor(void);
static void mildev_log_internal(const char *level, const char *format, va_list args);
static uint64_t mildev_get_timestamp_ms(void);

/*
 * Initialization and Cleanup Functions
 */

int mildev_init(void)
{
    pthread_mutex_lock(&g_mildev_mutex);
    
    if (g_initialized) {
        pthread_mutex_unlock(&g_mildev_mutex);
        return MILDEV_SUCCESS;
    }
    
    mildev_log_info("Initializing Military Device Interface v%s", MILDEV_VERSION_STRING);
    
    /* Open kernel device */
    int result = mildev_open_kernel_device();
    if (result != MILDEV_SUCCESS) {
        mildev_log_error("Failed to open kernel device: %s", mildev_error_string(result));
        pthread_mutex_unlock(&g_mildev_mutex);
        return result;
    }
    
    /* Verify thermal safety */
    int temp = mildev_get_thermal_celsius();
    if (temp < 0 || temp >= MILDEV_MAX_THERMAL_C) {
        mildev_log_error("Thermal conditions unsafe: %d°C (limit: %d°C)", temp, MILDEV_MAX_THERMAL_C);
        close(g_dsmil_fd);
        g_dsmil_fd = -1;
        pthread_mutex_unlock(&g_mildev_mutex);
        return MILDEV_ERROR_THERMAL_LIMIT;
    }
    
    mildev_log_info("Thermal conditions safe: %d°C", temp);
    
    /* Initialize memory mapping */
    result = mildev_mmap_init();
    if (result != MILDEV_SUCCESS) {
        mildev_log_warning("Memory mapping initialization failed, continuing with IOCTL mode");
    }
    
    g_initialized = true;
    mildev_log_info("Military Device Interface initialized successfully");
    
    pthread_mutex_unlock(&g_mildev_mutex);
    return MILDEV_SUCCESS;
}

void mildev_cleanup(void)
{
    pthread_mutex_lock(&g_mildev_mutex);
    
    if (!g_initialized) {
        pthread_mutex_unlock(&g_mildev_mutex);
        return;
    }
    
    mildev_log_info("Cleaning up Military Device Interface");
    
    /* Cleanup memory mapping */
    mildev_mmap_cleanup();
    
    /* Close kernel device */
    if (g_dsmil_fd >= 0) {
        close(g_dsmil_fd);
        g_dsmil_fd = -1;
    }
    
    g_initialized = false;
    mildev_log_info("Military Device Interface cleanup complete");
    
    pthread_mutex_unlock(&g_mildev_mutex);
}

/*
 * Device Discovery and Information Functions
 */

int mildev_scan_devices(mildev_discovery_result_t *result)
{
    if (!result) {
        return MILDEV_ERROR_INVALID_DEVICE;
    }
    
    if (!g_initialized) {
        return MILDEV_ERROR_KERNEL_MODULE;
    }
    
    MILDEV_VALIDATE_THERMAL();
    
    pthread_mutex_lock(&g_mildev_mutex);
    
    memset(result, 0, sizeof(mildev_discovery_result_t));
    result->last_scan_timestamp = mildev_get_timestamp_ms();
    
    mildev_log_info("Scanning military devices in range 0x%04X-0x%04X", 
                    MILDEV_BASE_ADDR, MILDEV_END_ADDR);
    
    /* Scan each device in the target range */
    for (uint16_t device_id = MILDEV_BASE_ADDR; device_id <= MILDEV_END_ADDR; device_id++) {
        int device_index = device_id - MILDEV_BASE_ADDR;
        mildev_device_info_t *device = &result->devices[device_index];
        
        /* Initialize device info */
        device->device_id = device_id;
        device->timestamp = mildev_get_timestamp_ms();
        device->thermal_celsius = mildev_get_thermal_celsius();
        
        /* Check quarantine status */
        device->is_quarantined = mildev_check_quarantine(device_id);
        if (device->is_quarantined) {
            device->state = MILDEV_STATE_QUARANTINED;
            device->access = MILDEV_ACCESS_NONE;
            result->quarantined_devices_found++;
            mildev_log_warning("Device 0x%04X quarantined", device_id);
            continue;
        }
        
        /* Attempt safe read to detect device presence */
        uint32_t response;
        int read_result = mildev_read_device_safe(device_id, &response);
        
        if (read_result == MILDEV_SUCCESS) {
            device->state = MILDEV_STATE_SAFE;
            device->access = MILDEV_ACCESS_READ;
            device->last_response = response;
            result->safe_devices_found++;
        } else if (read_result == MILDEV_ERROR_THERMAL_LIMIT) {
            device->state = MILDEV_STATE_THERMAL_LIMIT;
            device->access = MILDEV_ACCESS_NONE;
            mildev_emergency_stop("Thermal limit exceeded during device scan");
            break;
        } else {
            device->state = MILDEV_STATE_OFFLINE;
            device->access = MILDEV_ACCESS_NONE;
        }
        
        result->total_devices_found++;
    }
    
    mildev_log_info("Device scan complete: %d total, %d safe, %d quarantined", 
                    result->total_devices_found, result->safe_devices_found, 
                    result->quarantined_devices_found);
    
    pthread_mutex_unlock(&g_mildev_mutex);
    return MILDEV_SUCCESS;
}

int mildev_get_device_info(uint16_t device_id, mildev_device_info_t *info)
{
    if (!info) {
        return MILDEV_ERROR_INVALID_DEVICE;
    }
    
    MILDEV_VALIDATE_DEVICE_ID(device_id);
    MILDEV_VALIDATE_THERMAL();
    
    pthread_mutex_lock(&g_mildev_mutex);
    
    memset(info, 0, sizeof(mildev_device_info_t));
    info->device_id = device_id;
    info->timestamp = mildev_get_timestamp_ms();
    info->thermal_celsius = mildev_get_thermal_celsius();
    info->is_quarantined = mildev_check_quarantine(device_id);
    
    if (info->is_quarantined) {
        info->state = MILDEV_STATE_QUARANTINED;
        info->access = MILDEV_ACCESS_NONE;
        pthread_mutex_unlock(&g_mildev_mutex);
        return MILDEV_ERROR_QUARANTINED;
    }
    
    /* Attempt to read device */
    uint32_t response;
    int result = mildev_read_device_safe(device_id, &response);
    
    if (result == MILDEV_SUCCESS) {
        info->state = MILDEV_STATE_SAFE;
        info->access = MILDEV_ACCESS_READ;
        info->last_response = response;
    } else {
        info->state = MILDEV_STATE_ERROR;
        info->access = MILDEV_ACCESS_NONE;
    }
    
    pthread_mutex_unlock(&g_mildev_mutex);
    return result;
}

int mildev_get_system_status(mildev_system_status_t *status)
{
    if (!status) {
        return MILDEV_ERROR_INVALID_DEVICE;
    }
    
    memset(status, 0, sizeof(mildev_system_status_t));
    
    /* Check kernel module status */
    status->kernel_module_loaded = (g_dsmil_fd >= 0);
    
    /* Get thermal status */
    status->current_temp_celsius = mildev_get_thermal_celsius();
    status->thermal_safe = MILDEV_IS_THERMAL_SAFE(status->current_temp_celsius);
    
    /* Quick device count (without full scan) */
    if (status->kernel_module_loaded && status->thermal_safe) {
        mildev_discovery_result_t quick_scan;
        if (mildev_scan_devices(&quick_scan) == MILDEV_SUCCESS) {
            status->safe_device_count = quick_scan.safe_devices_found;
            status->quarantined_count = quick_scan.quarantined_devices_found;
            status->last_scan_timestamp = quick_scan.last_scan_timestamp;
        }
    }
    
    return MILDEV_SUCCESS;
}

/*
 * Safe Read Operations (READ-ONLY)
 */

int mildev_read_device_safe(uint16_t device_id, uint32_t *response)
{
    if (!response) {
        return MILDEV_ERROR_INVALID_DEVICE;
    }
    
    MILDEV_VALIDATE_DEVICE_ID(device_id);
    MILDEV_VALIDATE_THERMAL();
    
    if (!g_initialized) {
        return MILDEV_ERROR_KERNEL_MODULE;
    }
    
    /* Try memory-mapped I/O first */
    int result = mildev_mmap_read_device(device_id, response);
    if (result == MILDEV_SUCCESS) {
        return MILDEV_SUCCESS;
    }
    
    /* Fallback to IOCTL interface */
    mildev_device_info_t device_info = {0};
    device_info.device_id = device_id;
    
    if (ioctl(g_dsmil_fd, MILDEV_IOC_READ_DEVICE, &device_info) < 0) {
        mildev_log_error("IOCTL read failed for device 0x%04X: %s", device_id, strerror(errno));
        return MILDEV_ERROR_HARDWARE_FAULT;
    }
    
    *response = device_info.last_response;
    return MILDEV_SUCCESS;
}

int mildev_read_device_with_retry(uint16_t device_id, uint32_t *response, int max_retries)
{
    int result;
    int attempts = 0;
    
    do {
        result = mildev_read_device_safe(device_id, response);
        if (result == MILDEV_SUCCESS) {
            if (attempts > 0) {
                mildev_log_info("Device 0x%04X read succeeded on attempt %d", device_id, attempts + 1);
            }
            return MILDEV_SUCCESS;
        }
        
        /* Don't retry on critical errors */
        if (result == MILDEV_ERROR_THERMAL_LIMIT || 
            result == MILDEV_ERROR_QUARANTINED ||
            result == MILDEV_ERROR_INVALID_RANGE) {
            return result;
        }
        
        attempts++;
        if (attempts < max_retries) {
            mildev_log_warning("Device 0x%04X read attempt %d failed, retrying...", device_id, attempts);
            usleep(100000); /* 100ms delay between retries */
        }
        
    } while (attempts < max_retries);
    
    mildev_log_error("Device 0x%04X read failed after %d attempts", device_id, max_retries);
    return result;
}

/*
 * Safety and Validation Functions
 */

bool mildev_check_quarantine(uint16_t device_id)
{
    for (int i = 0; i < MILDEV_QUARANTINE_COUNT; i++) {
        if (device_id == MILDEV_QUARANTINE_LIST[i]) {
            return true;
        }
    }
    return false;
}

int mildev_get_thermal_celsius(void)
{
    return mildev_read_thermal_sensor();
}

bool mildev_is_system_safe(void)
{
    if (!g_initialized) {
        return false;
    }
    
    int temp = mildev_get_thermal_celsius();
    return MILDEV_IS_THERMAL_SAFE(temp);
}

/*
 * Memory-Mapped I/O Functions (READ-ONLY)
 */

int mildev_mmap_init(void)
{
    if (g_mmap_base != NULL) {
        return MILDEV_SUCCESS;  /* Already initialized */
    }
    
    int mem_fd = open("/dev/mem", O_RDONLY);
    if (mem_fd < 0) {
        mildev_log_warning("Cannot open /dev/mem for memory mapping: %s", strerror(errno));
        return MILDEV_ERROR_ACCESS_DENIED;
    }
    
    g_mmap_base = mmap(NULL, MILDEV_MMAP_SIZE, PROT_READ, MAP_SHARED, mem_fd, MILDEV_DEVICE_BASE_PHYS);
    close(mem_fd);
    
    if (g_mmap_base == MAP_FAILED) {
        mildev_log_warning("Memory mapping failed: %s", strerror(errno));
        g_mmap_base = NULL;
        return MILDEV_ERROR_HARDWARE_FAULT;
    }
    
    mildev_log_info("Memory mapping initialized at physical 0x%08X", MILDEV_DEVICE_BASE_PHYS);
    return MILDEV_SUCCESS;
}

void mildev_mmap_cleanup(void)
{
    if (g_mmap_base != NULL) {
        munmap(g_mmap_base, MILDEV_MMAP_SIZE);
        g_mmap_base = NULL;
        mildev_log_info("Memory mapping cleaned up");
    }
}

int mildev_mmap_read_device(uint16_t device_id, uint32_t *value)
{
    if (g_mmap_base == NULL || value == NULL) {
        return MILDEV_ERROR_HARDWARE_FAULT;
    }
    
    /* Calculate offset within mapped region */
    uint32_t offset = (device_id - MILDEV_BASE_ADDR) * sizeof(uint32_t);
    if (offset >= MILDEV_MMAP_SIZE) {
        return MILDEV_ERROR_INVALID_RANGE;
    }
    
    /* Read from mapped memory */
    volatile uint32_t *device_ptr = (volatile uint32_t *)((char *)g_mmap_base + offset);
    *value = *device_ptr;
    
    return MILDEV_SUCCESS;
}

/*
 * Utility Functions
 */

const char* mildev_error_string(mildev_error_t error)
{
    switch (error) {
        case MILDEV_SUCCESS:           return "Success";
        case MILDEV_ERROR_INVALID_DEVICE:  return "Invalid device";
        case MILDEV_ERROR_QUARANTINED: return "Device quarantined";
        case MILDEV_ERROR_THERMAL_LIMIT: return "Thermal limit exceeded";
        case MILDEV_ERROR_KERNEL_MODULE: return "Kernel module error";
        case MILDEV_ERROR_ACCESS_DENIED: return "Access denied";
        case MILDEV_ERROR_TIMEOUT:     return "Operation timeout";
        case MILDEV_ERROR_HARDWARE_FAULT: return "Hardware fault";
        case MILDEV_ERROR_INVALID_RANGE: return "Invalid range";
        case MILDEV_ERROR_SYSTEM_UNSAFE: return "System unsafe";
        default:                       return "Unknown error";
    }
}

const char* mildev_state_string(mildev_state_t state)
{
    switch (state) {
        case MILDEV_STATE_UNKNOWN:     return "Unknown";
        case MILDEV_STATE_OFFLINE:     return "Offline";
        case MILDEV_STATE_SAFE:        return "Safe";
        case MILDEV_STATE_QUARANTINED: return "Quarantined";
        case MILDEV_STATE_ERROR:       return "Error";
        case MILDEV_STATE_THERMAL_LIMIT: return "Thermal Limit";
        default:                       return "Invalid State";
    }
}

void mildev_print_device_info(const mildev_device_info_t *info)
{
    if (!info) return;
    
    printf("Device 0x%04X:\n", info->device_id);
    printf("  State: %s\n", mildev_state_string(info->state));
    printf("  Access: %s\n", (info->access == MILDEV_ACCESS_READ) ? "Read" : "None");
    printf("  Quarantined: %s\n", info->is_quarantined ? "Yes" : "No");
    printf("  Last Response: 0x%08X\n", info->last_response);
    printf("  Temperature: %d°C\n", info->thermal_celsius);
    printf("  Timestamp: %llu ms\n", (unsigned long long)info->timestamp);
}

void mildev_print_discovery_summary(const mildev_discovery_result_t *result)
{
    if (!result) return;
    
    printf("=== Military Device Discovery Summary ===\n");
    printf("Total devices scanned: %d\n", result->total_devices_found);
    printf("Safe devices found: %d\n", result->safe_devices_found);
    printf("Quarantined devices: %d\n", result->quarantined_devices_found);
    printf("Scan timestamp: %llu ms\n", (unsigned long long)result->last_scan_timestamp);
    printf("Range: 0x%04X - 0x%04X (%d devices)\n", 
           MILDEV_BASE_ADDR, MILDEV_END_ADDR, MILDEV_RANGE_SIZE);
}

/*
 * Logging and Debug Functions
 */

void mildev_log_info(const char *format, ...)
{
    va_list args;
    va_start(args, format);
    mildev_log_internal("INFO", format, args);
    va_end(args);
}

void mildev_log_warning(const char *format, ...)
{
    va_list args;
    va_start(args, format);
    mildev_log_internal("WARN", format, args);
    va_end(args);
}

void mildev_log_error(const char *format, ...)
{
    va_list args;
    va_start(args, format);
    mildev_log_internal("ERROR", format, args);
    va_end(args);
}

void mildev_emergency_stop(const char *reason)
{
    mildev_log_error("EMERGENCY STOP: %s", reason);
    printf("\n*** EMERGENCY STOP ***\n");
    printf("Reason: %s\n", reason);
    printf("System halted for safety\n");
    mildev_cleanup();
    exit(1);
}

void mildev_print_version(void)
{
    printf("Military Device Interface v%s\n", MILDEV_VERSION_STRING);
    printf("Dell Latitude 5450 MIL-SPEC DSMIL Interface\n");
    printf("PHASE 1: Safe Foundation Interface\n");
    printf("Range: 0x%04X-0x%04X (%d devices)\n", 
           MILDEV_BASE_ADDR, MILDEV_END_ADDR, MILDEV_RANGE_SIZE);
    printf("Safety: READ-ONLY with thermal monitoring\n");
}

uint32_t mildev_get_version_code(void)
{
    return (MILDEV_VERSION_MAJOR << 16) | (MILDEV_VERSION_MINOR << 8) | MILDEV_VERSION_PATCH;
}

/*
 * Internal Helper Functions
 */

static int mildev_open_kernel_device(void)
{
    g_dsmil_fd = open(MILDEV_DEVICE_PATH, O_RDONLY);
    if (g_dsmil_fd < 0) {
        mildev_log_error("Cannot open %s: %s", MILDEV_DEVICE_PATH, strerror(errno));
        return MILDEV_ERROR_KERNEL_MODULE;
    }
    
    /* Verify it's the correct device */
    uint32_t version;
    if (ioctl(g_dsmil_fd, MILDEV_IOC_GET_VERSION, &version) < 0) {
        mildev_log_warning("Cannot get device version, continuing anyway");
    }
    
    mildev_log_info("Opened DSMIL kernel device successfully");
    return MILDEV_SUCCESS;
}

static int mildev_read_thermal_sensor(void)
{
    /* Read thermal sensor from kernel module */
    if (g_dsmil_fd >= 0) {
        int temp;
        if (ioctl(g_dsmil_fd, MILDEV_IOC_GET_THERMAL, &temp) == 0) {
            return temp;
        }
    }
    
    /* Fallback: read from system thermal zone */
    FILE *thermal_file = fopen("/sys/class/thermal/thermal_zone0/temp", "r");
    if (thermal_file) {
        int temp_millidegrees;
        if (fscanf(thermal_file, "%d", &temp_millidegrees) == 1) {
            fclose(thermal_file);
            return temp_millidegrees / 1000;  /* Convert to Celsius */
        }
        fclose(thermal_file);
    }
    
    /* Default safe temperature if cannot read */
    mildev_log_warning("Cannot read thermal sensor, assuming 50°C");
    return 50;
}

static void mildev_log_internal(const char *level, const char *format, va_list args)
{
    char timestamp_str[32];
    time_t now = time(NULL);
    struct tm *tm_info = localtime(&now);
    strftime(timestamp_str, sizeof(timestamp_str), "%Y-%m-%d %H:%M:%S", tm_info);
    
    char buffer[MILDEV_LOG_BUFFER_SIZE];
    vsnprintf(buffer, sizeof(buffer), format, args);
    
    printf("[%s] MILDEV-%s: %s\n", timestamp_str, level, buffer);
    fflush(stdout);
}

static uint64_t mildev_get_timestamp_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)(ts.tv_sec * 1000) + (ts.tv_nsec / 1000000);
}
/*
 * Basic IOCTL test program for dsmil-72dev kernel module
 * Tests the new MILDEV_IOC_READ_DEVICE functionality
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <errno.h>
#include <string.h>
#include "military_device_interface.h"

void test_version(int fd)
{
    uint32_t version;
    int ret;
    
    printf("=== Testing MILDEV_IOC_GET_VERSION ===\n");
    ret = ioctl(fd, MILDEV_IOC_GET_VERSION, &version);
    if (ret == 0) {
        printf("✅ Version: 0x%08x (Major: %d, Minor: %d, Patch: %d)\n", 
               version, (version >> 16) & 0xFF, (version >> 8) & 0xFF, version & 0xFF);
    } else {
        printf("❌ Version request failed: %s\n", strerror(errno));
    }
}

void test_thermal(int fd)
{
    int thermal;
    int ret;
    
    printf("\n=== Testing MILDEV_IOC_GET_THERMAL ===\n");
    ret = ioctl(fd, MILDEV_IOC_GET_THERMAL, &thermal);
    if (ret == 0) {
        printf("✅ Thermal: %d°C\n", thermal);
    } else {
        printf("❌ Thermal request failed: %s\n", strerror(errno));
    }
}

void test_system_status(int fd)
{
    struct mildev_system_status status;
    int ret;
    
    printf("\n=== Testing MILDEV_IOC_GET_STATUS ===\n");
    ret = ioctl(fd, MILDEV_IOC_GET_STATUS, &status);
    if (ret == 0) {
        printf("✅ System Status:\n");
        printf("   Module loaded: %s\n", status.kernel_module_loaded ? "Yes" : "No");
        printf("   Thermal safe: %s\n", status.thermal_safe ? "Yes" : "No");
        printf("   Current temp: %d°C\n", status.current_temp_celsius);
        printf("   Safe devices: %u\n", status.safe_device_count);
        printf("   Quarantined: %u\n", status.quarantined_count);
        printf("   Last scan: %llu\n", (unsigned long long)status.last_scan_timestamp);
    } else {
        printf("❌ System status request failed: %s\n", strerror(errno));
    }
}

void test_device_read(int fd, uint16_t device_id)
{
    struct mildev_device_info dev_info;
    int ret;
    
    printf("\n=== Testing MILDEV_IOC_READ_DEVICE (0x%04X) ===\n", device_id);
    
    memset(&dev_info, 0, sizeof(dev_info));
    dev_info.device_id = device_id;
    
    ret = ioctl(fd, MILDEV_IOC_READ_DEVICE, &dev_info);
    
    printf("Device ID: 0x%04X\n", dev_info.device_id);
    printf("State: %u ", dev_info.state);
    switch(dev_info.state) {
        case 0: printf("(UNKNOWN)"); break;
        case 1: printf("(OFFLINE)"); break;
        case 2: printf("(SAFE)"); break;
        case 3: printf("(QUARANTINED)"); break;
        case 4: printf("(ERROR)"); break;
        case 5: printf("(THERMAL_LIMIT)"); break;
        default: printf("(UNKNOWN_%u)", dev_info.state); break;
    }
    printf("\n");
    
    printf("Access Level: %u ", dev_info.access);
    switch(dev_info.access) {
        case 0: printf("(NONE)"); break;
        case 1: printf("(READ)"); break;
        case 2: printf("(RESERVED)"); break;
        default: printf("(UNKNOWN_%u)", dev_info.access); break;
    }
    printf("\n");
    
    printf("Quarantined: %s\n", dev_info.is_quarantined ? "Yes" : "No");
    printf("Response: 0x%08X\n", dev_info.last_response);
    printf("Thermal: %d°C\n", dev_info.thermal_celsius);
    printf("Timestamp: %llu\n", (unsigned long long)dev_info.timestamp);
    
    if (ret == 0) {
        printf("✅ Device read successful\n");
    } else {
        printf("❌ Device read failed: %s (errno: %d)\n", strerror(errno), errno);
        if (errno == EACCES) {
            printf("   → Device is quarantined\n");
        } else if (errno == EINVAL) {
            printf("   → Invalid device ID range\n");
        } else if (errno == EBUSY) {
            printf("   → Thermal protection active\n");
        }
    }
}

int main()
{
    int fd;
    
    printf("DSMIL-72DEV IOCTL Test Program\n");
    printf("==============================\n");
    
    fd = open("/dev/dsmil-72dev", O_RDWR);
    if (fd < 0) {
        perror("Failed to open /dev/dsmil-72dev");
        return 1;
    }
    
    printf("Device opened successfully.\n");
    
    /* Test basic IOCTL functions */
    test_version(fd);
    test_thermal(fd);
    test_system_status(fd);
    
    /* Test safe device reads */
    printf("\n==================================================\n");
    printf("DEVICE READ TESTS\n");
    printf("==================================================\n");
    
    /* Test safe device */
    test_device_read(fd, 0x8000);  /* First device - should be safe */
    
    /* Test quarantined device */
    test_device_read(fd, 0x8009);  /* Quarantined device */
    
    /* Test another safe device */
    test_device_read(fd, 0x8001);  /* Second device - should be safe */
    
    /* Test invalid device */
    test_device_read(fd, 0x9000);  /* Outside range */
    
    /* Test edge case */
    test_device_read(fd, 0x806B);  /* Last valid device */
    
    close(fd);
    printf("\n✅ Test completed successfully!\n");
    
    return 0;
}
/*
 * Correct IOCTL Handler Discovery for DSMIL Kernel Module
 * Uses proper data structures from kernel module analysis
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <string.h>
#include <sys/ioctl.h>
#include <linux/types.h>

/* DSMIL data structures (from kernel module analysis) */
struct mildev_device_info {
    __u16 device_id;
    __u32 state;
    __u32 access;
    __u8 is_quarantined;
    __u32 last_response;
    __s32 thermal_celsius;
    __u64 timestamp;
};

struct mildev_system_status {
    __u8 kernel_module_loaded;
    __u8 thermal_safe;
    __s32 current_temp_celsius;
    __u32 safe_device_count;
    __u32 quarantined_count;
    __u64 last_scan_timestamp;
};

struct mildev_discovery_result {
    __u32 total_devices_found;
    __u32 safe_devices_found;
    __u32 quarantined_devices_found;
    __u64 last_scan_timestamp;
    struct mildev_device_info devices[108]; /* MILDEV_RANGE_SIZE */
};

/* DSMIL IOCTL commands with correct types */
#define MILDEV_IOC_MAGIC        'M'
#define MILDEV_IOC_GET_VERSION  _IOR(MILDEV_IOC_MAGIC, 1, __u32)
#define MILDEV_IOC_GET_STATUS   _IOR(MILDEV_IOC_MAGIC, 2, struct mildev_system_status)
#define MILDEV_IOC_SCAN_DEVICES _IOR(MILDEV_IOC_MAGIC, 3, struct mildev_discovery_result)
#define MILDEV_IOC_READ_DEVICE  _IOWR(MILDEV_IOC_MAGIC, 4, struct mildev_device_info)
#define MILDEV_IOC_GET_THERMAL  _IOR(MILDEV_IOC_MAGIC, 5, int)

/* Device ID constants from kernel analysis */
#define MILDEV_BASE_ADDR        0x8000
#define MILDEV_END_ADDR         0x806B

/* Test all IOCTL commands properly */
void test_all_ioctls(int fd) {
    printf("=== COMPREHENSIVE DSMIL IOCTL TEST ===\n\n");
    
    /* Test 1: GET_VERSION */
    printf("1. Testing GET_VERSION (0x%08lX):\n", (unsigned long)MILDEV_IOC_GET_VERSION);
    __u32 version = 0;
    int result = ioctl(fd, MILDEV_IOC_GET_VERSION, &version);
    if (result >= 0) {
        printf("   ✅ SUCCESS: Version = 0x%08X (%u.%u.%u)\n", 
               version, (version >> 16) & 0xFF, (version >> 8) & 0xFF, version & 0xFF);
    } else {
        printf("   ❌ FAILED: %s\n", strerror(errno));
    }
    printf("\n");
    
    /* Test 2: GET_STATUS */
    printf("2. Testing GET_STATUS (0x%08lX):\n", (unsigned long)MILDEV_IOC_GET_STATUS);
    struct mildev_system_status status;
    memset(&status, 0, sizeof(status));
    result = ioctl(fd, MILDEV_IOC_GET_STATUS, &status);
    if (result >= 0) {
        printf("   ✅ SUCCESS: System Status Retrieved\n");
        printf("     Module loaded: %s\n", status.kernel_module_loaded ? "YES" : "NO");
        printf("     Thermal safe: %s\n", status.thermal_safe ? "YES" : "NO");
        printf("     Current temperature: %d°C\n", status.current_temp_celsius);
        printf("     Safe device count: %u\n", status.safe_device_count);
        printf("     Quarantined count: %u\n", status.quarantined_count);
        printf("     Last scan: %lu ms\n", status.last_scan_timestamp);
    } else {
        printf("   ❌ FAILED: %s\n", strerror(errno));
    }
    printf("\n");
    
    /* Test 3: GET_THERMAL */
    printf("3. Testing GET_THERMAL (0x%08lX):\n", (unsigned long)MILDEV_IOC_GET_THERMAL);
    int thermal_temp = 0;
    result = ioctl(fd, MILDEV_IOC_GET_THERMAL, &thermal_temp);
    if (result >= 0) {
        printf("   ✅ SUCCESS: Thermal = %d°C\n", thermal_temp);
    } else {
        printf("   ❌ FAILED: %s\n", strerror(errno));
    }
    printf("\n");
    
    /* Test 4: SCAN_DEVICES */
    printf("4. Testing SCAN_DEVICES (0x%08lX):\n", (unsigned long)MILDEV_IOC_SCAN_DEVICES);
    printf("   Structure size: %zu bytes\n", sizeof(struct mildev_discovery_result));
    struct mildev_discovery_result *discovery = malloc(sizeof(struct mildev_discovery_result));
    if (!discovery) {
        printf("   ❌ FAILED: Cannot allocate memory\n");
    } else {
        memset(discovery, 0, sizeof(struct mildev_discovery_result));
        result = ioctl(fd, MILDEV_IOC_SCAN_DEVICES, discovery);
        if (result >= 0) {
            printf("   ✅ SUCCESS: Device scan completed\n");
            printf("     Total devices found: %u\n", discovery->total_devices_found);
            printf("     Safe devices: %u\n", discovery->safe_devices_found);
            printf("     Quarantined devices: %u\n", discovery->quarantined_devices_found);
            printf("     Scan timestamp: %lu ms\n", discovery->last_scan_timestamp);
            
            /* Show first few devices */
            int show_count = discovery->total_devices_found < 5 ? discovery->total_devices_found : 5;
            printf("     First %d devices:\n", show_count);
            for (int i = 0; i < show_count; i++) {
                printf("       Device 0x%04X: quarantined=%s, temp=%d°C\n",
                       discovery->devices[i].device_id,
                       discovery->devices[i].is_quarantined ? "YES" : "NO",
                       discovery->devices[i].thermal_celsius);
            }
        } else {
            printf("   ❌ FAILED: %s\n", strerror(errno));
        }
        free(discovery);
    }
    printf("\n");
    
    /* Test 5: READ_DEVICE */
    printf("5. Testing READ_DEVICE (0x%08lX):\n", (unsigned long)MILDEV_IOC_READ_DEVICE);
    struct mildev_device_info dev_info;
    memset(&dev_info, 0, sizeof(dev_info));
    
    /* Test with first device in range */
    dev_info.device_id = MILDEV_BASE_ADDR;  /* 0x8000 */
    printf("   Testing device ID: 0x%04X\n", dev_info.device_id);
    
    result = ioctl(fd, MILDEV_IOC_READ_DEVICE, &dev_info);
    if (result >= 0) {
        printf("   ✅ SUCCESS: Device info retrieved\n");
        printf("     Device ID: 0x%04X\n", dev_info.device_id);
        printf("     State: 0x%08X\n", dev_info.state);
        printf("     Access: 0x%08X\n", dev_info.access);
        printf("     Quarantined: %s\n", dev_info.is_quarantined ? "YES" : "NO");
        printf("     Last response: 0x%08X\n", dev_info.last_response);
        printf("     Thermal: %d°C\n", dev_info.thermal_celsius);
        printf("     Timestamp: %lu\n", dev_info.timestamp);
    } else {
        printf("   ❌ FAILED: %s\n", strerror(errno));
    }
    printf("\n");
    
    /* Test with a few more device IDs */
    printf("6. Testing additional device IDs:\n");
    __u16 test_device_ids[] = {0x8001, 0x8002, 0x8005, 0x800A, 0x806B};
    for (int i = 0; i < 5; i++) {
        memset(&dev_info, 0, sizeof(dev_info));
        dev_info.device_id = test_device_ids[i];
        
        result = ioctl(fd, MILDEV_IOC_READ_DEVICE, &dev_info);
        printf("   Device 0x%04X: ", test_device_ids[i]);
        if (result >= 0) {
            printf("✅ State=0x%X, Quarantined=%s, Temp=%d°C\n",
                   dev_info.state,
                   dev_info.is_quarantined ? "YES" : "NO",
                   dev_info.thermal_celsius);
        } else {
            printf("❌ %s\n", strerror(errno));
        }
    }
}

/* Discover undocumented IOCTL commands */
void discover_additional_commands(int fd) {
    printf("\n=== DISCOVERING ADDITIONAL IOCTL COMMANDS ===\n\n");
    
    int found_count = 0;
    
    /* Test adjacent command numbers around known commands */
    printf("Testing command numbers 0-20:\n");
    for (int nr = 0; nr <= 20; nr++) {
        if (nr >= 1 && nr <= 5) continue;  /* Skip known commands */
        
        uint32_t cmd = _IOR(MILDEV_IOC_MAGIC, nr, uint32_t);
        char buffer[1024];
        memset(buffer, 0, sizeof(buffer));
        
        int result = ioctl(fd, cmd, buffer);
        if (result >= 0) {
            printf("   ✅ Command %d (0x%08X): SUCCESS\n", nr, cmd);
            found_count++;
        } else if (errno != ENOTTY) {
            printf("   ⚠️  Command %d (0x%08X): HANDLER EXISTS (%s)\n", nr, cmd, strerror(errno));
            found_count++;
        }
    }
    
    /* Test different data transfer directions */
    printf("\nTesting different transfer directions:\n");
    for (int nr = 1; nr <= 10; nr++) {
        uint32_t cmd_none = _IO(MILDEV_IOC_MAGIC, nr);
        uint32_t cmd_write = _IOW(MILDEV_IOC_MAGIC, nr, uint32_t);
        
        /* Test _IO (no data) */
        int result = ioctl(fd, cmd_none, NULL);
        if (result >= 0 || errno != ENOTTY) {
            printf("   Command %d _IO: %s\n", nr, 
                   result >= 0 ? "SUCCESS" : strerror(errno));
            if (errno != ENOTTY) found_count++;
        }
        
        /* Test _IOW (write data) */
        uint32_t data = 0x12345678;
        result = ioctl(fd, cmd_write, &data);
        if (result >= 0 || errno != ENOTTY) {
            printf("   Command %d _IOW: %s\n", nr,
                   result >= 0 ? "SUCCESS" : strerror(errno));
            if (errno != ENOTTY) found_count++;
        }
    }
    
    printf("\nAdditional commands found: %d\n", found_count);
}

int main(int argc, char *argv[]) {
    const char *device_path = (argc > 1) ? argv[1] : "/dev/dsmil-72dev";
    int fd;
    
    printf("DSMIL Complete IOCTL Handler Discovery\n");
    printf("=====================================\n");
    printf("Device: %s\n", device_path);
    printf("Testing with correct data structures from kernel analysis\n\n");
    
    /* Open device */
    fd = open(device_path, O_RDWR);
    if (fd < 0) {
        printf("❌ Failed to open device %s: %s\n", device_path, strerror(errno));
        return 1;
    }
    
    printf("✅ Device opened successfully\n\n");
    
    /* Test all known IOCTLs */
    test_all_ioctls(fd);
    
    /* Discover additional commands */
    discover_additional_commands(fd);
    
    close(fd);
    return 0;
}
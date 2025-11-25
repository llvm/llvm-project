/*
 * Simple test for military device access
 * Tests the /dev/dsmil-72dev interface
 */

#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <errno.h>
#include <string.h>

// IOCTL commands from military_device_interface.h
#define MILDEV_IOC_MAGIC    0x4D
#define MILDEV_IOC_GET_VERSION     _IOR(MILDEV_IOC_MAGIC, 1, unsigned int)
#define MILDEV_IOC_READ_DEVICE      _IOWR(MILDEV_IOC_MAGIC, 4, struct mildev_device_info)

struct mildev_device_info {
    unsigned int device_id;
    unsigned int status;
    unsigned int response;
    unsigned int flags;
};

int main() {
    printf("=== Military Device Access Test ===\n");
    printf("Testing /dev/dsmil-72dev interface\n\n");
    
    // Open device
    int fd = open("/dev/dsmil-72dev", O_RDWR);
    if (fd < 0) {
        perror("Failed to open device");
        return 1;
    }
    printf("✓ Device opened successfully\n");
    
    // Test 1: Get version
    unsigned int version = 0;
    if (ioctl(fd, MILDEV_IOC_GET_VERSION, &version) == 0) {
        printf("✓ Module version: %u.%u.%u\n", 
               (version >> 16) & 0xFF,
               (version >> 8) & 0xFF,
               version & 0xFF);
    } else {
        printf("✗ Failed to get version: %s\n", strerror(errno));
    }
    
    // Test 2: Read some devices
    printf("\nTesting device reads:\n");
    
    // Test non-quarantined devices
    unsigned int test_devices[] = {0x8000, 0x8001, 0x8005, 0x8007};
    for (int i = 0; i < 4; i++) {
        struct mildev_device_info info = {0};
        info.device_id = test_devices[i];
        
        if (ioctl(fd, MILDEV_IOC_READ_DEVICE, &info) == 0) {
            printf("  Device 0x%04X: status=0x%08X response=0x%08X\n",
                   info.device_id, info.status, info.response);
        } else {
            printf("  Device 0x%04X: %s\n", 
                   info.device_id, strerror(errno));
        }
    }
    
    // Test quarantined device (should fail)
    printf("\nTesting quarantined device (should be blocked):\n");
    struct mildev_device_info quarantined = {0};
    quarantined.device_id = 0x8009;  // Known quarantined device
    
    if (ioctl(fd, MILDEV_IOC_READ_DEVICE, &quarantined) < 0) {
        if (errno == EACCES) {
            printf("✓ Device 0x%04X correctly blocked (quarantined)\n", 
                   quarantined.device_id);
        } else {
            printf("  Device 0x%04X: %s\n", 
                   quarantined.device_id, strerror(errno));
        }
    } else {
        printf("✗ WARNING: Quarantined device was accessible!\n");
    }
    
    close(fd);
    printf("\n✓ Test complete\n");
    return 0;
}
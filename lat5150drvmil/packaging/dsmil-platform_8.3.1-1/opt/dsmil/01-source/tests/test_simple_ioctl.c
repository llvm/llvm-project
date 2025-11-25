/*
 * Simple IOCTL Test - Quick verification that device is accessible
 * Minimal test to verify basic functionality before running full discovery
 */

#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <errno.h>
#include <string.h>

/* Known working IOCTL command */
#define MILDEV_IOC_MAGIC        'M'
#define MILDEV_IOC_GET_VERSION  _IOR(MILDEV_IOC_MAGIC, 1, unsigned int)

int main(int argc, char *argv[]) {
    const char *device_path = (argc > 1) ? argv[1] : "/dev/dsmil0";
    int fd;
    unsigned int version = 0;
    int result;
    
    printf("Simple DSMIL IOCTL Test\n");
    printf("======================\n");
    printf("Device: %s\n", device_path);
    printf("Testing command: 0x%08lX (GET_VERSION)\n", (unsigned long)MILDEV_IOC_GET_VERSION);
    printf("\n");
    
    /* Open device */
    fd = open(device_path, O_RDWR);
    if (fd < 0) {
        printf("❌ Failed to open device: %s\n", strerror(errno));
        
        /* Try alternatives */
        const char *alternatives[] = {"/dev/dsmil1", "/dev/dsmil2", "/dev/mildev"};
        for (int i = 0; i < 3; i++) {
            fd = open(alternatives[i], O_RDWR);
            if (fd >= 0) {
                printf("✅ Opened alternative: %s\n", alternatives[i]);
                device_path = alternatives[i];
                break;
            }
        }
        
        if (fd < 0) {
            printf("No accessible device found. Ensure DSMIL module is loaded:\n");
            printf("  sudo insmod 01-source/kernel/dsmil-72dev.ko\n");
            return 1;
        }
    } else {
        printf("✅ Device opened successfully (fd=%d)\n", fd);
    }
    
    /* Test the IOCTL call */
    printf("Calling IOCTL...\n");
    result = ioctl(fd, MILDEV_IOC_GET_VERSION, &version);
    
    if (result >= 0) {
        printf("✅ IOCTL successful!\n");
        printf("   Return value: %d\n", result);
        printf("   Version returned: 0x%08X (%u)\n", version, version);
    } else {
        printf("❌ IOCTL failed: %s (errno=%d)\n", strerror(errno), errno);
        
        /* Analyze the error */
        switch (errno) {
        case ENOTTY:
            printf("   → Handler does not exist for this command\n");
            break;
        case EINVAL:
            printf("   → Invalid arguments (handler exists)\n");
            break;
        case EFAULT:
            printf("   → Bad user space pointer (handler exists)\n");
            break;
        case EACCES:
        case EPERM:
            printf("   → Permission denied (handler exists)\n");
            break;
        default:
            printf("   → Other error - handler status unclear\n");
            break;
        }
    }
    
    close(fd);
    printf("\nDevice access test complete.\n");
    
    if (result >= 0) {
        printf("✅ Ready for full IOCTL discovery!\n");
        printf("Run: sudo ./run_ioctl_discovery.sh\n");
        return 0;
    } else {
        printf("⚠️  Basic test failed. Check module and permissions.\n");
        return 1;
    }
}
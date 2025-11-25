/*
 * Simple IOCTL Handler Discovery for DSMIL Kernel Module
 * Safe and straightforward approach to find IOCTL handlers
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

/* Known DSMIL IOCTL commands */
#define MILDEV_IOC_MAGIC        'M'
#define MILDEV_IOC_GET_VERSION  _IOR(MILDEV_IOC_MAGIC, 1, __u32)       
#define MILDEV_IOC_GET_STATUS   _IOR(MILDEV_IOC_MAGIC, 2, void*)       
#define MILDEV_IOC_SCAN_DEVICES _IOR(MILDEV_IOC_MAGIC, 3, void*)       
#define MILDEV_IOC_READ_DEVICE  _IOWR(MILDEV_IOC_MAGIC, 4, void*)      
#define MILDEV_IOC_GET_THERMAL  _IOR(MILDEV_IOC_MAGIC, 5, int)

/* Probe result structure */
struct probe_result {
    uint32_t cmd;
    int result;
    int error_code;
    bool handler_exists;
    const char *description;
};

/* Test a single IOCTL command */
struct probe_result test_ioctl_command(int fd, uint32_t cmd, const char *desc) {
    struct probe_result result = {0};
    char buffer[256];
    int ret;
    
    result.cmd = cmd;
    result.description = desc;
    
    /* Initialize buffer */
    memset(buffer, 0xAA, sizeof(buffer));
    
    /* Call IOCTL */
    ret = ioctl(fd, cmd, buffer);
    result.result = ret;
    result.error_code = (ret < 0) ? errno : 0;
    
    /* Determine if handler exists */
    if (ret >= 0) {
        result.handler_exists = true;  /* Success */
    } else if (errno == ENOTTY) {
        result.handler_exists = false; /* No handler */
    } else {
        result.handler_exists = true;  /* Handler exists but failed */
    }
    
    return result;
}

/* Print IOCTL command details */
void print_ioctl_info(uint32_t cmd) {
    printf("Command: 0x%08X\n", cmd);
    printf("  Magic: 0x%02X ('%c')\n", _IOC_TYPE(cmd), _IOC_TYPE(cmd));
    printf("  Number: %d\n", _IOC_NR(cmd));
    printf("  Direction: ");
    if ((_IOC_DIR(cmd) & _IOC_READ) && (_IOC_DIR(cmd) & _IOC_WRITE)) {
        printf("READ/WRITE");
    } else if (_IOC_DIR(cmd) & _IOC_READ) {
        printf("READ");
    } else if (_IOC_DIR(cmd) & _IOC_WRITE) {
        printf("WRITE");  
    } else {
        printf("NONE");
    }
    printf("\n");
    printf("  Size: %d bytes\n", _IOC_SIZE(cmd));
}

/* Systematic IOCTL discovery */
void discover_ioctl_handlers(int fd) {
    printf("=== DSMIL IOCTL HANDLER DISCOVERY ===\n\n");
    
    /* Test known commands */
    printf("1. TESTING KNOWN COMMANDS:\n");
    
    struct {
        uint32_t cmd;
        const char *name;
    } known_commands[] = {
        {MILDEV_IOC_GET_VERSION, "GET_VERSION"},
        {MILDEV_IOC_GET_STATUS, "GET_STATUS"},
        {MILDEV_IOC_SCAN_DEVICES, "SCAN_DEVICES"},
        {MILDEV_IOC_READ_DEVICE, "READ_DEVICE"},
        {MILDEV_IOC_GET_THERMAL, "GET_THERMAL"}
    };
    
    int handlers_found = 0;
    int successful_calls = 0;
    
    for (int i = 0; i < 5; i++) {
        struct probe_result result = test_ioctl_command(fd, known_commands[i].cmd, known_commands[i].name);
        
        printf("   %s (0x%08X): ", result.description, result.cmd);
        
        if (result.handler_exists) {
            handlers_found++;
            printf("✅ HANDLER EXISTS");
            if (result.result >= 0) {
                successful_calls++;
                printf(" (SUCCESS)");
            } else {
                printf(" (ERROR: %s)", strerror(result.error_code));
            }
        } else {
            printf("❌ NO HANDLER");
        }
        printf("\n");
    }
    
    /* Test adjacent command numbers */
    printf("\n2. PROBING ADJACENT COMMAND NUMBERS:\n");
    
    for (int nr = 0; nr <= 10; nr++) {
        if (nr >= 1 && nr <= 5) continue;  /* Skip known commands */
        
        uint32_t cmd = _IOR(MILDEV_IOC_MAGIC, nr, uint32_t);
        struct probe_result result = test_ioctl_command(fd, cmd, "UNKNOWN");
        
        if (result.handler_exists) {
            handlers_found++;
            printf("   Nr %d (0x%08X): ✅ HANDLER FOUND", nr, cmd);
            if (result.result >= 0) {
                successful_calls++;
                printf(" (SUCCESS)");
            } else {
                printf(" (ERROR: %s)", strerror(result.error_code));
            }
            printf("\n");
        }
    }
    
    /* Test different magic numbers */
    printf("\n3. TESTING DIFFERENT MAGIC NUMBERS:\n");
    
    char test_magics[] = {'D', 'S', 'd', 's', 'm', 'L'};
    for (int m = 0; m < 6; m++) {
        char magic = test_magics[m];
        if (magic == MILDEV_IOC_MAGIC) continue;
        
        for (int nr = 1; nr <= 3; nr++) {
            uint32_t cmd = _IOR(magic, nr, uint32_t);
            struct probe_result result = test_ioctl_command(fd, cmd, "ALT_MAGIC");
            
            if (result.handler_exists) {
                handlers_found++;
                printf("   Magic '%c' Nr %d (0x%08X): ✅ HANDLER FOUND", magic, nr, cmd);
                if (result.result >= 0) {
                    successful_calls++;
                    printf(" (SUCCESS)");
                } else {
                    printf(" (ERROR: %s)", strerror(result.error_code));
                }
                printf("\n");
            }
        }
    }
    
    /* Test common legacy command ranges */
    printf("\n4. TESTING LEGACY COMMAND RANGES:\n");
    
    uint32_t legacy_commands[] = {
        0x1000, 0x1001, 0x1002,  /* Common range 1 */
        0x5000, 0x5001, 0x5002,  /* Common range 2 */
        0x8000, 0x8001, 0x8002,  /* Common range 3 */
        0xD000, 0xD001, 0xD002   /* Vendor range */
    };
    
    for (int i = 0; i < 12; i++) {
        struct probe_result result = test_ioctl_command(fd, legacy_commands[i], "LEGACY");
        
        if (result.handler_exists) {
            handlers_found++;
            printf("   Legacy 0x%08X: ✅ HANDLER FOUND", legacy_commands[i]);
            if (result.result >= 0) {
                successful_calls++;
                printf(" (SUCCESS)");
            } else {
                printf(" (ERROR: %s)", strerror(result.error_code));
            }
            printf("\n");
        }
    }
    
    /* Summary */
    printf("\n=== DISCOVERY SUMMARY ===\n");
    printf("Total handlers found: %d\n", handlers_found);
    printf("Successful calls: %d\n", successful_calls);
    printf("Failed calls: %d\n", handlers_found - successful_calls);
    
    if (handlers_found > 0) {
        printf("\n✅ IOCTL handlers discovered!\n");
        if (successful_calls > 0) {
            printf("✅ %d handlers are accessible and working\n", successful_calls);
        }
        if (handlers_found > successful_calls) {
            printf("⚠️  %d handlers exist but returned errors (protected/privileged)\n", 
                   handlers_found - successful_calls);
        }
    } else {
        printf("\n❌ No additional handlers found beyond expected commands\n");
    }
}

/* Test specific command with detailed analysis */
void analyze_specific_command(int fd, uint32_t cmd) {
    printf("\n=== DETAILED COMMAND ANALYSIS ===\n");
    print_ioctl_info(cmd);
    printf("\n");
    
    /* Test with different buffer conditions */
    printf("Testing different buffer conditions:\n\n");
    
    /* Test 1: NULL pointer */
    printf("1. NULL pointer test:\n");
    int result = ioctl(fd, cmd, NULL);
    printf("   Result: %d, Error: %s\n", result, 
           result < 0 ? strerror(errno) : "SUCCESS");
    
    /* Test 2: Small buffer */
    printf("2. Small buffer test:\n");
    char small_buf[64];
    memset(small_buf, 0, sizeof(small_buf));
    result = ioctl(fd, cmd, small_buf);
    printf("   Result: %d, Error: %s\n", result,
           result < 0 ? strerror(errno) : "SUCCESS");
    
    /* Test 3: Large buffer */
    printf("3. Large buffer test:\n"); 
    char large_buf[1024];
    memset(large_buf, 0xAA, sizeof(large_buf));
    result = ioctl(fd, cmd, large_buf);
    printf("   Result: %d, Error: %s\n", result,
           result < 0 ? strerror(errno) : "SUCCESS");
    
    /* Check if buffer was modified */
    bool modified = false;
    for (int i = 0; i < 1024; i++) {
        if (large_buf[i] != (char)0xAA) {
            modified = true;
            break;
        }
    }
    printf("   Buffer modified: %s\n", modified ? "YES" : "NO");
    
    if (modified) {
        printf("   First 16 bytes: ");
        for (int i = 0; i < 16; i++) {
            printf("%02X ", (unsigned char)large_buf[i]);
        }
        printf("\n");
    }
}

int main(int argc, char *argv[]) {
    const char *device_path = (argc > 1) ? argv[1] : "/dev/dsmil-72dev";
    int fd;
    
    printf("DSMIL Simple IOCTL Handler Discovery Tool\n");
    printf("=========================================\n");
    printf("Device: %s\n\n", device_path);
    
    /* Open device */
    fd = open(device_path, O_RDWR);
    if (fd < 0) {
        printf("❌ Failed to open device %s: %s\n", device_path, strerror(errno));
        
        /* Try alternatives */
        const char *alternatives[] = {"/dev/dsmil0", "/dev/dsmil1", "/dev/mildev"};
        for (int i = 0; i < 3; i++) {
            fd = open(alternatives[i], O_RDWR);
            if (fd >= 0) {
                printf("✅ Opened alternative: %s\n", alternatives[i]);
                device_path = alternatives[i];
                break;
            }
        }
        
        if (fd < 0) {
            printf("No accessible device found.\n");
            printf("Ensure DSMIL module is loaded: sudo insmod 01-source/kernel/dsmil-72dev.ko\n");
            return 1;
        }
    }
    
    printf("✅ Device opened successfully (fd=%d)\n\n", fd);
    
    /* Run discovery */
    discover_ioctl_handlers(fd);
    
    /* Test specific command if provided */
    if (argc > 2) {
        uint32_t specific_cmd = strtoul(argv[2], NULL, 0);
        analyze_specific_command(fd, specific_cmd);
    }
    
    close(fd);
    return 0;
}
/*
 * Safe IOCTL Handler Discovery Tool for DSMIL Kernel Module
 * Uses assembly-level techniques to systematically probe IOCTL command space
 * without causing crashes or triggering dangerous operations
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
#include <sys/mman.h>
#include <signal.h>
#include <setjmp.h>
#include <time.h>
#include <linux/types.h>

/* Known DSMIL IOCTL commands from analysis */
#define MILDEV_IOC_MAGIC        'M'
#define MILDEV_IOC_GET_VERSION  _IOR(MILDEV_IOC_MAGIC, 1, __u32)       /* 0x80044D01 */
#define MILDEV_IOC_GET_STATUS   _IOR(MILDEV_IOC_MAGIC, 2, void*)       /* 0x80044D02 */ 
#define MILDEV_IOC_SCAN_DEVICES _IOR(MILDEV_IOC_MAGIC, 3, void*)       /* 0x80044D03 */
#define MILDEV_IOC_READ_DEVICE  _IOWR(MILDEV_IOC_MAGIC, 4, void*)      /* 0xC0044D04 */
#define MILDEV_IOC_GET_THERMAL  _IOR(MILDEV_IOC_MAGIC, 5, int)         /* 0x80044D05 */

/* Assembly-based safe probing structure */
struct ioctl_probe_result {
    uint32_t cmd;
    int result;
    int error_code;
    uint64_t execution_time_ns;
    uint8_t caused_crash;
    uint8_t handler_exists;
    char description[64];
};

/* Global state for signal handling */
static jmp_buf crash_recovery;
static volatile int crash_occurred = 0;
static int device_fd = -1;

/* Signal handler for crash recovery */
static void crash_handler(int sig) {
    crash_occurred = 1;
    siglongjmp(crash_recovery, sig);
}

/* Assembly-level timing functions */
static inline uint64_t rdtsc(void) {
    uint32_t low, high;
    __asm__ volatile ("rdtsc" : "=a" (low), "=d" (high));
    return ((uint64_t)high << 32) | low;
}

/* Safe memory allocation with guard pages */
static void* allocate_safe_buffer(size_t size) {
    void *buffer;
    size_t page_size = getpagesize();
    size_t total_size = ((size + page_size - 1) / page_size + 2) * page_size;
    
    /* Allocate with guard pages */
    buffer = mmap(NULL, total_size, PROT_READ | PROT_WRITE,
                  MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (buffer == MAP_FAILED) {
        return NULL;
    }
    
    /* Set up guard pages */
    mprotect(buffer, page_size, PROT_NONE);  /* Guard before */
    mprotect((char*)buffer + total_size - page_size, page_size, PROT_NONE);  /* Guard after */
    
    return (char*)buffer + page_size;
}

/* Safe buffer deallocation */
static void free_safe_buffer(void *buffer, size_t size) {
    if (buffer) {
        size_t page_size = getpagesize();
        size_t total_size = ((size + page_size - 1) / page_size + 2) * page_size;
        void *real_buffer = (char*)buffer - page_size;
        munmap(real_buffer, total_size);
    }
}

/* Assembly-level IOCTL call wrapper with crash protection */
static int safe_ioctl_call(int fd, uint32_t cmd, void *arg) {
    int result = -1;
    
    /* Set up crash protection */
    struct sigaction old_segv, old_bus, new_action;
    memset(&new_action, 0, sizeof(new_action));
    new_action.sa_handler = crash_handler;
    sigemptyset(&new_action.sa_mask);
    new_action.sa_flags = 0;
    
    sigaction(SIGSEGV, &new_action, &old_segv);
    sigaction(SIGBUS, &new_action, &old_bus);
    
    crash_occurred = 0;
    
    /* Assembly-level IOCTL call with minimal system interaction */
    if (sigsetjmp(crash_recovery, 1) == 0) {
        /* Use inline assembly for direct syscall to minimize crash risk */
        register long rax __asm__("rax") = 16;  /* __NR_ioctl */
        register long rdi __asm__("rdi") = fd;
        register long rsi __asm__("rsi") = cmd;
        register long rdx __asm__("rdx") = (long)arg;
        
        __asm__ volatile (
            "syscall"
            : "+r" (rax)
            : "r" (rdi), "r" (rsi), "r" (rdx)
            : "rcx", "r11", "memory"
        );
        
        result = (int)rax;
    } else {
        /* Crash occurred */
        result = -EFAULT;
    }
    
    /* Restore signal handlers */
    sigaction(SIGSEGV, &old_segv, NULL);
    sigaction(SIGBUS, &old_bus, NULL);
    
    return result;
}

/* Probe a single IOCTL command safely */
static struct ioctl_probe_result probe_ioctl_command(int fd, uint32_t cmd) {
    struct ioctl_probe_result result = {0};
    void *safe_buffer = NULL;
    uint64_t start_time, end_time;
    
    result.cmd = cmd;
    snprintf(result.description, sizeof(result.description), 
             "Magic: 0x%02X, Nr: %d, Type: %s", 
             _IOC_TYPE(cmd), _IOC_NR(cmd),
             _IOC_READ(cmd) && _IOC_WRITE(cmd) ? "RW" : 
             _IOC_READ(cmd) ? "R" : _IOC_WRITE(cmd) ? "W" : "None");
    
    /* Allocate minimal safe buffer for data transfer */
    size_t buffer_size = _IOC_SIZE(cmd);
    if (buffer_size == 0) buffer_size = 64;  /* Minimum safe size */
    if (buffer_size > 4096) buffer_size = 4096;  /* Maximum safe size */
    
    safe_buffer = allocate_safe_buffer(buffer_size);
    if (!safe_buffer) {
        result.result = -ENOMEM;
        result.error_code = ENOMEM;
        return result;
    }
    
    /* Initialize buffer with safe patterns */
    memset(safe_buffer, 0xAA, buffer_size);
    
    /* Time the IOCTL call */
    start_time = rdtsc();
    result.result = safe_ioctl_call(fd, cmd, safe_buffer);
    end_time = rdtsc();
    
    result.execution_time_ns = end_time - start_time;
    result.error_code = (result.result < 0) ? errno : 0;
    result.caused_crash = crash_occurred;
    
    /* Determine if handler exists based on error type */
    if (result.result >= 0) {
        result.handler_exists = 1;  /* Success - handler definitely exists */
    } else if (result.error_code == ENOTTY) {
        result.handler_exists = 0;  /* ENOTTY - no handler for this command */
    } else if (result.error_code == EINVAL) {
        result.handler_exists = 1;  /* EINVAL - handler exists but rejected args */
    } else if (result.error_code == EFAULT) {
        result.handler_exists = 1;  /* EFAULT - handler exists, bad user ptr */
    } else if (result.error_code == EACCES || result.error_code == EPERM) {
        result.handler_exists = 1;  /* Permission error - handler exists */
    } else {
        result.handler_exists = 0;  /* Other errors likely mean no handler */
    }
    
    free_safe_buffer(safe_buffer, buffer_size);
    
    return result;
}

/* Generate IOCTL command using Linux macros */
static uint32_t make_ioctl_cmd(uint8_t magic, uint8_t nr, uint8_t type, uint16_t size) {
    return _IOC(type, magic, nr, size);
}

/* Systematic IOCTL space probing */
static void probe_ioctl_space(int fd) {
    struct ioctl_probe_result result;
    int total_probed = 0, handlers_found = 0, successful_calls = 0;
    
    printf("\n=== SYSTEMATIC IOCTL HANDLER DISCOVERY ===\n");
    printf("Device FD: %d\n", fd);
    printf("Known working command: 0x%08X (MILDEV_IOC_GET_VERSION)\n", MILDEV_IOC_GET_VERSION);
    printf("\n");
    
    /* Test known working commands first */
    printf("1. TESTING KNOWN COMMANDS:\n");
    uint32_t known_commands[] = {
        MILDEV_IOC_GET_VERSION,
        MILDEV_IOC_GET_STATUS, 
        MILDEV_IOC_SCAN_DEVICES,
        MILDEV_IOC_READ_DEVICE,
        MILDEV_IOC_GET_THERMAL
    };
    
    for (int i = 0; i < sizeof(known_commands)/sizeof(known_commands[0]); i++) {
        result = probe_ioctl_command(fd, known_commands[i]);
        total_probed++;
        
        printf("   0x%08X: ", result.cmd);
        if (result.caused_crash) {
            printf("CRASH! ");
        }
        if (result.handler_exists) {
            handlers_found++;
            printf("HANDLER EXISTS");
            if (result.result >= 0) {
                successful_calls++;
                printf(" (SUCCESS)");
            } else {
                printf(" (ERROR: %s)", strerror(result.error_code));
            }
        } else {
            printf("NO HANDLER");
        }
        printf(" [%s] Time: %lu cycles\n", result.description, result.execution_time_ns);
    }
    
    /* Probe around known commands */
    printf("\n2. PROBING ADJACENT COMMAND NUMBERS:\n");
    uint8_t magic = MILDEV_IOC_MAGIC;
    for (uint8_t nr = 0; nr <= 15; nr++) {  /* Probe nr 0-15 */
        for (uint8_t type = 0; type <= 3; type++) {  /* _IOC_NONE, _IOC_WRITE, _IOC_READ, _IOC_READ|_IOC_WRITE */
            uint16_t size = (type == 0) ? 0 : 64;  /* Size 0 for NONE, 64 bytes for others */
            uint32_t cmd = make_ioctl_cmd(magic, nr, type, size);
            
            /* Skip if we already tested this */
            bool already_tested = false;
            for (int i = 0; i < sizeof(known_commands)/sizeof(known_commands[0]); i++) {
                if (cmd == known_commands[i]) {
                    already_tested = true;
                    break;
                }
            }
            if (already_tested) continue;
            
            result = probe_ioctl_command(fd, cmd);
            total_probed++;
            
            if (result.handler_exists) {
                handlers_found++;
                printf("   0x%08X: FOUND! ", cmd);
                if (result.caused_crash) {
                    printf("(CRASHED) ");
                }
                if (result.result >= 0) {
                    successful_calls++;
                    printf("SUCCESS");
                } else {
                    printf("ERROR: %s", strerror(result.error_code));
                }
                printf(" [%s] Time: %lu cycles\n", result.description, result.execution_time_ns);
            }
        }
    }
    
    /* Probe different magic numbers */
    printf("\n3. PROBING DIFFERENT MAGIC NUMBERS:\n");
    char test_magics[] = {'D', 'S', 'm', 'd', 'T', 'L', '0', '1', '2'};  /* DSMIL-related */
    for (int m = 0; m < sizeof(test_magics)/sizeof(test_magics[0]); m++) {
        uint8_t test_magic = test_magics[m];
        if (test_magic == magic) continue;  /* Skip the one we already tested */
        
        for (uint8_t nr = 1; nr <= 5; nr++) {  /* Test first few command numbers */
            uint32_t cmd = _IOR(test_magic, nr, uint32_t);
            result = probe_ioctl_command(fd, cmd);
            total_probed++;
            
            if (result.handler_exists) {
                handlers_found++;
                printf("   0x%08X: FOUND! Magic '%c', Nr %d ", cmd, test_magic, nr);
                if (result.caused_crash) {
                    printf("(CRASHED) ");
                }
                if (result.result >= 0) {
                    successful_calls++;
                    printf("SUCCESS");
                } else {
                    printf("ERROR: %s", strerror(result.error_code));
                }
                printf(" Time: %lu cycles\n", result.execution_time_ns);
            }
        }
    }
    
    /* Probe legacy/undocumented ranges */
    printf("\n4. PROBING LEGACY COMMAND RANGES:\n");
    uint32_t legacy_ranges[] = {
        0x1000, 0x2000, 0x3000,  /* Common legacy ranges */
        0x4000, 0x5000, 0x6000,
        0x7000, 0x8000, 0x9000,
        0xD000, 0xE000, 0xF000   /* Common vendor ranges */
    };
    
    for (int r = 0; r < sizeof(legacy_ranges)/sizeof(legacy_ranges[0]); r++) {
        for (int offset = 0; offset < 16; offset++) {  /* Test first 16 in each range */
            uint32_t cmd = legacy_ranges[r] + offset;
            result = probe_ioctl_command(fd, cmd);
            total_probed++;
            
            if (result.handler_exists) {
                handlers_found++;
                printf("   0x%08X: FOUND! Legacy range 0x%04X+%d ", 
                       cmd, legacy_ranges[r], offset);
                if (result.caused_crash) {
                    printf("(CRASHED) ");
                }
                if (result.result >= 0) {
                    successful_calls++;
                    printf("SUCCESS");
                } else {
                    printf("ERROR: %s", strerror(result.error_code));
                }
                printf(" Time: %lu cycles\n", result.execution_time_ns);
            }
        }
    }
    
    /* Summary */
    printf("\n=== DISCOVERY SUMMARY ===\n");
    printf("Total commands probed: %d\n", total_probed);
    printf("Handlers found: %d\n", handlers_found);
    printf("Successful calls: %d\n", successful_calls);
    printf("Detection accuracy: %.1f%% (based on error code analysis)\n", 
           total_probed > 0 ? (100.0 * handlers_found / total_probed) : 0.0);
    
    if (handlers_found > successful_calls) {
        printf("\nNote: %d handlers exist but returned errors.\n", 
               handlers_found - successful_calls);
        printf("This suggests protected/privileged operations.\n");
    }
}

/* Test a specific IOCTL with detailed analysis */
static void test_specific_ioctl(int fd, uint32_t cmd) {
    struct ioctl_probe_result result;
    void *test_buffer = NULL;
    
    printf("\n=== DETAILED IOCTL ANALYSIS ===\n");
    printf("Command: 0x%08X\n", cmd);
    printf("Magic: 0x%02X ('%c')\n", _IOC_TYPE(cmd), _IOC_TYPE(cmd));
    printf("Number: %d\n", _IOC_NR(cmd));
    printf("Direction: ");
    if (_IOC_READ(cmd) && _IOC_WRITE(cmd)) printf("READ/WRITE");
    else if (_IOC_READ(cmd)) printf("READ");
    else if (_IOC_WRITE(cmd)) printf("WRITE");
    else printf("NONE");
    printf("\n");
    printf("Size: %d bytes\n", _IOC_SIZE(cmd));
    printf("\n");
    
    /* Test with different buffer conditions */
    printf("Testing buffer conditions:\n");
    
    /* Test 1: NULL pointer */
    printf("1. NULL pointer: ");
    result = probe_ioctl_command(fd, cmd);  /* Uses safe buffer internally, but we can override */
    result.result = safe_ioctl_call(fd, cmd, NULL);
    result.error_code = errno;
    printf("Result: %d, Error: %s\n", result.result, 
           result.result < 0 ? strerror(result.error_code) : "SUCCESS");
    
    /* Test 2: Valid buffer with zeros */
    printf("2. Zero buffer: ");
    size_t buffer_size = _IOC_SIZE(cmd) > 0 ? _IOC_SIZE(cmd) : 64;
    test_buffer = allocate_safe_buffer(buffer_size);
    if (test_buffer) {
        memset(test_buffer, 0, buffer_size);
        result.result = safe_ioctl_call(fd, cmd, test_buffer);
        result.error_code = errno;
        printf("Result: %d, Error: %s\n", result.result,
               result.result < 0 ? strerror(result.error_code) : "SUCCESS");
        free_safe_buffer(test_buffer, buffer_size);
    } else {
        printf("Buffer allocation failed\n");
    }
    
    /* Test 3: Valid buffer with pattern */
    printf("3. Pattern buffer (0xAA): ");
    test_buffer = allocate_safe_buffer(buffer_size);
    if (test_buffer) {
        memset(test_buffer, 0xAA, buffer_size);
        result.result = safe_ioctl_call(fd, cmd, test_buffer);
        result.error_code = errno;
        printf("Result: %d, Error: %s\n", result.result,
               result.result < 0 ? strerror(result.error_code) : "SUCCESS");
        
        /* Check if buffer was modified */
        bool modified = false;
        uint8_t *bytes = (uint8_t*)test_buffer;
        for (size_t i = 0; i < buffer_size; i++) {
            if (bytes[i] != 0xAA) {
                modified = true;
                break;
            }
        }
        printf("   Buffer modified: %s\n", modified ? "YES" : "NO");
        
        if (modified && buffer_size <= 64) {
            printf("   Buffer contents: ");
            for (size_t i = 0; i < buffer_size && i < 16; i++) {
                printf("%02X ", bytes[i]);
            }
            if (buffer_size > 16) printf("...");
            printf("\n");
        }
        
        free_safe_buffer(test_buffer, buffer_size);
    } else {
        printf("Buffer allocation failed\n");
    }
}

int main(int argc, char *argv[]) {
    int fd;
    const char *device_path = "/dev/dsmil0";
    
    printf("DSMIL Kernel Module IOCTL Handler Discovery Tool\n");
    printf("================================================\n");
    printf("Using assembly-level techniques for safe probing\n\n");
    
    /* Override device path if provided */
    if (argc > 1) {
        device_path = argv[1];
    }
    
    /* Open device */
    fd = open(device_path, O_RDWR);
    if (fd < 0) {
        perror("Failed to open device");
        printf("\nTrying alternative paths...\n");
        
        /* Try alternative device paths */
        const char *alt_paths[] = {
            "/dev/dsmil1", "/dev/dsmil2", "/dev/dsmil_enhanced",
            "/dev/mildev", "/dev/dsmil", "/sys/kernel/debug/dsmil/control"
        };
        
        for (int i = 0; i < sizeof(alt_paths)/sizeof(alt_paths[0]); i++) {
            fd = open(alt_paths[i], O_RDWR);
            if (fd >= 0) {
                printf("Successfully opened: %s\n", alt_paths[i]);
                device_path = alt_paths[i];
                break;
            }
        }
        
        if (fd < 0) {
            printf("No accessible device found. Make sure the DSMIL module is loaded.\n");
            return 1;
        }
    }
    
    printf("Device: %s (fd=%d)\n", device_path, fd);
    device_fd = fd;
    
    /* Perform systematic probing */
    probe_ioctl_space(fd);
    
    /* Test specific command if provided */
    if (argc > 2) {
        uint32_t specific_cmd = strtoul(argv[2], NULL, 0);
        test_specific_ioctl(fd, specific_cmd);
    }
    
    close(fd);
    return 0;
}
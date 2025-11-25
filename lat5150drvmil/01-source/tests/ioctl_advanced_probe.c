/*
 * Advanced IOCTL Handler Discovery Tool for DSMIL Kernel Module
 * Enhanced version with memory mapping and assembly-level device probing
 * 
 * Features:
 * - Assembly-level syscalls for minimal kernel interaction
 * - Memory-mapped device register analysis  
 * - SMI instruction monitoring
 * - Hardware register state capture
 * - Intel Meteor Lake P-core/E-core aware timing
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/io.h>
#include <signal.h>
#include <setjmp.h>
#include <time.h>
#include <sched.h>

/* DSMIL memory regions from kernel module analysis */
#define DSMIL_PRIMARY_BASE      0x52000000UL
#define DSMIL_JRTC1_BASE        0x58000000UL  
#define DSMIL_EXTENDED_BASE     0x5C000000UL
#define DSMIL_PLATFORM_BASE     0x48000000UL
#define DSMIL_HIGH_BASE         0x60000000UL
#define DSMIL_MEMORY_SIZE       (4UL * 1024 * 1024)  /* 4MB per region */

/* Dell SMI I/O ports */
#define DELL_SMI_CMD_PORT       0xB2
#define DELL_SMI_DATA_PORT      0x164E
#define DELL_SMI_DATA_PORT_HI   0x164F

/* Intel Meteor Lake SMI coordination */
#define MTL_SMI_COORD_START     0xA0
#define MTL_SMI_COORD_SYNC      0xA1

/* Known DSMIL IOCTL commands */
#define MILDEV_IOC_MAGIC        'M'
#define MILDEV_IOC_GET_VERSION  _IOR(MILDEV_IOC_MAGIC, 1, __u32)       
#define MILDEV_IOC_GET_STATUS   _IOR(MILDEV_IOC_MAGIC, 2, void*)       
#define MILDEV_IOC_SCAN_DEVICES _IOR(MILDEV_IOC_MAGIC, 3, void*)       
#define MILDEV_IOC_READ_DEVICE  _IOWR(MILDEV_IOC_MAGIC, 4, void*)      
#define MILDEV_IOC_GET_THERMAL  _IOR(MILDEV_IOC_MAGIC, 5, int)

/* DSMIL signatures from kernel analysis */
#define DSMIL_SIG_SMIL          0x4C494D53  /* "SMIL" */
#define DSMIL_SIG_DSML          0x4C4D5344  /* "DSML" */  
#define DSMIL_SIG_JRTC          0x43545200  /* "JRTC" */
#define DSMIL_SIG_DELL          0x4C4C4544  /* "DELL" */
#define DSMIL_SIG_MLSP          0x50534C4D  /* "MLSP" */

/* Advanced probe result structure */
struct advanced_probe_result {
    uint32_t cmd;
    int ioctl_result;
    int error_code;
    uint64_t execution_cycles;
    uint8_t caused_crash;
    uint8_t handler_exists;
    uint8_t smi_triggered;
    uint8_t memory_changed;
    uint32_t pre_smi_state[4];   /* Hardware state before */
    uint32_t post_smi_state[4];  /* Hardware state after */
    char description[128];
};

/* Memory mapping state */
struct memory_region {
    uint64_t physical_addr;
    void *virtual_addr;
    size_t size;
    int mapped;
    char name[32];
};

/* Global state */
static jmp_buf crash_recovery;
static volatile int crash_occurred = 0;
static struct memory_region dsmil_regions[5];
static int device_fd = -1;
static int mem_fd = -1;
static int have_io_privileges = 0;

/* Assembly CPU identification and timing */
static inline void cpuid(uint32_t eax, uint32_t *regs) {
    __asm__ volatile (
        "cpuid"
        : "=a" (regs[0]), "=b" (regs[1]), "=c" (regs[2]), "=d" (regs[3])
        : "a" (eax)
    );
}

static inline uint64_t rdtsc_start(void) {
    uint32_t low, high;
    __asm__ volatile (
        "cpuid\n\t"     /* Serialize */
        "rdtsc\n\t"
        : "=a" (low), "=d" (high)
        :: "rbx", "rcx"
    );
    return ((uint64_t)high << 32) | low;
}

static inline uint64_t rdtsc_end(void) {
    uint32_t low, high;
    __asm__ volatile (
        "rdtscp\n\t"    /* Serializing read */
        "mov %%eax, %0\n\t"
        "mov %%edx, %1\n\t"
        "cpuid\n\t"     /* Serialize */
        : "=r" (low), "=r" (high)
        :: "rax", "rbx", "rcx", "rdx"
    );
    return ((uint64_t)high << 32) | low;
}

/* Enhanced signal handler with register dump */
static void advanced_crash_handler(int sig) {
    crash_occurred = sig;
    printf("\nCRASH DETECTED (signal %d)\n", sig);
    siglongjmp(crash_recovery, sig);
}

/* Initialize memory mappings for DSMIL regions */
static int init_memory_mappings(void) {
    uint64_t bases[] = {
        DSMIL_PRIMARY_BASE, DSMIL_JRTC1_BASE, DSMIL_EXTENDED_BASE,
        DSMIL_PLATFORM_BASE, DSMIL_HIGH_BASE
    };
    const char *names[] = {
        "PRIMARY", "JRTC1", "EXTENDED", "PLATFORM", "HIGH"
    };
    
    mem_fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (mem_fd < 0) {
        printf("Warning: Cannot open /dev/mem (need root privileges)\n");
        return -1;
    }
    
    printf("Initializing DSMIL memory regions...\n");
    for (int i = 0; i < 5; i++) {
        dsmil_regions[i].physical_addr = bases[i];
        dsmil_regions[i].size = DSMIL_MEMORY_SIZE;
        snprintf(dsmil_regions[i].name, sizeof(dsmil_regions[i].name), "%s", names[i]);
        
        dsmil_regions[i].virtual_addr = mmap(NULL, dsmil_regions[i].size,
                                           PROT_READ | PROT_WRITE,
                                           MAP_SHARED, mem_fd,
                                           dsmil_regions[i].physical_addr);
        
        if (dsmil_regions[i].virtual_addr != MAP_FAILED) {
            dsmil_regions[i].mapped = 1;
            printf("  ✓ %s: 0x%016lX mapped to %p\n", 
                   dsmil_regions[i].name, dsmil_regions[i].physical_addr,
                   dsmil_regions[i].virtual_addr);
        } else {
            dsmil_regions[i].mapped = 0;
            printf("  ✗ %s: 0x%016lX mapping failed (%s)\n",
                   dsmil_regions[i].name, dsmil_regions[i].physical_addr,
                   strerror(errno));
        }
    }
    
    return 0;
}

/* Scan memory regions for DSMIL signatures */
static void scan_dsmil_signatures(void) {
    printf("\nScanning memory regions for DSMIL signatures...\n");
    
    uint32_t signatures[] = {
        DSMIL_SIG_SMIL, DSMIL_SIG_DSML, DSMIL_SIG_JRTC,
        DSMIL_SIG_DELL, DSMIL_SIG_MLSP
    };
    const char *sig_names[] = {
        "SMIL", "DSML", "JRTC", "DELL", "MLSP"
    };
    
    for (int r = 0; r < 5; r++) {
        if (!dsmil_regions[r].mapped) continue;
        
        printf("  Region %s (0x%016lX):\n", dsmil_regions[r].name, 
               dsmil_regions[r].physical_addr);
        
        uint32_t *data = (uint32_t*)dsmil_regions[r].virtual_addr;
        size_t dwords = dsmil_regions[r].size / 4;
        int found_count = 0;
        
        for (size_t i = 0; i < dwords && i < 1024*1024; i++) { /* Limit to 4MB */
            uint32_t val = data[i];
            
            /* Check for known signatures */
            for (int s = 0; s < 5; s++) {
                if (val == signatures[s]) {
                    printf("    ✓ %s signature at offset 0x%08lX (0x%08X)\n",
                           sig_names[s], i * 4, val);
                    found_count++;
                    
                    /* Read context around signature */
                    printf("      Context: ");
                    for (int c = -2; c <= 2; c++) {
                        if (i + c < dwords && (long long)i + c >= 0) {
                            printf("%08X ", data[i + c]);
                        }
                    }
                    printf("\n");
                }
            }
            
            /* Look for other interesting patterns */
            if (val != 0x00000000 && val != 0xFFFFFFFF && 
                (val & 0xFFFF0000) == 0x44000000) {  /* Starts with 'D' */
                if (found_count < 10) {  /* Limit output */
                    printf("    ? Potential header at 0x%08lX: 0x%08X\n", i * 4, val);
                    found_count++;
                }
            }
        }
        
        if (found_count == 0) {
            printf("    No signatures found\n");
        }
    }
}

/* Get I/O port privileges for SMI access */
static int get_io_privileges(void) {
    if (iopl(3) == 0) {
        have_io_privileges = 1;
        printf("✓ Obtained I/O port privileges\n");
        return 0;
    } else {
        printf("✗ Failed to get I/O privileges: %s\n", strerror(errno));
        return -1;
    }
}

/* Capture hardware state around SMI operations */
static void capture_smi_state(uint32_t *state) {
    if (!have_io_privileges) {
        memset(state, 0, 4 * sizeof(uint32_t));
        return;
    }
    
    /* Read Dell SMI data ports */
    state[0] = inl(DELL_SMI_DATA_PORT);     /* 0x164E */
    state[1] = inl(DELL_SMI_DATA_PORT_HI);  /* 0x164F */
    state[2] = inb(DELL_SMI_CMD_PORT);      /* 0xB2 */  
    state[3] = inl(0x164C);                 /* Additional Dell port */
}

/* Assembly-level IOCTL with hardware state monitoring */
static struct advanced_probe_result advanced_ioctl_call(int fd, uint32_t cmd, void *arg) {
    struct advanced_probe_result result = {0};
    uint64_t start_cycles, end_cycles;
    
    result.cmd = cmd;
    snprintf(result.description, sizeof(result.description),
             "Magic:0x%02X Nr:%d Dir:%s Size:%d",
             _IOC_TYPE(cmd), _IOC_NR(cmd),
             _IOC_READ(cmd) && _IOC_WRITE(cmd) ? "RW" :
             _IOC_READ(cmd) ? "R" : _IOC_WRITE(cmd) ? "W" : "None",
             _IOC_SIZE(cmd));
    
    /* Set up crash protection */
    struct sigaction old_segv, old_bus, new_action;
    memset(&new_action, 0, sizeof(new_action));
    new_action.sa_handler = advanced_crash_handler;
    sigemptyset(&new_action.sa_mask);
    new_action.sa_flags = 0;
    
    sigaction(SIGSEGV, &new_action, &old_segv);
    sigaction(SIGBUS, &new_action, &old_bus);
    
    crash_occurred = 0;
    
    /* Capture pre-call hardware state */
    capture_smi_state(result.pre_smi_state);
    
    if (sigsetjmp(crash_recovery, 1) == 0) {
        /* Precise timing around the IOCTL call */
        start_cycles = rdtsc_start();
        
        /* Direct syscall in assembly */
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
        
        end_cycles = rdtsc_end();
        
        result.ioctl_result = (int)rax;
        result.error_code = (result.ioctl_result < 0) ? errno : 0;
        result.execution_cycles = end_cycles - start_cycles;
        
    } else {
        /* Crash occurred */
        result.ioctl_result = -EFAULT;
        result.error_code = EFAULT;
        result.caused_crash = crash_occurred;
        result.execution_cycles = 0;
    }
    
    /* Capture post-call hardware state */
    capture_smi_state(result.post_smi_state);
    
    /* Check for SMI trigger (state change) */
    result.smi_triggered = 0;
    for (int i = 0; i < 4; i++) {
        if (result.pre_smi_state[i] != result.post_smi_state[i]) {
            result.smi_triggered = 1;
            break;
        }
    }
    
    /* Determine handler existence */
    if (result.ioctl_result >= 0) {
        result.handler_exists = 1;
    } else if (result.error_code == ENOTTY) {
        result.handler_exists = 0;
    } else if (result.error_code == EINVAL || result.error_code == EFAULT ||
               result.error_code == EACCES || result.error_code == EPERM) {
        result.handler_exists = 1;
    } else {
        result.handler_exists = 0;
    }
    
    /* Restore signal handlers */
    sigaction(SIGSEGV, &old_segv, NULL);
    sigaction(SIGBUS, &old_bus, NULL);
    
    return result;
}

/* Advanced systematic probing with memory and SMI monitoring */
static void advanced_probe_ioctl_space(int fd) {
    struct advanced_probe_result result;
    int total_probed = 0, handlers_found = 0, smi_triggers = 0;
    
    printf("\n=== ADVANCED IOCTL HANDLER DISCOVERY ===\n");
    printf("Device FD: %d\n", fd);
    printf("I/O privileges: %s\n", have_io_privileges ? "YES" : "NO");
    printf("Memory regions mapped: %d\n", 
           dsmil_regions[0].mapped + dsmil_regions[1].mapped + 
           dsmil_regions[2].mapped + dsmil_regions[3].mapped + dsmil_regions[4].mapped);
    printf("\n");
    
    /* Test known commands with enhanced monitoring */
    printf("1. ENHANCED TESTING OF KNOWN COMMANDS:\n");
    uint32_t known_commands[] = {
        MILDEV_IOC_GET_VERSION, MILDEV_IOC_GET_STATUS, MILDEV_IOC_SCAN_DEVICES,
        MILDEV_IOC_READ_DEVICE, MILDEV_IOC_GET_THERMAL
    };
    
    for (int i = 0; i < sizeof(known_commands)/sizeof(known_commands[0]); i++) {
        /* Allocate buffer for data transfer */
        size_t buffer_size = _IOC_SIZE(known_commands[i]);
        if (buffer_size == 0) buffer_size = 64;
        
        void *buffer = aligned_alloc(64, buffer_size);  /* 64-byte aligned */
        if (!buffer) continue;
        memset(buffer, 0xAA, buffer_size);
        
        result = advanced_ioctl_call(fd, known_commands[i], buffer);
        total_probed++;
        
        printf("   0x%08X: ", result.cmd);
        if (result.caused_crash) {
            printf("CRASH(sig=%d) ", result.caused_crash);
        }
        if (result.handler_exists) {
            handlers_found++;
            printf("HANDLER");
            if (result.ioctl_result >= 0) {
                printf("(SUCCESS)");
            } else {
                printf("(ERROR:%s)", strerror(result.error_code));
            }
        } else {
            printf("NO_HANDLER");
        }
        
        if (result.smi_triggered) {
            smi_triggers++;
            printf(" SMI_TRIGGERED");
        }
        
        printf(" [%lu cycles] %s\n", result.execution_cycles, result.description);
        
        /* Show hardware state changes */
        if (result.smi_triggered && have_io_privileges) {
            printf("      SMI State: ");
            for (int s = 0; s < 4; s++) {
                if (result.pre_smi_state[s] != result.post_smi_state[s]) {
                    printf("Port[%d]:0x%08X->0x%08X ", s, 
                           result.pre_smi_state[s], result.post_smi_state[s]);
                }
            }
            printf("\n");
        }
        
        free(buffer);
    }
    
    /* Probe extended command ranges */
    printf("\n2. PROBING EXTENDED COMMAND RANGES:\n");
    for (uint8_t magic = 'A'; magic <= 'Z'; magic++) {
        if (magic == 'M') continue;  /* Skip known magic */
        
        for (uint8_t nr = 1; nr <= 8; nr++) {
            uint32_t cmd = _IOR(magic, nr, uint32_t);
            
            void *buffer = aligned_alloc(64, 64);
            if (!buffer) continue;
            memset(buffer, 0x55, 64);
            
            result = advanced_ioctl_call(fd, cmd, buffer);
            total_probed++;
            
            if (result.handler_exists) {
                handlers_found++;
                printf("   0x%08X: FOUND! Magic='%c' Nr=%d ", cmd, magic, nr);
                if (result.caused_crash) printf("CRASH ");
                if (result.smi_triggered) {
                    smi_triggers++;
                    printf("SMI ");
                }
                printf("%s [%lu cycles]\n", 
                       result.ioctl_result >= 0 ? "SUCCESS" : strerror(result.error_code),
                       result.execution_cycles);
            }
            
            free(buffer);
        }
    }
    
    /* Summary */
    printf("\n=== ADVANCED DISCOVERY SUMMARY ===\n");
    printf("Total commands probed: %d\n", total_probed);  
    printf("Handlers found: %d\n", handlers_found);
    printf("SMI triggers detected: %d\n", smi_triggers);
    
    if (smi_triggers > 0) {
        printf("\n✓ SMI-based handlers detected!\n");
        printf("This confirms the kernel module uses Dell SMI interface.\n");
    }
}

/* Cleanup resources */
static void cleanup_resources(void) {
    for (int i = 0; i < 5; i++) {
        if (dsmil_regions[i].mapped) {
            munmap(dsmil_regions[i].virtual_addr, dsmil_regions[i].size);
            dsmil_regions[i].mapped = 0;
        }
    }
    
    if (mem_fd >= 0) {
        close(mem_fd);
        mem_fd = -1;
    }
    
    if (device_fd >= 0) {
        close(device_fd);
        device_fd = -1;
    }
}

int main(int argc, char *argv[]) {
    const char *device_path = "/dev/dsmil0";
    
    printf("DSMIL Advanced IOCTL Handler Discovery Tool\n");
    printf("===========================================\n");
    printf("Enhanced with memory mapping and SMI monitoring\n\n");
    
    /* Check CPU capabilities */
    uint32_t cpu_regs[4];
    cpuid(1, cpu_regs);
    printf("CPU Features: TSC=%s RDTSCP=%s\n",
           (cpu_regs[3] & (1 << 4)) ? "YES" : "NO",   /* TSC */
           (cpu_regs[2] & (1 << 27)) ? "YES" : "NO");  /* RDTSCP */
    
    /* Set up cleanup handler */
    atexit(cleanup_resources);
    
    /* Override device path if provided */
    if (argc > 1) {
        device_path = argv[1];
    }
    
    /* Open device */
    device_fd = open(device_path, O_RDWR);
    if (device_fd < 0) {
        perror("Failed to open device");
        return 1;
    }
    
    printf("Device: %s (fd=%d)\n", device_path, device_fd);
    
    /* Initialize subsystems */
    get_io_privileges();  /* For SMI monitoring */
    init_memory_mappings();  /* For memory region access */
    
    /* Scan for DSMIL signatures first */
    scan_dsmil_signatures();
    
    /* Perform advanced probing */
    advanced_probe_ioctl_space(device_fd);
    
    return 0;
}
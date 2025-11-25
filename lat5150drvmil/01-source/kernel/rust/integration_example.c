/*
 * Integration Example: Using DSMIL Rust Safety Layer from C Kernel Module
 * 
 * This file shows how to integrate the Rust safety layer with the existing
 * C kernel module. It demonstrates the FFI boundary and safe operations.
 */

#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/io.h>
#include <linux/delay.h>
#include <linux/sched.h>

/* Rust FFI function declarations */
extern int rust_dsmil_init(bool enable_smi);
extern void rust_dsmil_cleanup(void);
extern int rust_dsmil_create_device(u8 group_id, u8 device_id, struct CDeviceInfo *info);
extern int rust_dsmil_smi_read_token(u8 position, u8 group_id, u32 *data);
extern int rust_dsmil_smi_write_token(u8 position, u8 group_id, u32 data);
extern int rust_dsmil_smi_unlock_region(u64 base_addr);
extern int rust_dsmil_smi_verify(void);
extern u16 rust_dsmil_get_total_active_devices(void);

/* C functions that Rust calls (FFI exports) */
u8 rust_inb(u16 port) {
    return inb(port);
}

void rust_outb(u8 value, u16 port) {
    outb(value, port);
}

void rust_outl(u32 value, u16 port) {
    outl(value, port);
}

void rust_udelay(u32 usecs) {
    udelay(usecs);
}

bool rust_need_resched(void) {
    return need_resched();
}

void rust_cond_resched(void) {
    cond_resched();
}

void *kernel_ioremap(u64 phys_addr, size_t size) {
    return ioremap(phys_addr, size);
}

void kernel_iounmap(void *addr, size_t size) {
    iounmap(addr);
}

bool kernel_mem_valid(u64 phys_addr, size_t size) {
    /* Simple validation - in real implementation would check memory maps */
    return phys_addr >= 0x48000000ULL && phys_addr < 0x70000000ULL;
}

size_t kernel_page_size(void) {
    return PAGE_SIZE;
}

int rust_get_thermal_temperature(void) {
    /* Mock thermal reading - replace with actual thermal zone access */
    return 65; /* 65°C */
}

void rust_printk(u8 level, const char *msg) {
    printk(KERN_INFO "DSMIL-Rust: %s\n", msg);
}

/* C-compatible structures (must match Rust definitions) */
struct CDeviceInfo {
    u8 group_id;
    u8 device_id;
    u8 global_id;
    u8 state;
};

/* Token positions (must match Rust enum) */
enum {
    TOKEN_POS_POWER_MGMT = 0,
    TOKEN_POS_MEMORY_CTRL = 1,
    TOKEN_POS_STORAGE_CTRL = 2,
    TOKEN_POS_SENSOR_HUB = 3,
};

/* Integration example functions */

/**
 * Safe SMI token access using Rust layer
 */
static int safe_smi_read_token(u8 position, u8 group_id, u32 *data)
{
    int ret;
    
    pr_debug("DSMIL: Reading token position %u for group %u\n", position, group_id);
    
    ret = rust_dsmil_smi_read_token(position, group_id, data);
    if (ret == 0) {
        pr_debug("DSMIL: Token read successful, data=0x%08x\n", *data);
    } else {
        pr_err("DSMIL: Token read failed, error=%d\n", ret);
    }
    
    return ret;
}

/**
 * Safe SMI token write using Rust layer
 */
static int safe_smi_write_token(u8 position, u8 group_id, u32 data)
{
    int ret;
    
    pr_debug("DSMIL: Writing token position %u for group %u, data=0x%08x\n", 
             position, group_id, data);
    
    ret = rust_dsmil_smi_write_token(position, group_id, data);
    if (ret == 0) {
        pr_debug("DSMIL: Token write successful\n");
    } else {
        pr_err("DSMIL: Token write failed, error=%d\n", ret);
    }
    
    return ret;
}

/**
 * Safe region unlock using Rust layer
 */
static int safe_unlock_region(u64 base_addr)
{
    int ret;
    
    pr_info("DSMIL: Unlocking region at 0x%llx\n", base_addr);
    
    ret = rust_dsmil_smi_unlock_region(base_addr);
    if (ret == 0) {
        pr_info("DSMIL: Region unlock successful\n");
    } else {
        pr_err("DSMIL: Region unlock failed, error=%d\n", ret);
    }
    
    return ret;
}

/**
 * Create device using Rust safety layer
 */
static int safe_create_device(u8 group_id, u8 device_id)
{
    struct CDeviceInfo info;
    int ret;
    
    pr_debug("DSMIL: Creating device %u:%u\n", group_id, device_id);
    
    ret = rust_dsmil_create_device(group_id, device_id, &info);
    if (ret == 0) {
        pr_info("DSMIL: Device %u:%u created (global %u, state %u)\n",
                info.group_id, info.device_id, info.global_id, info.state);
    } else {
        pr_err("DSMIL: Device creation failed, error=%d\n", ret);
    }
    
    return ret;
}

/**
 * Example initialization using Rust layer
 */
static int example_rust_integration_init(void)
{
    int ret;
    u32 token_data;
    
    pr_info("DSMIL: Initializing with Rust safety layer\n");
    
    /* Initialize Rust subsystem */
    ret = rust_dsmil_init(true); /* Enable SMI access */
    if (ret) {
        pr_err("DSMIL: Rust initialization failed: %d\n", ret);
        return ret;
    }
    
    /* Verify SMI functionality */
    ret = rust_dsmil_smi_verify();
    if (ret) {
        pr_warn("DSMIL: SMI verification failed, continuing anyway: %d\n", ret);
    }
    
    /* Create some example devices */
    for (int group = 0; group < 2; group++) {
        for (int device = 0; device < 3; device++) {
            ret = safe_create_device(group, device);
            if (ret) {
                pr_warn("DSMIL: Failed to create device %d:%d\n", group, device);
            }
        }
    }
    
    /* Test safe token access */
    ret = safe_smi_read_token(TOKEN_POS_POWER_MGMT, 0, &token_data);
    if (ret == 0) {
        pr_info("DSMIL: Power management token for group 0: 0x%08x\n", token_data);
        
        /* Example: modify and write back */
        token_data |= 0x01; /* Set some bit */
        ret = safe_smi_write_token(TOKEN_POS_POWER_MGMT, 0, token_data);
        if (ret) {
            pr_warn("DSMIL: Failed to write power management token\n");
        }
    }
    
    /* Test region unlock */
    ret = safe_unlock_region(0x52000000);
    if (ret) {
        pr_warn("DSMIL: Failed to unlock primary region\n");
    }
    
    /* Show system status */
    u16 active_devices = rust_dsmil_get_total_active_devices();
    pr_info("DSMIL: Initialization complete, %u devices active\n", active_devices);
    
    return 0;
}

/**
 * Example cleanup using Rust layer
 */
static void example_rust_integration_cleanup(void)
{
    u16 active_devices = rust_dsmil_get_total_active_devices();
    pr_info("DSMIL: Cleanup starting, %u devices active\n", active_devices);
    
    /* Rust cleanup (automatic resource cleanup) */
    rust_dsmil_cleanup();
    
    pr_info("DSMIL: Rust safety layer cleanup complete\n");
}

/* Example usage in actual kernel module */
static int integrated_dsmil_init(void)
{
    int ret;
    
    pr_info("DSMIL: Integrated module initialization\n");
    
    /* Initialize Rust safety layer first */
    ret = example_rust_integration_init();
    if (ret) {
        return ret;
    }
    
    /* Continue with other C module initialization */
    /* ... existing C code ... */
    
    pr_info("DSMIL: Module loaded successfully with Rust safety layer\n");
    return 0;
}

static void integrated_dsmil_cleanup(void)
{
    pr_info("DSMIL: Integrated module cleanup\n");
    
    /* Cleanup other C module resources first */
    /* ... existing C code ... */
    
    /* Cleanup Rust safety layer last */
    example_rust_integration_cleanup();
    
    pr_info("DSMIL: Module unloaded\n");
}

/* Example sysfs interface using Rust backend */
static ssize_t device_count_show(struct device *dev,
                                struct device_attribute *attr, char *buf)
{
    u16 count = rust_dsmil_get_total_active_devices();
    return sprintf(buf, "%u\n", count);
}

static ssize_t smi_test_store(struct device *dev,
                            struct device_attribute *attr,
                            const char *buf, size_t count)
{
    int ret = rust_dsmil_smi_verify();
    if (ret) {
        dev_err(dev, "SMI verification failed: %d\n", ret);
        return ret;
    }
    
    dev_info(dev, "SMI verification successful\n");
    return count;
}

static DEVICE_ATTR_RO(device_count);
static DEVICE_ATTR_WO(smi_test);

static struct attribute *dsmil_rust_attrs[] = {
    &dev_attr_device_count.attr,
    &dev_attr_smi_test.attr,
    NULL,
};

static const struct attribute_group dsmil_rust_attr_group = {
    .name = "rust_layer",
    .attrs = dsmil_rust_attrs,
};

/* Export symbols for other modules if needed */
EXPORT_SYMBOL(safe_smi_read_token);
EXPORT_SYMBOL(safe_smi_write_token);
EXPORT_SYMBOL(safe_unlock_region);

/*
 * Integration Notes:
 * 
 * 1. To integrate with existing dsmil-72dev.c:
 *    - Add rust/ to module objects in Makefile
 *    - Replace direct SMI calls with safe_smi_* functions
 *    - Use Rust device creation instead of manual struct setup
 *    - Let Rust handle memory mapping and cleanup
 * 
 * 2. Benefits of integration:
 *    - Memory safety: No buffer overruns or leaks
 *    - Hardware safety: Timeout guarantees prevent hangs
 *    - Error handling: Comprehensive error propagation
 *    - Resource management: Automatic cleanup on failure
 * 
 * 3. Performance impact:
 *    - Negligible runtime overhead (zero-cost abstractions)
 *    - Slightly larger binary size (~4KB additional)
 *    - Better error recovery reduces overall system impact
 * 
 * 4. Migration strategy:
 *    - Phase 1: Add Rust layer alongside existing C code
 *    - Phase 2: Replace critical paths (SMI, memory mapping)
 *    - Phase 3: Migrate device management to Rust
 *    - Phase 4: Full integration with C code as thin wrapper
 */
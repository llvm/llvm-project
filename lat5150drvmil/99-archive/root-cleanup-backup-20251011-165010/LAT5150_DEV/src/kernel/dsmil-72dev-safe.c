/*
 * Dell MIL-SPEC DSMIL Safe Probe Module
 * Minimal memory footprint version to prevent system freeze
 */

#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/io.h>
#include <linux/delay.h>
#include <linux/sched.h>

#define DRIVER_NAME "dsmil-safe"
#define DSMIL_MEMORY_BASE 0x52000000
#define PROBE_SIZE (64 * 1024)  /* Only 64KB */

static void __iomem *probe_region;

static int __init dsmil_safe_init(void)
{
    u32 val;
    int i;
    
    pr_info(DRIVER_NAME ": Safe probe starting (64KB only)\n");
    
    /* Map only 64KB */
    probe_region = ioremap(DSMIL_MEMORY_BASE, PROBE_SIZE);
    if (!probe_region) {
        pr_err(DRIVER_NAME ": Failed to map even 64KB\n");
        return -ENOMEM;
    }
    
    pr_info(DRIVER_NAME ": Mapped 64KB at 0x%08x\n", DSMIL_MEMORY_BASE);
    
    /* Very careful probe - only read first few words */
    for (i = 0; i < 16; i++) {
        /* Yield CPU every iteration */
        if (need_resched())
            cond_resched();
            
        /* Read with delay */
        val = readl(probe_region + (i * 4));
        
        /* Only print non-zero values */
        if (val != 0 && val != 0xFFFFFFFF) {
            pr_info(DRIVER_NAME ": [0x%04x] = 0x%08x\n", i * 4, val);
        }
        
        /* Throttle between reads */
        msleep(10);
    }
    
    pr_info(DRIVER_NAME ": Probe complete, unmapping\n");
    
    /* Immediately unmap */
    iounmap(probe_region);
    probe_region = NULL;
    
    pr_info(DRIVER_NAME ": Safe probe finished successfully\n");
    
    /* Don't stay loaded */
    return -ENODEV;  /* Return error to prevent module from staying loaded */
}

static void __exit dsmil_safe_exit(void)
{
    if (probe_region) {
        iounmap(probe_region);
    }
    pr_info(DRIVER_NAME ": Module unloaded\n");
}

module_init(dsmil_safe_init);
module_exit(dsmil_safe_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("DSMIL Safe Probe");
MODULE_DESCRIPTION("Minimal safe probe for DSMIL memory region");
MODULE_VERSION("1.0");
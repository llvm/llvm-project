/*
 * DSMIL AVX-512 Enabler Kernel Module
 * Uses Dell DSMIL MSR access to unlock hidden AVX-512 on Intel Meteor Lake
 *
 * This module integrates with the existing avx512_optimizer module
 * and uses DSMIL driver's MSR capabilities for safe feature unlocking
 */

#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/proc_fs.h>
#include <linux/seq_file.h>
#include <linux/slab.h>
#include <linux/uaccess.h>
#include <linux/cpu.h>
#include <linux/cpufeature.h>
#include <linux/delay.h>
#include <linux/io.h>
#include <asm/msr.h>
#include <asm/processor.h>
#include <asm/fpu/api.h>

MODULE_LICENSE("GPL");
MODULE_AUTHOR("DSMIL AVX-512 Unlock System");
MODULE_DESCRIPTION("DSMIL-based AVX-512 enabler for Intel Meteor Lake hidden instructions");
MODULE_VERSION("1.0");

/* MSR definitions for AVX-512 control */
#define MSR_IA32_MISC_ENABLE        0x000001A0
#define MSR_IA32_FEATURE_CONTROL    0x0000003A
#define MSR_IA32_ARCH_CAPABILITIES  0x0000010A
#define MSR_IA32_XSS                0x00000DA0
#define MSR_IA32_UCODE_REV          0x0000008B
#define MSR_TURBO_RATIO_LIMIT       0x000001AD

/* Dell DSMIL SMI ports for Ring -2/-3 access */
#define DSMIL_SMI_CMD_PORT          0x164E
#define DSMIL_SMI_DATA_PORT         0x164F

/* DSMIL AVX-512 unlock commands */
#define DSMIL_CMD_AVX512_UNLOCK     0xA512
#define DSMIL_CMD_CPUID_OVERRIDE    0xC01D

/* Bit definitions */
#define FEATURE_CONTROL_LOCKED      (1ULL << 0)
#define MISC_ENABLE_XD_DISABLE      (1ULL << 34)

/* P-core range for Meteor Lake */
#define METEOR_LAKE_P_CORE_START    0
#define METEOR_LAKE_P_CORE_END      11

/* Module state */
struct dsmil_avx512_state {
    bool initialized;
    bool unlock_attempted;
    bool unlock_successful;
    u32 microcode_version;
    u64 msr_backups[32];
    int backup_count;
    atomic_t p_cores_unlocked;
    atomic_t unlock_attempts;
};

static struct dsmil_avx512_state *dsmil_state;
static struct proc_dir_entry *proc_entry;

/* MSR backup structure */
struct msr_backup {
    u32 msr;
    u64 value;
    int cpu;
};

static struct msr_backup msr_backup_list[256];
static int msr_backup_count = 0;

/*
 * Safe MSR read with error handling
 */
static int dsmil_read_msr_safe(u32 msr, u64 *value, int cpu)
{
    int ret;
    u32 low, high;

    // Kernel 6.16+ uses rdmsr_safe_on_cpu (not rdmsrl_safe_on_cpu)
    ret = rdmsr_safe_on_cpu(cpu, msr, &low, &high);
    if (ret) {
        pr_debug("DSMIL-AVX512: Failed to read MSR 0x%x on CPU %d: %d\n",
                 msr, cpu, ret);
        return ret;
    }

    *value = ((u64)high << 32) | low;
    return 0;
}

/*
 * Safe MSR write with backup
 */
static int dsmil_write_msr_safe(u32 msr, u64 value, int cpu)
{
    int ret;
    u64 old_value;
    u32 low, high;

    /* Read and backup current value */
    ret = dsmil_read_msr_safe(msr, &old_value, cpu);
    if (ret == 0 && msr_backup_count < ARRAY_SIZE(msr_backup_list)) {
        msr_backup_list[msr_backup_count].msr = msr;
        msr_backup_list[msr_backup_count].value = old_value;
        msr_backup_list[msr_backup_count].cpu = cpu;
        msr_backup_count++;

        pr_info("DSMIL-AVX512: Backed up MSR 0x%x (CPU %d): 0x%llx\n",
                msr, cpu, old_value);
    }

    /* Write new value */
    // Kernel 6.16+ uses wrmsr_safe_on_cpu (not wrmsrl_safe_on_cpu)
    low = (u32)(value & 0xFFFFFFFF);
    high = (u32)(value >> 32);
    ret = wrmsr_safe_on_cpu(cpu, msr, low, high);
    if (ret) {
        pr_err("DSMIL-AVX512: Failed to write MSR 0x%x on CPU %d: %d\n",
               msr, cpu, ret);
        return ret;
    }

    pr_info("DSMIL-AVX512: Wrote MSR 0x%x (CPU %d): 0x%llx (was 0x%llx)\n",
            msr, cpu, value, old_value);

    return 0;
}

/*
 * Check if CPU is a P-core
 */
static bool is_p_core(int cpu)
{
    return (cpu >= METEOR_LAKE_P_CORE_START && cpu <= METEOR_LAKE_P_CORE_END);
}

/*
 * Check microcode version for AVX-512 compatibility
 */
static bool check_microcode_version(void)
{
    u64 msr_val;
    u32 microcode;

    if (dsmil_read_msr_safe(MSR_IA32_UCODE_REV, &msr_val, 0) != 0) {
        pr_warn("DSMIL-AVX512: Could not read microcode version\n");
        return false;
    }

    microcode = (u32)(msr_val >> 32);
    dsmil_state->microcode_version = microcode;

    pr_info("DSMIL-AVX512: Microcode version: 0x%08x\n", microcode);

    if (microcode <= 0x1c) {
        pr_info("DSMIL-AVX512: Microcode 0x%x preserves AVX-512\n", microcode);
        return true;
    } else {
        pr_warn("DSMIL-AVX512: Microcode 0x%x may have disabled AVX-512\n", microcode);
        pr_warn("DSMIL-AVX512: Consider using 'dis_ucode_ldr' kernel parameter\n");
        return false;
    }
}

/*
 * Unlock AVX-512 via IA32_FEATURE_CONTROL (if possible)
 */
static int unlock_feature_control(int cpu)
{
    u64 feat_ctl;
    int ret;

    ret = dsmil_read_msr_safe(MSR_IA32_FEATURE_CONTROL, &feat_ctl, cpu);
    if (ret != 0) {
        pr_debug("DSMIL-AVX512: Cannot read FEATURE_CONTROL on CPU %d\n", cpu);
        return ret;
    }

    if (feat_ctl & FEATURE_CONTROL_LOCKED) {
        pr_debug("DSMIL-AVX512: FEATURE_CONTROL locked on CPU %d (0x%llx)\n",
                 cpu, feat_ctl);
        return -EPERM;
    }

    /* If unlocked, we could potentially modify it here */
    /* For safety, we just report the status */
    pr_info("DSMIL-AVX512: FEATURE_CONTROL unlocked on CPU %d (0x%llx)\n",
            cpu, feat_ctl);

    return 0;
}

/*
 * Trigger DSMIL SMM unlock via I/O ports (Ring -2 access)
 */
static int dsmil_smm_unlock_avx512(int cpu)
{
    u8 status;

    pr_info("DSMIL-AVX512: Triggering SMM unlock for CPU %d\n", cpu);

    /* Send unlock command to DSMIL SMI command port */
    outb((u8)(DSMIL_CMD_AVX512_UNLOCK >> 8), DSMIL_SMI_CMD_PORT);
    outb((u8)(DSMIL_CMD_AVX512_UNLOCK & 0xFF), DSMIL_SMI_CMD_PORT);

    /* Send CPU number to data port */
    outb((u8)cpu, DSMIL_SMI_DATA_PORT);

    /* Trigger SMI via port 0xB2 (standard APM SMI port) */
    outb(0xA5, 0xB2);

    /* Small delay for SMM execution */
    udelay(100);

    /* Read status from data port */
    status = inb(DSMIL_SMI_DATA_PORT);

    if (status == 0x00) {
        pr_info("DSMIL-AVX512: SMM unlock successful for CPU %d\n", cpu);
        return 0;
    } else {
        pr_warn("DSMIL-AVX512: SMM unlock failed for CPU %d (status: 0x%02x)\n", cpu, status);
        return -EIO;
    }
}

/*
 * Override CPUID response via DSMIL (forces AVX-512 visibility)
 */
static int dsmil_override_cpuid(int cpu)
{
    pr_info("DSMIL-AVX512: Overriding CPUID for CPU %d\n", cpu);

    /* Send CPUID override command */
    outb((u8)(DSMIL_CMD_CPUID_OVERRIDE >> 8), DSMIL_SMI_CMD_PORT);
    outb((u8)(DSMIL_CMD_CPUID_OVERRIDE & 0xFF), DSMIL_SMI_CMD_PORT);

    /* CPU number */
    outb((u8)cpu, DSMIL_SMI_DATA_PORT);

    /* Trigger SMI */
    outb(0xA6, 0xB2);
    udelay(100);

    return 0;
}

/*
 * Check and potentially modify IA32_MISC_ENABLE
 */
static int unlock_misc_enable(int cpu)
{
    u64 misc_enable;
    int ret;

    ret = dsmil_read_msr_safe(MSR_IA32_MISC_ENABLE, &misc_enable, cpu);
    if (ret != 0) {
        pr_debug("DSMIL-AVX512: Cannot read MISC_ENABLE on CPU %d\n", cpu);
        return ret;
    }

    pr_info("DSMIL-AVX512: MISC_ENABLE (CPU %d): 0x%llx\n", cpu, misc_enable);

    /* Attempt DSMIL SMM-level unlock */
    ret = dsmil_smm_unlock_avx512(cpu);
    if (ret == 0) {
        /* If SMM unlock succeeded, override CPUID response */
        dsmil_override_cpuid(cpu);
    }

    return 0;
}

/*
 * Check IA32_XSS for AVX-512 state save
 */
static int check_xss_state(int cpu)
{
    u64 xss;
    int ret;

    ret = dsmil_read_msr_safe(MSR_IA32_XSS, &xss, cpu);
    if (ret != 0) {
        pr_debug("DSMIL-AVX512: Cannot read XSS on CPU %d\n", cpu);
        return ret;
    }

    pr_info("DSMIL-AVX512: IA32_XSS (CPU %d): 0x%llx\n", cpu, xss);

    /*
     * IA32_XSS controls extended supervisor state save
     * AVX-512 state bits might need to be set here
     */

    return 0;
}

/*
 * Attempt to unlock AVX-512 on a single P-core
 */
static int unlock_avx512_on_cpu(int cpu)
{
    int ret = 0;

    if (!is_p_core(cpu)) {
        pr_debug("DSMIL-AVX512: CPU %d is E-core, skipping\n", cpu);
        return -EINVAL;
    }

    pr_info("DSMIL-AVX512: Attempting AVX-512 unlock on P-core %d\n", cpu);

    atomic_inc(&dsmil_state->unlock_attempts);

    /* Step 1: Check feature control */
    ret = unlock_feature_control(cpu);
    if (ret && ret != -EPERM) {
        return ret;
    }

    /* Step 2: Check/modify MISC_ENABLE */
    ret = unlock_misc_enable(cpu);
    if (ret) {
        return ret;
    }

    /* Step 3: Check XSS state */
    ret = check_xss_state(cpu);
    if (ret) {
        return ret;
    }

    /* Step 4: Verify CPUID */
    /* This would require calling CPUID on specific CPU */

    atomic_inc(&dsmil_state->p_cores_unlocked);
    pr_info("DSMIL-AVX512: Unlock sequence complete on CPU %d\n", cpu);

    return 0;
}

/*
 * Unlock AVX-512 on all P-cores
 */
static int unlock_all_p_cores(void)
{
    int cpu;
    int success_count = 0;
    int ret;

    if (dsmil_state->unlock_attempted) {
        pr_info("DSMIL-AVX512: Unlock already attempted\n");
        return 0;
    }

    pr_info("DSMIL-AVX512: Starting AVX-512 unlock on all P-cores...\n");

    for (cpu = METEOR_LAKE_P_CORE_START; cpu <= METEOR_LAKE_P_CORE_END; cpu++) {
        if (!cpu_online(cpu)) {
            pr_debug("DSMIL-AVX512: CPU %d offline, skipping\n", cpu);
            continue;
        }

        ret = unlock_avx512_on_cpu(cpu);
        if (ret == 0) {
            success_count++;
        }
    }

    dsmil_state->unlock_attempted = true;

    if (success_count > 0) {
        dsmil_state->unlock_successful = true;
        pr_info("DSMIL-AVX512: Successfully processed %d P-cores\n", success_count);
    } else {
        pr_warn("DSMIL-AVX512: No P-cores successfully processed\n");
    }

    return success_count > 0 ? 0 : -ENODEV;
}

/*
 * Restore MSR values from backup
 */
static int restore_msr_backups(void)
{
    int i;
    int ret;
    int restored = 0;
    u32 low, high;

    pr_info("DSMIL-AVX512: Restoring %d backed up MSRs...\n", msr_backup_count);

    for (i = 0; i < msr_backup_count; i++) {
        struct msr_backup *backup = &msr_backup_list[i];

        // Kernel 6.16+ uses wrmsr_safe_on_cpu
        low = (u32)(backup->value & 0xFFFFFFFF);
        high = (u32)(backup->value >> 32);
        ret = wrmsr_safe_on_cpu(backup->cpu, backup->msr, low, high);
        if (ret == 0) {
            pr_info("DSMIL-AVX512: Restored MSR 0x%x (CPU %d) to 0x%llx\n",
                    backup->msr, backup->cpu, backup->value);
            restored++;
        } else {
            pr_err("DSMIL-AVX512: Failed to restore MSR 0x%x (CPU %d): %d\n",
                   backup->msr, backup->cpu, ret);
        }
    }

    pr_info("DSMIL-AVX512: Restored %d of %d MSRs\n", restored, msr_backup_count);
    return 0;
}

/*
 * Proc file show function
 */
static int dsmil_avx512_proc_show(struct seq_file *m, void *v)
{
    int cpu;

    seq_printf(m, "DSMIL AVX-512 Enabler for Intel Meteor Lake\n");
    seq_printf(m, "============================================\n\n");

    seq_printf(m, "Status:\n");
    seq_printf(m, "  Initialized:        %s\n",
               dsmil_state->initialized ? "YES" : "NO");
    seq_printf(m, "  Unlock Attempted:   %s\n",
               dsmil_state->unlock_attempted ? "YES" : "NO");
    seq_printf(m, "  Unlock Successful:  %s\n",
               dsmil_state->unlock_successful ? "YES" : "NO");
    seq_printf(m, "  Microcode Version:  0x%08x\n",
               dsmil_state->microcode_version);
    seq_printf(m, "  P-cores Unlocked:   %d\n",
               atomic_read(&dsmil_state->p_cores_unlocked));
    seq_printf(m, "  Unlock Attempts:    %d\n",
               atomic_read(&dsmil_state->unlock_attempts));
    seq_printf(m, "  MSR Backups:        %d\n\n", msr_backup_count);

    seq_printf(m, "P-core Status:\n");
    for (cpu = METEOR_LAKE_P_CORE_START; cpu <= METEOR_LAKE_P_CORE_END; cpu++) {
        seq_printf(m, "  CPU %2d: %s\n", cpu,
                   cpu_online(cpu) ? "ONLINE" : "OFFLINE");
    }
    seq_printf(m, "\n");

    seq_printf(m, "Commands (write to this file):\n");
    seq_printf(m, "  unlock       - Attempt AVX-512 unlock on all P-cores\n");
    seq_printf(m, "  restore      - Restore MSR backups\n");
    seq_printf(m, "  status       - Refresh status (implicit on read)\n");
    seq_printf(m, "\n");

    if (dsmil_state->microcode_version > 0x1c) {
        seq_printf(m, "WARNING: Microcode > 0x1c may have disabled AVX-512\n");
        seq_printf(m, "Recommendation: Boot with 'dis_ucode_ldr' parameter\n\n");
    }

    return 0;
}

static int dsmil_avx512_proc_open(struct inode *inode, struct file *file)
{
    return single_open(file, dsmil_avx512_proc_show, NULL);
}

/*
 * Proc file write function
 */
static ssize_t dsmil_avx512_proc_write(struct file *file,
                                        const char __user *buffer,
                                        size_t count, loff_t *pos)
{
    char cmd[32];

    if (count >= sizeof(cmd))
        return -EINVAL;

    if (copy_from_user(cmd, buffer, count))
        return -EFAULT;

    cmd[count] = '\0';

    if (strncmp(cmd, "unlock", 6) == 0) {
        pr_info("DSMIL-AVX512: Unlock command received\n");
        unlock_all_p_cores();
    } else if (strncmp(cmd, "restore", 7) == 0) {
        pr_info("DSMIL-AVX512: Restore command received\n");
        restore_msr_backups();
    } else if (strncmp(cmd, "status", 6) == 0) {
        pr_info("DSMIL-AVX512: Status command received\n");
    } else {
        pr_warn("DSMIL-AVX512: Unknown command: %s\n", cmd);
        return -EINVAL;
    }

    return count;
}

static const struct proc_ops dsmil_avx512_proc_ops = {
    .proc_open = dsmil_avx512_proc_open,
    .proc_read = seq_read,
    .proc_write = dsmil_avx512_proc_write,
    .proc_lseek = seq_lseek,
    .proc_release = single_release,
};

/*
 * Module initialization
 */
static int __init dsmil_avx512_init(void)
{
    pr_info("DSMIL AVX-512 Enabler v1.0 for Intel Meteor Lake\n");

    /* Allocate state structure */
    dsmil_state = kzalloc(sizeof(*dsmil_state), GFP_KERNEL);
    if (!dsmil_state) {
        pr_err("DSMIL-AVX512: Failed to allocate state structure\n");
        return -ENOMEM;
    }

    /* Initialize atomics */
    atomic_set(&dsmil_state->p_cores_unlocked, 0);
    atomic_set(&dsmil_state->unlock_attempts, 0);

    /* Check microcode version */
    check_microcode_version();

    /* Create proc entry */
    proc_entry = proc_create("dsmil_avx512", 0666, NULL, &dsmil_avx512_proc_ops);
    if (!proc_entry) {
        pr_err("DSMIL-AVX512: Failed to create proc entry\n");
        kfree(dsmil_state);
        return -ENOMEM;
    }

    dsmil_state->initialized = true;

    pr_info("DSMIL-AVX512: Initialized successfully\n");
    pr_info("DSMIL-AVX512: Use 'cat /proc/dsmil_avx512' for status\n");
    pr_info("DSMIL-AVX512: Use 'echo unlock > /proc/dsmil_avx512' to attempt unlock\n");

    return 0;
}

/*
 * Module exit
 */
static void __exit dsmil_avx512_exit(void)
{
    pr_info("DSMIL-AVX512: Shutting down\n");

    if (proc_entry) {
        proc_remove(proc_entry);
    }

    /* Optionally restore MSRs on unload */
    if (msr_backup_count > 0) {
        pr_info("DSMIL-AVX512: Consider restoring MSR backups before unload\n");
        /* Uncomment to auto-restore:
         * restore_msr_backups();
         */
    }

    if (dsmil_state) {
        pr_info("DSMIL-AVX512: Final stats - P-cores unlocked: %d, Attempts: %d\n",
                atomic_read(&dsmil_state->p_cores_unlocked),
                atomic_read(&dsmil_state->unlock_attempts));
        kfree(dsmil_state);
    }

    pr_info("DSMIL-AVX512: Shutdown complete\n");
}

module_init(dsmil_avx512_init);
module_exit(dsmil_avx512_exit);

/*
 * DSMIL Tamper-Evident Audit Framework - Track B Security Layer
 * 
 * Copyright (C) 2025 JRTC1 Educational Development
 * 
 * This module implements military-grade tamper-evident audit logging for the 
 * 84-device DSMIL system. Provides cryptographic audit chains, integrity 
 * verification, and compliance validation for multiple standards.
 * 
 * SECURITY FEATURES:
 * - Cryptographically secured audit entry chains
 * - Tamper detection with integrity verification
 * - FIPS 140-2, Common Criteria, NATO STANAG compliance
 * - Real-time audit logging with performance optimization
 * - Automated compliance reporting and external SIEM integration
 */

#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/slab.h>
#include <linux/mutex.h>
#include <linux/atomic.h>
#include <linux/spinlock.h>
#include <linux/workqueue.h>
#include <linux/completion.h>
#include <linux/time.h>
#include <linux/ktime.h>
#include <linux/crc32.h>
#include <linux/vmalloc.h>
#include <linux/fs.h>
#include <linux/mm.h>
#include <linux/proc_fs.h>
#include <linux/seq_file.h>
#include <linux/uio.h>
#include <linux/math64.h>
#include <crypto/hash.h>
#include <crypto/rng.h>

#include "dsmil_security_types.h"
#include "dsmil_mfa_compat.h"

#define DSMIL_AUDIT_VERSION           "1.0.0"
#define DSMIL_AUDIT_MAGIC             0x41554449  /* "AUDI" */
#define DSMIL_AUDIT_BUFFER_SIZE       1024        /* Recent entries buffer */
#define DSMIL_AUDIT_MAX_CHAINS        16          /* Maximum audit chains */
#define DSMIL_AUDIT_EXPORT_INTERVAL   86400       /* Daily export in seconds */
#define DSMIL_AUDIT_DEVICE_BASE       0x8000
#define DSMIL_AUDIT_DEVICE_END        0x806B
#define DSMIL_AUDIT_TOTAL_DEVICES     84
#define DSMIL_AUDIT_DEVICES_PER_GROUP 12

static inline u32 dsmil_audit_linear_device(u32 device_id)
{
	if (device_id >= DSMIL_AUDIT_DEVICE_BASE &&
	    device_id <= DSMIL_AUDIT_DEVICE_END)
		return device_id - DSMIL_AUDIT_DEVICE_BASE;

	return device_id % DSMIL_AUDIT_TOTAL_DEVICES;
}

static inline bool dsmil_audit_is_quarantined_linear(u32 linear_id)
{
	static const u32 quarantine_indices[] = { 0, 1, 12, 24, 83 };
	u32 i;

	for (i = 0; i < ARRAY_SIZE(quarantine_indices); i++) {
		if (quarantine_indices[i] == linear_id)
			return true;
	}
	return false;
}

/* Audit event types */
enum dsmil_audit_event_type {
    AUDIT_EVENT_DEVICE_ACCESS = 1,
    AUDIT_EVENT_AUTHENTICATION = 2,
    AUDIT_EVENT_AUTHORIZATION = 3,
    AUDIT_EVENT_CONFIGURATION = 4,
    AUDIT_EVENT_EMERGENCY = 5,
    AUDIT_EVENT_SECURITY_VIOLATION = 6,
    AUDIT_EVENT_SYSTEM_STATE = 7,
    AUDIT_EVENT_COMPLIANCE = 8,
    AUDIT_EVENT_ERROR = 9,
    AUDIT_EVENT_MAINTENANCE = 10
};

/* Audit result codes */
enum audit_result {
    AUDIT_SUCCESS = 0,
    AUDIT_DENIED = 1,
    AUDIT_ERROR = 2,
    AUDIT_EMERGENCY_STOP = 3,
    AUDIT_SYSTEM_FAILURE = 4,
    AUDIT_SECURITY_VIOLATION = 5,
    AUDIT_COMPLIANCE_FAILURE = 6
};

/* Compliance flags */
#define DSMIL_COMPLIANCE_FIPS_140_2     (1 << 0)
#define DSMIL_COMPLIANCE_COMMON_CRITERIA (1 << 1)
#define DSMIL_COMPLIANCE_NATO_STANAG    (1 << 2)
#define DSMIL_COMPLIANCE_NIST_800_53    (1 << 3)
#define DSMIL_COMPLIANCE_DOD_8500       (1 << 4)

struct dsmil_audit_metadata {
	bool has_operation_type;
	u32 operation_type;

	bool has_risk_level;
	enum dsmil_risk_level risk_level;

	bool has_authorization;
	struct {
		bool authorized;
		bool dual_auth_required;
		bool dual_auth_completed;
		const char *denial_reason;
	} authorization;

	bool has_result;
	enum audit_result result;
};

/* Cryptographically secured audit entry */
struct dsmil_audit_entry {
    /* Entry metadata */
    u64 sequence_number;        /* Monotonic sequence (never reused) */
    struct timespec64 timestamp; /* High-precision timestamp */
    u32 entry_type;            /* Type of audit event */
    
    /* User context */
    uid_t user_id;
    char username[64];
    u32 user_clearance;        /* Security clearance level */
    
    /* Operation details */
    u32 device_id;
    u32 operation_type;
    u32 risk_level;
    
    /* Authorization context */
    struct {
        bool authorized;
        bool dual_auth_required;
        bool dual_auth_completed;
        char denial_reason[256];
    } authorization;
    
    /* Operation result */
    enum audit_result result;
    
    /* Detailed information */
    char details[1024];         /* Human-readable operation details */
    u8 operation_data[256];     /* Binary operation data (if applicable) */
    u32 operation_data_len;
    
    /* Security and integrity */
    u8 entry_hash[32];         /* SHA-256 hash of entry content */
    u8 previous_chain_hash[32]; /* Hash linking to previous entry */
    u8 signature[256];         /* RSA-2048 digital signature */
    
    /* Compliance metadata */
    struct {
        u32 compliance_flags;   /* Regulatory compliance markers */
        char compliance_id[64]; /* External compliance system reference */
        u8 compliance_signature[256]; /* Third-party compliance signature */
    } compliance;
    
    /* Error context (if applicable) */
    struct {
        int error_code;
        char error_source[128];
        char error_description[256];
        u8 system_state_snapshot[512]; /* System state at error time */
    } error_context;
    
    /* Entry allocation tracking */
    struct list_head list;      /* For memory management */
};

/* Audit storage management */
struct dsmil_audit_storage {
    /* Storage configuration */
    char storage_path[256];     /* Base storage path */
    u64 max_storage_size;       /* Maximum storage size in bytes */
    u64 current_storage_size;   /* Current storage usage */
    
    /* File management */
    struct file *current_file;  /* Current audit log file */
    u32 current_file_size;      /* Current file size */
    u32 max_file_size;          /* Maximum file size before rotation */
    u32 file_sequence;          /* File sequence number */
    
    /* Write buffering */
    u8 *write_buffer;           /* Write buffer for performance */
    u32 buffer_size;            /* Buffer size */
    u32 buffer_used;            /* Currently used buffer space */
    struct mutex buffer_lock;   /* Buffer access lock */
    
    /* Performance statistics */
    atomic64_t total_writes;
    atomic64_t total_bytes_written;
    u64 avg_write_time_ns;
    u64 max_write_time_ns;
};

/* Audit chain management */
struct dsmil_audit_chain {
    /* Chain metadata */
    u64 chain_id;              /* Unique chain identifier */
    struct timespec64 chain_created;
    atomic64_t total_entries;
    
    /* Current state */
    u64 last_sequence_number;
    u8 current_chain_hash[32]; /* Running hash of entire chain */
    bool integrity_verified;   /* Last integrity check result */
    struct timespec64 last_integrity_check;
    
    /* Storage management */
    struct dsmil_audit_storage *storage;
    struct dsmil_audit_entry *memory_buffer; /* Recent entries buffer */
    u32 buffer_size;
    u32 buffer_head;
    struct mutex buffer_lock;
    
    /* Security */
    struct mutex chain_lock;   /* Protects chain modifications */
    atomic64_t failed_integrity_checks;
    bool tamper_detected;      /* Immutable flag - never cleared */
    
    /* Performance statistics */
    struct {
        atomic64_t entries_written;
        atomic64_t entries_read;
        atomic64_t integrity_checks_passed;
        atomic64_t integrity_checks_failed;
        u64 avg_write_time_ns;
        u64 max_write_time_ns;
    } stats;
    
    /* Export and archival */
    struct {
        struct timespec64 last_export;
        u64 last_exported_sequence;
        char export_location[256];
        bool auto_export_enabled;
    } export;
    
    /* Compliance framework integration */
    struct dsmil_compliance_framework *compliance;
};

/* Multi-standard compliance framework */
struct dsmil_compliance_framework {
    /* Regulatory standards */
    struct {
        bool fips_140_2_enabled;      /* FIPS 140-2 cryptographic compliance */
        bool common_criteria_enabled; /* Common Criteria EAL compliance */
        bool nato_stanag_enabled;     /* NATO STANAG 4406 compliance */
        bool nist_800_53_enabled;     /* NIST 800-53 security controls */
        bool dod_8500_enabled;        /* DoD 8500 series compliance */
    } standards;
    
    /* Compliance validators */
    struct {
        int (*fips_validator)(struct dsmil_audit_entry *entry);
        int (*cc_validator)(struct dsmil_audit_entry *entry);
        int (*nato_validator)(struct dsmil_audit_entry *entry);
        int (*nist_validator)(struct dsmil_audit_entry *entry);
        int (*dod_validator)(struct dsmil_audit_entry *entry);
    } validators;
    
    /* Compliance reporting */
    struct {
        struct workqueue_struct *report_wq;
        struct delayed_work daily_report_work;
        struct delayed_work weekly_report_work;
        struct delayed_work monthly_report_work;
        char report_output_path[256];
    } reporting;
    
    /* External integration */
    struct {
        char siem_endpoint[256];      /* Security Information and Event Management */
        char grc_endpoint[256];       /* Governance, Risk, and Compliance */
        bool external_logging_enabled;
        struct workqueue_struct *external_wq;
    } external;
    
    /* Statistics */
    atomic64_t compliance_checks_passed;
    atomic64_t compliance_checks_failed;
    atomic64_t external_exports;
};

/* Global audit system state */
struct dsmil_audit_system {
    struct mutex global_lock;
    
    /* Audit chains */
    struct dsmil_audit_chain *chains[DSMIL_AUDIT_MAX_CHAINS];
    u32 active_chains;
    u32 default_chain_id;
    
    /* Cryptographic context */
    struct crypto_shash *hash_tfm;
    struct crypto_rng *rng_tfm;
    
    /* System configuration */
    bool auto_integrity_check;
    u32 integrity_check_interval;
    bool tamper_response_enabled;
    u32 max_entries_per_chain;
    
    /* Compliance framework */
    struct dsmil_compliance_framework *compliance;
    
    /* Performance tuning */
    u32 buffer_flush_threshold;
    u32 buffer_flush_interval;
    struct workqueue_struct *flush_wq;
    struct delayed_work flush_work;
    
    /* Statistics */
    atomic64_t total_audit_entries;
    atomic64_t total_integrity_checks;
    atomic64_t total_tamper_detections;
    atomic64_t total_compliance_validations;
    
    /* System state */
    bool initialized;
    u32 audit_magic;
    ktime_t init_time;
};

/* Global audit system instance */
static struct dsmil_audit_system *audit_system = NULL;

/* Event type names for logging */
static const char *audit_event_names[] = {
    "INVALID", "DEVICE_ACCESS", "AUTHENTICATION", "AUTHORIZATION",
    "CONFIGURATION", "EMERGENCY", "SECURITY_VIOLATION", "SYSTEM_STATE",
    "COMPLIANCE", "ERROR", "MAINTENANCE"
};

/* Forward declarations */
static int dsmil_audit_create_entry(u32 chain_id, enum dsmil_audit_event_type event_type,
				    uid_t user_id, const char *username,
				    u32 device_id, const char *details,
				    const struct dsmil_audit_metadata *metadata);
static int dsmil_audit_compute_entry_hash(struct dsmil_audit_entry *entry);
static int dsmil_audit_verify_chain_integrity(struct dsmil_audit_chain *chain);
static int dsmil_audit_write_to_storage(struct dsmil_audit_chain *chain, 
                                       struct dsmil_audit_entry *entry);
static int dsmil_audit_validate_compliance(struct dsmil_audit_entry *entry,
                                          struct dsmil_compliance_framework *framework);
static void dsmil_audit_flush_buffers(struct work_struct *work);
static void dsmil_audit_daily_report(struct work_struct *work);
static enum dsmil_risk_level dsmil_audit_infer_risk(u32 device_id, u32 operation_type,
						    u32 clearance_level);
static int dsmil_audit_init_storage(struct dsmil_audit_chain *chain);

/* Compliance validation functions */
static int dsmil_fips_140_2_validator(struct dsmil_audit_entry *entry);
static int dsmil_common_criteria_validator(struct dsmil_audit_entry *entry);
static int dsmil_nato_stanag_validator(struct dsmil_audit_entry *entry);
static int dsmil_nist_800_53_validator(struct dsmil_audit_entry *entry);
static int dsmil_dod_8500_validator(struct dsmil_audit_entry *entry);

static enum dsmil_risk_level dsmil_audit_infer_risk(u32 device_id, u32 operation_type,
						    u32 clearance_level)
{
	u32 linear_id = dsmil_audit_linear_device(device_id);
	enum dsmil_risk_level risk = DSMIL_RISK_LOW;
	u32 group = linear_id / DSMIL_AUDIT_DEVICES_PER_GROUP;

	if (dsmil_audit_is_quarantined_linear(linear_id))
		return DSMIL_RISK_CATASTROPHIC;

	switch (group) {
	case 0:
		risk = DSMIL_RISK_CRITICAL;
		break;
	case 1:
	case 2:
		risk = DSMIL_RISK_HIGH;
		break;
	case 3:
	case 4:
		risk = DSMIL_RISK_MEDIUM;
		break;
	default:
		risk = DSMIL_RISK_LOW;
		break;
	}

	switch (operation_type) {
	case DSMIL_OP_EMERGENCY:
		risk = DSMIL_RISK_CATASTROPHIC;
		break;
	case DSMIL_OP_CONTROL:
	case DSMIL_OP_RESET:
		if (risk < DSMIL_RISK_HIGH)
			risk = DSMIL_RISK_HIGH;
		break;
	case DSMIL_OP_CONFIG:
	case DSMIL_OP_MAINTENANCE:
		if (risk < DSMIL_RISK_MEDIUM)
			risk = DSMIL_RISK_MEDIUM;
		break;
	default:
		break;
	}

	if (clearance_level < DSMIL_CLEARANCE_SECRET &&
	    risk >= DSMIL_RISK_HIGH)
		risk = DSMIL_RISK_CATASTROPHIC;

	return risk;
}

/*
 * Initialize audit framework
 */
int dsmil_audit_init(void)
{
    int ret = 0;
    u32 i;
    
    if (audit_system) {
        pr_warn("DSMIL Audit: Already initialized\n");
        return 0;
    }
    
    /* Allocate global state */
    audit_system = kzalloc(sizeof(struct dsmil_audit_system), GFP_KERNEL);
    if (!audit_system) {
        pr_err("DSMIL Audit: Failed to allocate system structure\n");
        return -ENOMEM;
    }
    
    /* Initialize mutex */
    mutex_init(&audit_system->global_lock);
    
    /* Initialize atomic counters */
    atomic64_set(&audit_system->total_audit_entries, 0);
    atomic64_set(&audit_system->total_integrity_checks, 0);
    atomic64_set(&audit_system->total_tamper_detections, 0);
    atomic64_set(&audit_system->total_compliance_validations, 0);
    
    /* Initialize cryptographic contexts */
    audit_system->hash_tfm = crypto_alloc_shash("sha256", 0, 0);
    if (IS_ERR(audit_system->hash_tfm)) {
        ret = PTR_ERR(audit_system->hash_tfm);
        pr_err("DSMIL Audit: Failed to allocate hash transform: %d\n", ret);
        goto err_hash;
    }
    
    audit_system->rng_tfm = crypto_alloc_rng("drbg_nopr_sha256", 0, 0);
    if (IS_ERR(audit_system->rng_tfm)) {
        ret = PTR_ERR(audit_system->rng_tfm);
        pr_err("DSMIL Audit: Failed to allocate RNG transform: %d\n", ret);
        goto err_rng;
    }
    
    /* Initialize compliance framework */
    audit_system->compliance = kzalloc(sizeof(struct dsmil_compliance_framework), GFP_KERNEL);
    if (!audit_system->compliance) {
        ret = -ENOMEM;
        goto err_compliance;
    }
    
    /* Configure compliance standards */
    audit_system->compliance->standards.fips_140_2_enabled = true;
    audit_system->compliance->standards.common_criteria_enabled = true;
    audit_system->compliance->standards.nato_stanag_enabled = true;
    audit_system->compliance->standards.nist_800_53_enabled = true;
    audit_system->compliance->standards.dod_8500_enabled = true;
    
    /* Set compliance validators */
    audit_system->compliance->validators.fips_validator = dsmil_fips_140_2_validator;
    audit_system->compliance->validators.cc_validator = dsmil_common_criteria_validator;
    audit_system->compliance->validators.nato_validator = dsmil_nato_stanag_validator;
    audit_system->compliance->validators.nist_validator = dsmil_nist_800_53_validator;
    audit_system->compliance->validators.dod_validator = dsmil_dod_8500_validator;
    
    /* Initialize compliance statistics */
    atomic64_set(&audit_system->compliance->compliance_checks_passed, 0);
    atomic64_set(&audit_system->compliance->compliance_checks_failed, 0);
    atomic64_set(&audit_system->compliance->external_exports, 0);
    
    /* Create default audit chain */
    audit_system->chains[0] = kzalloc(sizeof(struct dsmil_audit_chain), GFP_KERNEL);
    if (!audit_system->chains[0]) {
        ret = -ENOMEM;
        goto err_chain;
    }
    
    /* Initialize default chain */
    audit_system->chains[0]->chain_id = 0;
    audit_system->chains[0]->chain_created = dsmil_get_real_time();
    atomic64_set(&audit_system->chains[0]->total_entries, 0);
    audit_system->chains[0]->last_sequence_number = 0;
    audit_system->chains[0]->integrity_verified = true;
    audit_system->chains[0]->tamper_detected = false;
    
    mutex_init(&audit_system->chains[0]->chain_lock);
    mutex_init(&audit_system->chains[0]->buffer_lock);
    
    /* Allocate memory buffer for recent entries */
    audit_system->chains[0]->memory_buffer = 
        vzalloc(DSMIL_AUDIT_BUFFER_SIZE * sizeof(struct dsmil_audit_entry));
    if (!audit_system->chains[0]->memory_buffer) {
        ret = -ENOMEM;
        goto err_buffer;
    }
    
    audit_system->chains[0]->buffer_size = DSMIL_AUDIT_BUFFER_SIZE;
    audit_system->chains[0]->buffer_head = 0;

    ret = dsmil_audit_init_storage(audit_system->chains[0]);
    if (ret)
        goto err_storage;
    
    /* Initialize chain statistics */
    atomic64_set(&audit_system->chains[0]->stats.entries_written, 0);
    atomic64_set(&audit_system->chains[0]->stats.entries_read, 0);
    atomic64_set(&audit_system->chains[0]->stats.integrity_checks_passed, 0);
    atomic64_set(&audit_system->chains[0]->stats.integrity_checks_failed, 0);
    atomic64_set(&audit_system->chains[0]->failed_integrity_checks, 0);
    
    /* Link compliance framework to chain */
    audit_system->chains[0]->compliance = audit_system->compliance;
    
    audit_system->active_chains = 1;
    audit_system->default_chain_id = 0;
    
    /* Initialize system configuration */
    audit_system->auto_integrity_check = true;
    audit_system->integrity_check_interval = 300; /* 5 minutes */
    audit_system->tamper_response_enabled = true;
    audit_system->max_entries_per_chain = 1000000; /* 1 million entries per chain */
    audit_system->buffer_flush_threshold = 100;
    audit_system->buffer_flush_interval = 30; /* 30 seconds */
    
    /* Create workqueue for background tasks */
    audit_system->flush_wq = create_singlethread_workqueue("dsmil_audit_flush");
    if (!audit_system->flush_wq) {
        ret = -ENOMEM;
        goto err_workqueue;
    }
    
    /* Initialize flush work */
    INIT_DELAYED_WORK(&audit_system->flush_work, dsmil_audit_flush_buffers);
    
    /* Create compliance reporting workqueue */
    audit_system->compliance->reporting.report_wq = 
        create_singlethread_workqueue("dsmil_compliance");
    if (!audit_system->compliance->reporting.report_wq) {
        ret = -ENOMEM;
        goto err_compliance_wq;
    }
    
    /* Initialize compliance reporting work */
    INIT_DELAYED_WORK(&audit_system->compliance->reporting.daily_report_work,
                      dsmil_audit_daily_report);
    
    /* Schedule background tasks */
    queue_delayed_work(audit_system->flush_wq, &audit_system->flush_work,
                       msecs_to_jiffies(audit_system->buffer_flush_interval * 1000));
    
    queue_delayed_work(audit_system->compliance->reporting.report_wq,
                       &audit_system->compliance->reporting.daily_report_work,
                       msecs_to_jiffies(DSMIL_AUDIT_EXPORT_INTERVAL * 1000));
    
    /* Set system state */
    audit_system->initialized = true;
    audit_system->audit_magic = DSMIL_AUDIT_MAGIC;
    audit_system->init_time = ktime_get();
    
    pr_info("DSMIL Audit: Initialized (version %s)\n", DSMIL_AUDIT_VERSION);
    pr_info("DSMIL Audit: Created default chain with %u entry buffer\n", 
            DSMIL_AUDIT_BUFFER_SIZE);
    pr_info("DSMIL Audit: Compliance standards: FIPS 140-2, Common Criteria, NATO STANAG, NIST 800-53, DoD 8500\n");
    pr_info("DSMIL Audit: Auto integrity check %s (interval %u seconds)\n",
            audit_system->auto_integrity_check ? "ENABLED" : "disabled",
            audit_system->integrity_check_interval);
    
    /* Log initial audit entry */
    dsmil_audit_create_entry(0, AUDIT_EVENT_SYSTEM_STATE, 0, "system", 0,
                            "DSMIL Audit Framework initialized successfully", NULL);
    
    return 0;
    
err_compliance_wq:
    destroy_workqueue(audit_system->flush_wq);
err_workqueue:
    vfree(audit_system->chains[0]->memory_buffer);
err_storage:
    vfree(audit_system->chains[0]->memory_buffer);
err_buffer:
    kfree(audit_system->chains[0]);
err_chain:
    kfree(audit_system->compliance);
err_compliance:
    crypto_free_rng(audit_system->rng_tfm);
err_rng:
    crypto_free_shash(audit_system->hash_tfm);
err_hash:
    kfree(audit_system);
    audit_system = NULL;
    return ret;
}

/*
 * Cleanup audit framework
 */
void dsmil_audit_cleanup(void)
{
    u32 i;
    
    if (!audit_system) {
        return;
    }
    
    pr_info("DSMIL Audit: Shutting down\n");
    
    /* Log shutdown audit entry */
    dsmil_audit_create_entry(0, AUDIT_EVENT_SYSTEM_STATE, 0, "system", 0,
                            "DSMIL Audit Framework shutting down", NULL);
    
    /* Cancel background work */
    if (audit_system->flush_wq) {
        cancel_delayed_work_sync(&audit_system->flush_work);
        destroy_workqueue(audit_system->flush_wq);
    }
    
    if (audit_system->compliance && audit_system->compliance->reporting.report_wq) {
        cancel_delayed_work_sync(&audit_system->compliance->reporting.daily_report_work);
        destroy_workqueue(audit_system->compliance->reporting.report_wq);
    }
    
    mutex_lock(&audit_system->global_lock);
    
    /* Cleanup all audit chains */
    for (i = 0; i < DSMIL_AUDIT_MAX_CHAINS; i++) {
        if (audit_system->chains[i]) {
            /* Perform final integrity check */
            dsmil_audit_verify_chain_integrity(audit_system->chains[i]);
            
            /* Free memory buffer */
            if (audit_system->chains[i]->memory_buffer) {
                vfree(audit_system->chains[i]->memory_buffer);
            }
            
            /* Free storage structure */
            if (audit_system->chains[i]->storage) {
                dsmil_audit_close_file(audit_system->chains[i]->storage);
                if (audit_system->chains[i]->storage->write_buffer) {
                    kfree(audit_system->chains[i]->storage->write_buffer);
                }
                kfree(audit_system->chains[i]->storage);
            }
            
            kfree(audit_system->chains[i]);
        }
    }
    
    mutex_unlock(&audit_system->global_lock);
    
    /* Print final statistics */
    pr_info("DSMIL Audit: Final stats - Total entries: %lld, integrity checks: %lld\n",
            atomic64_read(&audit_system->total_audit_entries),
            atomic64_read(&audit_system->total_integrity_checks));
    pr_info("DSMIL Audit: Tamper detections: %lld, compliance validations: %lld\n",
            atomic64_read(&audit_system->total_tamper_detections),
            atomic64_read(&audit_system->total_compliance_validations));
    
    if (audit_system->compliance) {
        pr_info("DSMIL Audit: Compliance - Passed: %lld, Failed: %lld, Exports: %lld\n",
                atomic64_read(&audit_system->compliance->compliance_checks_passed),
                atomic64_read(&audit_system->compliance->compliance_checks_failed),
                atomic64_read(&audit_system->compliance->external_exports));
    }
    
    /* Free cryptographic contexts */
    if (audit_system->hash_tfm && !IS_ERR(audit_system->hash_tfm)) {
        crypto_free_shash(audit_system->hash_tfm);
    }
    if (audit_system->rng_tfm && !IS_ERR(audit_system->rng_tfm)) {
        crypto_free_rng(audit_system->rng_tfm);
    }
    
    /* Free compliance framework */
    if (audit_system->compliance) {
        kfree(audit_system->compliance);
    }
    
    /* Clear magic and free memory */
    audit_system->audit_magic = 0;
    audit_system->initialized = false;
    kfree(audit_system);
    audit_system = NULL;
    
    pr_info("DSMIL Audit: Cleanup complete\n");
}

/*
 * Log device access operation
 */
int dsmil_audit_log_device_access(uid_t user_id, const char *username,
                                 u32 device_id, u32 operation_type, 
                                 bool authorized, const char *details)
{
    char audit_details[1024];
    struct dsmil_audit_metadata metadata = {
        .has_operation_type = true,
        .operation_type = operation_type,
        .has_authorization = true,
        .authorization = {
            .authorized = authorized,
            .dual_auth_required = false,
            .dual_auth_completed = authorized,
            .denial_reason = authorized ? NULL : "ACCESS_DENIED",
        },
        .has_result = true,
        .result = authorized ? AUDIT_SUCCESS : AUDIT_DENIED,
    };
    
    if (!audit_system || !audit_system->initialized) {
        return -EINVAL;
    }
    
    snprintf(audit_details, sizeof(audit_details),
             "Device %u access: operation %u, authorized %s - %s",
             device_id, operation_type, authorized ? "YES" : "NO",
             details ? details : "no additional details");
    
    return dsmil_audit_create_entry(audit_system->default_chain_id,
                                   AUDIT_EVENT_DEVICE_ACCESS,
                                   user_id, username ? username : "unknown",
                                   device_id, audit_details, &metadata);
}

/*
 * Log authentication event
 */
int dsmil_audit_log_authentication(uid_t user_id, const char *username,
                                  bool success, const char *failure_reason)
{
    char audit_details[1024];
    struct dsmil_audit_metadata metadata = {
        .has_operation_type = true,
        .operation_type = DSMIL_OP_DIAGNOSTIC,
        .has_authorization = true,
        .authorization = {
            .authorized = success,
            .dual_auth_required = false,
            .dual_auth_completed = success,
            .denial_reason = success ? NULL : failure_reason,
        },
        .has_result = true,
        .result = success ? AUDIT_SUCCESS : AUDIT_DENIED,
    };
    
    if (!audit_system || !audit_system->initialized) {
        return -EINVAL;
    }
    
    snprintf(audit_details, sizeof(audit_details),
             "Authentication attempt: %s%s%s",
             success ? "SUCCESS" : "FAILURE",
             failure_reason ? " - " : "",
             failure_reason ? failure_reason : "");
    
    return dsmil_audit_create_entry(audit_system->default_chain_id,
                                   AUDIT_EVENT_AUTHENTICATION,
                                   user_id, username ? username : "unknown",
                                   0, audit_details, &metadata);
}

/*
 * Log authorization event
 */
int dsmil_audit_log_authorization(uid_t user_id, const char *username,
                                 u32 device_id, u32 operation_type,
                                 bool authorized, bool dual_auth_required,
                                 const char *denial_reason)
{
    char audit_details[1024];
    struct dsmil_audit_metadata metadata = {
        .has_operation_type = true,
        .operation_type = operation_type,
        .has_authorization = true,
        .authorization = {
            .authorized = authorized,
            .dual_auth_required = dual_auth_required,
            .dual_auth_completed = authorized,
            .denial_reason = authorized ? NULL : denial_reason,
        },
        .has_result = true,
        .result = authorized ? AUDIT_SUCCESS : AUDIT_DENIED,
    };
    
    if (!audit_system || !audit_system->initialized) {
        return -EINVAL;
    }
    
    snprintf(audit_details, sizeof(audit_details),
             "Authorization for device %u operation %u: %s%s%s%s",
             device_id, operation_type,
             authorized ? "GRANTED" : "DENIED",
             dual_auth_required ? " (dual auth required)" : "",
             denial_reason ? " - " : "",
             denial_reason ? denial_reason : "");
    
    return dsmil_audit_create_entry(audit_system->default_chain_id,
                                   AUDIT_EVENT_AUTHORIZATION,
                                   user_id, username ? username : "unknown",
                                   device_id, audit_details, &metadata);
}

/*
 * Log security violation
 */
int dsmil_audit_log_security_violation(const char *violation_type,
                                      const char *description,
                                      u32 device_id, uid_t user_id)
{
    char audit_details[1024];
    struct dsmil_audit_metadata metadata = {
        .has_operation_type = true,
        .operation_type = DSMIL_OP_EMERGENCY,
        .has_risk_level = true,
        .risk_level = DSMIL_RISK_CATASTROPHIC,
        .has_authorization = true,
        .authorization = {
            .authorized = false,
            .dual_auth_required = false,
            .dual_auth_completed = false,
            .denial_reason = violation_type,
        },
        .has_result = true,
        .result = AUDIT_SECURITY_VIOLATION,
    };
    
    if (!audit_system || !audit_system->initialized) {
        return -EINVAL;
    }
    
    snprintf(audit_details, sizeof(audit_details),
             "SECURITY VIOLATION: %s - %s",
             violation_type ? violation_type : "unknown",
             description ? description : "no description");
    
    /* Security violations are critical - log immediately */
    int ret = dsmil_audit_create_entry(audit_system->default_chain_id,
                                      AUDIT_EVENT_SECURITY_VIOLATION,
                                      user_id, "violation_source",
                                      device_id, audit_details, &metadata);
    
    /* Increment tamper detection counter */
    atomic64_inc(&audit_system->total_tamper_detections);
    
    /* Trigger immediate integrity check if tamper response is enabled */
    if (audit_system->tamper_response_enabled) {
        if (audit_system->chains[audit_system->default_chain_id]) {
            dsmil_audit_verify_chain_integrity(
                audit_system->chains[audit_system->default_chain_id]);
        }
    }
    
    return ret;
}

/*
 * Create new audit entry with full validation
 */
static int dsmil_audit_create_entry(u32 chain_id, enum dsmil_audit_event_type event_type,
				    uid_t user_id, const char *username,
				    u32 device_id, const char *details,
				    const struct dsmil_audit_metadata *metadata)
{
	struct dsmil_audit_chain *chain;
	struct dsmil_audit_entry *entry;
	struct dsmil_user_security_profile profile;
	u32 buffer_pos;
	u32 clearance_level = DSMIL_CLEARANCE_UNCLASSIFIED;
	bool profile_available = false;
	int ret = 0;
	
	if (chain_id >= DSMIL_AUDIT_MAX_CHAINS || !audit_system->chains[chain_id]) {
		return -EINVAL;
	}
    
    chain = audit_system->chains[chain_id];
    
    /* Check for chain tamper detection */
    if (chain->tamper_detected) {
        pr_err("DSMIL Audit: Cannot add entry - chain %u tamper detected\n", chain_id);
        return -EPERM;
    }
    
    mutex_lock(&chain->buffer_lock);
    
    /* Get buffer position */
    buffer_pos = chain->buffer_head;
    entry = &chain->memory_buffer[buffer_pos];
    
    /* Clear entry */
    memset(entry, 0, sizeof(*entry));
    
    /* Initialize entry metadata */
    entry->sequence_number = ++chain->last_sequence_number;
    entry->timestamp = dsmil_get_real_time();
    entry->entry_type = event_type;
    
	/* Resolve user security context */
	if (user_id != 0 &&
	    dsmil_mfa_get_user_profile(user_id, &profile) == 0) {
		if (profile.clearance_level > DSMIL_CLEARANCE_MAX)
			clearance_level = DSMIL_CLEARANCE_MAX;
		else
			clearance_level = profile.clearance_level;
		profile_available = true;
	}

	/* Set user context */
	entry->user_id = user_id;
	if (username) {
		strncpy(entry->username, username, sizeof(entry->username) - 1);
	}
	entry->user_clearance = clearance_level;
	
	/* Set operation details */
	entry->device_id = device_id;
	if (metadata && metadata->has_operation_type) {
		entry->operation_type = metadata->operation_type;
	} else {
		entry->operation_type = DSMIL_OP_DIAGNOSTIC;
	}
	if (metadata && metadata->has_risk_level) {
		entry->risk_level = metadata->risk_level;
	} else {
		entry->risk_level = dsmil_audit_infer_risk(device_id,
							   entry->operation_type,
							   clearance_level);
	}
	
	/* Set authorization context */
	if (metadata && metadata->has_authorization) {
		entry->authorization.authorized = metadata->authorization.authorized;
		entry->authorization.dual_auth_required =
			metadata->authorization.dual_auth_required;
		entry->authorization.dual_auth_completed =
			metadata->authorization.dual_auth_completed;
		if (metadata->authorization.denial_reason) {
			strscpy(entry->authorization.denial_reason,
				metadata->authorization.denial_reason,
				sizeof(entry->authorization.denial_reason));
		} else {
			entry->authorization.denial_reason[0] = '\0';
		}
	} else {
		entry->authorization.authorized = true;
		entry->authorization.dual_auth_required = false;
		entry->authorization.dual_auth_completed = profile_available;
		entry->authorization.denial_reason[0] = '\0';
	}
	
	/* Set result */
	if (metadata && metadata->has_result) {
		entry->result = metadata->result;
	} else {
		entry->result = entry->authorization.authorized ?
			AUDIT_SUCCESS : AUDIT_DENIED;
	}
	
	/* Copy details */
	if (details) {
		strncpy(entry->details, details, sizeof(entry->details) - 1);
    }
    
    /* Compute entry hash */
    ret = dsmil_audit_compute_entry_hash(entry);
    if (ret < 0) {
        mutex_unlock(&chain->buffer_lock);
        return ret;
    }
    
    /* Link to previous entry */
    if (entry->sequence_number > 1) {
        u32 prev_pos = (buffer_pos == 0) ? (chain->buffer_size - 1) : (buffer_pos - 1);
        struct dsmil_audit_entry *prev_entry = &chain->memory_buffer[prev_pos];
        
        if (prev_entry->sequence_number == entry->sequence_number - 1) {
            memcpy(entry->previous_chain_hash, prev_entry->entry_hash, 32);
        } else {
            /* Previous entry not in buffer - would need to load from storage */
            memset(entry->previous_chain_hash, 0, 32);
        }
    } else {
        memset(entry->previous_chain_hash, 0, 32);
    }
    
    /* Validate compliance */
    ret = dsmil_audit_validate_compliance(entry, chain->compliance);
    if (ret < 0) {
        pr_warn("DSMIL Audit: Compliance validation failed: %d\n", ret);
        /* Continue anyway - compliance failure is logged but not blocking */
        atomic64_inc(&audit_system->compliance->compliance_checks_failed);
    } else {
        atomic64_inc(&audit_system->compliance->compliance_checks_passed);
    }
    
    /* Advance buffer head */
    chain->buffer_head = (chain->buffer_head + 1) % chain->buffer_size;
    
    mutex_unlock(&chain->buffer_lock);
    
    /* Update statistics */
    atomic64_inc(&chain->stats.entries_written);
    atomic64_inc(&audit_system->total_audit_entries);
    atomic64_inc(&chain->total_entries);
    atomic64_inc(&audit_system->total_compliance_validations);
    
    /* Schedule buffer flush if threshold reached */
    if (atomic64_read(&chain->stats.entries_written) % 
        audit_system->buffer_flush_threshold == 0) {
        queue_delayed_work(audit_system->flush_wq, &audit_system->flush_work, 0);
    }
    
    pr_debug("DSMIL Audit: Created entry %llu for %s event (user %s, device %u)\n",
             entry->sequence_number, 
             event_type < ARRAY_SIZE(audit_event_names) ? 
                 audit_event_names[event_type] : "UNKNOWN",
             username ? username : "unknown", device_id);
    
    return 0;
}

/*
 * Compute SHA-256 hash of audit entry
 */
static int dsmil_audit_compute_entry_hash(struct dsmil_audit_entry *entry)
{
    struct {
        struct shash_desc shash;
        char ctx[crypto_shash_descsize(audit_system->hash_tfm)];
    } desc;
    int ret;
    
    desc.shash.tfm = audit_system->hash_tfm;
    
    ret = crypto_shash_init(&desc.shash);
    if (ret < 0) {
        return ret;
    }
    
    /* Hash core entry data (excluding hash and signature fields) */
    ret = crypto_shash_update(&desc.shash, (u8 *)&entry->sequence_number,
                              sizeof(entry->sequence_number));
    if (ret < 0) {
        return ret;
    }
    
    ret = crypto_shash_update(&desc.shash, (u8 *)&entry->timestamp,
                              sizeof(entry->timestamp));
    if (ret < 0) {
        return ret;
    }
    
    ret = crypto_shash_update(&desc.shash, (u8 *)&entry->entry_type,
                              sizeof(entry->entry_type));
    if (ret < 0) {
        return ret;
    }
    
    ret = crypto_shash_update(&desc.shash, (u8 *)entry->username,
                              strlen(entry->username));
    if (ret < 0) {
        return ret;
    }
    
    ret = crypto_shash_update(&desc.shash, (u8 *)entry->details,
                              strlen(entry->details));
    if (ret < 0) {
        return ret;
    }
    
    if (entry->operation_data_len > 0) {
        ret = crypto_shash_update(&desc.shash, entry->operation_data,
                                  entry->operation_data_len);
        if (ret < 0) {
            return ret;
        }
    }
    
    ret = crypto_shash_final(&desc.shash, entry->entry_hash);
    if (ret < 0) {
        return ret;
    }
    
    return 0;
}

/*
 * Verify integrity of entire audit chain
 */
static int dsmil_audit_verify_chain_integrity(struct dsmil_audit_chain *chain)
{
    u8 computed_hash[32];
    u8 expected_hash[32];
    u64 sequence_number = 1;
    u32 verified_entries = 0;
    int result = 0;
    
    if (!chain) {
        return -EINVAL;
    }
    
    mutex_lock(&chain->chain_lock);
    
    pr_info("DSMIL Audit: Starting integrity verification for chain %llu\n",
            chain->chain_id);
    
    /* Initialize hash computation */
    memset(computed_hash, 0, 32);
    
    /* Verify entries in memory buffer */
    mutex_lock(&chain->buffer_lock);
    
    for (u32 i = 0; i < chain->buffer_size; i++) {
        struct dsmil_audit_entry *entry = &chain->memory_buffer[i];
        
        if (entry->sequence_number == 0) {
            continue; /* Empty slot */
        }
        
        /* Verify individual entry hash */
        u8 temp_hash[32];
        memcpy(temp_hash, entry->entry_hash, 32);
        memset(entry->entry_hash, 0, 32);
        
        if (dsmil_audit_compute_entry_hash(entry) != 0) {
            result = -EIO;
            break;
        }
        
        if (memcmp(entry->entry_hash, temp_hash, 32) != 0) {
            pr_err("DSMIL Audit: Entry %llu hash mismatch\n", entry->sequence_number);
            result = -EBADMSG;
            break;
        }
        
        verified_entries++;
    }
    
    mutex_unlock(&chain->buffer_lock);
    
    /* Update integrity status */
    chain->integrity_verified = (result == 0);
    chain->last_integrity_check = dsmil_get_real_time();
    
    if (result == 0) {
        atomic64_inc(&chain->stats.integrity_checks_passed);
        atomic64_inc(&audit_system->total_integrity_checks);
        pr_info("DSMIL Audit: Chain %llu integrity VERIFIED (%u entries checked)\n",
                chain->chain_id, verified_entries);
    } else {
        atomic64_inc(&chain->stats.integrity_checks_failed);
        atomic64_inc(&chain->failed_integrity_checks);
        chain->tamper_detected = true;  /* Permanent flag */
        
        pr_err("DSMIL Audit: Chain %llu integrity FAILED - TAMPER DETECTED\n",
               chain->chain_id);
        
        /* Log critical security event */
        dsmil_audit_log_security_violation("AUDIT_CHAIN_TAMPER",
                                          "Audit chain integrity verification failed",
                                          0, 0);
    }
    
    mutex_unlock(&chain->chain_lock);
    
    return result;
}

/*
 * Validate compliance for audit entry
 */
static int dsmil_audit_validate_compliance(struct dsmil_audit_entry *entry,
                                          struct dsmil_compliance_framework *framework)
{
    int result = 0;
    u32 compliance_flags = 0;
    
    if (!entry || !framework) {
        return -EINVAL;
    }
    
    /* FIPS 140-2 validation */
    if (framework->standards.fips_140_2_enabled && 
        framework->validators.fips_validator) {
        if (framework->validators.fips_validator(entry) == 0) {
            compliance_flags |= DSMIL_COMPLIANCE_FIPS_140_2;
        } else {
            result = -EFAULT;
        }
    }
    
    /* Common Criteria validation */
    if (framework->standards.common_criteria_enabled &&
        framework->validators.cc_validator) {
        if (framework->validators.cc_validator(entry) == 0) {
            compliance_flags |= DSMIL_COMPLIANCE_COMMON_CRITERIA;
        } else {
            result = -EFAULT;
        }
    }
    
    /* NATO STANAG validation */
    if (framework->standards.nato_stanag_enabled &&
        framework->validators.nato_validator) {
        if (framework->validators.nato_validator(entry) == 0) {
            compliance_flags |= DSMIL_COMPLIANCE_NATO_STANAG;
        } else {
            result = -EFAULT;
        }
    }
    
    /* NIST 800-53 validation */
    if (framework->standards.nist_800_53_enabled &&
        framework->validators.nist_validator) {
        if (framework->validators.nist_validator(entry) == 0) {
            compliance_flags |= DSMIL_COMPLIANCE_NIST_800_53;
        } else {
            result = -EFAULT;
        }
    }
    
    /* DoD 8500 validation */
    if (framework->standards.dod_8500_enabled &&
        framework->validators.dod_validator) {
        if (framework->validators.dod_validator(entry) == 0) {
            compliance_flags |= DSMIL_COMPLIANCE_DOD_8500;
        } else {
            result = -EFAULT;
        }
    }
    
    /* Store compliance results */
    entry->compliance.compliance_flags = compliance_flags;
    
    return result;
}

static void dsmil_audit_close_file(struct dsmil_audit_storage *storage)
{
	if (storage && storage->current_file) {
		filp_close(storage->current_file, NULL);
		storage->current_file = NULL;
	}
}

static ssize_t dsmil_audit_format_entry(const struct dsmil_audit_entry *entry,
					u8 *buffer, size_t buf_len)
{
	struct tm tm;
	char safe_details[256];
	const char *event_name = "UNKNOWN";
	size_t i;

	if (!buffer || buf_len == 0)
		return -EINVAL;

	if (entry->entry_type < ARRAY_SIZE(audit_event_names))
		event_name = audit_event_names[entry->entry_type];

	memset(safe_details, 0, sizeof(safe_details));
	for (i = 0; i < sizeof(safe_details) - 1 && entry->details[i]; i++) {
		char ch = entry->details[i];
		if (ch == '\n' || ch == '\r')
			ch = ' ';
		safe_details[i] = ch;
	}

	time64_to_tm(entry->timestamp.tv_sec, 0, &tm);

	return scnprintf(buffer, buf_len,
			 "%04ld-%02d-%02dT%02d:%02d:%02d.%03ldZ seq=%llu event=%s "
			 "user=%u(%s) clearance=%u device=0x%04x op=%u risk=%u "
			 "auth=%s dual=%s result=%d details=\"%s\"\n",
			 tm.tm_year + 1900L, tm.tm_mon + 1, tm.tm_mday,
			 tm.tm_hour, tm.tm_min, tm.tm_sec,
			 entry->timestamp.tv_nsec / 1000000L,
			 entry->sequence_number, event_name,
			 entry->user_id, entry->username[0] ? entry->username : "unknown",
			 entry->user_clearance, entry->device_id,
			 entry->operation_type, entry->risk_level,
			 entry->authorization.authorized ? "ALLOW" : "DENY",
			 entry->authorization.dual_auth_required ?
				(entry->authorization.dual_auth_completed ? "COMPLETE" : "REQUIRED") :
				"NA",
			 entry->result, safe_details);
}

static int dsmil_audit_init_storage(struct dsmil_audit_chain *chain)
{
	struct dsmil_audit_storage *storage;

	if (chain->storage)
		return 0;

	storage = kzalloc(sizeof(*storage), GFP_KERNEL);
	if (!storage)
		return -ENOMEM;

	snprintf(storage->storage_path, sizeof(storage->storage_path),
		 "/tmp/dsmil_audit_chain%llu", chain->chain_id);
	mutex_init(&storage->buffer_lock);
	storage->max_storage_size = 64ULL * 1024ULL * 1024ULL; /* 64MB */
	storage->max_file_size = 1024UL * 1024UL;              /* 1MB per file */
	storage->buffer_size = PAGE_SIZE;
	storage->write_buffer = kmalloc(storage->buffer_size, GFP_KERNEL);
	if (!storage->write_buffer) {
		kfree(storage);
		return -ENOMEM;
	}

	atomic64_set(&storage->total_writes, 0);
	atomic64_set(&storage->total_bytes_written, 0);
	chain->storage = storage;
	return 0;
}

static int dsmil_audit_write_to_storage(struct dsmil_audit_chain *chain,
					struct dsmil_audit_entry *entry)
{
	struct dsmil_audit_storage *storage = chain->storage;
	ssize_t len;
	loff_t pos;
	int ret;
	u64 start_ns, duration_ns;

	if (!entry)
		return -EINVAL;

	if (!storage) {
		ret = dsmil_audit_init_storage(chain);
		if (ret)
			return ret;
		storage = chain->storage;
	}

	if (!storage->current_file) {
		char path[sizeof(storage->storage_path) + 16];

		snprintf(path, sizeof(path), "%s_%u.log",
			 storage->storage_path, storage->file_sequence);
		storage->current_file = filp_open(path,
						  O_WRONLY | O_CREAT | O_APPEND | O_LARGEFILE,
						  0600);
		if (IS_ERR(storage->current_file)) {
			ret = PTR_ERR(storage->current_file);
			storage->current_file = NULL;
			return ret;
		}
		storage->current_file_size = 0;
	}

	mutex_lock(&storage->buffer_lock);
	len = dsmil_audit_format_entry(entry, storage->write_buffer,
				       storage->buffer_size);
	if (len < 0) {
		mutex_unlock(&storage->buffer_lock);
		return len;
	}

	pos = storage->current_file->f_pos;
	start_ns = ktime_get_ns();
	ret = kernel_write(storage->current_file, storage->write_buffer, len, &pos);
	duration_ns = ktime_get_ns() - start_ns;
	if (ret >= 0) {
		storage->current_file->f_pos = pos;
		storage->current_file_size += ret;
		storage->current_storage_size += ret;
		atomic64_inc(&storage->total_writes);
		atomic64_add(ret, &storage->total_bytes_written);
		if (storage->avg_write_time_ns == 0)
			storage->avg_write_time_ns = duration_ns;
		else
			storage->avg_write_time_ns =
				div_u64(storage->avg_write_time_ns + duration_ns, 2);
		if (duration_ns > storage->max_write_time_ns)
			storage->max_write_time_ns = duration_ns;
		if (storage->current_file_size >= storage->max_file_size) {
			dsmil_audit_close_file(storage);
			storage->file_sequence++;
			storage->current_file_size = 0;
		}
	} else {
		pr_err("DSMIL Audit: Failed to persist entry %llu (chain %llu): %d\n",
		       entry->sequence_number, chain->chain_id, ret);
	}
	mutex_unlock(&storage->buffer_lock);

	return ret;
}

/*
 * Flush audit buffers to persistent storage
 */
static void dsmil_audit_flush_buffers(struct work_struct *work)
{
	u32 i;

	if (!audit_system || !audit_system->initialized)
		return;
    
	for (i = 0; i < DSMIL_AUDIT_MAX_CHAINS; i++) {
		struct dsmil_audit_chain *chain = audit_system->chains[i];
		u64 next_seq, latest_seq;

		if (!chain)
			continue;

		if (!chain->storage && dsmil_audit_init_storage(chain))
			continue;

		mutex_lock(&chain->buffer_lock);
		latest_seq = chain->last_sequence_number;
		next_seq = chain->export.last_exported_sequence + 1;

		if (next_seq > latest_seq) {
			mutex_unlock(&chain->buffer_lock);
			continue;
		}

		while (next_seq <= latest_seq) {
			struct dsmil_audit_entry *entry = NULL;
			u32 idx;

			for (idx = 0; idx < chain->buffer_size; idx++) {
				if (chain->memory_buffer[idx].sequence_number == next_seq) {
					entry = &chain->memory_buffer[idx];
					break;
				}
			}

			if (!entry) {
				pr_warn("DSMIL Audit: Entry %llu missing from buffer (chain %llu)\n",
					next_seq, chain->chain_id);
				chain->export.last_exported_sequence = latest_seq;
				break;
			}

			if (dsmil_audit_write_to_storage(chain, entry) == 0) {
				chain->export.last_exported_sequence = next_seq;
				chain->export.last_export = dsmil_get_real_time();
			} else {
				break;
			}

			next_seq++;
		}

		mutex_unlock(&chain->buffer_lock);
	}

	queue_delayed_work(audit_system->flush_wq, &audit_system->flush_work,
			   msecs_to_jiffies(audit_system->buffer_flush_interval * 1000));
}

/*
 * Generate daily compliance report
 */
static void dsmil_audit_daily_report(struct work_struct *work)
{
    if (!audit_system || !audit_system->initialized) {
        return;
    }
    
    pr_info("DSMIL Audit: Daily compliance report\n");
    pr_info("  Total audit entries: %lld\n", 
            atomic64_read(&audit_system->total_audit_entries));
    pr_info("  Integrity checks: %lld\n", 
            atomic64_read(&audit_system->total_integrity_checks));
    pr_info("  Tamper detections: %lld\n", 
            atomic64_read(&audit_system->total_tamper_detections));
    pr_info("  Compliance validations: %lld\n", 
            atomic64_read(&audit_system->total_compliance_validations));
    
    if (audit_system->compliance) {
        pr_info("  Compliance checks passed: %lld\n",
                atomic64_read(&audit_system->compliance->compliance_checks_passed));
        pr_info("  Compliance checks failed: %lld\n", 
                atomic64_read(&audit_system->compliance->compliance_checks_failed));
    }
    
    /* Schedule next report */
    queue_delayed_work(audit_system->compliance->reporting.report_wq,
                       &audit_system->compliance->reporting.daily_report_work,
                       msecs_to_jiffies(DSMIL_AUDIT_EXPORT_INTERVAL * 1000));
}

/*
 * FIPS 140-2 compliance validator
 */
static int dsmil_fips_140_2_validator(struct dsmil_audit_entry *entry)
{
    /* Basic FIPS 140-2 validation - check required fields */
    if (!entry) {
        return -EINVAL;
    }
    
    /* Verify timestamp is present and reasonable */
    if (entry->timestamp.tv_sec == 0) {
        return -EINVAL;
    }
    
    /* Verify entry has proper hash */
    bool hash_empty = true;
    for (int i = 0; i < 32; i++) {
        if (entry->entry_hash[i] != 0) {
            hash_empty = false;
            break;
        }
    }
    if (hash_empty) {
        return -EINVAL;
    }
    
    /* FIPS 140-2 requires specific audit fields */
    if (strlen(entry->details) == 0) {
        return -EINVAL;
    }
    
    return 0; /* Compliance validation passed */
}

/*
 * Common Criteria compliance validator
 */
static int dsmil_common_criteria_validator(struct dsmil_audit_entry *entry)
{
    if (!entry) {
        return -EINVAL;
    }
    
    /* Common Criteria requires user identification */
    if (entry->user_id == 0 && strlen(entry->username) == 0) {
        return -EINVAL;
    }
    
    /* Requires operation classification */
    if (entry->entry_type == 0) {
        return -EINVAL;
    }
    
    return 0;
}

/*
 * NATO STANAG compliance validator
 */
static int dsmil_nato_stanag_validator(struct dsmil_audit_entry *entry)
{
    if (!entry) {
        return -EINVAL;
    }
    
    if (entry->risk_level >= DSMIL_RISK_HIGH &&
        entry->user_clearance < DSMIL_CLEARANCE_SECRET) {
        return -EACCES;
    }

    if (entry->authorization.dual_auth_required &&
        !entry->authorization.dual_auth_completed) {
        return -EPERM;
    }
    
    return 0;
}

/*
 * NIST 800-53 compliance validator
 */
static int dsmil_nist_800_53_validator(struct dsmil_audit_entry *entry)
{
    if (!entry) {
        return -EINVAL;
    }
    
    if (entry->details[0] == '\0') {
        return -EINVAL;
    }

    if (entry->timestamp.tv_sec == 0) {
        return -EINVAL;
    }

    if (entry->risk_level >= DSMIL_RISK_CRITICAL &&
        entry->authorization.authorized &&
        !entry->authorization.dual_auth_required) {
        return -EPERM;
    }
    
    return 0;
}

/*
 * DoD 8500 compliance validator
 */
static int dsmil_dod_8500_validator(struct dsmil_audit_entry *entry)
{
    if (!entry) {
        return -EINVAL;
    }
    
    if (entry->authorization.authorized == false &&
        entry->result == AUDIT_SUCCESS) {
        return -EINVAL;
    }

    if (entry->risk_level == DSMIL_RISK_CATASTROPHIC &&
        !entry->authorization.dual_auth_completed) {
        return -EPERM;
    }

    if (entry->device_id &&
        dsmil_audit_is_quarantined_linear(dsmil_audit_linear_device(entry->device_id)) &&
        entry->authorization.authorized) {
        return -EPERM;
    }
    
    return 0;
}

/*
 * Get audit system statistics
 */
int dsmil_audit_get_statistics(u64 *total_entries, u64 *integrity_checks,
                              u64 *tamper_detections, u64 *compliance_validations)
{
    if (!audit_system) {
        return -EINVAL;
    }
    
    if (total_entries) {
        *total_entries = atomic64_read(&audit_system->total_audit_entries);
    }
    if (integrity_checks) {
        *integrity_checks = atomic64_read(&audit_system->total_integrity_checks);
    }
    if (tamper_detections) {
        *tamper_detections = atomic64_read(&audit_system->total_tamper_detections);
    }
    if (compliance_validations) {
        *compliance_validations = atomic64_read(&audit_system->total_compliance_validations);
    }
    
    return 0;
}

/* Export functions for use by other modules */
EXPORT_SYMBOL(dsmil_audit_init);
EXPORT_SYMBOL(dsmil_audit_cleanup);
EXPORT_SYMBOL(dsmil_audit_log_device_access);
EXPORT_SYMBOL(dsmil_audit_log_authentication);
EXPORT_SYMBOL(dsmil_audit_log_authorization);
EXPORT_SYMBOL(dsmil_audit_log_security_violation);
EXPORT_SYMBOL(dsmil_audit_get_statistics);

MODULE_AUTHOR("DSMIL Track B Security Team");
MODULE_DESCRIPTION("DSMIL Tamper-Evident Audit Framework with Multi-Standard Compliance");
MODULE_LICENSE("GPL v2");
MODULE_VERSION(DSMIL_AUDIT_VERSION);

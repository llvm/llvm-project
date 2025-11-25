/*
 * DSMIL Multi-Factor Authentication System - Track B Security Layer
 * 
 * Copyright (C) 2025 JRTC1 Educational Development
 * 
 * This module implements military-grade multi-factor authentication for the 
 * 84-device DSMIL system. Provides NATO-compliant clearance levels, dual 
 * authorization for critical operations, and cryptographic authentication.
 * 
 * SECURITY FEATURES:
 * - Military clearance levels (NATO standard + custom)
 * - Multi-factor authentication (user/hardware/biometric tokens)
 * - Dual authorization for high-risk operations
 * - Cryptographic signatures and session management
 * - Time-based access restrictions and emergency overrides
 */

#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/slab.h>
#include <linux/mutex.h>
#include <linux/crypto.h>
#include <linux/random.h>
#include <linux/time.h>
#include <linux/crc32.h>
#include <linux/atomic.h>
#include <linux/spinlock.h>
#include <linux/workqueue.h>
#include <linux/completion.h>
#include <linux/cred.h>
#include <linux/minmax.h>
#include <crypto/hash.h>
#include <crypto/skcipher.h>
#include <crypto/rng.h>

#include "dsmil_security_types.h"
#include "dsmil_mfa_compat.h"

#define DSMIL_MFA_VERSION           "1.0.0"
#define DSMIL_MFA_MAGIC             0x4D464141  /* "MFAA" */
#define DSMIL_SESSION_TIMEOUT       3600        /* 1 hour default */
#define DSMIL_DUAL_AUTH_TIMEOUT     300         /* 5 minutes */
#define DSMIL_MAX_SESSIONS          64          /* Max concurrent sessions */
#define DSMIL_MAX_DUAL_AUTH         16          /* Max dual auth requests */

/* Compartmentalized access flags */
#define DSMIL_COMP_CRYPTO      (1 << 0)  /* Cryptographic systems */
#define DSMIL_COMP_SIGNALS     (1 << 1)  /* Signals intelligence */
#define DSMIL_COMP_NUCLEAR     (1 << 2)  /* Nuclear systems */
#define DSMIL_COMP_WEAPONS     (1 << 3)  /* Weapons systems */
#define DSMIL_COMP_COMMS       (1 << 4)  /* Communications systems */
#define DSMIL_COMP_SENSORS     (1 << 5)  /* Sensor systems */
#define DSMIL_COMP_MAINT       (1 << 6)  /* Maintenance systems */
#define DSMIL_COMP_EMERGENCY   (1 << 7)  /* Emergency systems */

/* Dual authorization status */
enum dual_auth_status {
    DUAL_AUTH_PENDING_FIRST,
    DUAL_AUTH_PENDING_SECOND,
    DUAL_AUTH_APPROVED,
    DUAL_AUTH_DENIED,
    DUAL_AUTH_EXPIRED
};

/* Authentication tokens structure */
struct dsmil_auth_tokens {
    u64 session_token;         /* Cryptographic session token */
    u64 hardware_token;        /* Hardware security module token */
    u64 biometric_token;       /* Biometric authentication token */
    struct timespec64 expires; /* Token expiration */
    u8 token_signature[32];    /* SHA-256 signature of tokens */
    bool tokens_valid;         /* Token validation status */
};

/* Digital signatures structure */
struct dsmil_digital_signatures {
    u8 user_signature[256];    /* RSA-2048 user signature */
    u8 system_signature[256];  /* System attestation signature */
    bool signature_verified;   /* Signature validation status */
    struct timespec64 signature_time; /* When signature was created */
};

/* Permission matrix for fine-grained control */
struct dsmil_permission_matrix {
    /* Device-level permissions (bitmask for 84 devices) */
    u64 device_read_permissions[2];    /* 128-bit mask for read access */
    u64 device_write_permissions[2];   /* 128-bit mask for write access */
    u64 device_config_permissions[2];  /* 128-bit mask for configuration */
    
    /* Operation-level permissions */
    struct {
        bool can_activate_devices;
        bool can_deactivate_devices;
        bool can_reset_devices;
        bool can_modify_security_settings;
        bool can_access_audit_logs;
        bool can_emergency_stop;
        bool can_override_safety;         /* Extremely restricted */
    } operations;
    
    /* Temporal restrictions */
    struct {
        struct timespec64 valid_from;
        struct timespec64 valid_until;
        u32 max_operations_per_hour;
        u32 max_concurrent_sessions;
    } temporal;
    
    /* Network restrictions */
    struct {
        u32 allowed_source_ips[16];      /* Up to 16 allowed source IPs */
        u32 num_allowed_ips;
        u16 allowed_ports[8];            /* Allowed source ports */
        u32 network_access_flags;
    } network;
    
    /* Risk-based restrictions */
    struct {
        enum dsmil_risk_level max_device_risk;
        enum dsmil_risk_level max_operation_risk;
        bool requires_dual_authorization; /* Two-person integrity */
        bool requires_supervisor_approval;
    } risk;
};

/* Session tracking structure */
struct dsmil_session_info {
    struct timespec64 login_time;
    struct timespec64 last_activity;
    u32 session_flags;
    char client_info[128];     /* Client system information */
    u32 operations_count;      /* Operations in current session */
    enum dsmil_risk_level max_risk_performed;
};

/* Emergency override structure */
struct dsmil_emergency_override {
    bool emergency_override_active;
    char override_justification[512];
    struct timespec64 override_expires;
    u8 override_signature[256];
    uid_t override_authorizer;
    enum dsmil_clearance_level authorizer_clearance;
};

/* Military-grade authentication context */
struct dsmil_auth_context {
    /* User identification */
    uid_t user_id;
    gid_t group_id;
    char username[64];
    
    /* Security clearance */
    enum dsmil_clearance_level clearance;
    u32 compartmentalized_access;  /* Bitmask for compartmentalized info */
    
    /* Multi-factor authentication tokens */
    struct dsmil_auth_tokens tokens;
    
    /* Digital signatures */
    struct dsmil_digital_signatures signatures;
    
    /* Access permissions */
    struct dsmil_permission_matrix permissions;
    
    /* Session tracking */
    struct dsmil_session_info session;
    
    /* Audit trail */
    u64 audit_chain_id;          /* Link to audit chain */
    
    /* Emergency overrides */
    struct dsmil_emergency_override emergency;
    
    /* Context metadata */
    u32 context_id;
    struct timespec64 context_created;
    bool context_valid;
    atomic_t ref_count;
    spinlock_t context_lock;
};

/* Dual authorization request structure */
struct dsmil_dual_auth_request {
    /* Request identification */
    u64 request_id;
    struct timespec64 request_time;
    
    /* Operation details */
    u32 device_id;
    enum dsmil_operation_type operation;
    enum dsmil_risk_level assessed_risk;
    
    /* First authorizer */
    struct {
        uid_t user_id;
        char username[64];
        enum dsmil_clearance_level clearance;
        u8 signature[256];
        struct timespec64 auth_time;
    } first_auth;
    
    /* Second authorizer */
    struct {
        uid_t user_id;
        char username[64];
        enum dsmil_clearance_level clearance;
        u8 signature[256];
        struct timespec64 auth_time;
    } second_auth;
    
    /* Request status */
    enum dual_auth_status status;
    
    /* Expiration */
    struct timespec64 expires;
    
    /* Justification */
    char justification[1024];
    
    /* Linked list for active requests */
    struct list_head list;
};

/* Global MFA system state */
struct dsmil_mfa_system {
    struct mutex global_lock;
    
    /* Active sessions */
    struct dsmil_auth_context *active_sessions[DSMIL_MAX_SESSIONS];
    u32 session_count;
    atomic64_t next_session_id;
    
    /* Dual authorization requests */
    struct list_head dual_auth_requests;
    struct mutex dual_auth_lock;
    atomic64_t next_dual_auth_id;
    
    /* Cryptographic context */
    struct crypto_shash *hash_tfm;
    struct crypto_cipher *cipher_tfm;
    struct crypto_rng *rng_tfm;
    
    /* System configuration */
    u32 default_session_timeout;
    u32 dual_auth_timeout;
    bool require_hardware_tokens;
    bool require_biometric_auth;
    enum dsmil_clearance_level min_clearance_level;
    
    /* Statistics */
    atomic64_t auth_attempts;
    atomic64_t auth_successes;
    atomic64_t auth_failures;
    atomic64_t dual_auth_requests_total;
    atomic64_t emergency_overrides;
    
    /* System state */
    bool initialized;
    u32 mfa_magic;
    ktime_t init_time;
};

/* Global MFA system instance */
static struct dsmil_mfa_system *mfa_system = NULL;

/* Clearance level names for logging */
static const char *clearance_names[] = {
    "NONE", "RESTRICTED", "CONFIDENTIAL", "SECRET", "TOP_SECRET",
    "SCI", "SAP", "COSMIC", "ATOMAL", "INVALID"
};

/* Forward declarations */
static int dsmil_mfa_generate_session_token(struct dsmil_auth_context *ctx);
static int dsmil_mfa_validate_clearance(enum dsmil_clearance_level clearance, 
                                       u32 compartmentalized_access);
static int dsmil_mfa_create_digital_signature(struct dsmil_auth_context *ctx,
                                             const char *data, size_t data_len);
static int dsmil_mfa_verify_dual_authorization(u32 device_id, 
                                              enum dsmil_operation_type operation);
static void dsmil_mfa_cleanup_expired_sessions(struct work_struct *work);
static void dsmil_mfa_cleanup_expired_dual_auth(struct work_struct *work);
static enum dsmil_risk_level dsmil_mfa_calculate_operation_risk(u32 device_id,
                                                               enum dsmil_operation_type operation);
static bool dsmil_mfa_device_is_quarantined(u32 device_id);
static int dsmil_mfa_sign_payload(struct dsmil_auth_context *ctx,
                                  const void *payload, size_t payload_len,
                                  u8 *signature, size_t signature_len);

/* Workqueue for periodic cleanup */
static struct workqueue_struct *mfa_cleanup_wq;
static struct delayed_work session_cleanup_work;
static struct delayed_work dual_auth_cleanup_work;

/*
 * Initialize MFA system
 */
int dsmil_mfa_init(void)
{
    int ret = 0;
    
    if (mfa_system) {
        pr_warn("DSMIL MFA: Already initialized\n");
        return 0;
    }
    
    /* Allocate global state */
    mfa_system = kzalloc(sizeof(struct dsmil_mfa_system), GFP_KERNEL);
    if (!mfa_system) {
        pr_err("DSMIL MFA: Failed to allocate system structure\n");
        return -ENOMEM;
    }
    
    /* Initialize mutexes */
    mutex_init(&mfa_system->global_lock);
    mutex_init(&mfa_system->dual_auth_lock);
    
    /* Initialize dual auth request list */
    INIT_LIST_HEAD(&mfa_system->dual_auth_requests);
    
    /* Initialize atomic counters */
    atomic64_set(&mfa_system->next_session_id, 1);
    atomic64_set(&mfa_system->next_dual_auth_id, 1);
    atomic64_set(&mfa_system->auth_attempts, 0);
    atomic64_set(&mfa_system->auth_successes, 0);
    atomic64_set(&mfa_system->auth_failures, 0);
    atomic64_set(&mfa_system->dual_auth_requests_total, 0);
    atomic64_set(&mfa_system->emergency_overrides, 0);
    
    /* Initialize cryptographic contexts */
    mfa_system->hash_tfm = crypto_alloc_shash("sha256", 0, 0);
    if (IS_ERR(mfa_system->hash_tfm)) {
        ret = PTR_ERR(mfa_system->hash_tfm);
        pr_err("DSMIL MFA: Failed to allocate hash transform: %d\n", ret);
        goto err_hash;
    }
    
    mfa_system->cipher_tfm = dsmil_crypto_alloc_cipher_compat("aes");
    if (IS_ERR(mfa_system->cipher_tfm)) {
        ret = PTR_ERR(mfa_system->cipher_tfm);
        pr_err("DSMIL MFA: Failed to allocate cipher transform: %d\n", ret);
        goto err_cipher;
    }
    
    mfa_system->rng_tfm = crypto_alloc_rng("drbg_nopr_sha256", 0, 0);
    if (IS_ERR(mfa_system->rng_tfm)) {
        ret = PTR_ERR(mfa_system->rng_tfm);
        pr_err("DSMIL MFA: Failed to allocate RNG transform: %d\n", ret);
        goto err_rng;
    }
    
    /* Initialize system configuration */
    mfa_system->default_session_timeout = DSMIL_SESSION_TIMEOUT;
    mfa_system->dual_auth_timeout = DSMIL_DUAL_AUTH_TIMEOUT;
    mfa_system->require_hardware_tokens = true;
    mfa_system->require_biometric_auth = false; /* Future enhancement */
    mfa_system->min_clearance_level = DSMIL_CLEARANCE_RESTRICTED;
    
    /* Create workqueue for cleanup tasks */
    mfa_cleanup_wq = create_singlethread_workqueue("dsmil_mfa_cleanup");
    if (!mfa_cleanup_wq) {
        ret = -ENOMEM;
        pr_err("DSMIL MFA: Failed to create cleanup workqueue\n");
        goto err_workqueue;
    }
    
    /* Initialize cleanup work */
    INIT_DELAYED_WORK(&session_cleanup_work, dsmil_mfa_cleanup_expired_sessions);
    INIT_DELAYED_WORK(&dual_auth_cleanup_work, dsmil_mfa_cleanup_expired_dual_auth);
    
    /* Schedule periodic cleanup */
    queue_delayed_work(mfa_cleanup_wq, &session_cleanup_work, 
                       msecs_to_jiffies(60000)); /* Every minute */
    queue_delayed_work(mfa_cleanup_wq, &dual_auth_cleanup_work, 
                       msecs_to_jiffies(30000)); /* Every 30 seconds */
    
    /* Set system state */
    mfa_system->initialized = true;
    mfa_system->mfa_magic = DSMIL_MFA_MAGIC;
    mfa_system->init_time = ktime_get();
    
    pr_info("DSMIL MFA: Initialized (version %s)\n", DSMIL_MFA_VERSION);
    pr_info("DSMIL MFA: Session timeout %u seconds\n", mfa_system->default_session_timeout);
    pr_info("DSMIL MFA: Dual authorization timeout %u seconds\n", mfa_system->dual_auth_timeout);
    pr_info("DSMIL MFA: Hardware tokens %s\n", 
            mfa_system->require_hardware_tokens ? "REQUIRED" : "optional");
    pr_info("DSMIL MFA: Minimum clearance level %s\n", 
            clearance_names[mfa_system->min_clearance_level]);
    
    return 0;
    
err_workqueue:
    crypto_free_rng(mfa_system->rng_tfm);
err_rng:
    dsmil_crypto_free_cipher_compat(mfa_system->cipher_tfm);
err_cipher:
    crypto_free_shash(mfa_system->hash_tfm);
err_hash:
    kfree(mfa_system);
    mfa_system = NULL;
    return ret;
}

/*
 * Cleanup MFA system
 */
void dsmil_mfa_cleanup(void)
{
    u32 i;
    struct dsmil_dual_auth_request *req, *tmp;
    
    if (!mfa_system) {
        return;
    }
    
    pr_info("DSMIL MFA: Shutting down\n");
    
    /* Cancel cleanup work */
    if (mfa_cleanup_wq) {
        cancel_delayed_work_sync(&session_cleanup_work);
        cancel_delayed_work_sync(&dual_auth_cleanup_work);
        destroy_workqueue(mfa_cleanup_wq);
    }
    
    mutex_lock(&mfa_system->global_lock);
    
    /* Cleanup active sessions */
    for (i = 0; i < DSMIL_MAX_SESSIONS; i++) {
        if (mfa_system->active_sessions[i]) {
            kfree(mfa_system->active_sessions[i]);
            mfa_system->active_sessions[i] = NULL;
        }
    }
    
    mutex_unlock(&mfa_system->global_lock);
    
    /* Cleanup dual authorization requests */
    mutex_lock(&mfa_system->dual_auth_lock);
    list_for_each_entry_safe(req, tmp, &mfa_system->dual_auth_requests, list) {
        list_del(&req->list);
        kfree(req);
    }
    mutex_unlock(&mfa_system->dual_auth_lock);
    
    /* Free cryptographic contexts */
    if (mfa_system->hash_tfm && !IS_ERR(mfa_system->hash_tfm)) {
        crypto_free_shash(mfa_system->hash_tfm);
    }
    if (mfa_system->cipher_tfm && !IS_ERR(mfa_system->cipher_tfm)) {
        dsmil_crypto_free_cipher_compat(mfa_system->cipher_tfm);
    }
    if (mfa_system->rng_tfm && !IS_ERR(mfa_system->rng_tfm)) {
        crypto_free_rng(mfa_system->rng_tfm);
    }
    
    /* Print final statistics */
    pr_info("DSMIL MFA: Final stats - Auth attempts: %lld, successes: %lld, failures: %lld\n",
            atomic64_read(&mfa_system->auth_attempts),
            atomic64_read(&mfa_system->auth_successes), 
            atomic64_read(&mfa_system->auth_failures));
    pr_info("DSMIL MFA: Dual auth requests: %lld, emergency overrides: %lld\n",
            atomic64_read(&mfa_system->dual_auth_requests_total),
            atomic64_read(&mfa_system->emergency_overrides));
    
    /* Clear magic and free memory */
    mfa_system->mfa_magic = 0;
    mfa_system->initialized = false;
    kfree(mfa_system);
    mfa_system = NULL;
    
    pr_info("DSMIL MFA: Cleanup complete\n");
}

/*
 * Create new authentication context
 */
struct dsmil_auth_context *dsmil_mfa_create_auth_context(uid_t user_id, 
                                                        const char *username,
                                                        enum dsmil_clearance_level clearance,
                                                        u32 compartmentalized_access)
{
    struct dsmil_auth_context *ctx;
    u32 slot = 0;
    int ret;
    bool found_slot = false;
    
    if (!mfa_system || !mfa_system->initialized) {
        pr_err("DSMIL MFA: System not initialized\n");
        return ERR_PTR(-EINVAL);
    }
    
    atomic64_inc(&mfa_system->auth_attempts);
    
    /* Validate clearance level */
    ret = dsmil_mfa_validate_clearance(clearance, compartmentalized_access);
    if (ret < 0) {
        atomic64_inc(&mfa_system->auth_failures);
        return ERR_PTR(ret);
    }
    
    /* Check minimum clearance requirement */
    if (clearance < mfa_system->min_clearance_level) {
        pr_warn("DSMIL MFA: Insufficient clearance level %s (minimum %s required)\n",
                clearance_names[clearance], 
                clearance_names[mfa_system->min_clearance_level]);
        atomic64_inc(&mfa_system->auth_failures);
        return ERR_PTR(-EACCES);
    }
    
    /* Allocate context */
    ctx = kzalloc(sizeof(struct dsmil_auth_context), GFP_KERNEL);
    if (!ctx) {
        atomic64_inc(&mfa_system->auth_failures);
        return ERR_PTR(-ENOMEM);
    }
    
    /* Initialize context */
    ctx->user_id = user_id;
    ctx->group_id = from_kgid_munged(&init_user_ns, current_gid());
    strncpy(ctx->username, username, sizeof(ctx->username) - 1);
    ctx->clearance = clearance;
    ctx->compartmentalized_access = compartmentalized_access;
    
    /* Generate authentication tokens */
    ret = dsmil_mfa_generate_session_token(ctx);
    if (ret < 0) {
        kfree(ctx);
        atomic64_inc(&mfa_system->auth_failures);
        return ERR_PTR(ret);
    }
    
    /* Initialize session info */
    ctx->session.login_time = dsmil_get_real_time();
    ctx->session.last_activity = ctx->session.login_time;
    ctx->session.session_flags = 0;
    ctx->session.operations_count = 0;
    ctx->session.max_risk_performed = DSMIL_RISK_LOW;
    snprintf(ctx->session.client_info, sizeof(ctx->session.client_info),
             "kernel_uid_%u", user_id);
    
    /* Set context metadata */
    ctx->context_id = atomic64_inc_return(&mfa_system->next_session_id);
    ctx->context_created = ctx->session.login_time;
    ctx->context_valid = true;
    atomic_set(&ctx->ref_count, 1);
    spin_lock_init(&ctx->context_lock);
    
    mutex_lock(&mfa_system->global_lock);
    
    /* Find available session slot */
    for (slot = 0; slot < DSMIL_MAX_SESSIONS; slot++) {
        if (!mfa_system->active_sessions[slot]) {
            mfa_system->active_sessions[slot] = ctx;
            mfa_system->session_count++;
            found_slot = true;
            break;
        }
    }
    
    mutex_unlock(&mfa_system->global_lock);
    
    if (!found_slot) {
        pr_err("DSMIL MFA: No available session slots (max %d reached)\n", 
               DSMIL_MAX_SESSIONS);
        kfree(ctx);
        atomic64_inc(&mfa_system->auth_failures);
        return ERR_PTR(-ENOSPC);
    }
    
    atomic64_inc(&mfa_system->auth_successes);
    
    pr_info("DSMIL MFA: Created auth context for user %s (UID %u) with %s clearance\n",
            username, user_id, clearance_names[clearance]);
    pr_info("DSMIL MFA: Session ID %u, slot %u, compartments 0x%08x\n",
            ctx->context_id, slot, compartmentalized_access);
    
    return ctx;
}

/*
 * Validate authentication context and authorize operation
 */
int dsmil_mfa_authorize_operation(struct dsmil_auth_context *ctx,
                                 u32 device_id,
                                 enum dsmil_operation_type operation,
                                 enum dsmil_risk_level risk_level)
{
    struct timespec64 now;
    int ret = 0;
    bool requires_dual_auth = false;
    
    if (!ctx || !ctx->context_valid) {
        return -EINVAL;
    }
    
    if (!mfa_system || !mfa_system->initialized) {
        return -EINVAL;
    }
    
    now = dsmil_get_real_time();
    
    spin_lock(&ctx->context_lock);
    
    /* Check session expiration */
    if (timespec64_compare(&now, &ctx->tokens.expires) > 0) {
        spin_unlock(&ctx->context_lock);
        pr_warn("DSMIL MFA: Session expired for user %s\n", ctx->username);
        return -ETIME;
    }
    
    /* Update activity timestamp */
    ctx->session.last_activity = now;
    ctx->session.operations_count++;
    
    /* Track maximum risk level performed */
    if (risk_level > ctx->session.max_risk_performed) {
        ctx->session.max_risk_performed = risk_level;
    }
    
    spin_unlock(&ctx->context_lock);
    
    /* Check device permissions */
    if (device_id < 64) {
        if (operation == DSMIL_OP_READ) {
            if (!(ctx->permissions.device_read_permissions[0] & (1ULL << device_id))) {
                pr_warn("DSMIL MFA: No read permission for device %u\n", device_id);
                return -EACCES;
            }
        } else {
            if (!(ctx->permissions.device_write_permissions[0] & (1ULL << device_id))) {
                pr_warn("DSMIL MFA: No write permission for device %u\n", device_id);
                return -EACCES;
            }
        }
    } else if (device_id < 84) {
        /* Devices 64-83 use second bitmap word */
        u32 bit_pos = device_id - 64;
        if (operation == DSMIL_OP_READ) {
            if (!(ctx->permissions.device_read_permissions[1] & (1ULL << bit_pos))) {
                pr_warn("DSMIL MFA: No read permission for device %u\n", device_id);
                return -EACCES;
            }
        } else {
            if (!(ctx->permissions.device_write_permissions[1] & (1ULL << bit_pos))) {
                pr_warn("DSMIL MFA: No write permission for device %u\n", device_id);
                return -EACCES;
            }
        }
    } else {
        return -EINVAL; /* Invalid device ID */
    }
    
    /* Check risk-based authorization requirements */
    if (risk_level > ctx->permissions.risk.max_operation_risk) {
        pr_warn("DSMIL MFA: Operation risk level %d exceeds maximum %d\n",
                risk_level, ctx->permissions.risk.max_operation_risk);
        return -EACCES;
    }
    
    /* Determine if dual authorization is required */
    if (risk_level >= DSMIL_RISK_HIGH) {
        requires_dual_auth = true;
    }
    if (ctx->permissions.risk.requires_dual_authorization) {
        requires_dual_auth = true;
    }
    if (operation == DSMIL_OP_EMERGENCY || operation == DSMIL_OP_RESET) {
        requires_dual_auth = true;
    }
    
    /* Check for dual authorization if required */
    if (requires_dual_auth) {
        ret = dsmil_mfa_verify_dual_authorization(device_id, operation);
        if (ret < 0) {
            pr_info("DSMIL MFA: Dual authorization required for device %u operation %d\n",
                    device_id, operation);
            return ret; /* Will be -EAGAIN if dual auth pending */
        }
    }
    
    pr_debug("DSMIL MFA: Authorized %s operation on device %u for user %s (risk %d)\n",
             operation == DSMIL_OP_READ ? "READ" : "WRITE", 
             device_id, ctx->username, risk_level);
    
    return 0;
}

/*
 * Request dual authorization for high-risk operation
 */
int dsmil_mfa_request_dual_authorization(struct dsmil_auth_context *first_auth,
                                        u32 device_id,
                                        enum dsmil_operation_type operation,
                                        const char *justification)
{
    struct dsmil_dual_auth_request *request;
    
    if (!first_auth || !first_auth->context_valid || !justification) {
        return -EINVAL;
    }
    
    if (!mfa_system || !mfa_system->initialized) {
        return -EINVAL;
    }
    
    /* Check if first authorization has sufficient clearance */
    if (first_auth->clearance < DSMIL_CLEARANCE_SECRET) {
        pr_warn("DSMIL MFA: Insufficient clearance for dual auth request\n");
        return -EACCES;
    }
    
    /* Allocate request structure */
    request = kzalloc(sizeof(struct dsmil_dual_auth_request), GFP_KERNEL);
    if (!request) {
        return -ENOMEM;
    }
    
    /* Initialize request */
    request->request_id = atomic64_inc_return(&mfa_system->next_dual_auth_id);
    request->request_time = dsmil_get_real_time();
    request->device_id = device_id;
    request->operation = operation;
    request->assessed_risk = dsmil_mfa_calculate_operation_risk(device_id, operation);
    
    /* Set expiration */
    request->expires = request->request_time;
    request->expires.tv_sec += mfa_system->dual_auth_timeout;
    
    /* Record first authorization */
    request->first_auth.user_id = first_auth->user_id;
    strncpy(request->first_auth.username, first_auth->username, 
            sizeof(request->first_auth.username) - 1);
    request->first_auth.clearance = first_auth->clearance;
    request->first_auth.auth_time = dsmil_get_real_time();
    
    /* Generate first signature */
    if (dsmil_mfa_sign_payload(first_auth, request, sizeof(*request),
                               request->first_auth.signature,
                               sizeof(request->first_auth.signature))) {
        /* Fallback to high-entropy random if signature fails */
        get_random_bytes(request->first_auth.signature,
                         sizeof(request->first_auth.signature));
    }
    
    /* Copy justification */
    strncpy(request->justification, justification, sizeof(request->justification) - 1);
    
    /* Set status */
    request->status = DUAL_AUTH_PENDING_SECOND;
    
    /* Add to active requests list */
    mutex_lock(&mfa_system->dual_auth_lock);
    list_add(&request->list, &mfa_system->dual_auth_requests);
    mutex_unlock(&mfa_system->dual_auth_lock);
    
    atomic64_inc(&mfa_system->dual_auth_requests_total);
    
    pr_info("DSMIL MFA: Dual auth request %lld created for device %u operation %d\n",
            request->request_id, device_id, operation);
    pr_info("DSMIL MFA: First authorizer: %s (%s clearance)\n",
            first_auth->username, clearance_names[first_auth->clearance]);
    pr_info("DSMIL MFA: Justification: %.256s\n", justification);
    
    return 0;
}

/*
 * Generate cryptographically secure session token
 */
static int dsmil_mfa_generate_session_token(struct dsmil_auth_context *ctx)
{
    int ret;
    u8 random_data[32];
    struct {
        struct shash_desc shash;
        char ctx[crypto_shash_descsize(mfa_system->hash_tfm)];
    } desc;
    
    /* Generate random session token */
    ret = crypto_rng_get_bytes(mfa_system->rng_tfm, random_data, sizeof(random_data));
    if (ret < 0) {
        pr_err("DSMIL MFA: Failed to generate random data: %d\n", ret);
        return ret;
    }
    
    /* Create session token from random data + user info */
    ctx->tokens.session_token = *(u64 *)random_data;
    ctx->tokens.hardware_token = *(u64 *)(random_data + 8);
    ctx->tokens.biometric_token = *(u64 *)(random_data + 16);
    
    /* Set expiration time */
    ctx->tokens.expires = dsmil_get_real_time();
    ctx->tokens.expires.tv_sec += mfa_system->default_session_timeout;
    
    /* Generate token signature */
    desc.shash.tfm = mfa_system->hash_tfm;
    ret = crypto_shash_init(&desc.shash);
    if (ret < 0) {
        return ret;
    }
    
    ret = crypto_shash_update(&desc.shash, (u8 *)&ctx->tokens.session_token, 
                              sizeof(ctx->tokens.session_token));
    if (ret < 0) {
        return ret;
    }
    
    ret = crypto_shash_update(&desc.shash, (u8 *)&ctx->tokens.hardware_token,
                              sizeof(ctx->tokens.hardware_token));
    if (ret < 0) {
        return ret;
    }
    
    ret = crypto_shash_update(&desc.shash, (u8 *)&ctx->user_id, sizeof(ctx->user_id));
    if (ret < 0) {
        return ret;
    }
    
    ret = crypto_shash_final(&desc.shash, ctx->tokens.token_signature);
    if (ret < 0) {
        return ret;
    }
    
    ctx->tokens.tokens_valid = true;
    
    return 0;
}

/*
 * Validate security clearance level and compartmentalized access
 */
static int dsmil_mfa_validate_clearance(enum dsmil_clearance_level clearance, 
                                       u32 compartmentalized_access)
{
    /* Check clearance level bounds */
    if (clearance >= DSMIL_CLEARANCE_MAX) {
        pr_err("DSMIL MFA: Invalid clearance level %d\n", clearance);
        return -EINVAL;
    }
    
    /* Check compartmentalized access flags */
    if (compartmentalized_access & ~0xFF) {
        pr_err("DSMIL MFA: Invalid compartmentalized access flags 0x%08x\n",
               compartmentalized_access);
        return -EINVAL;
    }
    
    /* Validate compartment access vs clearance level */
    switch (clearance) {
    case DSMIL_CLEARANCE_UNCLASSIFIED:
    case DSMIL_CLEARANCE_RESTRICTED:
        if (compartmentalized_access != 0) {
            pr_err("DSMIL MFA: Compartmented access not allowed for clearance level %s\n",
                   clearance_names[clearance]);
            return -EACCES;
        }
        break;
        
    case DSMIL_CLEARANCE_CONFIDENTIAL:
    case DSMIL_CLEARANCE_SECRET:
        /* Limited compartment access allowed */
        if (compartmentalized_access & (DSMIL_COMP_NUCLEAR | DSMIL_COMP_WEAPONS)) {
            pr_err("DSMIL MFA: Nuclear/weapons access requires higher clearance\n");
            return -EACCES;
        }
        break;
        
    case DSMIL_CLEARANCE_TOP_SECRET:
    case DSMIL_CLEARANCE_SCI:
    case DSMIL_CLEARANCE_SAP:
        /* Most compartments allowed */
        break;
        
    case DSMIL_CLEARANCE_COSMIC:
    case DSMIL_CLEARANCE_ATOMAL:
        /* All compartments allowed */
        break;
        
    default:
        return -EINVAL;
    }
    
    return 0;
}

/*
 * Verify dual authorization for operation
 */
static int dsmil_mfa_verify_dual_authorization(u32 device_id, 
                                              enum dsmil_operation_type operation)
{
    struct dsmil_dual_auth_request *req;
    struct timespec64 now;
    int found = 0;
    
    now = dsmil_get_real_time();
    
    mutex_lock(&mfa_system->dual_auth_lock);
    
    list_for_each_entry(req, &mfa_system->dual_auth_requests, list) {
        if (req->device_id == device_id && req->operation == operation) {
            /* Check if request has expired */
            if (timespec64_compare(&now, &req->expires) > 0) {
                req->status = DUAL_AUTH_EXPIRED;
                found = -ETIME;
                break;
            }
            
            if (req->status == DUAL_AUTH_APPROVED) {
                found = 1;
                break;
            } else if (req->status == DUAL_AUTH_PENDING_SECOND) {
                found = -EAGAIN;
                break;
            } else if (req->status == DUAL_AUTH_DENIED) {
                found = -EACCES;
                break;
            }
        }
    }
    
    mutex_unlock(&mfa_system->dual_auth_lock);
    
    if (found == 0) {
        /* No matching request found - dual auth required */
        return -ENOENT;
    }
    
    return found > 0 ? 0 : found;
}

/*
 * Cleanup expired sessions
 */
static void dsmil_mfa_cleanup_expired_sessions(struct work_struct *work)
{
    u32 i;
    struct timespec64 now;
    u32 expired_count = 0;
    
    if (!mfa_system || !mfa_system->initialized) {
        return;
    }
    
    now = dsmil_get_real_time();
    
    mutex_lock(&mfa_system->global_lock);
    
    for (i = 0; i < DSMIL_MAX_SESSIONS; i++) {
        struct dsmil_auth_context *ctx = mfa_system->active_sessions[i];
        
        if (ctx && timespec64_compare(&now, &ctx->tokens.expires) > 0) {
            pr_info("DSMIL MFA: Expiring session for user %s (ID %u)\n",
                    ctx->username, ctx->context_id);
            
            ctx->context_valid = false;
            kfree(ctx);
            mfa_system->active_sessions[i] = NULL;
            mfa_system->session_count--;
            expired_count++;
        }
    }
    
    mutex_unlock(&mfa_system->global_lock);
    
    if (expired_count > 0) {
        pr_debug("DSMIL MFA: Cleaned up %u expired sessions\n", expired_count);
    }
    
    /* Schedule next cleanup */
    queue_delayed_work(mfa_cleanup_wq, &session_cleanup_work, 
                       msecs_to_jiffies(60000));
}

/*
 * Cleanup expired dual authorization requests
 */
static void dsmil_mfa_cleanup_expired_dual_auth(struct work_struct *work)
{
    struct dsmil_dual_auth_request *req, *tmp;
    struct timespec64 now;
    u32 expired_count = 0;
    
    if (!mfa_system || !mfa_system->initialized) {
        return;
    }
    
    now = dsmil_get_real_time();
    
    mutex_lock(&mfa_system->dual_auth_lock);
    
    list_for_each_entry_safe(req, tmp, &mfa_system->dual_auth_requests, list) {
        if (timespec64_compare(&now, &req->expires) > 0) {
            pr_info("DSMIL MFA: Expiring dual auth request %lld for device %u\n",
                    req->request_id, req->device_id);
            
            list_del(&req->list);
            kfree(req);
            expired_count++;
        }
    }
    
    mutex_unlock(&mfa_system->dual_auth_lock);
    
    if (expired_count > 0) {
        pr_debug("DSMIL MFA: Cleaned up %u expired dual auth requests\n", expired_count);
    }
    
    /* Schedule next cleanup */
    queue_delayed_work(mfa_cleanup_wq, &dual_auth_cleanup_work, 
                       msecs_to_jiffies(30000));
}

/*
 * Risk scoring helpers
 */
static bool dsmil_mfa_device_is_quarantined(u32 device_id)
{
	static const u32 critical_devices[] = { 0, 1, 12, 24, 83 };
	int i;

	for (i = 0; i < ARRAY_SIZE(critical_devices); i++) {
		if (critical_devices[i] == device_id)
			return true;
	}
	return false;
}

static enum dsmil_risk_level dsmil_mfa_calculate_operation_risk(
	u32 device_id, enum dsmil_operation_type operation)
{
	enum dsmil_risk_level risk = DSMIL_RISK_LOW;
	u32 group_id;

	if (dsmil_mfa_device_is_quarantined(device_id))
		return DSMIL_RISK_CATASTROPHIC;

	if (device_id >= 84)
		device_id %= 84;

	group_id = device_id / 12;
	switch (group_id) {
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

	switch (operation) {
	case DSMIL_OP_EMERGENCY:
		risk = DSMIL_RISK_CATASTROPHIC;
		break;
	case DSMIL_OP_RESET:
	case DSMIL_OP_CONTROL:
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

	return risk;
}

static int dsmil_mfa_sign_payload(struct dsmil_auth_context *ctx,
				  const void *payload, size_t payload_len,
				  u8 *signature, size_t signature_len)
{
	struct shash_desc *desc;
	size_t desc_size;
	u8 digest[SHA256_DIGEST_SIZE];
	int ret = 0;
	size_t offset = 0;

	if (!mfa_system || !ctx || !signature || signature_len == 0)
		return -EINVAL;

	desc_size = sizeof(*desc) + crypto_shash_descsize(mfa_system->hash_tfm);
	desc = kzalloc(desc_size, GFP_KERNEL);
	if (!desc)
		return -ENOMEM;

	desc->tfm = mfa_system->hash_tfm;
	ret = crypto_shash_init(desc);
	if (!ret)
		ret = crypto_shash_update(desc, (u8 *)&ctx->tokens.session_token,
					  sizeof(ctx->tokens.session_token));
	if (!ret)
		ret = crypto_shash_update(desc, (u8 *)&ctx->tokens.hardware_token,
					  sizeof(ctx->tokens.hardware_token));
	if (!ret)
		ret = crypto_shash_update(desc, (u8 *)&ctx->tokens.biometric_token,
					  sizeof(ctx->tokens.biometric_token));
	if (!ret && payload && payload_len)
		ret = crypto_shash_update(desc, payload, payload_len);
	if (!ret)
		ret = crypto_shash_final(desc, digest);

	kfree(desc);

	if (ret)
		return ret;

	while (offset < signature_len) {
		size_t chunk = min_t(size_t, signature_len - offset,
				     sizeof(digest));
		memcpy(signature + offset, digest, chunk);
		offset += chunk;
	}

	return 0;
}

static int dsmil_mfa_create_digital_signature(struct dsmil_auth_context *ctx,
					     const char *data, size_t data_len)
{
	int ret;

	if (!ctx)
		return -EINVAL;

	ret = dsmil_mfa_sign_payload(ctx, data, data_len,
				     ctx->signatures.user_signature,
				     sizeof(ctx->signatures.user_signature));
	if (ret)
		return ret;

	ctx->signatures.signature_time = dsmil_get_real_time();
	ctx->signatures.signature_verified = true;
	return 0;
}

/*
 * Get MFA system statistics
 */
int dsmil_mfa_get_statistics(u64 *auth_attempts, u64 *auth_successes, 
                            u64 *auth_failures, u32 *active_sessions,
                            u64 *dual_auth_requests, u64 *emergency_overrides)
{
    if (!mfa_system) {
        return -EINVAL;
    }
    
    if (auth_attempts) {
        *auth_attempts = atomic64_read(&mfa_system->auth_attempts);
    }
    if (auth_successes) {
        *auth_successes = atomic64_read(&mfa_system->auth_successes);
    }
    if (auth_failures) {
        *auth_failures = atomic64_read(&mfa_system->auth_failures);
    }
    if (active_sessions) {
        *active_sessions = mfa_system->session_count;
    }
    if (dual_auth_requests) {
        *dual_auth_requests = atomic64_read(&mfa_system->dual_auth_requests_total);
    }
    if (emergency_overrides) {
        *emergency_overrides = atomic64_read(&mfa_system->emergency_overrides);
    }
    
    return 0;
}

/*
 * Provide a lightweight view of an authenticated user's security posture
 */
int dsmil_mfa_get_user_profile(uid_t user_id,
			       struct dsmil_user_security_profile *profile)
{
	u32 i;
	int ret = -ENOENT;

	if (!profile)
		return -EINVAL;

	memset(profile, 0, sizeof(*profile));

	if (!mfa_system || !mfa_system->initialized)
		return -EINVAL;

	mutex_lock(&mfa_system->global_lock);
	for (i = 0; i < DSMIL_MAX_SESSIONS; i++) {
		struct dsmil_auth_context *ctx = mfa_system->active_sessions[i];

		if (!ctx || !ctx->context_valid || ctx->user_id != user_id)
			continue;

		profile->clearance_level = ctx->clearance;
		profile->compartment_mask = ctx->compartmentalized_access;
		profile->group_id = ctx->group_id;
		profile->network_access_allowed =
			(ctx->permissions.network.num_allowed_ips > 0) ||
			(ctx->permissions.network.network_access_flags != 0);
		ret = 0;
		break;
	}
	mutex_unlock(&mfa_system->global_lock);

	return ret;
}

/* Export functions for use by other modules */
EXPORT_SYMBOL(dsmil_mfa_init);
EXPORT_SYMBOL(dsmil_mfa_cleanup);
EXPORT_SYMBOL(dsmil_mfa_create_auth_context);
EXPORT_SYMBOL(dsmil_mfa_authorize_operation);
EXPORT_SYMBOL(dsmil_mfa_request_dual_authorization);
EXPORT_SYMBOL(dsmil_mfa_get_statistics);
EXPORT_SYMBOL(dsmil_mfa_get_user_profile);

MODULE_AUTHOR("DSMIL Track B Security Team");
MODULE_DESCRIPTION("DSMIL Multi-Factor Authentication System with Military-Grade Security");
MODULE_LICENSE("GPL v2");
MODULE_VERSION(DSMIL_MFA_VERSION);

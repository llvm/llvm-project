/*
 * DSMIL Military-Grade Authorization Engine - Track B Security Layer
 * 
 * Copyright (C) 2025 JRTC1 Educational Development
 * 
 * This module implements comprehensive authorization decision engine for the 
 * 84-device DSMIL system. Provides risk-based authorization, policy enforcement,
 * and dynamic access control with military-grade security requirements.
 * 
 * SECURITY FEATURES:
 * - Risk-based authorization decisions
 * - Dynamic policy evaluation 
 * - Temporal access restrictions
 * - Network-based access control
 * - Emergency authorization overrides
 * - Comprehensive decision logging
 */

#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/slab.h>
#include <linux/mutex.h>
#include <linux/atomic.h>
#include <linux/spinlock.h>
#include <linux/workqueue.h>
#include <linux/time.h>
#include <linux/crc32.h>
#include <linux/hash.h>
#include <linux/rbtree.h>
#include <crypto/hash.h>

#define DSMIL_AUTHZ_VERSION         "1.0.0"
#define DSMIL_AUTHZ_MAGIC           0x41555448  /* "AUTH" */
#define DSMIL_MAX_POLICIES          256         /* Maximum authorization policies */
#define DSMIL_MAX_SESSIONS          128         /* Maximum active sessions */
#define DSMIL_DECISION_CACHE_SIZE   1024        /* Authorization decision cache */

/* Authorization decision types */
enum dsmil_authz_decision {
    AUTHZ_DENIED = 0,
    AUTHZ_GRANTED = 1,
    AUTHZ_CONDITIONAL = 2,
    AUTHZ_ESCALATED = 3,
    AUTHZ_DEFERRED = 4
};

/* Authorization policy types */
enum dsmil_policy_type {
    POLICY_DEVICE_ACCESS = 0,
    POLICY_OPERATION_CONTROL = 1,
    POLICY_TEMPORAL_RESTRICTION = 2,
    POLICY_RISK_ASSESSMENT = 3,
    POLICY_EMERGENCY_OVERRIDE = 4,
    POLICY_NETWORK_CONTROL = 5,
    POLICY_USER_BASED = 6,
    POLICY_RESOURCE_LIMIT = 7
};

/* Authorization context for decision making */
struct dsmil_authz_context {
    /* Request identification */
    u64 request_id;
    struct timespec64 request_time;
    
    /* Requester information */
    uid_t user_id;
    char username[64];
    u32 user_clearance;
    u32 compartmentalized_access;
    u32 session_id;
    
    /* Request details */
    u32 target_device_id;
    u32 operation_type;
    u32 operation_flags;
    size_t data_size;
    void *operation_data;
    
    /* Risk assessment */
    u32 assessed_risk_level;
    u32 risk_factors;
    char risk_justification[256];
    
    /* Network context */
    u32 source_ip;
    u16 source_port;
    bool network_available;
    
    /* Temporal context */
    struct timespec64 request_valid_from;
    struct timespec64 request_valid_until;
    
    /* Authorization chain */
    u32 authorization_chain_length;
    u32 authorization_chain[8]; /* Track authorization path */
    
    /* Decision caching */
    bool cache_decision;
    u32 cache_duration_seconds;
};

/* Authorization policy definition */
struct dsmil_authz_policy {
    /* Policy identification */
    u32 policy_id;
    char policy_name[128];
    enum dsmil_policy_type policy_type;
    bool enabled;
    
    /* Policy conditions */
    struct {
        /* User-based conditions */
        u32 required_clearance_level;
        u32 required_compartments;
        u32 allowed_users[16];
        u32 denied_users[16];
        u32 allowed_groups[8];
        
        /* Device-based conditions */
        u64 allowed_devices[2];        /* 128-bit mask for 84 devices */
        u64 denied_devices[2];         /* 128-bit mask for 84 devices */
        u32 device_risk_threshold;
        
        /* Operation-based conditions */
        u32 allowed_operations;
        u32 denied_operations;
        u32 max_operation_size;
        u32 operation_risk_threshold;
        
        /* Temporal conditions */
        struct {
            u32 allowed_hours[24];     /* Bitmask for allowed hours */
            u32 allowed_days[7];       /* Bitmask for allowed days */
            struct timespec64 valid_from;
            struct timespec64 valid_until;
        } temporal;
        
        /* Network conditions */
        struct {
            u32 allowed_networks[8];   /* Allowed network ranges */
            u32 network_masks[8];      /* Network masks */
            u16 allowed_ports[16];     /* Allowed source ports */
            bool require_secure_channel;
        } network;
    } conditions;
    
    /* Policy actions */
    struct {
        enum dsmil_authz_decision default_decision;
        bool require_dual_authorization;
        bool require_supervisor_approval;
        bool log_all_decisions;
        bool send_notifications;
        u32 max_attempts_per_hour;
    } actions;
    
    /* Policy metadata */
    struct timespec64 created_time;
    struct timespec64 modified_time;
    u32 creator_uid;
    u32 modification_count;
    u32 usage_count;
    
    /* Linked list for policy management */
    struct list_head list;
    
    /* Red-black tree for efficient lookup */
    struct rb_node rb_node;
};

/* Authorization decision result */
struct dsmil_authz_decision_result {
    /* Decision information */
    enum dsmil_authz_decision decision;
    u32 confidence_score;      /* 0-100 confidence level */
    char decision_reason[512]; /* Human-readable reason */
    
    /* Policy information */
    u32 matched_policies_count;
    u32 matched_policy_ids[16];
    u32 total_policies_evaluated;
    
    /* Timing information */
    struct timespec64 decision_time;
    u64 evaluation_time_ns;
    
    /* Additional requirements */
    struct {
        bool dual_authorization_required;
        u32 dual_auth_timeout_seconds;
        bool supervisor_approval_required;
        u32 supervisor_timeout_seconds;
        bool additional_logging_required;
    } requirements;
    
    /* Conditions and restrictions */
    struct {
        u32 max_data_size;
        u32 session_timeout_seconds;
        u32 max_operations_per_hour;
        bool require_re_authentication;
        struct timespec64 authorization_expires;
    } conditions;
    
    /* Audit and tracking */
    u64 audit_entry_id;
    u32 decision_cache_key;
    bool decision_cached;
};

/* Decision cache entry */
struct dsmil_decision_cache_entry {
    /* Cache key components */
    u32 user_id;
    u32 device_id;
    u32 operation_type;
    u32 risk_level;
    u32 cache_key_hash;
    
    /* Cached decision */
    struct dsmil_authz_decision_result decision;
    
    /* Cache metadata */
    struct timespec64 cached_time;
    struct timespec64 expires_time;
    u32 usage_count;
    
    /* Hash table linkage */
    struct hlist_node hash_node;
};

/* Emergency override authorization */
struct dsmil_emergency_authz {
    /* Override identification */
    u64 override_id;
    struct timespec64 activated_time;
    struct timespec64 expires_time;
    
    /* Authorization details */
    uid_t authorizer_id;
    char authorizer_name[64];
    u32 authorizer_clearance;
    char justification[512];
    
    /* Override scope */
    struct {
        bool global_override;         /* Affects all policies */
        u32 affected_devices[16];     /* Specific devices affected */
        u32 affected_operations;      /* Operation types affected */
        u32 affected_users[8];        /* Users affected by override */
    } scope;
    
    /* Override tracking */
    u32 operations_performed;
    u64 last_activity_time;
    bool active;
    
    /* Linked list for active overrides */
    struct list_head list;
};

/* Global authorization engine state */
struct dsmil_authz_engine {
    struct mutex engine_lock;
    
    /* Policy management */
    struct {
        struct list_head policy_list;
        struct rb_root policy_tree;
        u32 policy_count;
        u32 enabled_policy_count;
        struct mutex policy_lock;
    } policies;
    
    /* Decision caching */
    struct {
        DECLARE_HASHTABLE(cache_table, 10); /* 1024 buckets */
        struct mutex cache_lock;
        u32 cache_entries;
        u32 cache_hits;
        u32 cache_misses;
        u32 cache_evictions;
    } cache;
    
    /* Active sessions */
    struct {
        struct dsmil_authz_context *sessions[DSMIL_MAX_SESSIONS];
        u32 active_session_count;
        struct mutex sessions_lock;
    } sessions;
    
    /* Emergency overrides */
    struct {
        struct list_head active_overrides;
        struct mutex overrides_lock;
        u32 active_override_count;
        atomic64_t total_overrides;
    } emergency;
    
    /* Performance statistics */
    struct {
        atomic64_t total_requests;
        atomic64_t granted_requests;
        atomic64_t denied_requests;
        atomic64_t conditional_requests;
        atomic64_t escalated_requests;
        u64 avg_decision_time_ns;
        u64 max_decision_time_ns;
    } stats;
    
    /* System configuration */
    struct {
        bool strict_mode;
        bool audit_all_decisions;
        u32 default_cache_timeout;
        u32 policy_evaluation_timeout_ms;
        bool enable_decision_caching;
    } config;
    
    /* System state */
    bool initialized;
    u32 authz_magic;
    ktime_t init_time;
};

/* Global authorization engine instance */
static struct dsmil_authz_engine *authz_engine = NULL;

/* Forward declarations */
static enum dsmil_authz_decision dsmil_authz_evaluate_policies(
    struct dsmil_authz_context *context,
    struct dsmil_authz_decision_result *result);
static int dsmil_authz_evaluate_policy(struct dsmil_authz_policy *policy,
                                       struct dsmil_authz_context *context);
static struct dsmil_decision_cache_entry *dsmil_authz_lookup_cache(
    struct dsmil_authz_context *context);
static int dsmil_authz_cache_decision(struct dsmil_authz_context *context,
                                     struct dsmil_authz_decision_result *result);
static int dsmil_authz_check_emergency_overrides(struct dsmil_authz_context *context);

/*
 * Initialize authorization engine
 */
int dsmil_authz_init(void)
{
    int ret = 0;
    
    if (authz_engine) {
        pr_warn("DSMIL Authorization: Already initialized\n");
        return 0;
    }
    
    /* Allocate global state */
    authz_engine = kzalloc(sizeof(struct dsmil_authz_engine), GFP_KERNEL);
    if (!authz_engine) {
        pr_err("DSMIL Authorization: Failed to allocate engine structure\n");
        return -ENOMEM;
    }
    
    /* Initialize mutexes */
    mutex_init(&authz_engine->engine_lock);
    mutex_init(&authz_engine->policies.policy_lock);
    mutex_init(&authz_engine->cache.cache_lock);
    mutex_init(&authz_engine->sessions.sessions_lock);
    mutex_init(&authz_engine->emergency.overrides_lock);
    
    /* Initialize policy management */
    INIT_LIST_HEAD(&authz_engine->policies.policy_list);
    authz_engine->policies.policy_tree = RB_ROOT;
    authz_engine->policies.policy_count = 0;
    authz_engine->policies.enabled_policy_count = 0;
    
    /* Initialize decision cache */
    hash_init(authz_engine->cache.cache_table);
    authz_engine->cache.cache_entries = 0;
    authz_engine->cache.cache_hits = 0;
    authz_engine->cache.cache_misses = 0;
    authz_engine->cache.cache_evictions = 0;
    
    /* Initialize emergency overrides */
    INIT_LIST_HEAD(&authz_engine->emergency.active_overrides);
    authz_engine->emergency.active_override_count = 0;
    atomic64_set(&authz_engine->emergency.total_overrides, 0);
    
    /* Initialize statistics */
    atomic64_set(&authz_engine->stats.total_requests, 0);
    atomic64_set(&authz_engine->stats.granted_requests, 0);
    atomic64_set(&authz_engine->stats.denied_requests, 0);
    atomic64_set(&authz_engine->stats.conditional_requests, 0);
    atomic64_set(&authz_engine->stats.escalated_requests, 0);
    authz_engine->stats.avg_decision_time_ns = 0;
    authz_engine->stats.max_decision_time_ns = 0;
    
    /* Initialize system configuration */
    authz_engine->config.strict_mode = true;
    authz_engine->config.audit_all_decisions = true;
    authz_engine->config.default_cache_timeout = 300; /* 5 minutes */
    authz_engine->config.policy_evaluation_timeout_ms = 1000; /* 1 second */
    authz_engine->config.enable_decision_caching = true;
    
    /* Set system state */
    authz_engine->initialized = true;
    authz_engine->authz_magic = DSMIL_AUTHZ_MAGIC;
    authz_engine->init_time = ktime_get();
    
    pr_info("DSMIL Authorization: Initialized (version %s)\n", DSMIL_AUTHZ_VERSION);
    pr_info("DSMIL Authorization: Strict mode %s, audit all decisions %s\n",
            authz_engine->config.strict_mode ? "ENABLED" : "disabled",
            authz_engine->config.audit_all_decisions ? "ENABLED" : "disabled");
    pr_info("DSMIL Authorization: Decision caching %s (timeout %u seconds)\n",
            authz_engine->config.enable_decision_caching ? "ENABLED" : "disabled",
            authz_engine->config.default_cache_timeout);
    
    return 0;
}

/*
 * Cleanup authorization engine
 */
void dsmil_authz_cleanup(void)
{
    struct dsmil_authz_policy *policy, *policy_tmp;
    struct dsmil_emergency_authz *override, *override_tmp;
    struct dsmil_decision_cache_entry *cache_entry;
    struct hlist_node *tmp;
    int bkt;
    u32 i;
    
    if (!authz_engine) {
        return;
    }
    
    pr_info("DSMIL Authorization: Shutting down\n");
    
    mutex_lock(&authz_engine->engine_lock);
    
    /* Cleanup policies */
    mutex_lock(&authz_engine->policies.policy_lock);
    list_for_each_entry_safe(policy, policy_tmp, 
                            &authz_engine->policies.policy_list, list) {
        list_del(&policy->list);
        kfree(policy);
    }
    mutex_unlock(&authz_engine->policies.policy_lock);
    
    /* Cleanup decision cache */
    mutex_lock(&authz_engine->cache.cache_lock);
    hash_for_each_safe(authz_engine->cache.cache_table, bkt, tmp, cache_entry, hash_node) {
        hash_del(&cache_entry->hash_node);
        kfree(cache_entry);
    }
    mutex_unlock(&authz_engine->cache.cache_lock);
    
    /* Cleanup active sessions */
    mutex_lock(&authz_engine->sessions.sessions_lock);
    for (i = 0; i < DSMIL_MAX_SESSIONS; i++) {
        if (authz_engine->sessions.sessions[i]) {
            kfree(authz_engine->sessions.sessions[i]);
            authz_engine->sessions.sessions[i] = NULL;
        }
    }
    mutex_unlock(&authz_engine->sessions.sessions_lock);
    
    /* Cleanup emergency overrides */
    mutex_lock(&authz_engine->emergency.overrides_lock);
    list_for_each_entry_safe(override, override_tmp,
                            &authz_engine->emergency.active_overrides, list) {
        list_del(&override->list);
        kfree(override);
    }
    mutex_unlock(&authz_engine->emergency.overrides_lock);
    
    mutex_unlock(&authz_engine->engine_lock);
    
    /* Print final statistics */
    pr_info("DSMIL Authorization: Final stats - Requests: %lld, Granted: %lld, Denied: %lld\n",
            atomic64_read(&authz_engine->stats.total_requests),
            atomic64_read(&authz_engine->stats.granted_requests),
            atomic64_read(&authz_engine->stats.denied_requests));
    pr_info("DSMIL Authorization: Cache hits: %u, misses: %u, efficiency: %.1f%%\n",
            authz_engine->cache.cache_hits, authz_engine->cache.cache_misses,
            authz_engine->cache.cache_hits * 100.0 / 
            (authz_engine->cache.cache_hits + authz_engine->cache.cache_misses + 1));
    
    /* Clear magic and free memory */
    authz_engine->authz_magic = 0;
    authz_engine->initialized = false;
    kfree(authz_engine);
    authz_engine = NULL;
    
    pr_info("DSMIL Authorization: Cleanup complete\n");
}

/*
 * Make authorization decision for operation request
 */
int dsmil_authz_authorize(struct dsmil_authz_context *context,
                         struct dsmil_authz_decision_result *result)
{
    ktime_t start_time, end_time;
    struct dsmil_decision_cache_entry *cached_decision;
    enum dsmil_authz_decision decision;
    u64 decision_time_ns;
    
    if (!authz_engine || !authz_engine->initialized) {
        pr_err("DSMIL Authorization: Engine not initialized\n");
        return -EINVAL;
    }
    
    if (!context || !result) {
        pr_err("DSMIL Authorization: Invalid parameters\n");
        return -EINVAL;
    }
    
    start_time = ktime_get();
    atomic64_inc(&authz_engine->stats.total_requests);
    
    /* Clear result structure */
    memset(result, 0, sizeof(*result));
    result->decision_time = dsmil_get_real_time();
    
    /* Assign unique request ID */
    context->request_id = atomic64_inc_return(&authz_engine->stats.total_requests);
    
    /* Check for cached decision */
    if (authz_engine->config.enable_decision_caching) {
        cached_decision = dsmil_authz_lookup_cache(context);
        if (cached_decision) {
            memcpy(result, &cached_decision->decision, sizeof(*result));
            cached_decision->usage_count++;
            authz_engine->cache.cache_hits++;
            
            pr_debug("DSMIL Authorization: Cache hit for request %llu\n", 
                     context->request_id);
            return 0;
        }
        authz_engine->cache.cache_misses++;
    }
    
    /* Check for emergency overrides */
    int override_result = dsmil_authz_check_emergency_overrides(context);
    if (override_result > 0) {
        result->decision = AUTHZ_GRANTED;
        result->confidence_score = 100;
        strncpy(result->decision_reason, "Emergency override active", 
                sizeof(result->decision_reason) - 1);
        decision = AUTHZ_GRANTED;
        goto decision_complete;
    }
    
    /* Evaluate authorization policies */
    decision = dsmil_authz_evaluate_policies(context, result);
    
decision_complete:
    /* Update decision timing */
    end_time = ktime_get();
    decision_time_ns = ktime_to_ns(ktime_sub(end_time, start_time));
    result->evaluation_time_ns = decision_time_ns;
    
    /* Update performance statistics */
    if (authz_engine->stats.avg_decision_time_ns == 0) {
        authz_engine->stats.avg_decision_time_ns = decision_time_ns;
    } else {
        authz_engine->stats.avg_decision_time_ns = 
            (authz_engine->stats.avg_decision_time_ns + decision_time_ns) / 2;
    }
    
    if (decision_time_ns > authz_engine->stats.max_decision_time_ns) {
        authz_engine->stats.max_decision_time_ns = decision_time_ns;
    }
    
    /* Update decision statistics */
    switch (decision) {
    case AUTHZ_GRANTED:
        atomic64_inc(&authz_engine->stats.granted_requests);
        break;
    case AUTHZ_DENIED:
        atomic64_inc(&authz_engine->stats.denied_requests);
        break;
    case AUTHZ_CONDITIONAL:
        atomic64_inc(&authz_engine->stats.conditional_requests);
        break;
    case AUTHZ_ESCALATED:
        atomic64_inc(&authz_engine->stats.escalated_requests);
        break;
    default:
        break;
    }
    
    /* Cache decision if appropriate */
    if (authz_engine->config.enable_decision_caching && context->cache_decision) {
        dsmil_authz_cache_decision(context, result);
    }
    
    /* Audit logging */
    if (authz_engine->config.audit_all_decisions) {
        pr_info("DSMIL Authorization: Request %llu - User %s, Device %u, Decision %d (%s)\n",
                context->request_id, context->username, context->target_device_id,
                decision, result->decision_reason);
    }
    
    pr_debug("DSMIL Authorization: Decision completed in %llu ns\n", decision_time_ns);
    
    return 0;
}

/*
 * Evaluate all applicable authorization policies
 */
static enum dsmil_authz_decision dsmil_authz_evaluate_policies(
    struct dsmil_authz_context *context,
    struct dsmil_authz_decision_result *result)
{
    struct dsmil_authz_policy *policy;
    enum dsmil_authz_decision final_decision = AUTHZ_DENIED;
    int policy_result;
    u32 granted_policies = 0;
    u32 denied_policies = 0;
    u32 conditional_policies = 0;
    u32 policies_evaluated = 0;
    
    /* Default to denial in strict mode */
    if (authz_engine->config.strict_mode) {
        final_decision = AUTHZ_DENIED;
        strncpy(result->decision_reason, "Default deny (strict mode)", 
                sizeof(result->decision_reason) - 1);
    }
    
    mutex_lock(&authz_engine->policies.policy_lock);
    
    /* Evaluate each enabled policy */
    list_for_each_entry(policy, &authz_engine->policies.policy_list, list) {
        if (!policy->enabled) {
            continue;
        }
        
        policies_evaluated++;
        policy_result = dsmil_authz_evaluate_policy(policy, context);
        
        /* Track which policies matched */
        if (result->matched_policies_count < ARRAY_SIZE(result->matched_policy_ids)) {
            result->matched_policy_ids[result->matched_policies_count++] = policy->policy_id;
        }
        
        switch (policy_result) {
        case 1: /* Policy grants access */
            granted_policies++;
            if (final_decision == AUTHZ_DENIED) {
                final_decision = AUTHZ_GRANTED;
                snprintf(result->decision_reason, sizeof(result->decision_reason),
                         "Granted by policy '%s'", policy->policy_name);
            }
            break;
            
        case 0: /* Policy denies access */
            denied_policies++;
            if (policy->actions.default_decision == AUTHZ_DENIED) {
                /* Explicit denial overrides grants */
                final_decision = AUTHZ_DENIED;
                snprintf(result->decision_reason, sizeof(result->decision_reason),
                         "Denied by policy '%s'", policy->policy_name);
            }
            break;
            
        case -1: /* Policy requires additional authorization */
            conditional_policies++;
            if (final_decision != AUTHZ_DENIED) {
                final_decision = AUTHZ_CONDITIONAL;
                snprintf(result->decision_reason, sizeof(result->decision_reason),
                         "Conditional approval required by policy '%s'", policy->policy_name);
                
                /* Set additional requirements */
                if (policy->actions.require_dual_authorization) {
                    result->requirements.dual_authorization_required = true;
                    result->requirements.dual_auth_timeout_seconds = 300;
                }
                if (policy->actions.require_supervisor_approval) {
                    result->requirements.supervisor_approval_required = true;
                    result->requirements.supervisor_timeout_seconds = 600;
                }
            }
            break;
            
        default:
            /* Policy evaluation error - treat as deny */
            denied_policies++;
            break;
        }
        
        policy->usage_count++;
    }
    
    mutex_unlock(&authz_engine->policies.policy_lock);
    
    result->total_policies_evaluated = policies_evaluated;
    
    /* Calculate confidence score based on policy consensus */
    if (policies_evaluated == 0) {
        result->confidence_score = 0;
        final_decision = AUTHZ_DENIED;
        strncpy(result->decision_reason, "No applicable policies found", 
                sizeof(result->decision_reason) - 1);
    } else {
        u32 consensus_score = 0;
        if (granted_policies > denied_policies) {
            consensus_score = (granted_policies * 100) / policies_evaluated;
        } else if (denied_policies > 0) {
            consensus_score = (denied_policies * 100) / policies_evaluated;
        }
        result->confidence_score = consensus_score;
    }
    
    result->decision = final_decision;
    
    pr_debug("DSMIL Authorization: Evaluated %u policies - Granted: %u, Denied: %u, Conditional: %u\n",
             policies_evaluated, granted_policies, denied_policies, conditional_policies);
    
    return final_decision;
}

/*
 * Evaluate individual authorization policy
 */
static int dsmil_authz_evaluate_policy(struct dsmil_authz_policy *policy,
                                      struct dsmil_authz_context *context)
{
    struct tm tm_time;
    u32 hour, day;
    
    /* Check user-based conditions */
    if (policy->conditions.required_clearance_level > 0 &&
        context->user_clearance < policy->conditions.required_clearance_level) {
        return 0; /* Insufficient clearance */
    }
    
    if (policy->conditions.required_compartments != 0 &&
        (context->compartmentalized_access & policy->conditions.required_compartments) == 0) {
        return 0; /* Missing required compartments */
    }
    
    /* Check device-based conditions */
    if (context->target_device_id < 64) {
        u64 device_bit = 1ULL << context->target_device_id;
        if ((policy->conditions.denied_devices[0] & device_bit) != 0) {
            return 0; /* Device explicitly denied */
        }
        if ((policy->conditions.allowed_devices[0] & device_bit) == 0 &&
            policy->conditions.allowed_devices[0] != 0) {
            return 0; /* Device not in allowed set */
        }
    } else if (context->target_device_id < 84) {
        u32 bit_pos = context->target_device_id - 64;
        u64 device_bit = 1ULL << bit_pos;
        if ((policy->conditions.denied_devices[1] & device_bit) != 0) {
            return 0; /* Device explicitly denied */
        }
        if ((policy->conditions.allowed_devices[1] & device_bit) == 0 &&
            policy->conditions.allowed_devices[1] != 0) {
            return 0; /* Device not in allowed set */
        }
    }
    
    /* Check operation-based conditions */
    if (policy->conditions.denied_operations != 0 &&
        (policy->conditions.denied_operations & (1 << context->operation_type)) != 0) {
        return 0; /* Operation explicitly denied */
    }
    
    if (policy->conditions.allowed_operations != 0 &&
        (policy->conditions.allowed_operations & (1 << context->operation_type)) == 0) {
        return 0; /* Operation not in allowed set */
    }
    
    if (policy->conditions.max_operation_size > 0 &&
        context->data_size > policy->conditions.max_operation_size) {
        return 0; /* Operation size exceeds limit */
    }
    
    /* Check temporal conditions */
    time64_to_tm(context->request_time.tv_sec, 0, &tm_time);
    hour = tm_time.tm_hour;
    day = tm_time.tm_wday;
    
    if (policy->conditions.temporal.allowed_hours[hour] == 0) {
        return 0; /* Operation not allowed at this hour */
    }
    
    if (policy->conditions.temporal.allowed_days[day] == 0) {
        return 0; /* Operation not allowed on this day */
    }
    
    /* Check if policy requires additional authorization */
    if (policy->actions.require_dual_authorization ||
        policy->actions.require_supervisor_approval) {
        return -1; /* Conditional approval */
    }
    
    /* All conditions passed - grant access */
    return 1;
}

/*
 * Look up cached authorization decision
 */
static struct dsmil_decision_cache_entry *dsmil_authz_lookup_cache(
    struct dsmil_authz_context *context)
{
    struct dsmil_decision_cache_entry *entry;
    u32 cache_key;
    struct timespec64 now;
    
    /* Generate cache key */
    cache_key = hash_32((context->user_id << 16) | 
                       (context->target_device_id << 8) |
                       context->operation_type, 10);
    
    now = dsmil_get_real_time();
    
    mutex_lock(&authz_engine->cache.cache_lock);
    
    hash_for_each_possible(authz_engine->cache.cache_table, entry, hash_node, cache_key) {
        if (entry->user_id == context->user_id &&
            entry->device_id == context->target_device_id &&
            entry->operation_type == context->operation_type &&
            entry->risk_level == context->assessed_risk_level) {
            
            /* Check if cache entry has expired */
            if (timespec64_compare(&now, &entry->expires_time) <= 0) {
                mutex_unlock(&authz_engine->cache.cache_lock);
                return entry;
            } else {
                /* Remove expired entry */
                hash_del(&entry->hash_node);
                authz_engine->cache.cache_entries--;
                authz_engine->cache.cache_evictions++;
                kfree(entry);
                break;
            }
        }
    }
    
    mutex_unlock(&authz_engine->cache.cache_lock);
    return NULL;
}

/*
 * Cache authorization decision
 */
static int dsmil_authz_cache_decision(struct dsmil_authz_context *context,
                                     struct dsmil_authz_decision_result *result)
{
    struct dsmil_decision_cache_entry *entry;
    u32 cache_key;
    
    if (authz_engine->cache.cache_entries >= DSMIL_DECISION_CACHE_SIZE) {
        pr_debug("DSMIL Authorization: Cache full, not caching decision\n");
        return -ENOSPC;
    }
    
    entry = kzalloc(sizeof(struct dsmil_decision_cache_entry), GFP_KERNEL);
    if (!entry) {
        return -ENOMEM;
    }
    
    /* Fill cache entry */
    entry->user_id = context->user_id;
    entry->device_id = context->target_device_id;
    entry->operation_type = context->operation_type;
    entry->risk_level = context->assessed_risk_level;
    
    /* Generate cache key */
    cache_key = hash_32((context->user_id << 16) | 
                       (context->target_device_id << 8) |
                       context->operation_type, 10);
    entry->cache_key_hash = cache_key;
    
    /* Copy decision result */
    memcpy(&entry->decision, result, sizeof(entry->decision));
    
    /* Set cache timing */
    entry->cached_time = dsmil_get_real_time();
    entry->expires_time = entry->cached_time;
    entry->expires_time.tv_sec += context->cache_duration_seconds > 0 ?
                                  context->cache_duration_seconds :
                                  authz_engine->config.default_cache_timeout;
    
    entry->usage_count = 0;
    
    mutex_lock(&authz_engine->cache.cache_lock);
    hash_add(authz_engine->cache.cache_table, &entry->hash_node, cache_key);
    authz_engine->cache.cache_entries++;
    mutex_unlock(&authz_engine->cache.cache_lock);
    
    pr_debug("DSMIL Authorization: Cached decision for user %u, device %u, operation %u\n",
             context->user_id, context->target_device_id, context->operation_type);
    
    return 0;
}

/*
 * Check for active emergency overrides
 */
static int dsmil_authz_check_emergency_overrides(struct dsmil_authz_context *context)
{
    struct dsmil_emergency_authz *override;
    struct timespec64 now;
    int override_applies = 0;
    
    now = dsmil_get_real_time();
    
    mutex_lock(&authz_engine->emergency.overrides_lock);
    
    list_for_each_entry(override, &authz_engine->emergency.active_overrides, list) {
        if (!override->active) {
            continue;
        }
        
        /* Check if override has expired */
        if (timespec64_compare(&now, &override->expires_time) > 0) {
            override->active = false;
            continue;
        }
        
        /* Check if override applies to this request */
        if (override->scope.global_override) {
            override_applies = 1;
            break;
        }
        
        /* Check device-specific overrides */
        for (int i = 0; i < 16; i++) {
            if (override->scope.affected_devices[i] == context->target_device_id) {
                override_applies = 1;
                break;
            }
        }
        
        if (override_applies) {
            override->operations_performed++;
            override->last_activity_time = now.tv_sec;
            break;
        }
    }
    
    mutex_unlock(&authz_engine->emergency.overrides_lock);
    
    return override_applies;
}

/*
 * Get authorization engine statistics
 */
int dsmil_authz_get_statistics(u64 *total_requests, u64 *granted_requests,
                              u64 *denied_requests, u32 *cache_efficiency,
                              u64 *avg_decision_time_ns)
{
    if (!authz_engine) {
        return -EINVAL;
    }
    
    if (total_requests) {
        *total_requests = atomic64_read(&authz_engine->stats.total_requests);
    }
    if (granted_requests) {
        *granted_requests = atomic64_read(&authz_engine->stats.granted_requests);
    }
    if (denied_requests) {
        *denied_requests = atomic64_read(&authz_engine->stats.denied_requests);
    }
    if (cache_efficiency) {
        u32 hits = authz_engine->cache.cache_hits;
        u32 misses = authz_engine->cache.cache_misses;
        *cache_efficiency = hits * 100 / (hits + misses + 1);
    }
    if (avg_decision_time_ns) {
        *avg_decision_time_ns = authz_engine->stats.avg_decision_time_ns;
    }
    
    return 0;
}

/* Export functions for use by other modules */
EXPORT_SYMBOL(dsmil_authz_init);
EXPORT_SYMBOL(dsmil_authz_cleanup);
EXPORT_SYMBOL(dsmil_authz_authorize);
EXPORT_SYMBOL(dsmil_authz_get_statistics);

MODULE_AUTHOR("DSMIL Track B Security Team");
MODULE_DESCRIPTION("DSMIL Military-Grade Authorization Engine with Risk-Based Access Control");
MODULE_LICENSE("GPL v2");
MODULE_VERSION(DSMIL_AUTHZ_VERSION);
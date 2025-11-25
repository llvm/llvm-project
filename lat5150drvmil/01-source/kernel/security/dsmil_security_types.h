/*
 * DSMIL Shared Security Types
 *
 * Centralized definitions for clearance levels, risk classifications,
 * and operation identifiers used across the kernel subsystems.
 */

#ifndef _DSMIL_SECURITY_TYPES_H
#define _DSMIL_SECURITY_TYPES_H

#include <linux/types.h>
#include <linux/uidgid.h>
#include <linux/stddef.h>

/* NATO / DoD inspired clearance levels */
enum dsmil_clearance_level {
	DSMIL_CLEARANCE_UNCLASSIFIED = 0,
	DSMIL_CLEARANCE_RESTRICTED,
	DSMIL_CLEARANCE_CONFIDENTIAL,
	DSMIL_CLEARANCE_SECRET,
	DSMIL_CLEARANCE_TOP_SECRET,
	DSMIL_CLEARANCE_SCI,
	DSMIL_CLEARANCE_SAP,
	DSMIL_CLEARANCE_COSMIC,
	DSMIL_CLEARANCE_ATOMAL
};

#define DSMIL_CLEARANCE_MAX DSMIL_CLEARANCE_ATOMAL

/* Risk ratings aligned with DSMIL safety documentation */
enum dsmil_risk_level {
	DSMIL_RISK_LOW = 0,
	DSMIL_RISK_MEDIUM,
	DSMIL_RISK_HIGH,
	DSMIL_RISK_CRITICAL,
	DSMIL_RISK_CATASTROPHIC
};

/* Operation semantic identifiers */
enum dsmil_operation_type {
	DSMIL_OP_READ = 0,
	DSMIL_OP_WRITE,
	DSMIL_OP_CONFIG,
	DSMIL_OP_CONTROL,
	DSMIL_OP_RESET,
	DSMIL_OP_EMERGENCY,
	DSMIL_OP_MAINTENANCE,
	DSMIL_OP_DIAGNOSTIC
};

/* Snapshot exported from the MFA subsystem for other modules */
struct dsmil_user_security_profile {
	u32 clearance_level;          /* enum dsmil_clearance_level */
	u32 compartment_mask;         /* Compartment authorization bitmask */
	gid_t group_id;               /* Kernel group identifier */
	bool network_access_allowed;  /* Whether network operations are permitted */
};

struct dsmil_auth_context;

/* MFA Authentication Functions */
int dsmil_mfa_init(void);
void dsmil_mfa_cleanup(void);

struct dsmil_auth_context *dsmil_mfa_create_auth_context(uid_t user_id,
							  const char *username,
							  enum dsmil_clearance_level clearance,
							  u32 compartmentalized_access);

int dsmil_mfa_authorize_operation(struct dsmil_auth_context *ctx,
				   u32 device_id,
				   enum dsmil_operation_type operation,
				   enum dsmil_risk_level risk_level);

int dsmil_mfa_request_dual_authorization(struct dsmil_auth_context *first_auth,
					  u32 device_id,
					  enum dsmil_operation_type operation,
					  const char *justification);

int dsmil_mfa_get_statistics(u64 *auth_attempts, u64 *auth_successes,
			      u64 *auth_failures, u32 *active_sessions,
			      u64 *dual_auth_requests, u64 *emergency_overrides);

int dsmil_mfa_get_user_profile(uid_t user_id,
			       struct dsmil_user_security_profile *profile);

#endif /* _DSMIL_SECURITY_TYPES_H */

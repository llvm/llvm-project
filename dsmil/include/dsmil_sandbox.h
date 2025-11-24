/**
 * @file dsmil_sandbox.h
 * @brief DSMIL Sandbox Runtime Support
 *
 * Defines structures and functions for role-based sandboxing using
 * libcap-ng and seccomp-bpf. Used by dsmil-sandbox-wrap pass.
 *
 * Version: 1.0
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#ifndef DSMIL_SANDBOX_H
#define DSMIL_SANDBOX_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <sys/types.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup DSMIL_SANDBOX_CONSTANTS Constants
 * @{
 */

/** Maximum profile name length */
#define DSMIL_SANDBOX_MAX_NAME      64

/** Maximum seccomp filter instructions */
#define DSMIL_SANDBOX_MAX_FILTER    512

/** Maximum number of allowed syscalls */
#define DSMIL_SANDBOX_MAX_SYSCALLS  256

/** Maximum number of capabilities */
#define DSMIL_SANDBOX_MAX_CAPS      64

/** Sandbox profile directory */
#define DSMIL_SANDBOX_PROFILE_DIR   "/etc/dsmil/sandbox"

/** @} */

/**
 * @defgroup DSMIL_SANDBOX_ENUMS Enumerations
 * @{
 */

/** Sandbox enforcement mode */
typedef enum {
    DSMIL_SANDBOX_MODE_ENFORCE = 0,  /**< Strict enforcement (default) */
    DSMIL_SANDBOX_MODE_WARN    = 1,  /**< Log violations, don't enforce */
    DSMIL_SANDBOX_MODE_DISABLED = 2,  /**< Sandbox disabled */
} dsmil_sandbox_mode_t;

/** Sandbox result codes */
typedef enum {
    DSMIL_SANDBOX_OK              = 0,  /**< Success */
    DSMIL_SANDBOX_NO_PROFILE      = 1,  /**< Profile not found */
    DSMIL_SANDBOX_MALFORMED       = 2,  /**< Malformed profile */
    DSMIL_SANDBOX_CAP_FAILED      = 3,  /**< Capability setup failed */
    DSMIL_SANDBOX_SECCOMP_FAILED  = 4,  /**< Seccomp setup failed */
    DSMIL_SANDBOX_RLIMIT_FAILED   = 5,  /**< Resource limit setup failed */
    DSMIL_SANDBOX_INVALID_MODE    = 6,  /**< Invalid enforcement mode */
} dsmil_sandbox_result_t;

/** @} */

/**
 * @defgroup DSMIL_SANDBOX_STRUCTS Data Structures
 * @{
 */

/** Capability bounding set */
typedef struct {
    uint32_t caps[DSMIL_SANDBOX_MAX_CAPS]; /**< Capability numbers (CAP_*) */
    uint32_t num_caps;                     /**< Number of capabilities */
} dsmil_cap_bset_t;

/** Seccomp BPF program */
typedef struct {
    struct sock_filter *filter;            /**< BPF instructions */
    uint16_t len;                          /**< Number of instructions */
} dsmil_seccomp_prog_t;

/** Allowed syscall list (alternative to full BPF program) */
typedef struct {
    uint32_t syscalls[DSMIL_SANDBOX_MAX_SYSCALLS]; /**< Syscall numbers */
    uint32_t num_syscalls;                 /**< Number of syscalls */
} dsmil_syscall_allowlist_t;

/** Resource limits */
typedef struct {
    uint64_t max_memory_bytes;             /**< RLIMIT_AS */
    uint64_t max_cpu_time_sec;             /**< RLIMIT_CPU */
    uint32_t max_open_files;               /**< RLIMIT_NOFILE */
    uint32_t max_processes;                /**< RLIMIT_NPROC */
    bool use_limits;                       /**< Apply resource limits */
} dsmil_resource_limits_t;

/** Network restrictions */
typedef struct {
    bool allow_network;                    /**< Allow any network access */
    bool allow_inet;                       /**< Allow IPv4 */
    bool allow_inet6;                      /**< Allow IPv6 */
    bool allow_unix;                       /**< Allow UNIX sockets */
    uint16_t allowed_ports[64];            /**< Allowed TCP/UDP ports */
    uint32_t num_allowed_ports;            /**< Number of allowed ports */
} dsmil_network_policy_t;

/** Filesystem restrictions */
typedef struct {
    char allowed_paths[32][256];           /**< Allowed filesystem paths */
    uint32_t num_allowed_paths;            /**< Number of allowed paths */
    bool readonly;                         /**< All paths read-only */
} dsmil_filesystem_policy_t;

/** Complete sandbox profile */
typedef struct {
    char name[DSMIL_SANDBOX_MAX_NAME];     /**< Profile name */
    char description[256];                 /**< Human-readable description */

    dsmil_cap_bset_t cap_bset;             /**< Capability bounding set */
    dsmil_seccomp_prog_t seccomp_prog;     /**< Seccomp BPF program */
    dsmil_syscall_allowlist_t syscall_allowlist; /**< Or use allowlist */
    dsmil_resource_limits_t limits;        /**< Resource limits */
    dsmil_network_policy_t network;        /**< Network policy */
    dsmil_filesystem_policy_t filesystem;  /**< Filesystem policy */

    dsmil_sandbox_mode_t mode;             /**< Enforcement mode */
} dsmil_sandbox_profile_t;

/** @} */

/**
 * @defgroup DSMIL_SANDBOX_API API Functions
 * @{
 */

/**
 * @brief Load sandbox profile by name
 *
 * Loads profile from /etc/dsmil/sandbox/<name>.profile
 *
 * @param[in] profile_name Profile name
 * @param[out] profile Output profile structure
 * @return Result code
 */
dsmil_sandbox_result_t dsmil_load_sandbox_profile(
    const char *profile_name,
    dsmil_sandbox_profile_t *profile);

/**
 * @brief Apply sandbox profile to current process
 *
 * Must be called before any privileged operations. Typically called
 * from injected main() wrapper.
 *
 * @param[in] profile Sandbox profile
 * @return Result code
 */
dsmil_sandbox_result_t dsmil_apply_sandbox(const dsmil_sandbox_profile_t *profile);

/**
 * @brief Apply sandbox by profile name
 *
 * Convenience function that loads and applies profile.
 *
 * @param[in] profile_name Profile name
 * @return Result code
 */
dsmil_sandbox_result_t dsmil_apply_sandbox_by_name(const char *profile_name);

/**
 * @brief Free sandbox profile resources
 *
 * @param[in] profile Profile to free
 */
void dsmil_free_sandbox_profile(dsmil_sandbox_profile_t *profile);

/**
 * @brief Get current sandbox enforcement mode
 *
 * Can be overridden by environment variable DSMIL_SANDBOX_MODE.
 *
 * @return Current enforcement mode
 */
dsmil_sandbox_mode_t dsmil_get_sandbox_mode(void);

/**
 * @brief Set sandbox enforcement mode
 *
 * @param[in] mode New enforcement mode
 */
void dsmil_set_sandbox_mode(dsmil_sandbox_mode_t mode);

/**
 * @brief Convert result code to string
 *
 * @param[in] result Result code
 * @return Human-readable string
 */
const char *dsmil_sandbox_result_str(dsmil_sandbox_result_t result);

/** @} */

/**
 * @defgroup DSMIL_SANDBOX_LOWLEVEL Low-Level Functions
 * @{
 */

/**
 * @brief Apply capability bounding set
 *
 * @param[in] cap_bset Capability set
 * @return 0 on success, negative error code on failure
 */
int dsmil_apply_capabilities(const dsmil_cap_bset_t *cap_bset);

/**
 * @brief Install seccomp BPF filter
 *
 * @param[in] prog BPF program
 * @return 0 on success, negative error code on failure
 */
int dsmil_apply_seccomp(const dsmil_seccomp_prog_t *prog);

/**
 * @brief Install seccomp filter from syscall allowlist
 *
 * Generates BPF program that allows only listed syscalls.
 *
 * @param[in] allowlist Syscall allowlist
 * @return 0 on success, negative error code on failure
 */
int dsmil_apply_seccomp_allowlist(const dsmil_syscall_allowlist_t *allowlist);

/**
 * @brief Apply resource limits
 *
 * @param[in] limits Resource limits
 * @return 0 on success, negative error code on failure
 */
int dsmil_apply_resource_limits(const dsmil_resource_limits_t *limits);

/**
 * @brief Check if current process is sandboxed
 *
 * @return true if sandboxed, false otherwise
 */
bool dsmil_is_sandboxed(void);

/** @} */

/**
 * @defgroup DSMIL_SANDBOX_PROFILES Well-Known Profiles
 * @{
 */

/**
 * @brief Get predefined LLM worker profile
 *
 * Layer 7 LLM inference worker with minimal privileges:
 * - Capabilities: None
 * - Syscalls: read, write, mmap, munmap, brk, exit, futex, etc.
 * - Network: None
 * - Filesystem: Read-only access to model directory
 * - Memory limit: 16 GB
 *
 * @param[out] profile Output profile
 * @return Result code
 */
dsmil_sandbox_result_t dsmil_get_profile_llm_worker(dsmil_sandbox_profile_t *profile);

/**
 * @brief Get predefined network daemon profile
 *
 * Layer 5 network service with network access:
 * - Capabilities: CAP_NET_BIND_SERVICE
 * - Syscalls: network I/O + basic syscalls
 * - Network: Full access
 * - Filesystem: Read-only /etc, writable /var/run
 * - Memory limit: 4 GB
 *
 * @param[out] profile Output profile
 * @return Result code
 */
dsmil_sandbox_result_t dsmil_get_profile_network_daemon(dsmil_sandbox_profile_t *profile);

/**
 * @brief Get predefined crypto worker profile
 *
 * Layer 3 cryptographic operations:
 * - Capabilities: None (uses unprivileged crypto APIs)
 * - Syscalls: Limited to crypto + memory operations
 * - Network: None
 * - Filesystem: Read-only access to keys
 * - Memory limit: 2 GB
 *
 * @param[out] profile Output profile
 * @return Result code
 */
dsmil_sandbox_result_t dsmil_get_profile_crypto_worker(dsmil_sandbox_profile_t *profile);

/**
 * @brief Get predefined telemetry agent profile
 *
 * Layer 5 observability/telemetry:
 * - Capabilities: CAP_SYS_PTRACE (for process inspection)
 * - Syscalls: ptrace, process_vm_readv, etc.
 * - Network: Outbound only (metrics export)
 * - Filesystem: Read-only /proc, /sys
 * - Memory limit: 1 GB
 *
 * @param[out] profile Output profile
 * @return Result code
 */
dsmil_sandbox_result_t dsmil_get_profile_telemetry_agent(dsmil_sandbox_profile_t *profile);

/** @} */

/**
 * @defgroup DSMIL_SANDBOX_UTIL Utility Functions
 * @{
 */

/**
 * @brief Generate seccomp BPF from syscall allowlist
 *
 * @param[in] allowlist Syscall allowlist
 * @param[out] prog Output BPF program (caller must free filter)
 * @return 0 on success, negative error code on failure
 */
int dsmil_generate_seccomp_bpf(const dsmil_syscall_allowlist_t *allowlist,
                                dsmil_seccomp_prog_t *prog);

/**
 * @brief Parse profile from JSON file
 *
 * @param[in] json_path Path to JSON profile file
 * @param[out] profile Output profile
 * @return Result code
 */
dsmil_sandbox_result_t dsmil_parse_profile_json(const char *json_path,
                                                 dsmil_sandbox_profile_t *profile);

/**
 * @brief Export profile to JSON
 *
 * @param[in] profile Profile to export
 * @param[out] json_out JSON string (caller must free)
 * @return 0 on success, negative error code on failure
 */
int dsmil_profile_to_json(const dsmil_sandbox_profile_t *profile, char **json_out);

/**
 * @brief Validate profile consistency
 *
 * Checks for conflicting settings, ensures all required fields are set.
 *
 * @param[in] profile Profile to validate
 * @return Result code
 */
dsmil_sandbox_result_t dsmil_validate_profile(const dsmil_sandbox_profile_t *profile);

/** @} */

/**
 * @defgroup DSMIL_SANDBOX_MACROS Convenience Macros
 * @{
 */

/**
 * @brief Apply sandbox and exit on failure
 *
 * Typical usage in injected main():
 * @code
 * DSMIL_SANDBOX_APPLY_OR_DIE("l7_llm_worker");
 * // Proceed with sandboxed execution
 * @endcode
 */
#define DSMIL_SANDBOX_APPLY_OR_DIE(profile_name) \
    do { \
        dsmil_sandbox_result_t __res = dsmil_apply_sandbox_by_name(profile_name); \
        if (__res != DSMIL_SANDBOX_OK) { \
            fprintf(stderr, "FATAL: Sandbox setup failed: %s\n", \
                    dsmil_sandbox_result_str(__res)); \
            exit(1); \
        } \
    } while (0)

/**
 * @brief Apply sandbox with warning on failure
 *
 * Non-fatal version for development builds.
 */
#define DSMIL_SANDBOX_APPLY_OR_WARN(profile_name) \
    do { \
        dsmil_sandbox_result_t __res = dsmil_apply_sandbox_by_name(profile_name); \
        if (__res != DSMIL_SANDBOX_OK) { \
            fprintf(stderr, "WARNING: Sandbox setup failed: %s\n", \
                    dsmil_sandbox_result_str(__res)); \
        } \
    } while (0)

/** @} */

#ifdef __cplusplus
}
#endif

#endif /* DSMIL_SANDBOX_H */

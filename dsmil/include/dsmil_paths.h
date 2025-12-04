/**
 * @file dsmil_paths.h
 * @brief Dynamic Path Resolution Utilities for DSLLVM
 *
 * This header provides utilities for resolving paths dynamically at runtime,
 * supporting portable installations and flexible deployment configurations.
 *
 * Version: 1.6.1
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#ifndef DSMIL_PATHS_H
#define DSMIL_PATHS_H

#include <stddef.h>
#include <stdbool.h>
#include <sys/types.h>
#include <limits.h>  /* For PATH_MAX */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup DSMIL_PATH_CONFIG Path Configuration
 * @{
 */

/**
 * @brief Get the DSMIL installation prefix
 *
 * Checks environment variables in order:
 * 1. DSMIL_PREFIX (highest priority)
 * 2. DSMIL_INSTALL_PREFIX
 * 3. Default: "/opt/dsmil"
 *
 * @return Path prefix string (do not free)
 */
const char *dsmil_get_prefix(void);

/**
 * @brief Get the DSMIL configuration directory
 *
 * Checks environment variables in order:
 * 1. DSMIL_CONFIG_DIR
 * 2. DSMIL_PREFIX + "/etc"
 * 3. Default: "/etc/dsmil"
 *
 * @return Configuration directory path (do not free)
 */
const char *dsmil_get_config_dir(void);

/**
 * @brief Get the DSMIL binary directory
 *
 * Checks environment variables in order:
 * 1. DSMIL_BIN_DIR
 * 2. DSMIL_PREFIX + "/bin"
 * 3. Default: "/opt/dsmil/bin"
 *
 * @return Binary directory path (do not free)
 */
const char *dsmil_get_bin_dir(void);

/**
 * @brief Get the DSMIL library directory
 *
 * Checks environment variables in order:
 * 1. DSMIL_LIB_DIR
 * 2. DSMIL_PREFIX + "/lib"
 * 3. Default: "/opt/dsmil/lib"
 *
 * @return Library directory path (do not free)
 */
const char *dsmil_get_lib_dir(void);

/**
 * @brief Get the DSMIL data directory
 *
 * Checks environment variables in order:
 * 1. DSMIL_DATA_DIR
 * 2. DSMIL_PREFIX + "/share"
 * 3. Default: "/opt/dsmil/share"
 *
 * @return Data directory path (do not free)
 */
const char *dsmil_get_data_dir(void);

/**
 * @brief Get the DSMIL runtime directory
 *
 * Checks environment variables in order:
 * 1. DSMIL_RUNTIME_DIR
 * 2. XDG_RUNTIME_DIR + "/dsmil"
 * 3. Default: "/var/run/dsmil"
 *
 * @return Runtime directory path (do not free)
 */
const char *dsmil_get_runtime_dir(void);

/**
 * @brief Get the DSMIL truststore directory
 *
 * Checks environment variables in order:
 * 1. DSMIL_TRUSTSTORE_DIR
 * 2. DSMIL_CONFIG_DIR + "/truststore"
 * 3. Default: "/etc/dsmil/truststore"
 *
 * @return Truststore directory path (do not free)
 */
const char *dsmil_get_truststore_dir(void);

/**
 * @brief Get the DSMIL log directory
 *
 * Checks environment variables in order:
 * 1. DSMIL_LOG_DIR
 * 2. DSMIL_PREFIX + "/var/log"
 * 3. Default: "/var/log/dsmil"
 *
 * @return Log directory path (do not free)
 */
const char *dsmil_get_log_dir(void);

/**
 * @brief Get the DSMIL cache directory
 *
 * Checks environment variables in order:
 * 1. DSMIL_CACHE_DIR
 * 2. XDG_CACHE_HOME + "/dsmil"
 * 3. HOME + "/.cache/dsmil"
 * 4. Default: "/var/cache/dsmil"
 *
 * @return Cache directory path (do not free)
 */
const char *dsmil_get_cache_dir(void);

/**
 * @brief Get the DSMIL temporary directory
 *
 * Checks environment variables in order:
 * 1. DSMIL_TMP_DIR
 * 2. TMPDIR
 * 3. TMP
 * 4. Default: "/tmp"
 *
 * @return Temporary directory path (do not free)
 */
const char *dsmil_get_tmp_dir(void);

/**
 * @brief Resolve a path relative to DSMIL prefix
 *
 * Constructs a full path by combining prefix with relative path.
 * The returned string is statically allocated and should not be freed.
 *
 * @param relative_path Path relative to prefix (e.g., "bin/dsmil-clang")
 * @param buffer Output buffer (must be at least PATH_MAX bytes)
 * @param buffer_size Size of output buffer
 * @return Pointer to buffer on success, NULL on failure
 */
char *dsmil_resolve_path(const char *relative_path, char *buffer, size_t buffer_size);

/**
 * @brief Resolve a configuration file path
 *
 * Searches for configuration files in standard locations:
 * 1. DSMIL_CONFIG_DIR
 * 2. $HOME/.config/dsmil
 * 3. /etc/dsmil
 *
 * @param filename Configuration filename (e.g., "mission-profiles.json")
 * @param buffer Output buffer (must be at least PATH_MAX bytes)
 * @param buffer_size Size of output buffer
 * @return Pointer to buffer on success, NULL if not found
 */
char *dsmil_resolve_config(const char *filename, char *buffer, size_t buffer_size);

/**
 * @brief Resolve a binary path
 *
 * Searches for binaries in standard locations:
 * 1. DSMIL_BIN_DIR
 * 2. PATH environment variable
 * 3. /opt/dsmil/bin
 *
 * @param binary_name Binary name (e.g., "dsmil-clang")
 * @param buffer Output buffer (must be at least PATH_MAX bytes)
 * @param buffer_size Size of output buffer
 * @return Pointer to buffer on success, NULL if not found
 */
char *dsmil_resolve_binary(const char *binary_name, char *buffer, size_t buffer_size);

/**
 * @brief Check if a path exists and is accessible
 *
 * @param path Path to check
 * @return true if path exists and is accessible, false otherwise
 */
bool dsmil_path_exists(const char *path);

/**
 * @brief Ensure a directory exists, creating it if necessary
 *
 * Creates parent directories as needed (like mkdir -p).
 *
 * @param path Directory path to ensure exists
 * @param mode Directory permissions (e.g., 0755)
 * @return 0 on success, -1 on error (errno set)
 */
int dsmil_ensure_dir(const char *path, mode_t mode);

/**
 * @brief Get user-specific config directory
 *
 * Returns $HOME/.config/dsmil or XDG_CONFIG_HOME/dsmil
 *
 * @param buffer Output buffer (must be at least PATH_MAX bytes)
 * @param buffer_size Size of output buffer
 * @return Pointer to buffer on success, NULL on failure
 */
char *dsmil_get_user_config_dir(char *buffer, size_t buffer_size);

/**
 * @brief Initialize path resolution system
 *
 * Call this once at program startup to initialize path resolution.
 * Safe to call multiple times (idempotent).
 *
 * @return 0 on success, -1 on error
 */
int dsmil_paths_init(void);

/**
 * @brief Cleanup path resolution system
 *
 * Call this at program shutdown to free resources.
 * Safe to call multiple times.
 */
void dsmil_paths_cleanup(void);

/** @} */

#ifdef __cplusplus
}
#endif

#endif /* DSMIL_PATHS_H */

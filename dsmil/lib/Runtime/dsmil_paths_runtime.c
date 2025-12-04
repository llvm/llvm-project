/**
 * @file dsmil_paths_runtime.c
 * @brief Dynamic Path Resolution Runtime Implementation
 *
 * Implementation of path resolution utilities for portable DSLLVM installations.
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include "dsmil_paths.h"
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>
#include <limits.h>
#include <errno.h>

#ifdef _WIN32
#include <windows.h>
#define PATH_SEP '\\'
#else
#define PATH_SEP '/'
#endif

/* Static path cache */
static struct {
    bool initialized;
    char prefix[PATH_MAX];
    char config_dir[PATH_MAX];
    char bin_dir[PATH_MAX];
    char lib_dir[PATH_MAX];
    char data_dir[PATH_MAX];
    char runtime_dir[PATH_MAX];
    char truststore_dir[PATH_MAX];
    char log_dir[PATH_MAX];
    char cache_dir[PATH_MAX];
    char tmp_dir[PATH_MAX];
} path_cache = {0};

static void init_path_cache(void) {
    if (path_cache.initialized)
        return;

    const char *env_prefix = getenv("DSMIL_PREFIX");
    if (!env_prefix)
        env_prefix = getenv("DSMIL_INSTALL_PREFIX");
    if (!env_prefix)
        env_prefix = "/opt/dsmil";

    strncpy(path_cache.prefix, env_prefix, sizeof(path_cache.prefix) - 1);
    path_cache.prefix[sizeof(path_cache.prefix) - 1] = '\0';

    /* Config directory */
    const char *env_config = getenv("DSMIL_CONFIG_DIR");
    if (env_config) {
        strncpy(path_cache.config_dir, env_config, sizeof(path_cache.config_dir) - 1);
    } else {
        snprintf(path_cache.config_dir, sizeof(path_cache.config_dir), "%s/etc", path_cache.prefix);
    }
    path_cache.config_dir[sizeof(path_cache.config_dir) - 1] = '\0';

    /* Binary directory */
    const char *env_bin = getenv("DSMIL_BIN_DIR");
    if (env_bin) {
        strncpy(path_cache.bin_dir, env_bin, sizeof(path_cache.bin_dir) - 1);
    } else {
        snprintf(path_cache.bin_dir, sizeof(path_cache.bin_dir), "%s/bin", path_cache.prefix);
    }
    path_cache.bin_dir[sizeof(path_cache.bin_dir) - 1] = '\0';

    /* Library directory */
    const char *env_lib = getenv("DSMIL_LIB_DIR");
    if (env_lib) {
        strncpy(path_cache.lib_dir, env_lib, sizeof(path_cache.lib_dir) - 1);
    } else {
        snprintf(path_cache.lib_dir, sizeof(path_cache.lib_dir), "%s/lib", path_cache.prefix);
    }
    path_cache.lib_dir[sizeof(path_cache.lib_dir) - 1] = '\0';

    /* Data directory */
    const char *env_data = getenv("DSMIL_DATA_DIR");
    if (env_data) {
        strncpy(path_cache.data_dir, env_data, sizeof(path_cache.data_dir) - 1);
    } else {
        snprintf(path_cache.data_dir, sizeof(path_cache.data_dir), "%s/share", path_cache.prefix);
    }
    path_cache.data_dir[sizeof(path_cache.data_dir) - 1] = '\0';

    /* Runtime directory */
    const char *env_runtime = getenv("DSMIL_RUNTIME_DIR");
    if (env_runtime) {
        strncpy(path_cache.runtime_dir, env_runtime, sizeof(path_cache.runtime_dir) - 1);
    } else {
        const char *xdg_runtime = getenv("XDG_RUNTIME_DIR");
        if (xdg_runtime) {
            snprintf(path_cache.runtime_dir, sizeof(path_cache.runtime_dir), "%s/dsmil", xdg_runtime);
        } else {
            strncpy(path_cache.runtime_dir, "/var/run/dsmil", sizeof(path_cache.runtime_dir) - 1);
        }
    }
    path_cache.runtime_dir[sizeof(path_cache.runtime_dir) - 1] = '\0';

    /* Truststore directory */
    const char *env_truststore = getenv("DSMIL_TRUSTSTORE_DIR");
    if (env_truststore) {
        strncpy(path_cache.truststore_dir, env_truststore, sizeof(path_cache.truststore_dir) - 1);
    } else {
        snprintf(path_cache.truststore_dir, sizeof(path_cache.truststore_dir), "%s/truststore", path_cache.config_dir);
    }
    path_cache.truststore_dir[sizeof(path_cache.truststore_dir) - 1] = '\0';

    /* Log directory */
    const char *env_log = getenv("DSMIL_LOG_DIR");
    if (env_log) {
        strncpy(path_cache.log_dir, env_log, sizeof(path_cache.log_dir) - 1);
    } else {
        snprintf(path_cache.log_dir, sizeof(path_cache.log_dir), "%s/var/log", path_cache.prefix);
    }
    path_cache.log_dir[sizeof(path_cache.log_dir) - 1] = '\0';

    /* Cache directory */
    const char *env_cache = getenv("DSMIL_CACHE_DIR");
    if (env_cache) {
        strncpy(path_cache.cache_dir, env_cache, sizeof(path_cache.cache_dir) - 1);
    } else {
        const char *xdg_cache = getenv("XDG_CACHE_HOME");
        if (xdg_cache) {
            snprintf(path_cache.cache_dir, sizeof(path_cache.cache_dir), "%s/dsmil", xdg_cache);
        } else {
            const char *home = getenv("HOME");
            if (home) {
                snprintf(path_cache.cache_dir, sizeof(path_cache.cache_dir), "%s/.cache/dsmil", home);
            } else {
                strncpy(path_cache.cache_dir, "/var/cache/dsmil", sizeof(path_cache.cache_dir) - 1);
            }
        }
    }
    path_cache.cache_dir[sizeof(path_cache.cache_dir) - 1] = '\0';

    /* Temporary directory */
    const char *env_tmp = getenv("DSMIL_TMP_DIR");
    if (env_tmp) {
        strncpy(path_cache.tmp_dir, env_tmp, sizeof(path_cache.tmp_dir) - 1);
    } else {
        env_tmp = getenv("TMPDIR");
        if (!env_tmp)
            env_tmp = getenv("TMP");
        if (!env_tmp)
            env_tmp = "/tmp";
        strncpy(path_cache.tmp_dir, env_tmp, sizeof(path_cache.tmp_dir) - 1);
    }
    path_cache.tmp_dir[sizeof(path_cache.tmp_dir) - 1] = '\0';

    path_cache.initialized = true;
}

const char *dsmil_get_prefix(void) {
    init_path_cache();
    return path_cache.prefix;
}

const char *dsmil_get_config_dir(void) {
    init_path_cache();
    return path_cache.config_dir;
}

const char *dsmil_get_bin_dir(void) {
    init_path_cache();
    return path_cache.bin_dir;
}

const char *dsmil_get_lib_dir(void) {
    init_path_cache();
    return path_cache.lib_dir;
}

const char *dsmil_get_data_dir(void) {
    init_path_cache();
    return path_cache.data_dir;
}

const char *dsmil_get_runtime_dir(void) {
    init_path_cache();
    return path_cache.runtime_dir;
}

const char *dsmil_get_truststore_dir(void) {
    init_path_cache();
    return path_cache.truststore_dir;
}

const char *dsmil_get_log_dir(void) {
    init_path_cache();
    return path_cache.log_dir;
}

const char *dsmil_get_cache_dir(void) {
    init_path_cache();
    return path_cache.cache_dir;
}

const char *dsmil_get_tmp_dir(void) {
    init_path_cache();
    return path_cache.tmp_dir;
}

char *dsmil_resolve_path(const char *relative_path, char *buffer, size_t buffer_size) {
    if (!relative_path || !buffer || buffer_size == 0)
        return NULL;

    init_path_cache();

    size_t prefix_len = strlen(path_cache.prefix);
    size_t rel_len = strlen(relative_path);

    if (prefix_len + 1 + rel_len >= buffer_size)
        return NULL;

    snprintf(buffer, buffer_size, "%s%c%s", path_cache.prefix, PATH_SEP, relative_path);
    return buffer;
}

char *dsmil_resolve_config(const char *filename, char *buffer, size_t buffer_size) {
    if (!filename || !buffer || buffer_size == 0)
        return NULL;

    init_path_cache();

    /* Try config directory first */
    size_t config_len = strlen(path_cache.config_dir);
    size_t filename_len = strlen(filename);

    if (config_len + 1 + filename_len < buffer_size) {
        snprintf(buffer, buffer_size, "%s%c%s", path_cache.config_dir, PATH_SEP, filename);
        if (dsmil_path_exists(buffer))
            return buffer;
    }

    /* Try user config directory */
    char user_config[PATH_MAX];
    if (dsmil_get_user_config_dir(user_config, sizeof(user_config))) {
        if (strlen(user_config) + 1 + filename_len < buffer_size) {
            snprintf(buffer, buffer_size, "%s%c%s", user_config, PATH_SEP, filename);
            if (dsmil_path_exists(buffer))
                return buffer;
        }
    }

    return NULL;
}

char *dsmil_resolve_binary(const char *binary_name, char *buffer, size_t buffer_size) {
    if (!binary_name || !buffer || buffer_size == 0)
        return NULL;

    init_path_cache();

    /* Try bin directory first */
    size_t bin_len = strlen(path_cache.bin_dir);
    size_t binary_len = strlen(binary_name);

    if (bin_len + 1 + binary_len < buffer_size) {
        snprintf(buffer, buffer_size, "%s%c%s", path_cache.bin_dir, PATH_SEP, binary_name);
        if (dsmil_path_exists(buffer))
            return buffer;
    }

    /* Try PATH */
    const char *path_env = getenv("PATH");
    if (path_env) {
        char *path_copy = strdup(path_env);
        if (path_copy) {
            char *token = strtok(path_copy, ":");
            while (token) {
                if (strlen(token) + 1 + binary_len < buffer_size) {
                    snprintf(buffer, buffer_size, "%s%c%s", token, PATH_SEP, binary_name);
                    if (dsmil_path_exists(buffer)) {
                        free(path_copy);
                        return buffer;
                    }
                }
                token = strtok(NULL, ":");
            }
            free(path_copy);
        }
    }

    return NULL;
}

bool dsmil_path_exists(const char *path) {
    if (!path)
        return false;

    struct stat st;
    return (stat(path, &st) == 0);
}

int dsmil_ensure_dir(const char *path, mode_t mode) {
    if (!path)
        return -1;

    /* Check if already exists */
    struct stat st;
    if (stat(path, &st) == 0) {
        if (S_ISDIR(st.st_mode))
            return 0;
        errno = ENOTDIR;
        return -1;
    }

    /* Create parent directories first */
    char parent[PATH_MAX];
    strncpy(parent, path, sizeof(parent) - 1);
    parent[sizeof(parent) - 1] = '\0';

    char *last_sep = strrchr(parent, PATH_SEP);
    if (last_sep && last_sep != parent) {
        *last_sep = '\0';
        if (dsmil_ensure_dir(parent, mode) != 0)
            return -1;
    }

    /* Create directory */
#ifdef _WIN32
    if (mkdir(path) != 0)
        return -1;
#else
    if (mkdir(path, mode) != 0)
        return -1;
#endif

    return 0;
}

char *dsmil_get_user_config_dir(char *buffer, size_t buffer_size) {
    if (!buffer || buffer_size == 0)
        return NULL;

    const char *xdg_config = getenv("XDG_CONFIG_HOME");
    if (xdg_config) {
        if (strlen(xdg_config) + 6 < buffer_size) {
            snprintf(buffer, buffer_size, "%s/dsmil", xdg_config);
            return buffer;
        }
    }

    const char *home = getenv("HOME");
    if (home) {
        if (strlen(home) + 15 < buffer_size) {
            snprintf(buffer, buffer_size, "%s/.config/dsmil", home);
            return buffer;
        }
    }

    return NULL;
}

int dsmil_paths_init(void) {
    init_path_cache();
    return 0;
}

void dsmil_paths_cleanup(void) {
    /* Currently no cleanup needed, but reserved for future use */
    (void)0;
}

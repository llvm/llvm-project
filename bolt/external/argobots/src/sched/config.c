/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

#include "abti.h"

#include <stdlib.h>
#include <stdarg.h>
#include <string.h>

static inline size_t sched_config_type_size(ABT_sched_config_type type);

/** @defgroup SCHED_CONFIG Scheduler config
 * This group is for Scheduler config.
 */

/* Global configurable parameters */
ABT_sched_config_var ABT_sched_config_var_end = { .idx = -1,
                                                  .type =
                                                      ABT_SCHED_CONFIG_INT };

ABT_sched_config_var ABT_sched_config_access = { .idx = -2,
                                                 .type = ABT_SCHED_CONFIG_INT };

ABT_sched_config_var ABT_sched_config_automatic = { .idx = -3,
                                                    .type =
                                                        ABT_SCHED_CONFIG_INT };

/**
 * @ingroup SCHED_CONFIG
 * @brief   Create a scheduler configuration.
 *
 * This function is used to create a specific configuration of a scheduler. The
 * dynamic parameters are a list of tuples composed of the variable of type \c
 * ABT_sched_config_var and a value for this variable. The list must end with a
 * single value \c ABT_sched_config_var_end.
 *
 * For now the parameters can be
 *   - for all the schedulers
 *     - ABT_sched_config_access: to choose the access type of the
 *     automatically created pools (ABT_POOL_ACCESS_MPSC by default)
 *     - ABT_sched_config_automatic: to automatically free the scheduler when
 *     unused (ABT_TRUE by default)
 *   - for the basic scheduler:
 *     - ABT_sched_basic_freq; to set the frequency on checking events
 *
 * If you want to write your own scheduler and use this function, you can find
 * a good example in the test called \c sched_config.
 *
 * For example, if you want to configure the basic scheduler to have a
 * frequency for checking events equal to 5, you will have this call:
 * ABT_sched_config_create(&config, ABT_sched_basic_freq, 5,
 * ABT_sched_config_var_end);
 *
 * @param[out] config   configuration to create
 * @param[in]  ...      list of arguments
 * @return Error code
 * @retval ABT_SUCCESS on success
 */
int ABT_sched_config_create(ABT_sched_config *config, ...)
{
    int abt_errno;
    ABTI_sched_config *p_config;

    char *buffer = NULL;
    size_t alloc_size = 8 * sizeof(size_t);

    int num_params = 0;
    size_t offset = sizeof(num_params);

    size_t buffer_size = alloc_size;
    abt_errno = ABTU_malloc(buffer_size, (void **)&buffer);
    ABTI_CHECK_ERROR(abt_errno);

    va_list varg_list;
    va_start(varg_list, config);

    /* We read each couple (var, value) until we find ABT_sched_config_var_end
     */
    while (1) {
        ABT_sched_config_var var = va_arg(varg_list, ABT_sched_config_var);
        if (var.idx == ABT_sched_config_var_end.idx)
            break;

        int param = var.idx;
        ABT_sched_config_type type = var.type;
        num_params++;

        size_t size = sched_config_type_size(type);
        if (offset + sizeof(param) + sizeof(type) + size > buffer_size) {
            size_t cur_buffer_size = buffer_size;
            buffer_size += alloc_size;
            abt_errno =
                ABTU_realloc(cur_buffer_size, buffer_size, (void **)&buffer);
            if (ABTI_IS_ERROR_CHECK_ENABLED && abt_errno != ABT_SUCCESS) {
                ABTU_free(buffer);
                ABTI_HANDLE_ERROR(abt_errno);
            }
        }
        /* Copy the parameter index */
        memcpy(buffer + offset, (void *)&param, sizeof(param));
        offset += sizeof(param);

        /* Copy the size of the argument */
        memcpy(buffer + offset, (void *)&size, sizeof(size));
        offset += sizeof(size);

        /* Copy the argument */
        void *ptr;
        int i;
        double d;
        void *p;
        switch (type) {
            case ABT_SCHED_CONFIG_INT:
                i = va_arg(varg_list, int);
                ptr = (void *)&i;
                break;
            case ABT_SCHED_CONFIG_DOUBLE:
                d = va_arg(varg_list, double);
                ptr = (void *)&d;
                break;
            case ABT_SCHED_CONFIG_PTR:
                p = va_arg(varg_list, void *);
                ptr = (void *)&p;
                break;
            default:
                ABTI_HANDLE_ERROR(ABT_ERR_SCHED_CONFIG);
        }

        memcpy(buffer + offset, ptr, size);
        offset += size;
    }
    va_end(varg_list);

    if (num_params) {
        memcpy(buffer, (int *)&num_params, sizeof(num_params));
    } else {
        ABTU_free(buffer);
        buffer = NULL;
    }

    p_config = (ABTI_sched_config *)buffer;
    *config = ABTI_sched_config_get_handle(p_config);
    return ABT_SUCCESS;
}

/**
 * @ingroup SCHED_CONFIG
 * @brief   Copy the set values from config into the variables passed in the
 *          dynamic list of arguments.
 *
 * The arguments in \c ... are the addresses of the variables where to copy the
 * packed data. The packed data are copied to their corresponding variables.
 * For a good example, see the test \c sched_config.
 *
 * @param[in] config    configuration to read
 * @param[in] num_vars  number of variable addresses in \c ...
 * @param[in] ...       list of arguments
 * @return Error code
 * @retval ABT_SUCCESS on success
 */
int ABT_sched_config_read(ABT_sched_config config, int num_vars, ...)
{
    int abt_errno;
    int v;

    /* We read all the variables and save the addresses */
    void **variables;
    abt_errno = ABTU_malloc(num_vars * sizeof(void *), (void **)&variables);
    ABTI_CHECK_ERROR(abt_errno);

    va_list varg_list;
    va_start(varg_list, num_vars);
    for (v = 0; v < num_vars; v++) {
        variables[v] = va_arg(varg_list, void *);
    }
    va_end(varg_list);

    abt_errno = ABTI_sched_config_read(config, 1, num_vars, variables);
    ABTI_CHECK_ERROR(abt_errno);

    ABTU_free(variables);
    return ABT_SUCCESS;
}

/**
 * @ingroup SCHED_CONFIG
 * @brief   Free the configuration.
 *
 * @param[in,out] config  configuration to free
 * @return Error code
 * @retval ABT_SUCCESS on success
 */
int ABT_sched_config_free(ABT_sched_config *config)
{
    ABTI_sched_config *p_config = ABTI_sched_config_get_ptr(*config);
    ABTU_free(p_config);

    *config = ABT_SCHED_CONFIG_NULL;

    return ABT_SUCCESS;
}

/*****************************************************************************/
/* Private APIs                                                              */
/*****************************************************************************/

ABTU_ret_err int ABTI_sched_config_read_global(ABT_sched_config config,
                                               ABT_pool_access *access,
                                               ABT_bool *automatic)
{
    int abt_errno;
    int num_vars = 2;
    /* We use XXX_i variables because va_list converts these types into int */
    int access_i = -1;
    int automatic_i = -1;

    void **variables;
    abt_errno = ABTU_malloc(num_vars * sizeof(void *), (void **)&variables);
    ABTI_CHECK_ERROR(abt_errno);

    variables[(ABT_sched_config_access.idx + 2) * (-1)] = &access_i;
    variables[(ABT_sched_config_automatic.idx + 2) * (-1)] = &automatic_i;

    abt_errno = ABTI_sched_config_read(config, 0, num_vars, variables);
    ABTU_free(variables);
    ABTI_CHECK_ERROR(abt_errno);

    if (access_i != -1)
        *access = (ABT_pool_access)access_i;
    if (automatic_i != -1)
        *automatic = (ABT_bool)automatic_i;

    return ABT_SUCCESS;
}

/* type is 0 if we read the private parameters, else 1 */
ABTU_ret_err int ABTI_sched_config_read(ABT_sched_config config, int type,
                                        int num_vars, void **variables)
{
    size_t offset = 0;
    int num_params;

    if (config == ABT_SCHED_CONFIG_NULL) {
        return ABT_SUCCESS;
    }

    ABTI_sched_config *p_config = ABTI_sched_config_get_ptr(config);

    char *buffer = (char *)p_config;

    /* Number of parameters in buffer */
    memcpy(&num_params, buffer, sizeof(num_params));
    offset += sizeof(num_params);

    /* Copy the data from buffer to the right variables */
    int p;
    for (p = 0; p < num_params; p++) {
        int var_idx;
        size_t size;

        /* Get the variable index of the next parameter */
        memcpy(&var_idx, buffer + offset, sizeof(var_idx));
        offset += sizeof(var_idx);
        /* Get the size of the next parameter */
        memcpy(&size, buffer + offset, sizeof(size));
        offset += sizeof(size);
        /* Get the next argument */
        /* We save it only if
         *   - the index is < 0  when type == 0
         *   - the index is >= 0 when type == 1
         */
        if (type == 0) {
            if (var_idx < 0) {
                var_idx = (var_idx + 2) * -1;
                if (var_idx >= num_vars)
                    return ABT_ERR_INV_SCHED_CONFIG;
                memcpy(variables[var_idx], buffer + offset, size);
            }
        } else {
            if (var_idx >= 0) {
                if (var_idx >= num_vars)
                    return ABT_ERR_INV_SCHED_CONFIG;
                memcpy(variables[var_idx], buffer + offset, size);
            }
        }
        offset += size;
    }
    return ABT_SUCCESS;
}

/*****************************************************************************/
/* Internal static functions                                                 */
/*****************************************************************************/

static inline size_t sched_config_type_size(ABT_sched_config_type type)
{
    switch (type) {
        case ABT_SCHED_CONFIG_INT:
            return sizeof(int);
        case ABT_SCHED_CONFIG_DOUBLE:
            return sizeof(double);
        case ABT_SCHED_CONFIG_PTR:
            return sizeof(void *);
        default:
            ABTI_ASSERT(0);
            ABTU_unreachable();
    }
}

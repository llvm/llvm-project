/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

#include "abti.h"

ABTU_ret_err static int pool_create(ABT_pool_def *def, ABT_pool_config config,
                                    ABT_bool automatic, ABTI_pool **pp_newpool);

/** @defgroup POOL Pool
 * This group is for Pool.
 */

/**
 * @ingroup POOL
 * @brief   Create a new pool and return its handle through \c newpool.
 *
 * This function creates a new pool, given by a definition (\c def) and a
 * configuration (\c config). The configuration can be \c ABT_SCHED_CONFIG_NULL
 * or obtained from a specific function of the pool defined by \c def. The
 * configuration will be passed as the parameter of the initialization function
 * of the pool.
 *
 * @param[in]  def     definition required for pool creation
 * @param[in]  config  specific config used during the pool creation
 * @param[out] newpool handle to a new pool
 * @return Error code
 * @retval ABT_SUCCESS on success
 */
int ABT_pool_create(ABT_pool_def *def, ABT_pool_config config,
                    ABT_pool *newpool)
{
    ABTI_pool *p_newpool;
    int abt_errno = pool_create(def, config, ABT_FALSE, &p_newpool);
    ABTI_CHECK_ERROR(abt_errno);

    *newpool = ABTI_pool_get_handle(p_newpool);
    return ABT_SUCCESS;
}

/**
 * @ingroup POOL
 * @brief   Create a new pool from a predefined type and return its handle
 *          through \c newpool.
 *
 * For more details see \c ABT_pool_create().
 *
 * @param[in]  kind      name of the predefined pool
 * @param[in]  access    access type of the predefined pool
 * @param[in]  automatic ABT_TRUE if the pool should be automatically freed
 * @param[out] newpool   handle to a new pool
 * @return Error code
 * @retval ABT_SUCCESS on success
 */
int ABT_pool_create_basic(ABT_pool_kind kind, ABT_pool_access access,
                          ABT_bool automatic, ABT_pool *newpool)
{
    ABTI_pool *p_newpool;
    int abt_errno = ABTI_pool_create_basic(kind, access, automatic, &p_newpool);
    ABTI_CHECK_ERROR(abt_errno);

    *newpool = ABTI_pool_get_handle(p_newpool);
    return ABT_SUCCESS;
}

/**
 * @ingroup POOL
 * @brief   Free the given pool, and modify its value to ABT_POOL_NULL
 *
 * @param[inout] pool handle
 * @return Error code
 * @retval ABT_SUCCESS on success
 */
int ABT_pool_free(ABT_pool *pool)
{
    ABT_pool h_pool = *pool;
    ABTI_pool *p_pool = ABTI_pool_get_ptr(h_pool);

    ABTI_CHECK_TRUE(p_pool != NULL && h_pool != ABT_POOL_NULL,
                    ABT_ERR_INV_POOL);
    ABTI_pool_free(p_pool);

    *pool = ABT_POOL_NULL;
    return ABT_SUCCESS;
}

/**
 * @ingroup POOL
 * @brief   Get the access type of target pool
 *
 * @param[in]  pool    handle to the pool
 * @param[out] access  access type
 * @return Error code
 * @retval ABT_SUCCESS on success
 */
int ABT_pool_get_access(ABT_pool pool, ABT_pool_access *access)
{
    ABTI_pool *p_pool = ABTI_pool_get_ptr(pool);
    ABTI_CHECK_NULL_POOL_PTR(p_pool);

    *access = p_pool->access;
    return ABT_SUCCESS;
}

/**
 * @ingroup POOL
 * @brief   Return the total size of a pool
 *
 * The returned size is the number of elements in the pool (provided by the
 * specific function in case of a user-defined pool), plus the number of
 * blocked ULTs and migrating ULTs.
 *
 * @param[in] pool handle to the pool
 * @param[out] size size of the pool
 * @return Error code
 * @retval ABT_SUCCESS on success
 */
int ABT_pool_get_total_size(ABT_pool pool, size_t *size)
{
    ABTI_pool *p_pool = ABTI_pool_get_ptr(pool);
    ABTI_CHECK_NULL_POOL_PTR(p_pool);

    *size = ABTI_pool_get_total_size(p_pool);
    return ABT_SUCCESS;
}

/**
 * @ingroup POOL
 * @brief   Return the size of a pool
 *
 * The returned size is the number of elements in the pool (provided by the
 * specific function in case of a user-defined pool).
 *
 * @param[in] pool handle to the pool
 * @param[out] size size of the pool
 * @return Error code
 * @retval ABT_SUCCESS on success
 */
int ABT_pool_get_size(ABT_pool pool, size_t *size)
{
    ABTI_pool *p_pool = ABTI_pool_get_ptr(pool);
    ABTI_CHECK_NULL_POOL_PTR(p_pool);

    *size = ABTI_pool_get_size(p_pool);
    return ABT_SUCCESS;
}

/**
 * @ingroup POOL
 * @brief   Pop a unit from the target pool
 *
 * @param[in] pool handle to the pool
 * @param[out] p_unit handle to the unit
 * @return Error code
 * @retval ABT_SUCCESS on success
 */
int ABT_pool_pop(ABT_pool pool, ABT_unit *p_unit)
{
    ABTI_pool *p_pool = ABTI_pool_get_ptr(pool);
    ABTI_CHECK_NULL_POOL_PTR(p_pool);

    *p_unit = ABTI_pool_pop(p_pool);
    return ABT_SUCCESS;
}

/**
 * @ingroup POOL
 * @brief   Pop a unit from the target pool with wait
 *
 * \c ABT_pool_pop_wait pops a unit from a pool \c pool if a unit is in a pool;
 * otherwise, it suspends an underlying execution stream and waits in a pool.
 * \c time_secs directs how long \c ABT_pool_pop_wait suspends the underlying
 * execution stream.  A work unit successfully popped from \c pool is returned
 * via \c p_unit.  If no work unit is available, it returns ABT_UNIT_NULL.
 *
 * In most cases, \c ABT_pool_pop() is more efficient, but \c ABT_pool_pop_wait
 * is useful in cases where users want to make execution streams active only
 * when is available.
 *
 * @param[in]  pool       handle to the pool
 * @param[out] p_unit     handle to the unit
 * @param[in]  time_secs  duration of waiting time (seconds)
 * @return Error code
 * @retval ABT_SUCCESS on success
 */
int ABT_pool_pop_wait(ABT_pool pool, ABT_unit *p_unit, double time_secs)
{
    ABTI_pool *p_pool = ABTI_pool_get_ptr(pool);
    ABTI_CHECK_NULL_POOL_PTR(p_pool);

    *p_unit = ABTI_pool_pop_wait(p_pool, time_secs);
    return ABT_SUCCESS;
}

int ABT_pool_pop_timedwait(ABT_pool pool, ABT_unit *p_unit, double abstime_secs)
{
    ABTI_pool *p_pool = ABTI_pool_get_ptr(pool);
    ABTI_CHECK_NULL_POOL_PTR(p_pool);

    *p_unit = ABTI_pool_pop_timedwait(p_pool, abstime_secs);
    return ABT_SUCCESS;
}

/**
 * @ingroup POOL
 * @brief   Push a unit to the target pool
 *
 * @param[in] pool handle to the pool
 * @param[in] unit handle to the unit
 * @return Error code
 * @retval ABT_SUCCESS on success
 */
int ABT_pool_push(ABT_pool pool, ABT_unit unit)
{
    ABTI_pool *p_pool = ABTI_pool_get_ptr(pool);
    ABTI_CHECK_NULL_POOL_PTR(p_pool);

    ABTI_CHECK_TRUE(unit != ABT_UNIT_NULL, ABT_ERR_UNIT);

    /* Save the producer ES information in the pool */
    ABTI_pool_push(p_pool, unit);
    return ABT_SUCCESS;
}

/**
 * @ingroup POOL
 * @brief   Remove a specified unit from the target pool
 *
 * @param[in] pool handle to the pool
 * @param[in] unit handle to the unit
 * @return Error code
 * @retval ABT_SUCCESS on success
 */
int ABT_pool_remove(ABT_pool pool, ABT_unit unit)
{
    ABTI_pool *p_pool = ABTI_pool_get_ptr(pool);
    ABTI_CHECK_NULL_POOL_PTR(p_pool);

    int abt_errno = ABTI_pool_remove(p_pool, unit);
    ABTI_CHECK_ERROR(abt_errno);
    return ABT_SUCCESS;
}

/**
 * @ingroup POOL
 * @brief   Apply a print function to every unit in a pool using a user-defined
 *          function.
 *
 * This function applies \c print_fn to every unit in \c pool. As the name of
 * the argument implies, \c print_fn may not have any side effect;
 * \c ABT_pool_print_all() is for the purpose of debugging and profiling.  For
 * example, changing the state of \c ABT_unit in \c print_fn is forbidden.
 *
 * When \c pool does not support the print-all feature, ABT_ERR_POOL is
 * returned.
 *
 * @param[in] pool     handle to the pool
 * @param[in] arg      argument passed to \c print_fn
 * @param[in] print_fn user-defined print function
 * @return Error code
 * @retval ABT_SUCCESS on success
 */
int ABT_pool_print_all(ABT_pool pool, void *arg,
                       void (*print_fn)(void *, ABT_unit))
{
    ABTI_pool *p_pool = ABTI_pool_get_ptr(pool);
    ABTI_CHECK_NULL_POOL_PTR(p_pool);
    if (!p_pool->p_print_all) {
        ABTI_HANDLE_ERROR(ABT_ERR_POOL);
    }
    p_pool->p_print_all(pool, arg, print_fn);
    return ABT_SUCCESS;
}

/**
 * @ingroup POOL
 * @brief   Set the specific data of the target user-defined pool
 *
 * This function will be called by the user during the initialization of his
 * user-defined pool.
 *
 * @param[in] pool handle to the pool
 * @param[in] data specific data of the pool
 * @return Error code
 * @retval ABT_SUCCESS on success
 */
int ABT_pool_set_data(ABT_pool pool, void *data)
{
    ABTI_pool *p_pool = ABTI_pool_get_ptr(pool);
    ABTI_CHECK_NULL_POOL_PTR(p_pool);

    p_pool->data = data;
    return ABT_SUCCESS;
}

/**
 * @ingroup POOL
 * @brief   Retrieve the specific data of the target user-defined pool
 *
 * This function will be called by the user in a user-defined function of his
 * user-defined pool.
 *
 * @param[in] pool handle to the pool
 * @param[in] data specific data of the pool
 * @return Error code
 * @retval ABT_SUCCESS on success
 */
int ABT_pool_get_data(ABT_pool pool, void **data)
{
    ABTI_pool *p_pool = ABTI_pool_get_ptr(pool);
    ABTI_CHECK_NULL_POOL_PTR(p_pool);

    *data = p_pool->data;
    return ABT_SUCCESS;
}

/**
 * @ingroup POOL
 * @brief   Push a scheduler to a pool
 *
 * By pushing a scheduler, the user can change the running scheduler: when the
 * top scheduler (the running scheduler) will pick it from the pool and run it,
 * it will become the new scheduler. This new scheduler will be in charge until
 * it explicitly yields, except if ABT_sched_finish() or ABT_sched_exit() are
 * called.
 *
 * The scheduler should have been created by ABT_sched_create or
 * ABT_sched_create_basic.
 *
 * @param[in] pool handle to the pool
 * @param[in] sched handle to the sched
 * @return Error code
 * @retval ABT_SUCCESS on success
 */
int ABT_pool_add_sched(ABT_pool pool, ABT_sched sched)
{
    ABTI_local *p_local = ABTI_local_get_local();

    ABTI_pool *p_pool = ABTI_pool_get_ptr(pool);
    ABTI_CHECK_NULL_POOL_PTR(p_pool);

    ABTI_sched *p_sched = ABTI_sched_get_ptr(sched);
    ABTI_CHECK_NULL_SCHED_PTR(p_sched);

    /* Mark the scheduler as it is used in pool */
    ABTI_CHECK_TRUE(p_sched->used == ABTI_SCHED_NOT_USED, ABT_ERR_INV_SCHED);
    p_sched->used = ABTI_SCHED_IN_POOL;

    /* In both ABT_SCHED_TYPE_ULT and ABT_SCHED_TYPE_TASK cases, we use ULT-type
     * scheduler to reduce the code maintenance cost.  ABT_SCHED_TYPE_TASK
     * should be removed in the future. */
    int abt_errno = ABTI_ythread_create_sched(p_local, p_pool, p_sched);
    ABTI_CHECK_ERROR(abt_errno);
    return ABT_SUCCESS;
}

/**
 * @ingroup POOL
 * @brief   Get the ID of the target pool
 *
 * \c ABT_pool_get_id() returns the ID of \c pool.
 *
 * @param[in]  pool  handle to the target pool
 * @param[out] id    pool id
 * @return Error code
 * @retval ABT_SUCCESS on success
 */
int ABT_pool_get_id(ABT_pool pool, int *id)
{
    ABTI_pool *p_pool = ABTI_pool_get_ptr(pool);
    ABTI_CHECK_NULL_POOL_PTR(p_pool);

    *id = (int)p_pool->id;
    return ABT_SUCCESS;
}

/*****************************************************************************/
/* Private APIs                                                              */
/*****************************************************************************/

ABTU_ret_err int ABTI_pool_create_basic(ABT_pool_kind kind,
                                        ABT_pool_access access,
                                        ABT_bool automatic,
                                        ABTI_pool **pp_newpool)
{
    int abt_errno;
    ABT_pool_def def;

    switch (kind) {
        case ABT_POOL_FIFO:
            abt_errno = ABTI_pool_get_fifo_def(access, &def);
            break;
        case ABT_POOL_FIFO_WAIT:
            abt_errno = ABTI_pool_get_fifo_wait_def(access, &def);
            break;
        default:
            abt_errno = ABT_ERR_INV_POOL_KIND;
            break;
    }
    ABTI_CHECK_ERROR(abt_errno);

    abt_errno = pool_create(&def, ABT_POOL_CONFIG_NULL, automatic, pp_newpool);
    ABTI_CHECK_ERROR(abt_errno);
    return ABT_SUCCESS;
}

void ABTI_pool_free(ABTI_pool *p_pool)
{
    LOG_DEBUG("[P%" PRIu64 "] freed\n", p_pool->id);
    ABT_pool h_pool = ABTI_pool_get_handle(p_pool);
    p_pool->p_free(h_pool);
    ABTU_free(p_pool);
}

void ABTI_pool_print(ABTI_pool *p_pool, FILE *p_os, int indent)
{
    if (p_pool == NULL) {
        fprintf(p_os, "%*s== NULL POOL ==\n", indent, "");
    } else {
        const char *access;

        switch (p_pool->access) {
            case ABT_POOL_ACCESS_PRIV:
                access = "PRIV";
                break;
            case ABT_POOL_ACCESS_SPSC:
                access = "SPSC";
                break;
            case ABT_POOL_ACCESS_MPSC:
                access = "MPSC";
                break;
            case ABT_POOL_ACCESS_SPMC:
                access = "SPMC";
                break;
            case ABT_POOL_ACCESS_MPMC:
                access = "MPMC";
                break;
            default:
                access = "UNKNOWN";
                break;
        }

        fprintf(p_os,
                "%*s== POOL (%p) ==\n"
                "%*sid            : %" PRIu64 "\n"
                "%*saccess        : %s\n"
                "%*sautomatic     : %s\n"
                "%*snum_scheds    : %d\n"
                "%*ssize          : %zu\n"
                "%*snum_blocked   : %d\n"
                "%*snum_migrations: %d\n"
                "%*sdata          : %p\n",
                indent, "", (void *)p_pool, indent, "", p_pool->id, indent, "",
                access, indent, "",
                (p_pool->automatic == ABT_TRUE) ? "TRUE" : "FALSE", indent, "",
                ABTD_atomic_acquire_load_int32(&p_pool->num_scheds), indent, "",
                ABTI_pool_get_size(p_pool), indent, "",
                ABTD_atomic_acquire_load_int32(&p_pool->num_blocked), indent,
                "", ABTD_atomic_acquire_load_int32(&p_pool->num_migrations),
                indent, "", p_pool->data);
    }
    fflush(p_os);
}

static ABTD_atomic_uint64 g_pool_id = ABTD_ATOMIC_UINT64_STATIC_INITIALIZER(0);
void ABTI_pool_reset_id(void)
{
    ABTD_atomic_release_store_uint64(&g_pool_id, 0);
}

/*****************************************************************************/
/* Internal static functions                                                 */
/*****************************************************************************/

static inline uint64_t pool_get_new_id(void);
ABTU_ret_err static int pool_create(ABT_pool_def *def, ABT_pool_config config,
                                    ABT_bool automatic, ABTI_pool **pp_newpool)
{
    int abt_errno;
    ABTI_pool *p_pool;
    abt_errno = ABTU_malloc(sizeof(ABTI_pool), (void **)&p_pool);
    ABTI_CHECK_ERROR(abt_errno);

    p_pool->access = def->access;
    p_pool->automatic = automatic;
    ABTD_atomic_release_store_int32(&p_pool->num_scheds, 0);
    ABTD_atomic_release_store_int32(&p_pool->num_blocked, 0);
    ABTD_atomic_release_store_int32(&p_pool->num_migrations, 0);
    p_pool->data = NULL;

    /* Set up the pool functions from def */
    p_pool->u_get_type = def->u_get_type;
    p_pool->u_get_thread = def->u_get_thread;
    p_pool->u_get_task = def->u_get_task;
    p_pool->u_is_in_pool = def->u_is_in_pool;
    p_pool->u_create_from_thread = def->u_create_from_thread;
    p_pool->u_create_from_task = def->u_create_from_task;
    p_pool->u_free = def->u_free;
    p_pool->p_init = def->p_init;
    p_pool->p_get_size = def->p_get_size;
    p_pool->p_push = def->p_push;
    p_pool->p_pop = def->p_pop;
    p_pool->p_pop_wait = def->p_pop_wait;
    p_pool->p_pop_timedwait = def->p_pop_timedwait;
    p_pool->p_remove = def->p_remove;
    p_pool->p_free = def->p_free;
    p_pool->p_print_all = def->p_print_all;
    p_pool->id = pool_get_new_id();
    LOG_DEBUG("[P%" PRIu64 "] created\n", p_pool->id);

    /* Configure the pool */
    if (p_pool->p_init) {
        abt_errno = p_pool->p_init(ABTI_pool_get_handle(p_pool), config);
        if (abt_errno != ABT_SUCCESS) {
            ABTU_free(p_pool);
            return abt_errno;
        }
    }
    *pp_newpool = p_pool;
    return ABT_SUCCESS;
}

static inline uint64_t pool_get_new_id(void)
{
    return (uint64_t)ABTD_atomic_fetch_add_uint64(&g_pool_id, 1);
}

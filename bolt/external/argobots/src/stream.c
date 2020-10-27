/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

#include "abti.h"

ABTU_ret_err static int xstream_create(ABTI_sched *p_sched,
                                       ABTI_xstream_type xstream_type, int rank,
                                       ABTI_xstream **pp_xstream);
ABTU_ret_err static int xstream_start(ABTI_xstream *p_xstream);
ABTU_ret_err static int xstream_join(ABTI_local **pp_local,
                                     ABTI_xstream *p_xstream);
static ABT_bool xstream_set_new_rank(ABTI_xstream *p_newxstream, int rank);
static void xstream_return_rank(ABTI_xstream *p_xstream);
static inline void xstream_schedule_ythread(ABTI_xstream **pp_local_xstream,
                                            ABTI_ythread *p_ythread);
static inline void xstream_schedule_task(ABTI_xstream *p_local_xstream,
                                         ABTI_thread *p_task);
static void xstream_init_main_sched(ABTI_xstream *p_xstream,
                                    ABTI_sched *p_sched);
ABTU_ret_err static int
xstream_update_main_sched(ABTI_xstream **pp_local_xstream,
                          ABTI_xstream *p_xstream, ABTI_sched *p_sched);
static void *xstream_launch_root_ythread(void *p_xstream);
#ifndef ABT_CONFIG_DISABLE_MIGRATION
ABTU_ret_err static int xstream_migrate_thread(ABTI_local *p_local,
                                               ABTI_thread *p_thread);
#endif

/** @defgroup ES Execution Stream (ES)
 * This group is for Execution Stream.
 */

/**
 * @ingroup ES
 * @brief   Create a new ES and return its handle through newxstream.
 *
 * @param[in]  sched  handle to the scheduler used for a new ES. If this is
 *                    ABT_SCHED_NULL, the runtime-provided scheduler is used.
 * @param[out] newxstream  handle to a newly created ES. This cannot be NULL
 *                    because unnamed ES is not allowed.
 * @return Error code
 * @retval ABT_SUCCESS on success
 */
int ABT_xstream_create(ABT_sched sched, ABT_xstream *newxstream)
{
    int abt_errno;
    ABTI_sched *p_sched;
    ABTI_xstream *p_newxstream;

    if (sched == ABT_SCHED_NULL) {
        abt_errno = ABTI_sched_create_basic(ABT_SCHED_DEFAULT, 0, NULL,
                                            ABT_SCHED_CONFIG_NULL, &p_sched);
        ABTI_CHECK_ERROR(abt_errno);
    } else {
        p_sched = ABTI_sched_get_ptr(sched);
        ABTI_CHECK_TRUE(p_sched->used == ABTI_SCHED_NOT_USED,
                        ABT_ERR_INV_SCHED);
    }

    abt_errno =
        xstream_create(p_sched, ABTI_XSTREAM_TYPE_SECONDARY, -1, &p_newxstream);
    ABTI_CHECK_ERROR(abt_errno);

    /* Start this ES */
    abt_errno = xstream_start(p_newxstream);
    ABTI_CHECK_ERROR(abt_errno);

    /* Return value */
    *newxstream = ABTI_xstream_get_handle(p_newxstream);
    return ABT_SUCCESS;
}

/**
 * @ingroup ES
 * @brief   Create a new ES with a predefined scheduler and return its handle
 *          through \c newxstream.
 *
 * If \c predef is a scheduler that includes automatic creation of pools,
 * \c pools will be equal to NULL.
 *
 * @param[in]  predef       predefined scheduler
 * @param[in]  num_pools    number of pools associated with this scheduler
 * @param[in]  pools        pools associated with this scheduler
 * @param[in]  config       specific config used during the scheduler creation
 * @param[out] newxstream   handle to the target ES
 * @return Error code
 * @retval ABT_SUCCESS on success
 */
int ABT_xstream_create_basic(ABT_sched_predef predef, int num_pools,
                             ABT_pool *pools, ABT_sched_config config,
                             ABT_xstream *newxstream)
{
    int abt_errno;
    ABTI_xstream *p_newxstream;

    ABTI_sched *p_sched;
    abt_errno =
        ABTI_sched_create_basic(predef, num_pools, pools, config, &p_sched);
    ABTI_CHECK_ERROR(abt_errno);

    abt_errno =
        xstream_create(p_sched, ABTI_XSTREAM_TYPE_SECONDARY, -1, &p_newxstream);
    ABTI_CHECK_ERROR(abt_errno);

    /* Start this ES */
    abt_errno = xstream_start(p_newxstream);
    ABTI_CHECK_ERROR(abt_errno);

    *newxstream = ABTI_xstream_get_handle(p_newxstream);
    return ABT_SUCCESS;
}

/**
 * @ingroup ES
 * @brief   Create a new ES with a specific rank.
 *
 * @param[in]  sched  handle to the scheduler used for a new ES. If this is
 *                    ABT_SCHED_NULL, the runtime-provided scheduler is used.
 * @param[in]  rank   target rank
 * @param[out] newxstream  handle to a newly created ES. This cannot be NULL
 *                    because unnamed ES is not allowed.
 * @return Error code
 * @retval ABT_SUCCESS               on success
 * @retval ABT_ERR_INV_XSTREAM_RANK  invalid rank
 */
int ABT_xstream_create_with_rank(ABT_sched sched, int rank,
                                 ABT_xstream *newxstream)
{
    int abt_errno;
    ABTI_sched *p_sched;
    ABTI_xstream *p_newxstream;

    ABTI_CHECK_TRUE(rank >= 0, ABT_ERR_INV_XSTREAM_RANK);

    if (sched == ABT_SCHED_NULL) {
        abt_errno = ABTI_sched_create_basic(ABT_SCHED_DEFAULT, 0, NULL,
                                            ABT_SCHED_CONFIG_NULL, &p_sched);
        ABTI_CHECK_ERROR(abt_errno);
    } else {
        p_sched = ABTI_sched_get_ptr(sched);
        ABTI_CHECK_TRUE(p_sched->used == ABTI_SCHED_NOT_USED,
                        ABT_ERR_INV_SCHED);
    }

    abt_errno = xstream_create(p_sched, ABTI_XSTREAM_TYPE_SECONDARY, rank,
                               &p_newxstream);
    if (ABTI_IS_ERROR_CHECK_ENABLED && abt_errno != ABT_SUCCESS) {
        if (sched == ABT_SCHED_NULL)
            ABTI_sched_free(ABTI_local_get_local_uninlined(), p_sched,
                            ABT_FALSE);
        ABTI_HANDLE_ERROR(abt_errno);
    }

    /* Start this ES */
    abt_errno = xstream_start(p_newxstream);
    ABTI_CHECK_ERROR(abt_errno);

    /* Return value */
    *newxstream = ABTI_xstream_get_handle(p_newxstream);
    return ABT_SUCCESS;
}

/**
 * @ingroup ES
 * @brief   Restart an ES that has been joined by \c ABT_xstream_join().
 *
 * @param[in] xstream  handle to an ES that has been joined but not freed.
 * @return Error code
 * @retval ABT_SUCCESS on success
 */
int ABT_xstream_revive(ABT_xstream xstream)
{
    ABTI_local *p_local = ABTI_local_get_local();
    ABTI_xstream *p_xstream = ABTI_xstream_get_ptr(xstream);
    ABTI_CHECK_NULL_XSTREAM_PTR(p_xstream);

    /* Revives the main scheduler thread. */
    ABTI_sched *p_main_sched = p_xstream->p_main_sched;
    ABTI_ythread *p_main_sched_ythread = p_main_sched->p_ythread;
    ABTI_CHECK_TRUE(ABTD_atomic_relaxed_load_int(
                        &p_main_sched_ythread->thread.state) ==
                        ABT_THREAD_STATE_TERMINATED,
                    ABT_ERR_INV_THREAD);

    ABTD_atomic_relaxed_store_uint32(&p_main_sched->request, 0);
    ABTI_tool_event_thread_join(p_local, &p_main_sched_ythread->thread,
                                ABTI_local_get_xstream_or_null(p_local)
                                    ? ABTI_local_get_xstream(p_local)->p_thread
                                    : NULL);

    ABTI_thread_revive(p_local, p_xstream->p_root_pool,
                       p_main_sched_ythread->thread.f_thread,
                       p_main_sched_ythread->thread.p_arg,
                       &p_main_sched_ythread->thread);

    ABTD_atomic_relaxed_store_int(&p_xstream->state, ABT_XSTREAM_STATE_RUNNING);
    ABTD_xstream_context_revive(&p_xstream->ctx);
    return ABT_SUCCESS;
}

/**
 * @ingroup ES
 * @brief   Release the ES object associated with ES handle.
 *
 * This routine deallocates memory used for the ES object. If the xstream
 * is still running when this routine is called, the deallocation happens
 * after the xstream terminates and then this routine returns. If it is
 * successfully processed, xstream is set as ABT_XSTREAM_NULL. The primary
 * ES cannot be freed with this routine.
 *
 * @param[in,out] xstream  handle to the target ES
 * @return Error code
 * @retval ABT_SUCCESS on success
 */
int ABT_xstream_free(ABT_xstream *xstream)
{
    ABTI_local *p_local = ABTI_local_get_local();
    ABT_xstream h_xstream = *xstream;

    ABTI_xstream *p_xstream = ABTI_xstream_get_ptr(h_xstream);
    if (p_xstream == NULL)
        return ABT_SUCCESS;

    /* We first need to check whether p_local_xstream is NULL because this
     * routine might be called by external threads. */
    ABTI_CHECK_TRUE_MSG(p_xstream != ABTI_local_get_xstream_or_null(p_local),
                        ABT_ERR_INV_XSTREAM,
                        "The current xstream cannot be freed.");

    ABTI_CHECK_TRUE_MSG(p_xstream->type != ABTI_XSTREAM_TYPE_PRIMARY,
                        ABT_ERR_INV_XSTREAM,
                        "The primary xstream cannot be freed explicitly.");

    /* Wait until xstream terminates */
    int abt_errno = xstream_join(&p_local, p_xstream);
    ABTI_CHECK_ERROR(abt_errno);

    /* Free the xstream object */
    ABTI_xstream_free(p_local, p_xstream, ABT_FALSE);

    /* Return value */
    *xstream = ABT_XSTREAM_NULL;
    return ABT_SUCCESS;
}

/**
 * @ingroup ES
 * @brief   Wait for xstream to terminate.
 *
 * The target xstream cannot be the same as the xstream associated with calling
 * thread. If they are identical, this routine returns immediately without
 * waiting for the xstream's termination.
 *
 * @param[in] xstream  handle to the target ES
 * @return Error code
 * @retval ABT_SUCCESS on success
 */

int ABT_xstream_join(ABT_xstream xstream)
{
    ABTI_local *p_local = ABTI_local_get_local();
    ABTI_xstream *p_xstream = ABTI_xstream_get_ptr(xstream);
    ABTI_CHECK_NULL_XSTREAM_PTR(p_xstream);

    int abt_errno = xstream_join(&p_local, p_xstream);
    ABTI_CHECK_ERROR(abt_errno);
    return ABT_SUCCESS;
}

/**
 * @ingroup ES
 * @brief   Terminate the ES associated with the calling ULT.
 *
 * Since the calling ULT's ES terminates, this routine never returns.
 * Tasklets are not allowed to call this routine.
 *
 * @return Error code
 * @retval ABT_SUCCESS           on success
 * @retval ABT_ERR_UNINITIALIZED Argobots has not been initialized
 * @retval ABT_ERR_INV_XSTREAM   called by an external thread
 * @retval ABT_ERR_INV_THREAD    called by a non-yieldable thread (tasklet)
 */
int ABT_xstream_exit(void)
{
    ABTI_xstream *p_local_xstream;
    ABTI_ythread *p_ythread;
    ABTI_SETUP_LOCAL_YTHREAD_WITH_INIT_CHECK(&p_local_xstream, &p_ythread);

    /* Terminate the main scheduler. */
    ABTD_atomic_fetch_or_uint32(&p_local_xstream->p_main_sched->p_ythread
                                     ->thread.request,
                                ABTI_THREAD_REQ_TERMINATE);
    /* Terminate this ULT */
    ABTI_ythread_exit(p_local_xstream, p_ythread);
    ABTU_unreachable();
    return ABT_SUCCESS;
}

/**
 * @ingroup ES
 * @brief   Request the cancellation of the target ES.
 *
 * @param[in] xstream  handle to the target ES
 * @return Error code
 * @retval ABT_SUCCESS on success
 */
int ABT_xstream_cancel(ABT_xstream xstream)
{
    ABTI_xstream *p_xstream = ABTI_xstream_get_ptr(xstream);

    ABTI_CHECK_TRUE_MSG(p_xstream->type != ABTI_XSTREAM_TYPE_PRIMARY,
                        ABT_ERR_INV_XSTREAM,
                        "The primary xstream cannot be canceled.");

    /* Terminate the main scheduler of the target xstream. */
    ABTD_atomic_fetch_or_uint32(&p_xstream->p_main_sched->p_ythread->thread
                                     .request,
                                ABTI_THREAD_REQ_TERMINATE);
    return ABT_SUCCESS;
}

/**
 * @ingroup ES
 * @brief   Return the ES handle associated with the caller work unit.
 *
 * \c ABT_xstream_self() returns the handle to ES object associated with
 * the caller work unit through \c xstream.
 *
 * At present \c xstream is set to \c ABT_XSTREAM_NULL when an error occurs,
 * but this behavior is deprecated.  The program should not rely on this
 * behavior.
 *
 * @param[out] xstream  ES handle
 * @return Error code
 * @retval ABT_SUCCESS           on success
 * @retval ABT_ERR_UNINITIALIZED Argobots has not been initialized
 * @retval ABT_ERR_INV_XSTREAM   called by an external thread
 */
int ABT_xstream_self(ABT_xstream *xstream)
{
    *xstream = ABT_XSTREAM_NULL;

    ABTI_xstream *p_local_xstream;
    ABTI_SETUP_LOCAL_XSTREAM_WITH_INIT_CHECK(&p_local_xstream);

    /* Return value */
    *xstream = ABTI_xstream_get_handle(p_local_xstream);
    return ABT_SUCCESS;
}

/**
 * @ingroup ES
 * @brief   Return the rank of ES associated with the caller work unit.
 *
 * @param[out] rank  ES rank
 * @return Error code
 * @retval ABT_SUCCESS           on success
 * @retval ABT_ERR_UNINITIALIZED Argobots has not been initialized
 * @retval ABT_ERR_INV_XSTREAM   called by an external thread, e.g., pthread
 */
int ABT_xstream_self_rank(int *rank)
{
    ABTI_xstream *p_local_xstream;
    ABTI_SETUP_LOCAL_XSTREAM_WITH_INIT_CHECK(&p_local_xstream);

    /* Return value */
    *rank = (int)p_local_xstream->rank;
    return ABT_SUCCESS;
}

/**
 * @ingroup ES
 * @brief   Set the rank for target ES
 *
 * @param[in] xstream  handle to the target ES
 * @param[in] rank     ES rank
 * @return Error code
 * @retval ABT_SUCCESS on success
 */
int ABT_xstream_set_rank(ABT_xstream xstream, int rank)
{
    ABTI_xstream *p_xstream = ABTI_xstream_get_ptr(xstream);
    ABTI_CHECK_NULL_XSTREAM_PTR(p_xstream);

    p_xstream->rank = rank;

    /* Set the CPU affinity for the ES */
    if (gp_ABTI_global->set_affinity == ABT_TRUE) {
        ABTD_affinity_cpuset_apply_default(&p_xstream->ctx, p_xstream->rank);
    }
    return ABT_SUCCESS;
}

/**
 * @ingroup ES
 * @brief   Return the rank of ES
 *
 * @param[in]  xstream  handle to the target ES
 * @param[out] rank     ES rank
 * @return Error code
 * @retval ABT_SUCCESS on success
 */
int ABT_xstream_get_rank(ABT_xstream xstream, int *rank)
{
    ABTI_xstream *p_xstream = ABTI_xstream_get_ptr(xstream);
    ABTI_CHECK_NULL_XSTREAM_PTR(p_xstream);

    *rank = (int)p_xstream->rank;
    return ABT_SUCCESS;
}

/**
 * @ingroup ES
 * @brief   Set the main scheduler of the target ES.
 *
 * \c ABT_xstream_set_main_sched() sets \c sched as the main scheduler for
 * \c xstream.  The scheduler \c sched will first run when the ES \c xstream is
 * started.  Only ULTs can call this routine.
 * If \c xstream is a handle to the primary ES, \c sched will be automatically
 * freed on \c ABT_finalize() or when the main scheduler of the primary ES is
 * changed again.  In this case, the explicit call \c ABT_sched_free() for
 * \c sched may cause undefined behavior.
 *
 * NOTE: The current implementation of this routine has some limitations.
 * 1. If the target ES \c xstream is running, the caller ULT must be running on
 * the same ES. However, if the target ES is not in the RUNNING state, the
 * caller can be any ULT that is running on any ES.
 * 2. If the current main scheduler of \c xstream has work units residing in
 * its associated pools, this routine will not be successful. In this case, the
 * user has to complete all work units in the main scheduler's pools or migrate
 * them to unassociated pools.
 *
 * @param[in] xstream  handle to the target ES
 * @param[in] sched    handle to the scheduler
 * @return Error code
 * @retval ABT_SUCCESS          on success
 * @retval ABT_ERR_XSTREAM      the current main scheduler of \c xstream has
 *                              work units in its associated pools
 * @retval ABT_ERR_INV_XSTREAM  called by an external thread
 * @retval ABT_ERR_INV_THREAD   called by a non-yieldable thread (tasklet)
 */
int ABT_xstream_set_main_sched(ABT_xstream xstream, ABT_sched sched)
{
    int abt_errno;

    ABTI_xstream *p_xstream = ABTI_xstream_get_ptr(xstream);
    ABTI_CHECK_NULL_XSTREAM_PTR(p_xstream);

    ABTI_xstream *p_local_xstream;
    ABTI_ythread *p_self;
    ABTI_SETUP_LOCAL_YTHREAD(&p_local_xstream, &p_self);

    /* For now, if the target ES is running, we allow to change the main
     * scheduler of the ES only when the caller is running on the same ES. */
    /* TODO: a new state representing that the scheduler is changed is needed
     * to avoid running xstreams while the scheduler is changed in this
     * function. */
    if (ABTD_atomic_acquire_load_int(&p_xstream->state) ==
        ABT_XSTREAM_STATE_RUNNING) {
        if (p_self->thread.p_last_xstream != p_xstream) {
            ABTI_HANDLE_ERROR(ABT_ERR_XSTREAM_STATE);
        }
    }

    /* TODO: permit to change the scheduler even when having work units in pools
     */
    if (p_xstream->p_main_sched) {
        /* We only allow to change the main scheduler when the current main
         * scheduler of p_xstream has no work unit in its associated pools. */
        if (ABTI_sched_get_effective_size(ABTI_xstream_get_local(
                                              p_local_xstream),
                                          p_xstream->p_main_sched) > 0) {
            ABTI_HANDLE_ERROR(ABT_ERR_XSTREAM);
        }
    }

    ABTI_sched *p_sched;
    if (sched == ABT_SCHED_NULL) {
        abt_errno = ABTI_sched_create_basic(ABT_SCHED_DEFAULT, 0, NULL,
                                            ABT_SCHED_CONFIG_NULL, &p_sched);
        ABTI_CHECK_ERROR(abt_errno);
    } else {
        p_sched = ABTI_sched_get_ptr(sched);
        ABTI_CHECK_TRUE(p_sched->used == ABTI_SCHED_NOT_USED,
                        ABT_ERR_INV_SCHED);
    }

    abt_errno = xstream_update_main_sched(&p_local_xstream, p_xstream, p_sched);
    ABTI_CHECK_ERROR(abt_errno);
    return ABT_SUCCESS;
}

/**
 * @ingroup ES
 * @brief   Set the main scheduler for \c xstream with a predefined scheduler.
 *
 * See \c ABT_xstream_set_main_sched() for more details.
 *
 * @param[in] xstream     handle to the target ES
 * @param[in] predef      predefined scheduler
 * @param[in] num_pools   number of pools associated with this scheduler
 * @param[in] pools       pools associated with this scheduler
 * @return Error code
 * @retval ABT_SUCCESS          on success
 * @retval ABT_ERR_INV_XSTREAM  called by an external thread
 * @retval ABT_ERR_INV_THREAD   called by a non-yieldable thread (tasklet)
 */
int ABT_xstream_set_main_sched_basic(ABT_xstream xstream,
                                     ABT_sched_predef predef, int num_pools,
                                     ABT_pool *pools)
{
    int abt_errno;

    ABTI_xstream *p_local_xstream;
    ABTI_SETUP_LOCAL_YTHREAD(&p_local_xstream, NULL);

    ABTI_xstream *p_xstream = ABTI_xstream_get_ptr(xstream);
    ABTI_CHECK_NULL_XSTREAM_PTR(p_xstream);

    ABTI_sched *p_sched;
    abt_errno = ABTI_sched_create_basic(predef, num_pools, pools,
                                        ABT_SCHED_CONFIG_NULL, &p_sched);
    ABTI_CHECK_ERROR(abt_errno);

    abt_errno = xstream_update_main_sched(&p_local_xstream, p_xstream, p_sched);
    ABTI_CHECK_ERROR(abt_errno);
    return ABT_SUCCESS;
}

/**
 * @ingroup ES
 * @brief   Get the main scheduler of the target ES.
 *
 * \c ABT_xstream_get_main_sched() gets the handle of the main scheduler
 * for the target ES \c xstream through \c sched.
 *
 * @param[in] xstream  handle to the target ES
 * @param[out] sched   handle to the scheduler
 * @return Error code
 * @retval ABT_SUCCESS on success
 */
int ABT_xstream_get_main_sched(ABT_xstream xstream, ABT_sched *sched)
{
    ABTI_xstream *p_xstream = ABTI_xstream_get_ptr(xstream);
    ABTI_CHECK_NULL_XSTREAM_PTR(p_xstream);

    *sched = ABTI_sched_get_handle(p_xstream->p_main_sched);
    return ABT_SUCCESS;
}

/**
 * @ingroup ES
 * @brief   Get the pools of the main scheduler of the target ES.
 *
 * This function is a convenient function that retrieves the associated pools of
 * the main scheduler.
 *
 * @param[in]  xstream   handle to the target ES
 * @param[in]  max_pools maximum number of pools
 * @param[out] pools     array of handles to the pools
 * @return Error code
 * @retval ABT_SUCCESS on success
 */
int ABT_xstream_get_main_pools(ABT_xstream xstream, int max_pools,
                               ABT_pool *pools)
{
    ABTI_xstream *p_xstream = ABTI_xstream_get_ptr(xstream);
    ABTI_CHECK_NULL_XSTREAM_PTR(p_xstream);

    ABTI_sched *p_sched = p_xstream->p_main_sched;
    max_pools = p_sched->num_pools > max_pools ? max_pools : p_sched->num_pools;
    memcpy(pools, p_sched->pools, sizeof(ABT_pool) * max_pools);
    return ABT_SUCCESS;
}

/**
 * @ingroup ES
 * @brief   Return the state of xstream.
 *
 * @param[in]  xstream  handle to the target ES
 * @param[out] state    the xstream's state
 * @return Error code
 * @retval ABT_SUCCESS on success
 */
int ABT_xstream_get_state(ABT_xstream xstream, ABT_xstream_state *state)
{
    ABTI_xstream *p_xstream = ABTI_xstream_get_ptr(xstream);
    ABTI_CHECK_NULL_XSTREAM_PTR(p_xstream);

    *state = (ABT_xstream_state)ABTD_atomic_acquire_load_int(&p_xstream->state);
    return ABT_SUCCESS;
}

/**
 * @ingroup ES
 * @brief   Compare two ES handles for equality.
 *
 * \c ABT_xstream_equal() compares two ES handles for equality. If two handles
 * are associated with the same ES, \c result will be set to \c ABT_TRUE.
 * Otherwise, \c result will be set to \c ABT_FALSE.
 *
 * @param[in]  xstream1  handle to the ES 1
 * @param[in]  xstream2  handle to the ES 2
 * @param[out] result    comparison result (<tt>ABT_TRUE</tt>: same,
 *                       <tt>ABT_FALSE</tt>: not same)
 * @return Error code
 * @retval ABT_SUCCESS on success
 */
int ABT_xstream_equal(ABT_xstream xstream1, ABT_xstream xstream2,
                      ABT_bool *result)
{
    ABTI_xstream *p_xstream1 = ABTI_xstream_get_ptr(xstream1);
    ABTI_xstream *p_xstream2 = ABTI_xstream_get_ptr(xstream2);
    *result = (p_xstream1 == p_xstream2) ? ABT_TRUE : ABT_FALSE;
    return ABT_SUCCESS;
}

/**
 * @ingroup ES
 * @brief   Return the number of current existing ESs.
 *
 * \c ABT_xstream_get_num() returns the number of ESs that exist in the current
 * Argobots environment through \c num_xstreams.
 *
 * @param[out] num_xstreams  the number of ESs
 * @return Error code
 * @retval ABT_SUCCESS           on success
 * @retval ABT_ERR_UNINITIALIZED Argobots has not been initialized
 */
int ABT_xstream_get_num(int *num_xstreams)
{
    /* In case that Argobots has not been initialized, return an error code
     * instead of making the call fail. */
    ABTI_SETUP_WITH_INIT_CHECK();

    *num_xstreams = gp_ABTI_global->num_xstreams;
    return ABT_SUCCESS;
}

/**
 * @ingroup ES
 * @brief   Check if the target ES is the primary ES.
 *
 * \c ABT_xstream_is_primary() checks whether the target ES is the primary ES.
 * If the ES \c xstream is the primary ES, \c flag is set to \c ABT_TRUE.
 * Otherwise, \c flag is set to \c ABT_FALSE.
 *
 * @param[in]  xstream  handle to the target ES
 * @param[out] flag     result (<tt>ABT_TRUE</tt>: primary ES,
 *                      <tt>ABT_FALSE</tt>: not)
 * @return Error code
 * @retval ABT_SUCCESS on success
 */
int ABT_xstream_is_primary(ABT_xstream xstream, ABT_bool *flag)
{
    ABTI_xstream *p_xstream;

    p_xstream = ABTI_xstream_get_ptr(xstream);
    ABTI_CHECK_NULL_XSTREAM_PTR(p_xstream);

    /* Return value */
    *flag =
        (p_xstream->type == ABTI_XSTREAM_TYPE_PRIMARY) ? ABT_TRUE : ABT_FALSE;
    return ABT_SUCCESS;
}

/**
 * @ingroup ES
 * @brief   Execute a unit on the local ES.
 *
 * This function can be called by a scheduler after picking one unit. So a user
 * will use it for his own defined scheduler.
 *
 * EXPERIMENTAL: this function can be called by a normal ULT, too.  The function
 * name could be changed in the future.
 *
 * @param[in] unit handle to the unit to run
 * @param[in] pool pool where unit is from
 * @return Error code
 * @retval ABT_SUCCESS          on success
 * @retval ABT_ERR_INV_XSTREAM  called by an external thread
 * @retval ABT_ERR_INV_THREAD   called by a non-yieldable thread (tasklet)
 */
int ABT_xstream_run_unit(ABT_unit unit, ABT_pool pool)
{
    ABTI_pool *p_pool = ABTI_pool_get_ptr(pool);

    ABTI_xstream *p_local_xstream;
    ABTI_SETUP_LOCAL_YTHREAD(&p_local_xstream, NULL);

    ABTI_xstream_run_unit(&p_local_xstream, unit, p_pool);
    return ABT_SUCCESS;
}

/**
 * @ingroup ES
 * @brief   Check the events and process them
 *
 * This function must be called by a scheduler periodically. Therefore, a user
 * will use it on his own defined scheduler.
 *
 * @param[in] sched handle to the scheduler where this call is from
 * @return Error code
 * @retval ABT_SUCCESS on success
 * @retval ABT_ERR_INV_XSTREAM  called by an external thread
 */
int ABT_xstream_check_events(ABT_sched sched)
{
    ABTI_xstream *p_local_xstream;
    ABTI_SETUP_LOCAL_XSTREAM_WITH_INIT_CHECK(&p_local_xstream);

    ABTI_sched *p_sched = ABTI_sched_get_ptr(sched);
    ABTI_CHECK_NULL_SCHED_PTR(p_sched);

    ABTI_xstream_check_events(p_local_xstream, p_sched);
    return ABT_SUCCESS;
}

/**
 * @ingroup ES
 * @brief   Bind the target ES to a target CPU.
 *
 * \c ABT_xstream_set_cpubind() binds the target ES \c xstream to the target
 * CPU whose ID is \c cpuid.  Here, the CPU ID corresponds to the processor
 * index used by OS.
 *
 * @param[in] xstream  handle to the target ES
 * @param[in] cpuid    CPU ID
 * @return Error code
 * @retval ABT_SUCCESS on success
 */
int ABT_xstream_set_cpubind(ABT_xstream xstream, int cpuid)
{
    ABTI_xstream *p_xstream = ABTI_xstream_get_ptr(xstream);
    ABTI_CHECK_NULL_XSTREAM_PTR(p_xstream);

    ABTD_affinity_cpuset cpuset;
    cpuset.num_cpuids = 1;
    cpuset.cpuids = &cpuid;
    int abt_errno = ABTD_affinity_cpuset_apply(&p_xstream->ctx, &cpuset);
    /* Do not free cpuset since cpuids points to a user pointer. */
    ABTI_CHECK_ERROR(abt_errno);
    return ABT_SUCCESS;
}

/**
 * @ingroup ES
 * @brief   Get the CPU binding for the target ES.
 *
 * \c ABT_xstream_get_cpubind() returns the ID of CPU, which the target ES
 * \c xstream is bound to.  If \c xstream is bound to more than one CPU, only
 * the first CPU ID is returned.
 *
 * @param[in] xstream  handle to the target ES
 * @param[out] cpuid   CPU ID
 * @return Error code
 * @retval ABT_SUCCESS on success
 */
int ABT_xstream_get_cpubind(ABT_xstream xstream, int *cpuid)
{
    int abt_errno;
    ABTI_xstream *p_xstream = ABTI_xstream_get_ptr(xstream);
    ABTI_CHECK_NULL_XSTREAM_PTR(p_xstream);

    ABTD_affinity_cpuset cpuset;
    cpuset.num_cpuids = 0;
    cpuset.cpuids = NULL;
    abt_errno = ABTD_affinity_cpuset_read(&p_xstream->ctx, &cpuset);
    ABTI_CHECK_ERROR(abt_errno);

    if (cpuset.num_cpuids != 0) {
        *cpuid = cpuset.cpuids[0];
    } else {
        abt_errno = ABT_ERR_FEATURE_NA;
    }
    ABTD_affinity_cpuset_destroy(&cpuset);
    ABTI_CHECK_ERROR(abt_errno);
    return ABT_SUCCESS;
}

/**
 * @ingroup ES
 * @brief   Set the CPU affinity of the target ES.
 *
 * \c ABT_xstream_set_cpubind() binds the target ES \c xstream on the given CPU
 * set, \c cpuset, which is an array of CPU IDs.  Here, the CPU IDs correspond
 * to processor indexes used by OS.
 *
 * @param[in] xstream      handle to the target ES
 * @param[in] cpuset_size  the number of \c cpuset entries
 * @param[in] cpuset       array of CPU IDs
 * @return Error code
 * @retval ABT_SUCCESS on success
 */
int ABT_xstream_set_affinity(ABT_xstream xstream, int cpuset_size, int *cpuset)
{
    ABTI_xstream *p_xstream = ABTI_xstream_get_ptr(xstream);
    ABTI_CHECK_NULL_XSTREAM_PTR(p_xstream);

    ABTD_affinity_cpuset affinity;
    affinity.num_cpuids = cpuset_size;
    affinity.cpuids = cpuset;
    int abt_errno = ABTD_affinity_cpuset_apply(&p_xstream->ctx, &affinity);
    /* Do not free affinity since cpuids may not be freed. */
    ABTI_CHECK_ERROR(abt_errno);
    return ABT_SUCCESS;
}

/**
 * @ingroup ES
 * @brief   Get the CPU affinity for the target ES.
 *
 * \c ABT_xstream_get_cpubind() writes CPU IDs (at most, \c cpuset_size) to
 * \c cpuset and returns the number of elements written to \c cpuset to
 * \c num_cpus.  If \c num_cpus is \c NULL, it is ignored.
 *
 * If \c cpuset is \c NULL, \c cpuset_size is ignored and the nubmer of all
 * CPUs on which \c xstream is bound is returned through \c num_cpus.
 * Otherwise, i.e., if \c cpuset is \c NULL, \c cpuset_size must be greater
 * than zero.
 *
 * @param[in]  xstream      handle to the target ES
 * @param[in]  cpuset_size  the number of \c cpuset entries
 * @param[out] cpuset       array of CPU IDs
 * @param[out] num_cpus     the number of total CPU IDs
 * @return Error code
 * @retval ABT_SUCCESS on success
 */
int ABT_xstream_get_affinity(ABT_xstream xstream, int cpuset_size, int *cpuset,
                             int *num_cpus)
{
    ABTI_xstream *p_xstream = ABTI_xstream_get_ptr(xstream);
    ABTI_CHECK_NULL_XSTREAM_PTR(p_xstream);

    ABTD_affinity_cpuset affinity;
    int abt_errno = ABTD_affinity_cpuset_read(&p_xstream->ctx, &affinity);
    ABTI_CHECK_ERROR(abt_errno);

    int i, n;
    n = affinity.num_cpuids > cpuset_size ? cpuset_size : affinity.num_cpuids;
    *num_cpus = n;
    for (i = 0; i < n; i++) {
        cpuset[i] = affinity.cpuids[i];
    }
    ABTD_affinity_cpuset_destroy(&affinity);
    return abt_errno;
}

/*****************************************************************************/
/* Private APIs                                                              */
/*****************************************************************************/

ABTU_ret_err int ABTI_xstream_create_primary(ABTI_xstream **pp_xstream)
{
    int abt_errno;
    ABTI_xstream *p_newxstream;
    ABTI_sched *p_sched;

    /* For the primary ES, a default scheduler is created. */
    abt_errno = ABTI_sched_create_basic(ABT_SCHED_DEFAULT, 0, NULL,
                                        ABT_SCHED_CONFIG_NULL, &p_sched);
    ABTI_CHECK_ERROR(abt_errno);

    abt_errno =
        xstream_create(p_sched, ABTI_XSTREAM_TYPE_PRIMARY, -1, &p_newxstream);
    ABTI_CHECK_ERROR(abt_errno);

    *pp_xstream = p_newxstream;
    return ABT_SUCCESS;
}

/* This routine starts the primary ES. It should be called in ABT_init. */
void ABTI_xstream_start_primary(ABTI_xstream **pp_local_xstream,
                                ABTI_xstream *p_xstream,
                                ABTI_ythread *p_ythread)
{
    /* p_ythread must be the main thread. */
    ABTI_ASSERT(p_ythread->thread.type & ABTI_THREAD_TYPE_MAIN);
    /* The ES's state must be running here. */
    ABTI_ASSERT(ABTD_atomic_relaxed_load_int(&p_xstream->state) ==
                ABT_XSTREAM_STATE_RUNNING);

    LOG_DEBUG("[E%d] start\n", p_xstream->rank);

    ABTD_xstream_context_set_self(&p_xstream->ctx);

    /* Set the CPU affinity for the ES */
    if (gp_ABTI_global->set_affinity == ABT_TRUE) {
        ABTD_affinity_cpuset_apply_default(&p_xstream->ctx, p_xstream->rank);
    }

    /* Context switch to the root thread. */
    p_xstream->p_root_ythread->thread.p_last_xstream = p_xstream;
    ABTD_ythread_context_switch(&p_ythread->ctx,
                                &p_xstream->p_root_ythread->ctx);
    /* Come back to the main thread.  Now this thread is executed on top of the
     * main scheduler, which is running on the root thread. */
    (*pp_local_xstream)->p_thread = &p_ythread->thread;
}

void ABTI_xstream_run_unit(ABTI_xstream **pp_local_xstream, ABT_unit unit,
                           ABTI_pool *p_pool)
{
    ABT_unit_type type = p_pool->u_get_type(unit);

    if (type == ABT_UNIT_TYPE_THREAD) {
        ABT_thread thread = p_pool->u_get_thread(unit);
        ABTI_ythread *p_ythread = ABTI_ythread_get_ptr(thread);
        /* Switch the context */
        xstream_schedule_ythread(pp_local_xstream, p_ythread);
    } else {
        ABTI_ASSERT(type == ABT_UNIT_TYPE_TASK);
        ABT_task task = p_pool->u_get_task(unit);
        ABTI_thread *p_task = ABTI_thread_get_ptr(task);
        /* Execute the task */
        xstream_schedule_task(*pp_local_xstream, p_task);
    }
}

void ABTI_xstream_check_events(ABTI_xstream *p_xstream, ABTI_sched *p_sched)
{
    ABTI_info_check_print_all_thread_stacks();

    uint32_t request = ABTD_atomic_acquire_load_uint32(
        &p_xstream->p_main_sched->p_ythread->thread.request);
    if (request & ABTI_THREAD_REQ_JOIN) {
        ABTI_sched_finish(p_sched);
    }

    if ((request & ABTI_THREAD_REQ_TERMINATE) ||
        (request & ABTI_THREAD_REQ_CANCEL)) {
        ABTI_sched_exit(p_sched);
    }
}

void ABTI_xstream_free(ABTI_local *p_local, ABTI_xstream *p_xstream,
                       ABT_bool force_free)
{
    LOG_DEBUG("[E%d] freed\n", p_xstream->rank);

    /* Clean up memory pool. */
    ABTI_mem_finalize_local(p_xstream);
    /* Return rank for reuse. rank must be returned prior to other free
     * functions so that other xstreams cannot refer to this xstream. */
    xstream_return_rank(p_xstream);

    /* Free the scheduler */
    ABTI_sched *p_cursched = p_xstream->p_main_sched;
    if (p_cursched != NULL) {
        /* Join a scheduler thread. */
        ABTI_tool_event_thread_join(p_local, &p_cursched->p_ythread->thread,
                                    ABTI_local_get_xstream_or_null(p_local)
                                        ? ABTI_local_get_xstream(p_local)
                                              ->p_thread
                                        : NULL);
        ABTI_sched_discard_and_free(p_local, p_cursched, force_free);
        /* The main scheduler thread is also freed. */
    }

    /* Free the root thread and pool. */
    ABTI_ythread_free_root(p_local, p_xstream->p_root_ythread);
    ABTI_pool_free(p_xstream->p_root_pool);

    /* Free the context if a given xstream is secondary. */
    if (p_xstream->type == ABTI_XSTREAM_TYPE_SECONDARY) {
        ABTD_xstream_context_free(&p_xstream->ctx);
    }

    ABTU_free(p_xstream);
}

void ABTI_xstream_print(ABTI_xstream *p_xstream, FILE *p_os, int indent,
                        ABT_bool print_sub)
{
    if (p_xstream == NULL) {
        fprintf(p_os, "%*s== NULL ES ==\n", indent, "");
    } else {
        const char *type, *state;
        switch (p_xstream->type) {
            case ABTI_XSTREAM_TYPE_PRIMARY:
                type = "PRIMARY";
                break;
            case ABTI_XSTREAM_TYPE_SECONDARY:
                type = "SECONDARY";
                break;
            default:
                type = "UNKNOWN";
                break;
        }
        switch (ABTD_atomic_acquire_load_int(&p_xstream->state)) {
            case ABT_XSTREAM_STATE_RUNNING:
                state = "RUNNING";
                break;
            case ABT_XSTREAM_STATE_TERMINATED:
                state = "TERMINATED";
                break;
            default:
                state = "UNKNOWN";
                break;
        }

        fprintf(p_os,
                "%*s== ES (%p) ==\n"
                "%*srank      : %d\n"
                "%*stype      : %s\n"
                "%*sstate     : %s\n"
                "%*smain_sched: %p\n",
                indent, "", (void *)p_xstream, indent, "", p_xstream->rank,
                indent, "", type, indent, "", state, indent, "",
                (void *)p_xstream->p_main_sched);

        if (print_sub == ABT_TRUE) {
            ABTI_sched_print(p_xstream->p_main_sched, p_os,
                             indent + ABTI_INDENT, ABT_TRUE);
        }
    }
    fflush(p_os);
}

static void *xstream_launch_root_ythread(void *p_xstream)
{
    ABTI_xstream *p_local_xstream = (ABTI_xstream *)p_xstream;

    /* Initialization of the local variables */
    ABTI_local_set_xstream(p_local_xstream);

    LOG_DEBUG("[E%d] start\n", p_local_xstream->rank);

    /* Set the root thread as the current thread */
    ABTI_ythread *p_root_ythread = p_local_xstream->p_root_ythread;
    p_local_xstream->p_thread = &p_local_xstream->p_root_ythread->thread;
    p_root_ythread->thread.f_thread(p_root_ythread->thread.p_arg);

    LOG_DEBUG("[E%d] end\n", p_local_xstream->rank);

    /* Reset the current ES and its local info. */
    ABTI_local_set_xstream(NULL);
    return NULL;
}

/*****************************************************************************/
/* Internal static functions                                                 */
/*****************************************************************************/

ABTU_ret_err static int xstream_create(ABTI_sched *p_sched,
                                       ABTI_xstream_type xstream_type, int rank,
                                       ABTI_xstream **pp_xstream)
{
    int abt_errno;
    ABTI_xstream *p_newxstream;

    abt_errno = ABTU_malloc(sizeof(ABTI_xstream), (void **)&p_newxstream);
    ABTI_CHECK_ERROR(abt_errno);

    p_newxstream->p_prev = NULL;
    p_newxstream->p_next = NULL;

    if (xstream_set_new_rank(p_newxstream, rank) == ABT_FALSE) {
        ABTU_free(p_newxstream);
        return ABT_ERR_INV_XSTREAM_RANK;
    }

    p_newxstream->type = xstream_type;
    ABTD_atomic_relaxed_store_int(&p_newxstream->state,
                                  ABT_XSTREAM_STATE_RUNNING);
    p_newxstream->p_main_sched = NULL;
    p_newxstream->p_thread = NULL;
    ABTI_mem_init_local(p_newxstream);

    /* Set the main scheduler */
    xstream_init_main_sched(p_newxstream, p_sched);

    /* Create the root thread. */
    abt_errno =
        ABTI_ythread_create_root(ABTI_xstream_get_local(p_newxstream),
                                 p_newxstream, &p_newxstream->p_root_ythread);
    ABTI_CHECK_ERROR(abt_errno);

    /* Create the root pool. */
    abt_errno = ABTI_pool_create_basic(ABT_POOL_FIFO, ABT_POOL_ACCESS_MPSC,
                                       ABT_FALSE, &p_newxstream->p_root_pool);
    ABTI_CHECK_ERROR(abt_errno);

    /* Create the main scheduler thread. */
    abt_errno =
        ABTI_ythread_create_main_sched(ABTI_xstream_get_local(p_newxstream),
                                       p_newxstream,
                                       p_newxstream->p_main_sched);
    ABTI_CHECK_ERROR(abt_errno);

    LOG_DEBUG("[E%d] created\n", p_newxstream->rank);

    /* Return value */
    *pp_xstream = p_newxstream;
    return ABT_SUCCESS;
}

ABTU_ret_err static int xstream_start(ABTI_xstream *p_xstream)
{
    /* The ES's state must be RUNNING */
    ABTI_ASSERT(ABTD_atomic_relaxed_load_int(&p_xstream->state) ==
                ABT_XSTREAM_STATE_RUNNING);
    ABTI_ASSERT(p_xstream->type != ABTI_XSTREAM_TYPE_PRIMARY);
    /* Start the main scheduler on a different ES */
    int abt_errno =
        ABTD_xstream_context_create(xstream_launch_root_ythread,
                                    (void *)p_xstream, &p_xstream->ctx);
    ABTI_CHECK_ERROR(abt_errno);

    /* Set the CPU affinity for the ES */
    if (gp_ABTI_global->set_affinity == ABT_TRUE) {
        ABTD_affinity_cpuset_apply_default(&p_xstream->ctx, p_xstream->rank);
    }
    return ABT_SUCCESS;
}

ABTU_ret_err static int xstream_join(ABTI_local **pp_local,
                                     ABTI_xstream *p_xstream)
{
    /* The primary ES cannot be joined. */
    ABTI_CHECK_TRUE(p_xstream->type != ABTI_XSTREAM_TYPE_PRIMARY,
                    ABT_ERR_INV_XSTREAM);
    /* The main scheduler cannot join itself. */
    ABTI_CHECK_TRUE(!ABTI_local_get_xstream_or_null(*pp_local) ||
                        &p_xstream->p_main_sched->p_ythread->thread !=
                            ABTI_local_get_xstream(*pp_local)->p_thread,
                    ABT_ERR_INV_THREAD);

    /* Wait until the target ES terminates */
    ABTI_sched_finish(p_xstream->p_main_sched);
    ABTI_thread_join(pp_local, &p_xstream->p_main_sched->p_ythread->thread);

    /* Normal join request */
    ABTD_xstream_context_join(&p_xstream->ctx);

    ABTI_ASSERT(ABTD_atomic_acquire_load_int(&p_xstream->state) ==
                ABT_XSTREAM_STATE_TERMINATED);
    return ABT_SUCCESS;
}

static inline void xstream_schedule_ythread(ABTI_xstream **pp_local_xstream,
                                            ABTI_ythread *p_ythread)
{
    ABTI_xstream *p_local_xstream = *pp_local_xstream;

#ifndef ABT_CONFIG_DISABLE_THREAD_CANCEL
    if (ABTD_atomic_acquire_load_uint32(&p_ythread->thread.request) &
        ABTI_THREAD_REQ_CANCEL) {
        LOG_DEBUG("[U%" PRIu64 ":E%d] canceled\n",
                  ABTI_thread_get_id(&p_ythread->thread),
                  p_local_xstream->rank);
        ABTD_ythread_cancel(p_local_xstream, p_ythread);
        ABTI_xstream_terminate_thread(ABTI_xstream_get_local(p_local_xstream),
                                      &p_ythread->thread);
        return;
    }
#endif

#ifndef ABT_CONFIG_DISABLE_MIGRATION
    if (ABTD_atomic_acquire_load_uint32(&p_ythread->thread.request) &
        ABTI_THREAD_REQ_MIGRATE) {
        int abt_errno =
            xstream_migrate_thread(ABTI_xstream_get_local(p_local_xstream),
                                   &p_ythread->thread);
        if (!ABTI_IS_ERROR_CHECK_ENABLED || abt_errno == ABT_SUCCESS) {
            /* Migration succeeded, so we do not need to schedule p_ythread. */
            return;
        }
    }
#endif

    /* Change the last ES */
    p_ythread->thread.p_last_xstream = p_local_xstream;

    /* Change the ULT state */
    ABTD_atomic_release_store_int(&p_ythread->thread.state,
                                  ABT_THREAD_STATE_RUNNING);

    /* Switch the context */
    LOG_DEBUG("[U%" PRIu64 ":E%d] start running\n",
              ABTI_thread_get_id(&p_ythread->thread), p_local_xstream->rank);

    /* Since the argument is pp_local_xstream, p_local_xstream->p_thread must be
     * yieldable. */
    ABTI_ythread *p_self = ABTI_thread_get_ythread(p_local_xstream->p_thread);
    p_ythread = ABTI_ythread_context_switch_to_child(pp_local_xstream, p_self,
                                                     p_ythread);
    /* The previous ULT (p_ythread) may not be the same as one to which the
     * context has been switched. */
    /* The scheduler continues from here. */
    p_local_xstream = *pp_local_xstream;

    LOG_DEBUG("[U%" PRIu64 ":E%d] stopped\n",
              ABTI_thread_get_id(&p_ythread->thread), p_local_xstream->rank);

    /* Request handling. */
    /* We do not need to acquire-load request since all critical requests
     * (BLOCK, ORPHAN, STOP, and NOPUSH) are written by p_ythread. CANCEL might
     * be delayed. */
    uint32_t request =
        ABTD_atomic_acquire_load_uint32(&p_ythread->thread.request);
    if (request & ABTI_THREAD_REQ_TERMINATE) {
        /* The ULT has completed its execution or it called the exit request. */
        LOG_DEBUG("[U%" PRIu64 ":E%d] finished\n",
                  ABTI_thread_get_id(&p_ythread->thread),
                  p_local_xstream->rank);
        ABTI_xstream_terminate_thread(ABTI_xstream_get_local(p_local_xstream),
                                      &p_ythread->thread);
#ifndef ABT_CONFIG_DISABLE_THREAD_CANCEL
    } else if (request & ABTI_THREAD_REQ_CANCEL) {
        LOG_DEBUG("[U%" PRIu64 ":E%d] canceled\n",
                  ABTI_thread_get_id(&p_ythread->thread),
                  p_local_xstream->rank);
        ABTD_ythread_cancel(p_local_xstream, p_ythread);
        ABTI_xstream_terminate_thread(ABTI_xstream_get_local(p_local_xstream),
                                      &p_ythread->thread);
#endif
    } else if (!(request & ABTI_THREAD_REQ_NON_YIELD)) {
        /* The ULT did not finish its execution.
         * Change the state of current running ULT and
         * add it to the pool again. */
        ABTI_pool_add_thread(&p_ythread->thread);
    } else if (request & ABTI_THREAD_REQ_BLOCK) {
        LOG_DEBUG("[U%" PRIu64 ":E%d] check blocked\n",
                  ABTI_thread_get_id(&p_ythread->thread),
                  p_local_xstream->rank);
        ABTI_thread_unset_request(&p_ythread->thread, ABTI_THREAD_REQ_BLOCK);
#ifndef ABT_CONFIG_DISABLE_MIGRATION
    } else if (request & ABTI_THREAD_REQ_MIGRATE) {
        /* This is the case when the ULT requests migration of itself. */
        int abt_errno =
            xstream_migrate_thread(ABTI_xstream_get_local(p_local_xstream),
                                   &p_ythread->thread);
        /* Migration is optional, so it is okay if it fails. */
        (void)abt_errno;
#endif
    } else if (request & ABTI_THREAD_REQ_ORPHAN) {
        /* The ULT is not pushed back to the pool and is disconnected from any
         * pool. */
        LOG_DEBUG("[U%" PRIu64 ":E%d] orphaned\n",
                  ABTI_thread_get_id(&p_ythread->thread),
                  p_local_xstream->rank);
        ABTI_thread_unset_request(&p_ythread->thread, ABTI_THREAD_REQ_ORPHAN);
        p_ythread->thread.p_pool->u_free(&p_ythread->thread.unit);
        p_ythread->thread.p_pool = NULL;
    } else if (request & ABTI_THREAD_REQ_NOPUSH) {
        /* The ULT is not pushed back to the pool */
        LOG_DEBUG("[U%" PRIu64 ":E%d] not pushed\n",
                  ABTI_thread_get_id(&p_ythread->thread),
                  p_local_xstream->rank);
        ABTI_thread_unset_request(&p_ythread->thread, ABTI_THREAD_REQ_NOPUSH);
    } else {
        ABTI_ASSERT(0);
        ABTU_unreachable();
    }
}

static inline void xstream_schedule_task(ABTI_xstream *p_local_xstream,
                                         ABTI_thread *p_task)
{
#ifndef ABT_CONFIG_DISABLE_TASK_CANCEL
    if (ABTD_atomic_acquire_load_uint32(&p_task->request) &
        ABTI_THREAD_REQ_CANCEL) {
        ABTI_tool_event_thread_cancel(p_local_xstream, p_task);
        ABTI_xstream_terminate_thread(ABTI_xstream_get_local(p_local_xstream),
                                      p_task);
        return;
    }
#endif

    /* Change the task state */
    ABTD_atomic_release_store_int(&p_task->state, ABT_THREAD_STATE_RUNNING);

    /* Set the associated ES */
    p_task->p_last_xstream = p_local_xstream;

    /* Execute the task function */
    LOG_DEBUG("[T%" PRIu64 ":E%d] running\n", ABTI_thread_get_id(p_task),
              p_local_xstream->rank);

    ABTI_thread *p_sched_thread = p_local_xstream->p_thread;
    p_local_xstream->p_thread = p_task;
    p_task->p_parent = p_sched_thread;

    /* Execute the task function */
    ABTI_tool_event_thread_run(p_local_xstream, p_task, p_sched_thread,
                               p_sched_thread);
    LOG_DEBUG("[T%" PRIu64 ":E%d] running\n", ABTI_thread_get_id(p_task),
              p_local_xstream->rank);
    p_task->f_thread(p_task->p_arg);
    ABTI_tool_event_thread_finish(p_local_xstream, p_task, p_sched_thread);
    LOG_DEBUG("[T%" PRIu64 ":E%d] stopped\n", ABTI_thread_get_id(p_task),
              p_local_xstream->rank);

    /* Set the current running scheduler's thread */
    p_local_xstream->p_thread = p_sched_thread;

    /* Terminate the tasklet */
    ABTI_xstream_terminate_thread(ABTI_xstream_get_local(p_local_xstream),
                                  p_task);
}

#ifndef ABT_CONFIG_DISABLE_MIGRATION
ABTU_ret_err static int xstream_migrate_thread(ABTI_local *p_local,
                                               ABTI_thread *p_thread)
{
    int abt_errno;
    ABTI_pool *p_pool;

    ABTI_thread_mig_data *p_mig_data;
    abt_errno = ABTI_thread_get_mig_data(p_local, p_thread, &p_mig_data);
    ABTI_CHECK_ERROR(abt_errno);

    /* callback function */
    if (p_mig_data->f_migration_cb) {
        ABTI_ythread *p_ythread = ABTI_thread_get_ythread_or_null(p_thread);
        if (p_ythread) {
            ABT_thread thread = ABTI_ythread_get_handle(p_ythread);
            p_mig_data->f_migration_cb(thread, p_mig_data->p_migration_cb_arg);
        }
    }

    /* If request is set, p_migration_pool has a valid pool pointer. */
    ABTI_ASSERT(ABTD_atomic_acquire_load_uint32(&p_thread->request) &
                ABTI_THREAD_REQ_MIGRATE);

    /* Extracting argument in migration request. */
    p_pool = ABTD_atomic_relaxed_load_ptr(&p_mig_data->p_migration_pool);
    ABTI_thread_unset_request(p_thread, ABTI_THREAD_REQ_MIGRATE);

    /* Change the associated pool */
    p_thread->p_pool = p_pool;

    /* Add the unit to the scheduler's pool */
    ABTI_pool_push(p_pool, p_thread->unit);

    ABTI_pool_dec_num_migrations(p_pool);

    return ABT_SUCCESS;
}
#endif

static void xstream_init_main_sched(ABTI_xstream *p_xstream,
                                    ABTI_sched *p_sched)
{
    ABTI_ASSERT(p_xstream->p_main_sched == NULL);
    /* The main scheduler will to be a ULT, not a tasklet */
    p_sched->type = ABT_SCHED_TYPE_ULT;
    /* Set the scheduler as a main scheduler */
    p_sched->used = ABTI_SCHED_MAIN;
    /* Set the scheduler */
    p_xstream->p_main_sched = p_sched;
}

ABTU_ret_err static int
xstream_update_main_sched(ABTI_xstream **pp_local_xstream,
                          ABTI_xstream *p_xstream, ABTI_sched *p_sched)
{
    ABTI_ythread *p_ythread = NULL;
    ABTI_sched *p_main_sched;
    ABTI_pool *p_tar_pool = NULL;
    int p;

    /* The main scheduler will to be a ULT, not a tasklet */
    p_sched->type = ABT_SCHED_TYPE_ULT;

    /* Set the scheduler as a main scheduler */
    p_sched->used = ABTI_SCHED_MAIN;

    p_main_sched = p_xstream->p_main_sched;
    if (p_main_sched == NULL) {
        /* Set the scheduler */
        p_xstream->p_main_sched = p_sched;
        return ABT_SUCCESS;
    }

    /* If the ES has a main scheduler, we have to free it */
    ABTI_CHECK_YIELDABLE((*pp_local_xstream)->p_thread, &p_ythread,
                         ABT_ERR_INV_THREAD);
    p_tar_pool = ABTI_pool_get_ptr(p_sched->pools[0]);

    /* If the caller ULT is associated with a pool of the current main
     * scheduler, it needs to be associated to a pool of new scheduler. */
    for (p = 0; p < p_main_sched->num_pools; p++) {
        if (p_ythread->thread.p_pool ==
            ABTI_pool_get_ptr(p_main_sched->pools[p])) {
            /* Associate the work unit to the first pool of new scheduler */
            p_ythread->thread.p_pool->u_free(&p_ythread->thread.unit);
            ABT_thread h_thread = ABTI_ythread_get_handle(p_ythread);
            p_ythread->thread.unit = p_tar_pool->u_create_from_thread(h_thread);
            p_ythread->thread.p_pool = p_tar_pool;
            break;
        }
    }
    if (p_xstream->type == ABTI_XSTREAM_TYPE_PRIMARY) {
        ABTI_CHECK_TRUE(p_ythread->thread.type & ABTI_THREAD_TYPE_MAIN,
                        ABT_ERR_THREAD);

        /* Since the primary ES does not finish its execution until ABT_finalize
         * is called, its main scheduler needs to be automatically freed when
         * it is freed in ABT_finalize. */
        p_sched->automatic = ABT_TRUE;
    }

    /* Finish the current main scheduler */
    ABTI_sched_set_request(p_main_sched, ABTI_SCHED_REQ_FINISH);

    /* If the ES is secondary, we should take the associated ULT of the
     * current main scheduler and keep it in the new scheduler. */
    p_sched->p_ythread = p_main_sched->p_ythread;
    /* The current ULT is pushed to the new scheduler's pool so that when
     * the new scheduler starts (see below), it can be scheduled by the new
     * scheduler. When the current ULT resumes its execution, it will free
     * the current main scheduler (see below). */
    ABTI_pool_push(p_tar_pool, p_ythread->thread.unit);

    /* Set the scheduler */
    p_xstream->p_main_sched = p_sched;

    /* Switch to the current main scheduler */
    ABTI_thread_set_request(&p_ythread->thread, ABTI_THREAD_REQ_NOPUSH);
    ABTI_ythread_context_switch_to_parent(pp_local_xstream, p_ythread,
                                          ABT_SYNC_EVENT_TYPE_OTHER, NULL);

    /* Now, we free the current main scheduler. p_main_sched->p_ythread must
     * be NULL to avoid freeing it in ABTI_sched_discard_and_free(). */
    p_main_sched->p_ythread = NULL;
    ABTI_sched_discard_and_free(ABTI_xstream_get_local(*pp_local_xstream),
                                p_main_sched, ABT_FALSE);
    return ABT_SUCCESS;
}

/* Set a new rank to ES */
static ABT_bool xstream_set_new_rank(ABTI_xstream *p_newxstream, int rank)
{
    ABTI_global *p_global = gp_ABTI_global;

    ABTI_spinlock_acquire(&p_global->xstream_list_lock);

    ABTI_xstream *p_prev_xstream = p_global->p_xstream_head;
    ABTI_xstream *p_xstream = p_prev_xstream;
    if (rank == -1) {
        /* Find an unused rank from 0. */
        rank = 0;
        while (p_xstream) {
            if (p_xstream->rank == rank) {
                rank++;
            } else {
                /* Use this rank. */
                break;
            }
            p_prev_xstream = p_xstream;
            p_xstream = p_xstream->p_next;
        }
    } else {
        /* Check if a certain rank is available */
        while (p_xstream) {
            if (p_xstream->rank == rank) {
                ABTI_spinlock_release(&p_global->xstream_list_lock);
                return ABT_FALSE;
            } else if (p_xstream->rank > rank) {
                /* Use this p_xstream. */
                break;
            }
            p_prev_xstream = p_xstream;
            p_xstream = p_xstream->p_next;
        }
    }
    if (!p_xstream) {
        /* p_newxstream is appended to p_prev_xstream */
        if (p_prev_xstream) {
            p_prev_xstream->p_next = p_newxstream;
            p_newxstream->p_prev = p_prev_xstream;
            p_newxstream->p_next = NULL;
        } else {
            ABTI_ASSERT(p_global->p_xstream_head == NULL);
            p_newxstream->p_prev = NULL;
            p_newxstream->p_next = NULL;
            p_global->p_xstream_head = p_newxstream;
        }
    } else {
        /* p_newxstream is inserted in the middle.
         * (p_xstream->p_prev) -> p_new_xstream -> p_xstream */
        if (p_xstream->p_prev) {
            p_xstream->p_prev->p_next = p_newxstream;
            p_newxstream->p_prev = p_xstream->p_prev;
        } else {
            /* This p_xstream is the first element */
            ABTI_ASSERT(p_global->p_xstream_head == p_xstream);
            p_global->p_xstream_head = p_newxstream;
        }
        p_xstream->p_prev = p_newxstream;
        p_newxstream->p_next = p_xstream;
    }
    p_global->num_xstreams++;
    if (rank >= p_global->max_xstreams) {
        static int max_xstreams_warning_once = 0;
        if (max_xstreams_warning_once == 0) {
            /* Because some Argobots functionalities depend on the runtime value
             * ABT_MAX_NUM_XSTREAMS (or gp_ABTI_global->max_xstreams), changing
             * this value at run-time can cause an error.  For example, using
             * ABT_mutex created before updating max_xstreams causes an error
             * since ABTI_thread_htable's array size depends on
             * ABT_MAX_NUM_XSTREAMS.  To fix this issue, please set a larger
             * number to ABT_MAX_NUM_XSTREAMS in advance. */
            char *warning_message;
            int abt_errno =
                ABTU_malloc(sizeof(char) * 1024, (void **)&warning_message);
            if (!ABTI_IS_ERROR_CHECK_ENABLED || abt_errno == ABT_SUCCESS) {
                snprintf(warning_message, 1024,
                         "Warning: the number of execution streams exceeds "
                         "ABT_MAX_NUM_XSTREAMS (=%d). This may cause an error.",
                         p_global->max_xstreams);
                HANDLE_WARNING(warning_message);
                ABTU_free(warning_message);
                max_xstreams_warning_once = 1;
            }
        }
        /* Anyway. let's increase max_xstreams. */
        p_global->max_xstreams = rank + 1;
    }

    ABTI_spinlock_release(&p_global->xstream_list_lock);

    /* Set the rank */
    p_newxstream->rank = rank;
    return ABT_TRUE;
}

static void xstream_return_rank(ABTI_xstream *p_xstream)
{
    ABTI_global *p_global = gp_ABTI_global;
    /* Remove this xstream from the global ES list */
    ABTI_spinlock_acquire(&p_global->xstream_list_lock);
    if (!p_xstream->p_prev) {
        ABTI_ASSERT(p_global->p_xstream_head == p_xstream);
        p_global->p_xstream_head = p_xstream->p_next;
    } else {
        p_xstream->p_prev->p_next = p_xstream->p_next;
    }
    if (p_xstream->p_next) {
        p_xstream->p_next->p_prev = p_xstream->p_prev;
    }
    p_global->num_xstreams--;
    ABTI_spinlock_release(&p_global->xstream_list_lock);
}

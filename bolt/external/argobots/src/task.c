/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

#include "abti.h"

ABTU_ret_err static int task_create(ABTI_local *p_local, ABTI_pool *p_pool,
                                    void (*task_func)(void *), void *arg,
                                    ABTI_sched *p_sched, int refcount,
                                    ABTI_thread **pp_newtask);

/** @defgroup TASK Tasklet
 * This group is for Tasklet.
 */

/**
 * @ingroup TASK
 * @brief   Create a new task and return its handle through newtask.
 *
 * \c ABT_task_create() creates a new tasklet that is pushed into \c pool. The
 * insertion is done from the ES where this call is made. Therefore, the access
 * type of \c pool should comply with that. The handle of the newly created
 * tasklet is obtained through \c newtask.
 *
 * If this is ABT_XSTREAM_NULL, the new task is managed globally and it can be
 * executed by any ES. Otherwise, the task is scheduled and runs in the
 * specified ES.
 * If newtask is NULL, the task object will be automatically released when
 * this \a unnamed task completes the execution of task_func. Otherwise,
 * ABT_task_free() can be used to explicitly release the task object.
 *
 * @param[in]  pool       handle to the associated pool
 * @param[in]  task_func  function to be executed by a new task
 * @param[in]  arg        argument for task_func
 * @param[out] newtask    handle to a newly created task
 * @return Error code
 * @retval ABT_SUCCESS on success
 */
int ABT_task_create(ABT_pool pool, void (*task_func)(void *), void *arg,
                    ABT_task *newtask)
{
    ABTI_local *p_local = ABTI_local_get_local();
    ABTI_thread *p_newtask;
    ABTI_pool *p_pool = ABTI_pool_get_ptr(pool);
    ABTI_CHECK_NULL_POOL_PTR(p_pool);

    int refcount = (newtask != NULL) ? 1 : 0;
    int abt_errno = task_create(p_local, p_pool, task_func, arg, NULL, refcount,
                                &p_newtask);
    ABTI_CHECK_ERROR(abt_errno);

    /* Return value */
    if (newtask)
        *newtask = ABTI_thread_get_handle(p_newtask);
    return ABT_SUCCESS;
}

/**
 * @ingroup TASK
 * @brief   Create a new tasklet associated with the target ES (\c xstream).
 *
 * \c ABT_task_create_on_xstream() creates a new tasklet associated with
 * the target ES and returns its handle through \c newtask. The new tasklet
 * is inserted into a proper pool associated with the main scheduler of
 * the target ES.
 *
 * This routine is only for convenience. If the user wants to focus on the
 * performance, we recommend to use \c ABT_task_create() with directly
 * dealing with pools. Pools are a right way to manage work units in Argobots.
 * ES is just an abstract, and it is not a mechanism for execution and
 * performance tuning.
 *
 * If \c newtask is \c NULL, this routine creates an unnamed tasklet.
 * The object for unnamed tasklet will be automatically freed when the unnamed
 * tasklet completes its execution. Otherwise, this routine creates a named
 * tasklet and \c ABT_task_free() can be used to explicitly free the tasklet
 * object.
 *
 * If \c newtask is not \c NULL and an error occurs in this routine, a non-zero
 * error code will be returned and \c newtask will be set to \c ABT_TASK_NULL.
 *
 * @param[in]  xstream    handle to the target ES
 * @param[in]  task_func  function to be executed by a new tasklet
 * @param[in]  arg        argument for <tt>task_func</tt>
 * @param[out] newtask    handle to a newly created tasklet
 * @return Error code
 * @retval ABT_SUCCESS on success
 */
int ABT_task_create_on_xstream(ABT_xstream xstream, void (*task_func)(void *),
                               void *arg, ABT_task *newtask)
{
    ABTI_local *p_local = ABTI_local_get_local();
    ABTI_thread *p_newtask;

    ABTI_xstream *p_xstream = ABTI_xstream_get_ptr(xstream);
    ABTI_CHECK_NULL_XSTREAM_PTR(p_xstream);

    /* TODO: need to consider the access type of target pool */
    ABTI_pool *p_pool = ABTI_xstream_get_main_pool(p_xstream);
    int refcount = (newtask != NULL) ? 1 : 0;
    int abt_errno = task_create(p_local, p_pool, task_func, arg, NULL, refcount,
                                &p_newtask);
    ABTI_CHECK_ERROR(abt_errno);

    /* Return value */
    if (newtask)
        *newtask = ABTI_thread_get_handle(p_newtask);
    return ABT_SUCCESS;
}

#ifdef ABT_CONFIG_USE_DOXYGEN
/**
 * @ingroup TASK
 * @brief   Revive the tasklet.
 *
 * \c ABT_task_revive() revives the tasklet, \c task, with \c task_func and
 * \arg and pushes the revived tasklet into \c pool.
 *
 * This function must be called with a valid tasklet handle, which has not been
 * freed by \c ABT_task_free().  However, the tasklet should have been joined
 * by \c ABT_task_join() before its handle is used in this routine.
 *
 * @param[in]     pool       handle to the associated pool
 * @param[in]     task_func  function to be executed by the tasklet
 * @param[in]     arg        argument for task_func
 * @param[in,out] task       handle to the tasklet
 * @return Error code
 * @retval ABT_SUCCESS on success
 */
int ABT_task_revive(ABT_pool pool, void (*task_func)(void *), void *arg,
                    ABT_task *task);
#endif

#ifdef ABT_CONFIG_USE_DOXYGEN
/**
 * @ingroup TASK
 * @brief   Release the task object associated with task handle.
 *
 * This routine deallocates memory used for the task object. If the task is
 * still running when this routine is called, the deallocation happens after
 * the task terminates and then this routine returns. If it is successfully
 * processed, task is set as ABT_TASK_NULL.
 *
 * @param[in,out] task  handle to the target task
 * @return Error code
 * @retval ABT_SUCCESS on success
 */
int ABT_task_free(ABT_task *task);
#endif

#ifdef ABT_CONFIG_USE_DOXYGEN
/**
 * @ingroup TASK
 * @brief   Wait for the tasklet to terminate.
 *
 * \c ABT_task_join() blocks until the target tasklet \c task terminates.
 * Since this routine blocks, only ULTs can call this routine.  If tasklets use
 * this routine, the behavior is undefined.
 *
 * @param[in] task  handle to the target tasklet
 * @return Error code
 * @retval ABT_SUCCESS on success
 */
int ABT_task_join(ABT_task task);
#endif

#ifdef ABT_CONFIG_USE_DOXYGEN
/**
 * @ingroup TASK
 * @brief   Request the cancellation of the target task.
 *
 * @param[in] task  handle to the target task
 * @return Error code
 * @retval ABT_SUCCESS on success
 */
int ABT_task_cancel(ABT_task task);
#endif

/**
 * @ingroup TASK
 * @brief   Return the handle of the calling tasklet.
 *
 * \c ABT_task_self() returns the handle of the calling tasklet.
 * If ULTs call this routine, \c ABT_TASK_NULL will be returned to \c task.
 *
 * At present \c task is set to \c ABT_TASK_NULL when an error occurs, but this
 * behavior is deprecated.  The program should not rely on this behavior.
 *
 * @param[out] task  tasklet handle
 * @return Error code
 * @retval ABT_SUCCESS           on success
 * @retval ABT_ERR_UNINITIALIZED Argobots has not been initialized
 * @retval ABT_ERR_INV_XSTREAM   called by an external thread
 * @retval ABT_ERR_INV_THREAD    called by a yieldable thread (ULT)
 */
int ABT_task_self(ABT_task *task)
{
    *task = ABT_TASK_NULL;

    ABTI_xstream *p_local_xstream;
    ABTI_SETUP_LOCAL_XSTREAM_WITH_INIT_CHECK(&p_local_xstream);

    ABTI_thread *p_thread = p_local_xstream->p_thread;
    if (p_thread->type & ABTI_THREAD_TYPE_YIELDABLE) {
        return ABT_ERR_INV_THREAD;
    } else {
        *task = ABTI_thread_get_handle(p_thread);
    }
    return ABT_SUCCESS;
}

/**
 * @ingroup TASK
 * @brief   Return the ID of the calling tasklet.
 *
 * \c ABT_task_self_id() returns the ID of the calling tasklet.
 *
 * @param[out] id  tasklet id
 * @return Error code
 * @retval ABT_SUCCESS           on success
 * @retval ABT_ERR_UNINITIALIZED Argobots has not been initialized
 * @retval ABT_ERR_INV_XSTREAM   called by an external thread
 * @retval ABT_ERR_INV_THREAD    called by a yieldable thread (ULT)
 */
int ABT_task_self_id(ABT_unit_id *id)
{
    ABTI_xstream *p_local_xstream;
    ABTI_SETUP_LOCAL_XSTREAM_WITH_INIT_CHECK(&p_local_xstream);

    ABTI_thread *p_thread = p_local_xstream->p_thread;
    ABTI_CHECK_TRUE(!(p_thread->type & ABTI_THREAD_TYPE_YIELDABLE),
                    ABT_ERR_INV_THREAD);
    *id = ABTI_thread_get_id(p_thread);
    return ABT_SUCCESS;
}

#ifdef ABT_CONFIG_USE_DOXYGEN
/**
 * @ingroup TASK
 * @brief   Get the ES associated with the target tasklet.
 *
 * \c ABT_task_get_xstream() returns the ES handle associated with the target
 * tasklet to \c xstream. If the target tasklet is not associated with any ES,
 * \c ABT_XSTREAM_NULL is returned to \c xstream.
 *
 * @param[in]  task     handle to the target tasklet
 * @param[out] xstream  ES handle
 * @return Error code
 * @retval ABT_SUCCESS on success
 */
int ABT_task_get_xstream(ABT_task task, ABT_xstream *xstream);
#endif

#ifdef ABT_CONFIG_USE_DOXYGEN
/**
 * @ingroup TASK
 * @brief   Return the state of task.
 *
 * @param[in]  task   handle to the target task
 * @param[out] state  the task's state
 * @return Error code
 * @retval ABT_SUCCESS on success
 */
int ABT_task_get_state(ABT_task task, ABT_task_state *state);
#endif

#ifdef ABT_CONFIG_USE_DOXYGEN
/**
 * @ingroup TASK
 * @brief   Return the last pool of task.
 *
 * If the task is not running, we get the pool where it is, else we get the
 * last pool where it was (the pool from the task was popped).
 *
 * @param[in]  task  handle to the target task
 * @param[out] pool  the last pool of the task
 * @return Error code
 * @retval ABT_SUCCESS on success
 */
int ABT_task_get_last_pool(ABT_task task, ABT_pool *pool);
#endif

#ifdef ABT_CONFIG_USE_DOXYGEN
/**
 * @ingroup TASK
 * @brief   Get the last pool's ID of the tasklet
 *
 * \c ABT_task_get_last_pool_id() returns the last pool's ID of \c task.  If
 * the tasklet is not running, this routine returns the ID of the pool where it
 * is residing.  Otherwise, it returns the ID of the last pool where the
 * tasklet was (i.e., the pool from which the tasklet was popped).
 *
 * @param[in]  task  handle to the target tasklet
 * @param[out] id    pool id
 * @return Error code
 * @retval ABT_SUCCESS on success
 */
int ABT_task_get_last_pool_id(ABT_task task, int *id);
#endif

#ifdef ABT_CONFIG_USE_DOXYGEN
/**
 * @ingroup TASK
 * @brief   Set the tasklet's migratability.
 *
 * \c ABT_task_set_migratable() sets the tasklet's migratability. By default,
 * all tasklets are migratable.
 * If \c flag is \c ABT_TRUE, the target tasklet becomes migratable. On the
 * other hand, if \c flag is \c ABT_FALSE, the target tasklet becomes
 * unmigratable.
 *
 * @param[in] task  handle to the target tasklet
 * @param[in] flag  migratability flag (<tt>ABT_TRUE</tt>: migratable,
 *                  <tt>ABT_FALSE</tt>: not)
 * @return Error code
 * @retval ABT_SUCCESS on success
 */
int ABT_task_set_migratable(ABT_task task, ABT_bool flag);
#endif

#ifdef ABT_CONFIG_USE_DOXYGEN
/**
 * @ingroup TASK
 * @brief   Get the tasklet's migratability.
 *
 * \c ABT_task_is_migratable() returns the tasklet's migratability through
 * \c flag. If the target tasklet is migratable, \c ABT_TRUE is returned to
 * \c flag. Otherwise, \c flag is set to \c ABT_FALSE.
 *
 * @param[in]  task  handle to the target tasklet
 * @param[out] flag  migratability flag (<tt>ABT_TRUE</tt>: migratable,
 *                   <tt>ABT_FALSE</tt>: not)
 * @return Error code
 * @retval ABT_SUCCESS on success
 */
int ABT_task_is_migratable(ABT_task task, ABT_bool *flag);
#endif

#ifdef ABT_CONFIG_USE_DOXYGEN
/**
 * @ingroup TASK
 * @brief   Check if the target task is unnamed
 *
 * \c ABT_task_is_unnamed() returns whether the target tasklet, \c task, is
 * unnamed or not.  Note that a handle of an unnamed tasklet can be obtained by,
 * for example, running \c ABT_task_self() on an unnamed tasklet.
 *
 * @param[in]  task  handle to the target tasklet
 * @param[out] flag  result (<tt>ABT_TRUE</tt> if unnamed)
 *
 * @return Error code
 * @retval ABT_SUCCESS  on success
 */
int ABT_task_is_unnamed(ABT_task task, ABT_bool *flag);
#endif

#ifdef ABT_CONFIG_USE_DOXYGEN
/**
 * @ingroup TASK
 * @brief   Compare two tasklet handles for equality.
 *
 * \c ABT_task_equal() compares two tasklet handles for equality. If two handles
 * are associated with the same tasklet object, \c result will be set to
 * \c ABT_TRUE. Otherwise, \c result will be set to \c ABT_FALSE.
 *
 * @param[in]  task1   handle to the tasklet 1
 * @param[in]  task2   handle to the tasklet 2
 * @param[out] result  comparison result (<tt>ABT_TRUE</tt>: same,
 *                     <tt>ABT_FALSE</tt>: not same)
 * @return Error code
 * @retval ABT_SUCCESS on success
 */
int ABT_task_equal(ABT_task task1, ABT_task task2, ABT_bool *result);
#endif

#ifdef ABT_CONFIG_USE_DOXYGEN
/**
 * @ingroup TASK
 * @brief   Get the tasklet's id
 *
 * \c ABT_task_get_id() returns the id of \c task.
 *
 * @param[in]  task     handle to the target tasklet
 * @param[out] task_id  tasklet id
 * @return Error code
 * @retval ABT_SUCCESS on success
 */
int ABT_task_get_id(ABT_task task, ABT_unit_id *task_id);
#endif

#ifdef ABT_CONFIG_USE_DOXYGEN
/**
 * @ingroup TASK
 * @brief   Retrieve the argument for the tasklet function
 *
 * \c ABT_task_get_arg() returns the argument for the taslet function, which was
 * passed to \c ABT_task_create() when the target tasklet \c task was created.
 *
 * @param[in]  task  handle to the target tasklet
 * @param[out] arg   argument for the tasklet function
 * @return Error code
 * @retval ABT_SUCCESS on success
 */
int ABT_task_get_arg(ABT_task task, void **arg);
#endif

#ifdef ABT_CONFIG_USE_DOXYGEN
/**
 * @ingroup TASK
 * @brief  Set the tasklet-specific value associated with the key
 *
 * \c ABT_task_set_specific() associates a value, \c value, with a work
 * unit-specific data key, \c key.  The target work unit is \c task.
 *
 * @param[in] task   handle to the target tasklet
 * @param[in] key    handle to the target key
 * @param[in] value  value for the key
 * @return Error code
 * @retval ABT_SUCCESS on success
 */
int ABT_task_set_specific(ABT_task task, ABT_key key, void *value);
#endif

#ifdef ABT_CONFIG_USE_DOXYGEN
/**
 * @ingroup TASK
 * @brief   Get the tasklet-specific value associated with the key
 *
 * \c ABT_task_get_specific() returns the value associated with a target work
 * unit-specific data key, \c key, through \c value.  The target work unit is
 * \c task.  If \c task has never set a value for the key, this routine returns
 * \c NULL to \c value.
 *
 * @param[in]  task   handle to the target tasklet
 * @param[in]  key    handle to the target key
 * @param[out] value  value for the key
 * @return Error code
 * @retval ABT_SUCCESS on success
 */
int ABT_task_get_specific(ABT_task task, ABT_key key, void **value);
#endif

/*****************************************************************************/
/* Internal static functions                                                 */
/*****************************************************************************/

ABTU_ret_err static int task_create(ABTI_local *p_local, ABTI_pool *p_pool,
                                    void (*task_func)(void *), void *arg,
                                    ABTI_sched *p_sched, int refcount,
                                    ABTI_thread **pp_newtask)
{
    ABTI_thread *p_newtask;
    ABT_task h_newtask;

    /* Allocate a task object */
    int abt_errno = ABTI_mem_alloc_nythread(p_local, &p_newtask);
    ABTI_CHECK_ERROR(abt_errno);

    p_newtask->p_last_xstream = NULL;
    p_newtask->p_parent = NULL;
    ABTD_atomic_relaxed_store_int(&p_newtask->state, ABT_THREAD_STATE_READY);
    ABTD_atomic_relaxed_store_uint32(&p_newtask->request, 0);
    p_newtask->f_thread = task_func;
    p_newtask->p_arg = arg;
    p_newtask->p_pool = p_pool;
    ABTD_atomic_relaxed_store_ptr(&p_newtask->p_keytable, NULL);
    p_newtask->id = ABTI_TASK_INIT_ID;

    /* Create a wrapper work unit */
    h_newtask = ABTI_thread_get_handle(p_newtask);
    ABTI_thread_type thread_type =
        refcount ? (ABTI_THREAD_TYPE_THREAD | ABTI_THREAD_TYPE_NAMED)
                 : ABTI_THREAD_TYPE_THREAD;
#ifndef ABT_CONFIG_DISABLE_MIGRATION
    thread_type |= ABTI_THREAD_TYPE_MIGRATABLE;
#endif
    p_newtask->type |= thread_type;
    p_newtask->unit = p_pool->u_create_from_task(h_newtask);

    ABTI_tool_event_thread_create(p_local, p_newtask,
                                  ABTI_local_get_xstream_or_null(p_local)
                                      ? ABTI_local_get_xstream(p_local)
                                            ->p_thread
                                      : NULL,
                                  p_pool);
    LOG_DEBUG("[T%" PRIu64 "] created\n", ABTI_thread_get_id(p_newtask));

    /* Add this task to the scheduler's pool */
    ABTI_pool_push(p_pool, p_newtask->unit);

    /* Return value */
    *pp_newtask = p_newtask;

    return ABT_SUCCESS;
}

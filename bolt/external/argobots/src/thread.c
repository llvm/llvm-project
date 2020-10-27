/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

#include "abti.h"

ABTU_ret_err static inline int
ythread_create(ABTI_local *p_local, ABTI_pool *p_pool,
               void (*thread_func)(void *), void *arg, ABTI_thread_attr *p_attr,
               ABTI_thread_type thread_type, ABTI_sched *p_sched,
               ABT_bool push_pool, ABTI_ythread **pp_newthread);
static inline void thread_join(ABTI_local **pp_local, ABTI_thread *p_thread);
static inline void thread_free(ABTI_local *p_local, ABTI_thread *p_thread,
                               ABT_bool free_unit);
static void thread_root_func(void *arg);
static void thread_main_sched_func(void *arg);
#ifndef ABT_CONFIG_DISABLE_MIGRATION
ABTU_ret_err static int thread_migrate_to_xstream(ABTI_local **pp_local,
                                                  ABTI_thread *p_thread,
                                                  ABTI_xstream *p_xstream);
ABTU_ret_err static int thread_migrate_to_pool(ABTI_local **p_local,
                                               ABTI_thread *p_thread,
                                               ABTI_pool *p_pool);
#endif
static inline ABT_unit_id thread_get_new_id(void);

static void thread_key_destructor_stackable_sched(void *p_value);
static ABTI_key g_thread_sched_key =
    ABTI_KEY_STATIC_INITIALIZER(thread_key_destructor_stackable_sched,
                                ABTI_KEY_ID_STACKABLE_SCHED);
static void thread_key_destructor_migration(void *p_value);
static ABTI_key g_thread_mig_data_key =
    ABTI_KEY_STATIC_INITIALIZER(thread_key_destructor_migration,
                                ABTI_KEY_ID_MIGRATION);

/** @defgroup ULT User-level Thread (ULT)
 * This group is for User-level Thread (ULT).
 */

/**
 * @ingroup ULT
 * @brief   Create a new thread and return its handle through newthread.
 *
 * \c ABT_thread_create() creates a new ULT that is pushed into \c pool. The
 * insertion is done from the ES where this call is made. Therefore, the access
 * type of \c pool should comply with that. Only a \a secondary ULT can be
 * created explicitly, and the \a primary ULT is created automatically.
 *
 * If newthread is NULL, the thread object will be automatically released when
 * this \a unnamed thread completes the execution of thread_func. Otherwise,
 * ABT_thread_free() can be used to explicitly release the thread object.
 *
 * @param[in]  pool         handle to the associated pool
 * @param[in]  thread_func  function to be executed by a new thread
 * @param[in]  arg          argument for thread_func
 * @param[in]  attr         thread attribute. If it is ABT_THREAD_ATTR_NULL,
 *                          the default attribute is used.
 * @param[out] newthread    handle to a newly created thread
 * @return Error code
 * @retval ABT_SUCCESS on success
 */
int ABT_thread_create(ABT_pool pool, void (*thread_func)(void *), void *arg,
                      ABT_thread_attr attr, ABT_thread *newthread)
{
    ABTI_local *p_local = ABTI_local_get_local();
    ABTI_ythread *p_newthread;

    ABTI_pool *p_pool = ABTI_pool_get_ptr(pool);
    ABTI_CHECK_NULL_POOL_PTR(p_pool);

    ABTI_thread_type unit_type =
        (newthread != NULL)
            ? (ABTI_THREAD_TYPE_YIELDABLE | ABTI_THREAD_TYPE_NAMED)
            : ABTI_THREAD_TYPE_YIELDABLE;
    int abt_errno = ythread_create(p_local, p_pool, thread_func, arg,
                                   ABTI_thread_attr_get_ptr(attr), unit_type,
                                   NULL, ABT_TRUE, &p_newthread);
    ABTI_CHECK_ERROR(abt_errno);

    /* Return value */
    if (newthread)
        *newthread = ABTI_ythread_get_handle(p_newthread);
    return ABT_SUCCESS;
}

/**
 * @ingroup ULT
 * @brief   Create a new ULT associated with the target ES (\c xstream).
 *
 * \c ABT_thread_create_on_xstream() creates a new ULT associated with the
 * target ES and returns its handle through \c newthread. The new ULT will be
 * inserted into a proper pool associated with the main scheduler of the target
 * ES.
 *
 * This routine is only for convenience. If the user wants to focus on the
 * performance, we recommend to use \c ABT_thread_create() with directly
 * dealing with pools. Pools are a right way to manage work units in Argobots.
 * ES is just an abstract, and it is not a mechanism for execution and
 * performance tuning.
 *
 * If \c attr is \c ABT_THREAD_ATTR_NULL, a new ULT is created with default
 * attributes. For example, the stack size of default attribute is 16KB.
 * If the attribute is specified, attribute values are saved in the ULT object.
 * After creating the ULT object, changes in the attribute object will not
 * affect attributes of the ULT object. A new attribute object can be created
 * with \c ABT_thread_attr_create().
 *
 * If \c newthread is \c NULL, this routine creates an unnamed ULT. The object
 * for unnamed ULT will be automatically freed when the unnamed ULT completes
 * its execution. Otherwise, this routine creates a named ULT and
 * \c ABT_thread_free() can be used to explicitly free the object for
 * the named ULT.
 *
 * If \c newthread is not \c NULL and an error occurs in this routine,
 * a non-zero error code will be returned and \c newthread will be set to
 * \c ABT_THREAD_NULL.
 *
 * @param[in]  xstream      handle to the target ES
 * @param[in]  thread_func  function to be executed by a new ULT
 * @param[in]  arg          argument for <tt>thread_func</tt>
 * @param[in]  attr         ULT attribute
 * @param[out] newthread    handle to a newly created ULT
 * @return Error code
 * @retval ABT_SUCCESS on success
 */
int ABT_thread_create_on_xstream(ABT_xstream xstream,
                                 void (*thread_func)(void *), void *arg,
                                 ABT_thread_attr attr, ABT_thread *newthread)
{
    ABTI_local *p_local = ABTI_local_get_local();
    ABTI_ythread *p_newthread;

    ABTI_xstream *p_xstream = ABTI_xstream_get_ptr(xstream);
    ABTI_CHECK_NULL_XSTREAM_PTR(p_xstream);

    /* TODO: need to consider the access type of target pool */
    ABTI_pool *p_pool = ABTI_xstream_get_main_pool(p_xstream);
    ABTI_thread_type unit_type =
        (newthread != NULL)
            ? (ABTI_THREAD_TYPE_YIELDABLE | ABTI_THREAD_TYPE_NAMED)
            : ABTI_THREAD_TYPE_YIELDABLE;
    int abt_errno = ythread_create(p_local, p_pool, thread_func, arg,
                                   ABTI_thread_attr_get_ptr(attr), unit_type,
                                   NULL, ABT_TRUE, &p_newthread);
    ABTI_CHECK_ERROR(abt_errno);

    /* Return value */
    if (newthread)
        *newthread = ABTI_ythread_get_handle(p_newthread);

    return ABT_SUCCESS;
}

/**
 * @ingroup ULT
 * @brief   Create a set of ULTs.
 *
 * \c ABT_thread_create_many() creates a set of ULTs, i.e., \c num ULTs, having
 * the same attribute and returns ULT handles to \c newthread_list.  Each newly
 * created ULT is pushed to each pool of \c pool_list.  That is, the \a i-th
 * ULT is pushed to \a i-th pool in \c pool_list.
 *
 * NOTE: Since this routine uses the same ULT attribute for creating all ULTs,
 * it does not support using the user-provided stack.  If \c attr contains the
 * user-provided stack, it will return an error. When \c newthread_list is NULL,
 * unnamed threads are created.
 *
 * @param[in] num               the number of array elements
 * @param[in] pool_list         array of pool handles
 * @param[in] thread_func_list  array of ULT functions
 * @param[in] arg_list          array of arguments for each ULT function
 * @param[in] attr              ULT attribute
 * @param[out] newthread_list   array of newly created ULT handles
 * @return Error code
 * @retval ABT_SUCCESS on success
 */
int ABT_thread_create_many(int num, ABT_pool *pool_list,
                           void (**thread_func_list)(void *), void **arg_list,
                           ABT_thread_attr attr, ABT_thread *newthread_list)
{
    ABTI_local *p_local = ABTI_local_get_local();
    int i;

    if (attr != ABT_THREAD_ATTR_NULL) {
        /* This implies that the stack is given by a user.  Since threads
         * cannot use the same stack region, this is illegal. */
        ABTI_CHECK_TRUE(!(ABTI_thread_attr_get_ptr(attr)->thread_type &
                          (ABTI_THREAD_TYPE_MEM_MEMPOOL_DESC |
                           ABTI_THREAD_TYPE_MEM_MALLOC_DESC)),
                        ABT_ERR_INV_THREAD_ATTR);
    }

    if (newthread_list == NULL) {
        for (i = 0; i < num; i++) {
            ABTI_ythread *p_newthread;
            ABT_pool pool = pool_list[i];
            ABTI_pool *p_pool = ABTI_pool_get_ptr(pool);
            ABTI_CHECK_NULL_POOL_PTR(p_pool);

            void (*thread_f)(void *) = thread_func_list[i];
            void *arg = arg_list ? arg_list[i] : NULL;
            int abt_errno = ythread_create(p_local, p_pool, thread_f, arg,
                                           ABTI_thread_attr_get_ptr(attr),
                                           ABTI_THREAD_TYPE_YIELDABLE, NULL,
                                           ABT_TRUE, &p_newthread);
            ABTI_CHECK_ERROR(abt_errno);
        }
    } else {
        for (i = 0; i < num; i++) {
            ABTI_ythread *p_newthread;
            ABT_pool pool = pool_list[i];
            ABTI_pool *p_pool = ABTI_pool_get_ptr(pool);
            ABTI_CHECK_NULL_POOL_PTR(p_pool);

            void (*thread_f)(void *) = thread_func_list[i];
            void *arg = arg_list ? arg_list[i] : NULL;
            int abt_errno = ythread_create(p_local, p_pool, thread_f, arg,
                                           ABTI_thread_attr_get_ptr(attr),
                                           ABTI_THREAD_TYPE_YIELDABLE |
                                               ABTI_THREAD_TYPE_NAMED,
                                           NULL, ABT_TRUE, &p_newthread);
            newthread_list[i] = ABTI_ythread_get_handle(p_newthread);
            /* TODO: Release threads that have been already created. */
            ABTI_CHECK_ERROR(abt_errno);
        }
    }

    return ABT_SUCCESS;
}

/**
 * @ingroup ULT
 * @brief   Revive the ULT.
 *
 * \c ABT_thread_revive() revives the ULT, \c thread, with \c thread_func and
 * \arg while it does not change the attributes used in creating \c thread.
 * The revived ULT is pushed into \c pool.
 *
 * This function must be called with a valid ULT handle, which has not been
 * freed by \c ABT_thread_free().  However, the ULT should have been joined by
 * \c ABT_thread_join() before its handle is used in this routine.
 *
 * @param[in]     pool         handle to the associated pool
 * @param[in]     thread_func  function to be executed by the ULT
 * @param[in]     arg          argument for thread_func
 * @param[in,out] thread       handle to the ULT
 * @return Error code
 * @retval ABT_SUCCESS on success
 */
int ABT_thread_revive(ABT_pool pool, void (*thread_func)(void *), void *arg,
                      ABT_thread *thread)
{
    ABTI_local *p_local = ABTI_local_get_local();

    ABTI_thread *p_thread = ABTI_thread_get_ptr(*thread);
    ABTI_CHECK_NULL_THREAD_PTR(p_thread);

    ABTI_CHECK_TRUE(ABTD_atomic_relaxed_load_int(&p_thread->state) ==
                        ABT_THREAD_STATE_TERMINATED,
                    ABT_ERR_INV_THREAD);

    ABTI_pool *p_pool = ABTI_pool_get_ptr(pool);
    ABTI_CHECK_NULL_POOL_PTR(p_pool);

    ABTI_thread_revive(p_local, p_pool, thread_func, arg, p_thread);

    return ABT_SUCCESS;
}

/**
 * @ingroup ULT
 * @brief   Release the thread object associated with thread handle.
 *
 * This routine deallocates memory used for the thread object. If the thread
 * is still running when this routine is called, the deallocation happens
 * after the thread terminates and then this routine returns. If it is
 * successfully processed, thread is set as ABT_THREAD_NULL.
 *
 * @param[in,out] thread  handle to the target thread
 * @return Error code
 * @retval ABT_SUCCESS on success
 */
int ABT_thread_free(ABT_thread *thread)
{
    ABTI_local *p_local = ABTI_local_get_local();
    ABT_thread h_thread = *thread;

    ABTI_thread *p_thread = ABTI_thread_get_ptr(h_thread);
    ABTI_CHECK_NULL_THREAD_PTR(p_thread);

    /* We first need to check whether p_local_xstream is NULL because external
     * threads might call this routine. */
    ABTI_CHECK_TRUE_MSG(!ABTI_local_get_xstream_or_null(p_local) ||
                            p_thread !=
                                ABTI_local_get_xstream(p_local)->p_thread,
                        ABT_ERR_INV_THREAD,
                        "The current thread cannot be freed.");

    ABTI_CHECK_TRUE_MSG(!(p_thread->type & (ABTI_THREAD_TYPE_MAIN |
                                            ABTI_THREAD_TYPE_MAIN_SCHED)),
                        ABT_ERR_INV_THREAD,
                        "The main thread cannot be freed explicitly.");

    /* Wait until the thread terminates */
    thread_join(&p_local, p_thread);
    /* Free the ABTI_thread structure */
    ABTI_thread_free(p_local, p_thread);

    /* Return value */
    *thread = ABT_THREAD_NULL;

    return ABT_SUCCESS;
}

/**
 * @ingroup ULT
 * @brief   Release a set of ULT objects.
 *
 * \c ABT_thread_free_many() releases a set of ULT objects listed in
 * \c thread_list. If it is successfully processed, all elements in
 * \c thread_list are set to \c ABT_THREAD_NULL.
 *
 * @param[in]     num          the number of array elements
 * @param[in,out] thread_list  array of ULT handles
 * @return Error code
 * @retval ABT_SUCCESS on success
 */
int ABT_thread_free_many(int num, ABT_thread *thread_list)
{
    ABTI_local *p_local = ABTI_local_get_local();
    int i;

    for (i = 0; i < num; i++) {
        ABTI_thread *p_thread = ABTI_thread_get_ptr(thread_list[i]);
        /* TODO: check input */
        thread_join(&p_local, p_thread);
        ABTI_thread_free(p_local, p_thread);
    }
    return ABT_SUCCESS;
}

/**
 * @ingroup ULT
 * @brief   Wait for thread to terminate.
 *
 * The target thread cannot be the same as the calling thread.
 *
 * @param[in] thread  handle to the target thread
 * @return Error code
 * @retval ABT_SUCCESS on success
 */
int ABT_thread_join(ABT_thread thread)
{
    ABTI_local *p_local = ABTI_local_get_local();
    ABTI_thread *p_thread = ABTI_thread_get_ptr(thread);
    ABTI_CHECK_NULL_THREAD_PTR(p_thread);

    ABTI_CHECK_TRUE_MSG(!ABTI_local_get_xstream_or_null(p_local) ||
                            p_thread !=
                                ABTI_local_get_xstream(p_local)->p_thread,
                        ABT_ERR_INV_THREAD,
                        "The current thread cannot be freed.");

    ABTI_CHECK_TRUE_MSG(!(p_thread->type & (ABTI_THREAD_TYPE_MAIN |
                                            ABTI_THREAD_TYPE_MAIN_SCHED)),
                        ABT_ERR_INV_THREAD,
                        "The main thread cannot be freed explicitly.");

    thread_join(&p_local, p_thread);
    return ABT_SUCCESS;
}

/**
 * @ingroup ULT
 * @brief   Wait for a number of ULTs to terminate.
 *
 * The caller of \c ABT_thread_join_many() waits until all ULTs in
 * \c thread_list, which should have \c num_threads ULT handles, are terminated.
 *
 * @param[in] num_threads  the number of ULTs to join
 * @param[in] thread_list  array of target ULT handles
 * @return Error code
 * @retval ABT_SUCCESS on success
 */
int ABT_thread_join_many(int num_threads, ABT_thread *thread_list)
{
    ABTI_local *p_local = ABTI_local_get_local();
    int i;
    for (i = 0; i < num_threads; i++) {
        /* TODO: check input */
        thread_join(&p_local, ABTI_thread_get_ptr(thread_list[i]));
    }
    return ABT_SUCCESS;
}

/**
 * @ingroup ULT
 * @brief   The calling ULT terminates its execution.
 *
 * Since the calling ULT terminates, this routine never returns.
 *
 * @return Error code
 * @retval ABT_SUCCESS           on success
 * @retval ABT_ERR_UNINITIALIZED Argobots has not been initialized
 * @retval ABT_ERR_INV_XSTREAM   called by an external thread
 * @retval ABT_ERR_INV_THREAD    called by a non-yieldable thread (tasklet)
 */
int ABT_thread_exit(void)
{
    ABTI_xstream *p_local_xstream;
    ABTI_ythread *p_ythread;
    ABTI_SETUP_LOCAL_YTHREAD_WITH_INIT_CHECK(&p_local_xstream, &p_ythread);

    ABTI_ythread_exit(p_local_xstream, p_ythread);
    return ABT_SUCCESS;
}

/**
 * @ingroup ULT
 * @brief   Request the cancellation of the target thread.
 *
 * @param[in] thread  handle to the target thread
 * @return Error code
 * @retval ABT_SUCCESS on success
 */
int ABT_thread_cancel(ABT_thread thread)
{
#ifdef ABT_CONFIG_DISABLE_THREAD_CANCEL
    ABTI_HANDLE_ERROR(ABT_ERR_FEATURE_NA);
#else
    ABTI_thread *p_thread = ABTI_thread_get_ptr(thread);
    ABTI_CHECK_NULL_THREAD_PTR(p_thread);
    ABTI_CHECK_TRUE_MSG(!(p_thread->type & (ABTI_THREAD_TYPE_MAIN |
                                            ABTI_THREAD_TYPE_MAIN_SCHED)),
                        ABT_ERR_INV_THREAD,
                        "The main thread cannot be canceled.");

    /* Set the cancel request */
    ABTI_thread_set_request(p_thread, ABTI_THREAD_REQ_CANCEL);
    return ABT_SUCCESS;
#endif
}

/**
 * @ingroup ULT
 * @brief   Return the handle of the calling ULT.
 *
 * \c ABT_thread_self() returns the handle of the calling ULT. Both the primary
 * ULT and secondary ULTs can get their handle through this routine.
 * If tasklets call this routine, \c ABT_THREAD_NULL will be returned to
 * \c thread.
 *
 * At present \c thread is set to \c ABT_THREAD_NULL when an error occurs, but
 * this behavior is deprecated.  The program should not rely on this behavior.
 *
 * @param[out] thread  ULT handle
 * @return Error code
 * @retval ABT_SUCCESS           on success
 * @retval ABT_ERR_UNINITIALIZED Argobots has not been initialized
 * @retval ABT_ERR_INV_XSTREAM   called by an external thread
 * @retval ABT_ERR_INV_THREAD    called by a non-yieldable thread (tasklet)
 */
int ABT_thread_self(ABT_thread *thread)
{
    *thread = ABT_THREAD_NULL;

    ABTI_xstream *p_local_xstream;
    ABTI_SETUP_LOCAL_XSTREAM_WITH_INIT_CHECK(&p_local_xstream);
    ABTI_thread *p_thread = p_local_xstream->p_thread;
    if (!(p_thread->type & ABTI_THREAD_TYPE_YIELDABLE)) {
        /* This is checked even if an error check is disabled. */
        ABTI_HANDLE_ERROR(ABT_ERR_INV_THREAD);
    }

    *thread = ABTI_thread_get_handle(p_thread);
    return ABT_SUCCESS;
}

/**
 * @ingroup ULT
 * @brief   Return the calling ULT's ID.
 *
 * \c ABT_thread_self_id() returns the ID of the calling ULT.
 *
 * @param[out] id  ULT id
 * @return Error code
 * @retval ABT_SUCCESS           on success
 * @retval ABT_ERR_UNINITIALIZED Argobots has not been initialized
 * @retval ABT_ERR_INV_XSTREAM   called by an external thread
 * @retval ABT_ERR_INV_THREAD    called by a non-yieldable thread (tasklet)
 */
int ABT_thread_self_id(ABT_unit_id *id)
{
    ABTI_ythread *p_self;
    ABTI_SETUP_LOCAL_YTHREAD_WITH_INIT_CHECK(NULL, &p_self);

    *id = ABTI_thread_get_id(&p_self->thread);
    return ABT_SUCCESS;
}

/**
 * @ingroup ULT
 * @brief   Get the ES associated with the target thread.
 *
 * \c ABT_thread_get_last_xstream() returns the last ES handle associated with
 * the target thread to \c xstream.  If the target thread is not associated
 * with any ES, \c ABT_XSTREAM_NULL is returned to \c xstream.
 *
 * @param[in]  thread   handle to the target thread
 * @param[out] xstream  ES handle
 * @return Error code
 * @retval ABT_SUCCESS on success
 */
int ABT_thread_get_last_xstream(ABT_thread thread, ABT_xstream *xstream)
{
    ABTI_thread *p_thread = ABTI_thread_get_ptr(thread);
    ABTI_CHECK_NULL_THREAD_PTR(p_thread);

    *xstream = ABTI_xstream_get_handle(p_thread->p_last_xstream);
    return ABT_SUCCESS;
}

/**
 * @ingroup ULT
 * @brief   Return the state of thread.
 *
 * @param[in]  thread  handle to the target thread
 * @param[out] state   the thread's state
 * @return Error code
 * @retval ABT_SUCCESS on success
 */
int ABT_thread_get_state(ABT_thread thread, ABT_thread_state *state)
{
    ABTI_thread *p_thread = ABTI_thread_get_ptr(thread);
    ABTI_CHECK_NULL_THREAD_PTR(p_thread);

    *state = (ABT_thread_state)ABTD_atomic_acquire_load_int(&p_thread->state);
    return ABT_SUCCESS;
}

/**
 * @ingroup ULT
 * @brief   Return the last pool of ULT.
 *
 * If the ULT is not running, we get the pool where it is, else we get the
 * last pool where it was (i.e., the pool from which the ULT was popped).
 *
 * @param[in]  thread handle to the target ULT
 * @param[out] pool   the last pool of the ULT
 * @return Error code
 * @retval ABT_SUCCESS on success
 */
int ABT_thread_get_last_pool(ABT_thread thread, ABT_pool *pool)
{
    ABTI_thread *p_thread = ABTI_thread_get_ptr(thread);
    ABTI_CHECK_NULL_THREAD_PTR(p_thread);

    *pool = ABTI_pool_get_handle(p_thread->p_pool);
    return ABT_SUCCESS;
}

/**
 * @ingroup ULT
 * @brief   Get the last pool's ID of the ULT
 *
 * \c ABT_thread_get_last_pool_id() returns the last pool's ID of \c thread.
 * If the ULT is not running, this routine returns the ID of the pool where it
 * is residing.  Otherwise, it returns the ID of the last pool where the ULT
 * was (i.e., the pool from which the ULT was popped).
 *
 * @param[in]  thread  handle to the target ULT
 * @param[out] id      pool id
 * @return Error code
 * @retval ABT_SUCCESS on success
 */
int ABT_thread_get_last_pool_id(ABT_thread thread, int *id)
{
    ABTI_thread *p_thread = ABTI_thread_get_ptr(thread);
    ABTI_CHECK_NULL_THREAD_PTR(p_thread);
    ABTI_ASSERT(p_thread->p_pool);

    *id = (int)(p_thread->p_pool->id);
    return ABT_SUCCESS;
}

/**
 * @ingroup ULT
 * @brief   Set the associated pool for the target ULT.
 *
 * \c ABT_thread_set_associated_pool() changes the associated pool of the target
 * ULT \c thread to \c pool.  This routine must be called after \c thread is
 * popped from its original associated pool (i.e., \c thread must not be inside
 * any pool), which is the pool where \c thread was residing in.
 *
 * NOTE: \c ABT_thread_migrate_to_pool() can be used to change the associated
 * pool of \c thread regardless of its location.
 *
 * @param[in] thread  handle to the target ULT
 * @param[in] pool    handle to the pool
 * @return Error code
 * @retval ABT_SUCCESS on success
 */
int ABT_thread_set_associated_pool(ABT_thread thread, ABT_pool pool)
{
    ABTI_thread *p_thread = ABTI_thread_get_ptr(thread);
    ABTI_CHECK_NULL_THREAD_PTR(p_thread);
    ABTI_pool *p_pool = ABTI_pool_get_ptr(pool);
    ABTI_CHECK_NULL_POOL_PTR(p_pool);

    p_thread->p_pool = p_pool;
    return ABT_SUCCESS;
}

/**
 * @ingroup ULT
 * @brief   Yield the processor from the current running thread to the
 *          specific thread.
 *
 * This function can be used for users to explicitly schedule the next thread
 * to execute.
 *
 * @param[in] thread  handle to the target thread
 * @return Error code
 * @retval ABT_SUCCESS          on success
 * @retval ABT_ERR_INV_XSTREAM  called by an external thread
 * @retval ABT_ERR_INV_THREAD   called by a non-yieldable thread (tasklet)
 */
int ABT_thread_yield_to(ABT_thread thread)
{
    ABTI_ythread *p_tar_ythread = ABTI_ythread_get_ptr(thread);
    ABTI_CHECK_NULL_YTHREAD_PTR(p_tar_ythread);

    ABTI_xstream *p_local_xstream;
    ABTI_ythread *p_cur_ythread;
    ABTI_SETUP_LOCAL_YTHREAD(&p_local_xstream, &p_cur_ythread);

    LOG_DEBUG("[U%" PRIu64 ":E%d] yield_to -> U%" PRIu64 "\n",
              ABTI_thread_get_id(&p_cur_ythread->thread),
              p_cur_ythread->thread.p_last_xstream->rank,
              ABTI_thread_get_id(&p_tar_ythread->thread));

    /* The target ULT must be different from the caller ULT. */
    ABTI_CHECK_TRUE_MSG(p_cur_ythread != p_tar_ythread, ABT_ERR_INV_THREAD,
                        "The caller and target ULTs are the same.");

    ABTI_CHECK_TRUE_MSG(ABTD_atomic_relaxed_load_int(
                            &p_tar_ythread->thread.state) !=
                            ABT_THREAD_STATE_TERMINATED,
                        ABT_ERR_INV_THREAD,
                        "Cannot yield to the terminated thread");

    /* Both threads must be associated with the same pool. */
    /* FIXME: instead of same pool, runnable by the same ES */
    ABTI_CHECK_TRUE_MSG(p_cur_ythread->thread.p_pool ==
                            p_tar_ythread->thread.p_pool,
                        ABT_ERR_INV_THREAD,
                        "The target thread's pool is not the same as mine.");

    /* If the target thread is not in READY, we don't yield.  Note that ULT can
     * be regarded as 'ready' only if its state is READY and it has been
     * pushed into a pool. Since we set ULT's state to READY and then push it
     * into a pool, we check them in the reverse order, i.e., check if the ULT
     * is inside a pool and the its state. */
    if (!(p_tar_ythread->thread.p_pool->u_is_in_pool(
              p_tar_ythread->thread.unit) == ABT_TRUE &&
          ABTD_atomic_acquire_load_int(&p_tar_ythread->thread.state) ==
              ABT_THREAD_STATE_READY)) {
        return ABT_SUCCESS;
    }

    /* Remove the target ULT from the pool */
    if (ABTI_IS_ERROR_CHECK_ENABLED) {
        /* This is necessary to prevent the size of this pool from 0. */
        ABTI_pool_inc_num_blocked(p_tar_ythread->thread.p_pool);
    }
    int abt_errno = ABTI_pool_remove(p_tar_ythread->thread.p_pool,
                                     p_tar_ythread->thread.unit);
    if (ABTI_IS_ERROR_CHECK_ENABLED) {
        ABTI_pool_dec_num_blocked(p_tar_ythread->thread.p_pool);
        ABTI_CHECK_ERROR(abt_errno);
    }

    ABTD_atomic_release_store_int(&p_cur_ythread->thread.state,
                                  ABT_THREAD_STATE_READY);

    /* This operation is corresponding to yield */
    ABTI_tool_event_ythread_yield(p_local_xstream, p_cur_ythread,
                                  p_cur_ythread->thread.p_parent,
                                  ABT_SYNC_EVENT_TYPE_USER, NULL);

    /* Add the current thread to the pool again. */
    ABTI_pool_push(p_cur_ythread->thread.p_pool, p_cur_ythread->thread.unit);

    /* We set the last ES */
    p_tar_ythread->thread.p_last_xstream = p_local_xstream;

    /* Switch the context */
    ABTD_atomic_release_store_int(&p_tar_ythread->thread.state,
                                  ABT_THREAD_STATE_RUNNING);
    ABTI_ythread *p_prev =
        ABTI_ythread_context_switch_to_sibling(&p_local_xstream, p_cur_ythread,
                                               p_tar_ythread);
    ABTI_tool_event_thread_run(p_local_xstream, &p_cur_ythread->thread,
                               &p_prev->thread, p_cur_ythread->thread.p_parent);
    return ABT_SUCCESS;
}

/**
 * @ingroup ULT
 * @brief   Yield the processor from the current running ULT back to the
 *          scheduler.
 *
 * The ULT that yields, goes back to its pool, and eventually will be
 * resumed automatically later.
 *
 * @return Error code
 * @retval ABT_SUCCESS on success
 * @retval ABT_ERR_UNINITIALIZED Argobots has not been initialized
 * @retval ABT_ERR_INV_XSTREAM   called by an external thread
 * @retval ABT_ERR_INV_THREAD    called by a non-yieldable thread (tasklet)
 */
int ABT_thread_yield(void)
{
    ABTI_xstream *p_local_xstream;
    ABTI_ythread *p_ythread;
    ABTI_SETUP_LOCAL_YTHREAD_WITH_INIT_CHECK(&p_local_xstream, &p_ythread);

    ABTI_ythread_yield(&p_local_xstream, p_ythread, ABT_SYNC_EVENT_TYPE_USER,
                       NULL);
    return ABT_SUCCESS;
}

/**
 * @ingroup ULT
 * @brief   Resume the target ULT.
 *
 * \c ABT_thread_resume() makes the blocked ULT schedulable by changing the
 * state of the target ULT to READY and pushing it to its associated pool.
 * The ULT will resume its execution when the scheduler schedules it.
 *
 * The ULT should have been blocked by \c ABT_self_suspend() or
 * \c ABT_thread_suspend().  Otherwise, the behavior of this routine is
 * undefined.
 *
 * @param[in] thread   handle to the target ULT
 * @return Error code
 * @retval ABT_SUCCESS on success
 */
int ABT_thread_resume(ABT_thread thread)
{
    ABTI_local *p_local = ABTI_local_get_local();

    ABTI_thread *p_thread = ABTI_thread_get_ptr(thread);
    ABTI_CHECK_NULL_THREAD_PTR(p_thread);
    ABTI_ythread *p_ythread;
    ABTI_CHECK_YIELDABLE(p_thread, &p_ythread, ABT_ERR_INV_THREAD);

    /* The ULT must be in BLOCKED state. */
    ABTI_CHECK_TRUE(ABTD_atomic_acquire_load_int(&p_ythread->thread.state) ==
                        ABT_THREAD_STATE_BLOCKED,
                    ABT_ERR_THREAD);

    ABTI_ythread_set_ready(p_local, p_ythread);
    return ABT_SUCCESS;
}

/**
 * @ingroup ULT
 * @brief   Migrate a thread to a specific ES.
 *
 * The actual migration occurs asynchronously with this function call.  In other
 * words, this function may return immediately without the thread being
 * migrated.  The migration request will be posted on the thread, such that next
 * time a scheduler picks it up, migration will happen.  The target pool is
 * chosen by the running scheduler of the target ES.
 *
 * Note that users must be responsible for keeping the target execution stream,
 * its main scheduler, and the associated pools available during this function
 * and, if this function returns ABT_SUCCESS, until the migration process
 * completes.
 *
 * The migration will fail if the running scheduler has no pool available for
 * migration.
 *
 * @param[in] thread   handle to the thread to migrate
 * @param[in] xstream  handle to the ES to migrate the thread to
 * @return Error code
 * @retval ABT_SUCCESS on success
 */
int ABT_thread_migrate_to_xstream(ABT_thread thread, ABT_xstream xstream)
{
#ifndef ABT_CONFIG_DISABLE_MIGRATION
    ABTI_local *p_local = ABTI_local_get_local();
    ABTI_thread *p_thread = ABTI_thread_get_ptr(thread);
    ABTI_CHECK_NULL_THREAD_PTR(p_thread);
    ABTI_xstream *p_xstream = ABTI_xstream_get_ptr(xstream);
    ABTI_CHECK_NULL_XSTREAM_PTR(p_xstream);

    int abt_errno = thread_migrate_to_xstream(&p_local, p_thread, p_xstream);
    ABTI_CHECK_ERROR(abt_errno);
    return ABT_SUCCESS;
#else
    ABTI_HANDLE_ERROR(ABT_ERR_MIGRATION_NA);
#endif
}

/**
 * @ingroup ULT
 * @brief   Migrate a thread to a specific scheduler.
 *
 * The actual migration occurs asynchronously with this function call.  In other
 * words, this function may return immediately without the thread being
 * migrated.  The migration request will be posted on the thread, such that next
 * time a scheduler picks it up, migration will happen.  The target pool is
 * chosen by the scheduler itself.
 *
 * Note that users must be responsible for keeping the target scheduler and its
 * associated pools available during this function and, if this function returns
 * ABT_SUCCESS, until the migration process completes.
 *
 * The migration will fail if the target scheduler has no pool available for
 * migration.
 *
 * @param[in] thread handle to the thread to migrate
 * @param[in] sched  handle to the sched to migrate the thread to
 * @return Error code
 * @retval ABT_SUCCESS on success
 */
int ABT_thread_migrate_to_sched(ABT_thread thread, ABT_sched sched)
{
#ifndef ABT_CONFIG_DISABLE_MIGRATION
    ABTI_local *p_local = ABTI_local_get_local();
    ABTI_thread *p_thread = ABTI_thread_get_ptr(thread);
    ABTI_CHECK_NULL_THREAD_PTR(p_thread);
    ABTI_sched *p_sched = ABTI_sched_get_ptr(sched);
    ABTI_CHECK_NULL_SCHED_PTR(p_sched);

    /* checking for cases when migration is not allowed */
    ABTI_CHECK_TRUE(!(p_thread->type &
                      (ABTI_THREAD_TYPE_MAIN | ABTI_THREAD_TYPE_MAIN_SCHED)),
                    ABT_ERR_INV_THREAD);
    ABTI_CHECK_TRUE(ABTD_atomic_acquire_load_int(&p_thread->state) !=
                        ABT_THREAD_STATE_TERMINATED,
                    ABT_ERR_INV_THREAD);

    /* Find a pool */
    ABTI_pool *p_pool;
    int abt_errno;
    abt_errno =
        ABTI_sched_get_migration_pool(p_sched, p_thread->p_pool, &p_pool);
    ABTI_CHECK_ERROR(abt_errno);

    abt_errno = thread_migrate_to_pool(&p_local, p_thread, p_pool);
    ABTI_CHECK_ERROR(abt_errno);

    ABTI_pool_inc_num_migrations(p_pool);
    return ABT_SUCCESS;
#else
    ABTI_HANDLE_ERROR(ABT_ERR_MIGRATION_NA);
#endif
}

/**
 * @ingroup ULT
 * @brief   Migrate a thread to a specific pool.
 *
 * The actual migration occurs asynchronously with this function call.
 * In other words, this function may return immediately without the thread
 * being migrated. The migration request will be posted on the thread, such that
 * next time a scheduler picks it up, migration will happen.
 *
 * Note that users must be responsible for keeping the target pool available
 * during this function and, if this function returns ABT_SUCCESS, until the
 * migration process completes.
 *
 * @param[in] thread handle to the thread to migrate
 * @param[in] pool   handle to the pool to migrate the thread to
 * @return Error code
 * @retval ABT_SUCCESS              on success
 * @retval ABT_ERR_MIGRATION_TARGET the same pool is used
 */
int ABT_thread_migrate_to_pool(ABT_thread thread, ABT_pool pool)
{
#ifndef ABT_CONFIG_DISABLE_MIGRATION
    ABTI_local *p_local = ABTI_local_get_local();
    ABTI_thread *p_thread = ABTI_thread_get_ptr(thread);
    ABTI_CHECK_NULL_THREAD_PTR(p_thread);
    ABTI_pool *p_pool = ABTI_pool_get_ptr(pool);
    ABTI_CHECK_NULL_POOL_PTR(p_pool);

    int abt_errno = thread_migrate_to_pool(&p_local, p_thread, p_pool);
    ABTI_CHECK_ERROR(abt_errno);

    ABTI_pool_inc_num_migrations(p_pool);
    return ABT_SUCCESS;
#else
    ABTI_HANDLE_ERROR(ABT_ERR_MIGRATION_NA);
#endif
}

/**
 * @ingroup ULT
 * @brief   Request migration of the thread to an any available ES.
 *
 * ABT_thread_migrate requests migration of the thread but does not specify
 * the target ES. The target ES will be determined among available ESs by the
 * runtime. Other semantics of this routine are the same as those of
 * \c ABT_thread_migrate_to_xstream().
 *
 * Note that users must be responsible for keeping all the execution streams,
 * their main schedulers, and the associated pools available (i.e., not freed)
 * during this function and, if this function returns ABT_SUCCESS, until the
 * whole migration process completes.
 *
 * NOTE: This function may have some bugs.
 *
 * @param[in] thread  handle to the thread
 * @return Error code
 * @retval ABT_SUCCESS          on success
 * @retval ABT_ERR_MIGRATION_NA no other available ES for migration
 */
int ABT_thread_migrate(ABT_thread thread)
{
#ifndef ABT_CONFIG_DISABLE_MIGRATION
    /* TODO: fix the bug(s) */
    ABTI_local *p_local = ABTI_local_get_local();

    ABTI_thread *p_thread = ABTI_thread_get_ptr(thread);
    ABTI_CHECK_NULL_THREAD_PTR(p_thread);
    ABTI_CHECK_TRUE(gp_ABTI_global->num_xstreams != 1, ABT_ERR_MIGRATION_NA);

    /* Choose the destination xstream */
    /* FIXME: Currently, the target xstream is linearly chosen. We need a
     * better selection strategy. */
    /* TODO: handle better when no pool accepts migration */

    ABTI_xstream *p_xstream = gp_ABTI_global->p_xstream_head;
    while (p_xstream) {
        if (p_xstream != p_thread->p_last_xstream) {
            if (ABTD_atomic_acquire_load_int(&p_xstream->state) ==
                ABT_XSTREAM_STATE_RUNNING) {
                int abt_errno =
                    thread_migrate_to_xstream(&p_local, p_thread, p_xstream);
                if (abt_errno != ABT_ERR_INV_XSTREAM &&
                    abt_errno != ABT_ERR_MIGRATION_TARGET) {
                    ABTI_CHECK_ERROR(abt_errno);
                    break;
                }
            }
        }
        p_xstream = p_xstream->p_next;
    }
    return ABT_SUCCESS;
#else
    ABTI_HANDLE_ERROR(ABT_ERR_MIGRATION_NA);
#endif
}

/**
 * @ingroup ULT
 * @brief   Set the callback function.
 *
 * \c ABT_thread_set_callback sets the callback function to be used when the
 * ULT is migrated.
 *
 * @param[in] thread   handle to the target ULT
 * @param[in] cb_func  callback function pointer
 * @param[in] cb_arg   argument for the callback function
 * @return Error code
 * @retval ABT_SUCCESS on success
 */
int ABT_thread_set_callback(ABT_thread thread,
                            void (*cb_func)(ABT_thread thread, void *cb_arg),
                            void *cb_arg)
{
#ifndef ABT_CONFIG_DISABLE_MIGRATION
    ABTI_local *p_local = ABTI_local_get_local();
    ABTI_thread *p_thread = ABTI_thread_get_ptr(thread);
    ABTI_CHECK_NULL_THREAD_PTR(p_thread);

    ABTI_thread_mig_data *p_mig_data;
    int abt_errno = ABTI_thread_get_mig_data(p_local, p_thread, &p_mig_data);
    ABTI_CHECK_ERROR(abt_errno);

    p_mig_data->f_migration_cb = cb_func;
    p_mig_data->p_migration_cb_arg = cb_arg;
    return ABT_SUCCESS;
#else
    ABTI_HANDLE_ERROR(ABT_ERR_FEATURE_NA);
#endif
}

/**
 * @ingroup ULT
 * @brief   Set the ULT's migratability.
 *
 * \c ABT_thread_set_migratable sets the secondary ULT's migratability. This
 * routine cannot be used for the primary ULT. If \c flag is \c ABT_TRUE, the
 * target ULT becomes migratable. On the other hand, if \c flag is \c
 * ABT_FALSE, the target ULT becomes unmigratable.
 *
 * @param[in] thread  handle to the target ULT
 * @param[in] flag    migratability flag (<tt>ABT_TRUE</tt>: migratable,
 *                    <tt>ABT_FALSE</tt>: not)
 * @return Error code
 * @retval ABT_SUCCESS on success
 */
int ABT_thread_set_migratable(ABT_thread thread, ABT_bool flag)
{
#ifndef ABT_CONFIG_DISABLE_MIGRATION
    ABTI_thread *p_thread = ABTI_thread_get_ptr(thread);
    ABTI_CHECK_NULL_THREAD_PTR(p_thread);

    if (!(p_thread->type &
          (ABTI_THREAD_TYPE_MAIN | ABTI_THREAD_TYPE_MAIN_SCHED))) {
        if (flag) {
            p_thread->type |= ABTI_THREAD_TYPE_MIGRATABLE;
        } else {
            p_thread->type &= ~ABTI_THREAD_TYPE_MIGRATABLE;
        }
    }
    return ABT_SUCCESS;
#else
    ABTI_HANDLE_ERROR(ABT_ERR_FEATURE_NA);
#endif
}

/**
 * @ingroup ULT
 * @brief   Get the ULT's migratability.
 *
 * \c ABT_thread_is_migratable returns the ULT's migratability through
 * \c flag. If the target ULT is migratable, \c ABT_TRUE is returned to
 * \c flag. Otherwise, \c flag is set to \c ABT_FALSE.
 *
 * @param[in]  thread  handle to the target ULT
 * @param[out] flag    migratability flag (<tt>ABT_TRUE</tt>: migratable,
 *                     <tt>ABT_FALSE</tt>: not)
 * @return Error code
 * @retval ABT_SUCCESS on success
 */
int ABT_thread_is_migratable(ABT_thread thread, ABT_bool *flag)
{
#ifndef ABT_CONFIG_DISABLE_MIGRATION
    ABTI_thread *p_thread = ABTI_thread_get_ptr(thread);
    ABTI_CHECK_NULL_THREAD_PTR(p_thread);

    *flag =
        (p_thread->type & ABTI_THREAD_TYPE_MIGRATABLE) ? ABT_TRUE : ABT_FALSE;
    return ABT_SUCCESS;
#else
    ABTI_HANDLE_ERROR(ABT_ERR_FEATURE_NA);
#endif
}

/**
 * @ingroup ULT
 * @brief   Check if the target ULT is the primary ULT.
 *
 * \c ABT_thread_is_primary confirms whether the target ULT, \c thread,
 * is the primary ULT and returns the result through \c flag.
 * If \c thread is a handle to the primary ULT, \c flag is set to \c ABT_TRUE.
 * Otherwise, \c flag is set to \c ABT_FALSE.
 *
 * @param[in]  thread  handle to the target ULT
 * @param[out] flag    result (<tt>ABT_TRUE</tt>: primary ULT,
 *                     <tt>ABT_FALSE</tt>: not)
 * @return Error code
 * @retval ABT_SUCCESS        on success
 * @retval ABT_ERR_INV_THREAD invalid ULT handle
 */
int ABT_thread_is_primary(ABT_thread thread, ABT_bool *flag)
{
    ABTI_thread *p_thread = ABTI_thread_get_ptr(thread);
    ABTI_CHECK_NULL_THREAD_PTR(p_thread);

    *flag = (p_thread->type & ABTI_THREAD_TYPE_MAIN) ? ABT_TRUE : ABT_FALSE;
    return ABT_SUCCESS;
}

/**
 * @ingroup ULT
 * @brief   Check if the target ULT is unnamed
 *
 * \c ABT_thread_is_unnamed() returns whether the target ULT, \c thread, is
 * unnamed or not.  Note that a handle of an unnamed ULT can be obtained by, for
 * example, running \c ABT_thread_self() on an unnamed ULT.
 *
 * @param[in]  thread  handle to the target ULT
 * @param[out] flag    result (<tt>ABT_TRUE</tt> if unnamed)
 *
 * @return Error code
 * @retval ABT_SUCCESS  on success
 */
int ABT_thread_is_unnamed(ABT_thread thread, ABT_bool *flag)
{
    ABTI_thread *p_thread = ABTI_thread_get_ptr(thread);
    ABTI_CHECK_NULL_THREAD_PTR(p_thread);

    *flag = (p_thread->type & ABTI_THREAD_TYPE_NAMED) ? ABT_FALSE : ABT_TRUE;
    return ABT_SUCCESS;
}

/**
 * @ingroup ULT
 * @brief   Compare two ULT handles for equality.
 *
 * \c ABT_thread_equal() compares two ULT handles for equality. If two handles
 * are associated with the same ULT object, \c result will be set to
 * \c ABT_TRUE. Otherwise, \c result will be set to \c ABT_FALSE.
 *
 * @param[in]  thread1  handle to the ULT 1
 * @param[in]  thread2  handle to the ULT 2
 * @param[out] result   comparison result (<tt>ABT_TRUE</tt>: same,
 *                      <tt>ABT_FALSE</tt>: not same)
 * @return Error code
 * @retval ABT_SUCCESS on success
 */
int ABT_thread_equal(ABT_thread thread1, ABT_thread thread2, ABT_bool *result)
{
    ABTI_thread *p_thread1 = ABTI_thread_get_ptr(thread1);
    ABTI_thread *p_thread2 = ABTI_thread_get_ptr(thread2);
    *result = (p_thread1 == p_thread2) ? ABT_TRUE : ABT_FALSE;
    return ABT_SUCCESS;
}

/**
 * @ingroup ULT
 * @brief   Get the ULT's stack size.
 *
 * \c ABT_thread_get_stacksize() returns the stack size of \c thread in bytes.
 *
 * @param[in]  thread     handle to the target thread
 * @param[out] stacksize  stack size in bytes
 * @return Error code
 * @retval ABT_SUCCESS on success
 */
int ABT_thread_get_stacksize(ABT_thread thread, size_t *stacksize)
{
    ABTI_thread *p_thread = ABTI_thread_get_ptr(thread);
    ABTI_CHECK_NULL_THREAD_PTR(p_thread);
    ABTI_ythread *p_ythread;
    ABTI_CHECK_YIELDABLE(p_thread, &p_ythread, ABT_ERR_INV_THREAD);

    *stacksize = p_ythread->stacksize;
    return ABT_SUCCESS;
}

/**
 * @ingroup ULT
 * @brief   Get the ULT's id
 *
 * \c ABT_thread_get_id() returns the id of \c a thread.
 *
 * @param[in]  thread     handle to the target thread
 * @param[out] thread_id  thread id
 * @return Error code
 * @retval ABT_SUCCESS on success
 */
int ABT_thread_get_id(ABT_thread thread, ABT_unit_id *thread_id)
{
    ABTI_thread *p_thread = ABTI_thread_get_ptr(thread);
    ABTI_CHECK_NULL_THREAD_PTR(p_thread);

    *thread_id = ABTI_thread_get_id(p_thread);
    return ABT_SUCCESS;
}

/**
 * @ingroup ULT
 * @brief   Set the argument for the ULT function
 *
 * \c ABT_thread_set_arg() sets the argument for the ULT function.
 *
 * @param[in] thread  handle to the target ULT
 * @param[in] arg     argument for the ULT function
 * @return Error code
 * @retval ABT_SUCCESS on success
 */
int ABT_thread_set_arg(ABT_thread thread, void *arg)
{
    ABTI_thread *p_thread = ABTI_thread_get_ptr(thread);
    ABTI_CHECK_NULL_THREAD_PTR(p_thread);

    p_thread->p_arg = arg;
    return ABT_SUCCESS;
}

/**
 * @ingroup ULT
 * @brief   Retrieve the argument for the ULT function
 *
 * \c ABT_thread_get_arg() returns the argument for the ULT function, which was
 * passed to \c ABT_thread_create() when the target ULT \c thread was created
 * or was set by \c ABT_thread_set_arg().
 *
 * @param[in]  thread  handle to the target ULT
 * @param[out] arg     argument for the ULT function
 * @return Error code
 * @retval ABT_SUCCESS on success
 */
int ABT_thread_get_arg(ABT_thread thread, void **arg)
{
    ABTI_thread *p_thread = ABTI_thread_get_ptr(thread);
    ABTI_CHECK_NULL_THREAD_PTR(p_thread);

    *arg = p_thread->p_arg;
    return ABT_SUCCESS;
}

/**
 * @ingroup ULT
 * @brief  Set the ULT-specific value associated with the key
 *
 * \c ABT_thread_set_specific() associates a value, \c value, with a work
 * unit-specific data key, \c key.  The target work unit is \c thread.
 *
 * @param[in] thread  handle to the target ULT
 * @param[in] key     handle to the target key
 * @param[in] value   value for the key
 * @return Error code
 * @retval ABT_SUCCESS on success
 */
int ABT_thread_set_specific(ABT_thread thread, ABT_key key, void *value)
{
    ABTI_local *p_local = ABTI_local_get_local();

    ABTI_thread *p_thread = ABTI_thread_get_ptr(thread);
    ABTI_CHECK_NULL_THREAD_PTR(p_thread);

    ABTI_key *p_key = ABTI_key_get_ptr(key);
    ABTI_CHECK_NULL_KEY_PTR(p_key);

    /* Set the value. */
    int abt_errno =
        ABTI_ktable_set(p_local, &p_thread->p_keytable, p_key, value);
    ABTI_CHECK_ERROR(abt_errno);
    return ABT_SUCCESS;
}

/**
 * @ingroup ULT
 * @brief   Get the ULT-specific value associated with the key
 *
 * \c ABT_thread_get_specific() returns the value associated with a target work
 * unit-specific data key, \c key, through \c value.  The target work unit is
 * \c thread.  If \c thread has never set a value for the key, this routine
 * returns \c NULL to \c value.
 *
 * @param[in]  thread  handle to the target ULT
 * @param[in]  key     handle to the target key
 * @param[out] value   value for the key
 * @return Error code
 * @retval ABT_SUCCESS on success
 */
int ABT_thread_get_specific(ABT_thread thread, ABT_key key, void **value)
{
    ABTI_thread *p_thread = ABTI_thread_get_ptr(thread);
    ABTI_CHECK_NULL_THREAD_PTR(p_thread);

    ABTI_key *p_key = ABTI_key_get_ptr(key);
    ABTI_CHECK_NULL_KEY_PTR(p_key);

    /* Get the value. */
    *value = ABTI_ktable_get(&p_thread->p_keytable, p_key);
    return ABT_SUCCESS;
}

/**
 * @ingroup ULT
 * @brief   Get attributes of the target ULT
 *
 * \c ABT_thread_get_attr() returns the attributes of the ULT \c thread to
 * \c attr.  \c attr contains actual attribute values that may be different
 * from those used to create \c thread.  Since this routine allocates an
 * attribute object, when \c attr is no longer used it should be destroyed
 * using \c ABT_thread_attr_free().
 *
 * @param[in]  thread  handle to the target ULT
 * @param[out] attr    ULT attributes
 * @return Error code
 * @retval ABT_SUCCESS on success
 */
int ABT_thread_get_attr(ABT_thread thread, ABT_thread_attr *attr)
{
    ABTI_thread *p_thread = ABTI_thread_get_ptr(thread);
    ABTI_CHECK_NULL_THREAD_PTR(p_thread);

    ABTI_thread_attr thread_attr, *p_attr;
    ABTI_ythread *p_ythread = ABTI_thread_get_ythread_or_null(p_thread);
    if (p_ythread) {
        thread_attr.p_stack = p_ythread->p_stack;
        thread_attr.stacksize = p_ythread->stacksize;
    } else {
        thread_attr.p_stack = NULL;
        thread_attr.stacksize = 0;
    }
    thread_attr.thread_type = p_thread->type;
#ifndef ABT_CONFIG_DISABLE_MIGRATION
    thread_attr.migratable =
        (p_thread->type & ABTI_THREAD_TYPE_MIGRATABLE) ? ABT_TRUE : ABT_FALSE;
    ABTI_thread_mig_data *p_mig_data =
        (ABTI_thread_mig_data *)ABTI_ktable_get(&p_thread->p_keytable,
                                                &g_thread_mig_data_key);
    if (p_mig_data) {
        thread_attr.f_cb = p_mig_data->f_migration_cb;
        thread_attr.p_cb_arg = p_mig_data->p_migration_cb_arg;
    } else {
        thread_attr.f_cb = NULL;
        thread_attr.p_cb_arg = NULL;
    }
#endif
    int abt_errno = ABTI_thread_attr_dup(&thread_attr, &p_attr);
    ABTI_CHECK_ERROR(abt_errno);

    *attr = ABTI_thread_attr_get_handle(p_attr);
    return ABT_SUCCESS;
}

/*****************************************************************************/
/* Private APIs                                                              */
/*****************************************************************************/

void ABTI_thread_revive(ABTI_local *p_local, ABTI_pool *p_pool,
                        void (*thread_func)(void *), void *arg,
                        ABTI_thread *p_thread)
{
    ABTI_ASSERT(ABTD_atomic_relaxed_load_int(&p_thread->state) ==
                ABT_THREAD_STATE_TERMINATED);
    p_thread->f_thread = thread_func;
    p_thread->p_arg = arg;

    ABTD_atomic_relaxed_store_int(&p_thread->state, ABT_THREAD_STATE_READY);
    ABTD_atomic_relaxed_store_uint32(&p_thread->request, 0);
    p_thread->p_last_xstream = NULL;
    p_thread->p_parent = NULL;

    ABTI_ythread *p_ythread = ABTI_thread_get_ythread_or_null(p_thread);
    if (p_thread->p_pool != p_pool) {
        /* Free the unit for the old pool */
        p_thread->p_pool->u_free(&p_thread->unit);

        /* Set the new pool */
        p_thread->p_pool = p_pool;

        /* Create a wrapper unit */
        if (p_ythread) {
            ABT_thread h_thread = ABTI_ythread_get_handle(p_ythread);
            p_thread->unit = p_pool->u_create_from_thread(h_thread);
        } else {
            ABT_task task = ABTI_thread_get_handle(p_thread);
            p_thread->unit = p_pool->u_create_from_task(task);
        }
    }

    if (p_ythread) {
        /* Create a ULT context */
        size_t stacksize = p_ythread->stacksize;
        ABTD_ythread_context_create(NULL, stacksize, p_ythread->p_stack,
                                    &p_ythread->ctx);
    }

    /* Invoke a thread revive event. */
    ABTI_tool_event_thread_revive(p_local, p_thread,
                                  ABTI_local_get_xstream_or_null(p_local)
                                      ? ABTI_local_get_xstream(p_local)
                                            ->p_thread
                                      : NULL,
                                  p_pool);

    LOG_DEBUG("[U%" PRIu64 "] revived\n", ABTI_thread_get_id(p_thread));

    /* Add this thread to the pool */
    ABTI_pool_push(p_pool, p_thread->unit);
}

ABTU_ret_err int ABTI_ythread_create_main(ABTI_local *p_local,
                                          ABTI_xstream *p_xstream,
                                          ABTI_ythread **p_ythread)
{
    ABTI_thread_attr attr;
    ABTI_pool *p_pool;

    /* Get the first pool of ES */
    p_pool = ABTI_pool_get_ptr(p_xstream->p_main_sched->pools[0]);

    /* Allocate a ULT object */

    /* TODO: Need to set the actual stack address and size for the main ULT */
    ABTI_thread_attr_init(&attr, NULL, 0, ABTI_THREAD_TYPE_MEM_MEMPOOL_DESC,
                          ABT_FALSE);

    /* Although this main ULT is running now, we add this main ULT to the pool
     * so that the scheduler can schedule the main ULT when the main ULT is
     * context switched to the scheduler for the first time. */
    ABT_bool push_pool = ABT_TRUE;
    int abt_errno =
        ythread_create(p_local, p_pool, NULL, NULL, &attr,
                       ABTI_THREAD_TYPE_YIELDABLE | ABTI_THREAD_TYPE_MAIN, NULL,
                       push_pool, p_ythread);
    ABTI_CHECK_ERROR(abt_errno);
    return ABT_SUCCESS;
}

ABTU_ret_err int ABTI_ythread_create_root(ABTI_local *p_local,
                                          ABTI_xstream *p_xstream,
                                          ABTI_ythread **pp_root_ythread)
{
    ABTI_thread_attr attr;
    /* Create a ULT context */
    if (p_xstream->type == ABTI_XSTREAM_TYPE_PRIMARY) {
        /* Create a thread with its stack */
        ABTI_thread_attr_init(&attr, NULL, gp_ABTI_global->sched_stacksize,
                              ABTI_THREAD_TYPE_MEM_MALLOC_DESC_STACK,
                              ABT_FALSE);
    } else {
        /* For secondary ESs, the stack of an OS thread is used. */
        ABTI_thread_attr_init(&attr, NULL, 0, ABTI_THREAD_TYPE_MEM_MEMPOOL_DESC,
                              ABT_FALSE);
    }
    ABTI_ythread *p_root_ythread;
    int abt_errno =
        ythread_create(p_local, NULL, thread_root_func, NULL, &attr,
                       ABTI_THREAD_TYPE_YIELDABLE | ABTI_THREAD_TYPE_ROOT, NULL,
                       ABT_FALSE, &p_root_ythread);
    ABTI_CHECK_ERROR(abt_errno);
    *pp_root_ythread = p_root_ythread;
    return ABT_SUCCESS;
}

ABTU_ret_err int ABTI_ythread_create_main_sched(ABTI_local *p_local,
                                                ABTI_xstream *p_xstream,
                                                ABTI_sched *p_sched)
{
    ABTI_thread_attr attr;

    /* Allocate a ULT object and its stack */
    ABTI_thread_attr_init(&attr, NULL, gp_ABTI_global->sched_stacksize,
                          ABTI_THREAD_TYPE_MEM_MALLOC_DESC_STACK, ABT_FALSE);
    int abt_errno =
        ythread_create(p_local, p_xstream->p_root_pool, thread_main_sched_func,
                       NULL, &attr,
                       ABTI_THREAD_TYPE_YIELDABLE |
                           ABTI_THREAD_TYPE_MAIN_SCHED | ABTI_THREAD_TYPE_NAMED,
                       p_sched, ABT_TRUE, &p_sched->p_ythread);
    ABTI_CHECK_ERROR(abt_errno);
    return ABT_SUCCESS;
}

/* This routine is to create a ULT for the scheduler. */
ABTU_ret_err int ABTI_ythread_create_sched(ABTI_local *p_local,
                                           ABTI_pool *p_pool,
                                           ABTI_sched *p_sched)
{
    ABTI_thread_attr attr;

    /* Allocate a ULT object and its stack */
    ABTI_thread_attr_init(&attr, NULL, gp_ABTI_global->sched_stacksize,
                          ABTI_THREAD_TYPE_MEM_MALLOC_DESC_STACK, ABT_FALSE);
    int abt_errno =
        ythread_create(p_local, p_pool, (void (*)(void *))p_sched->run,
                       (void *)ABTI_sched_get_handle(p_sched), &attr,
                       ABTI_THREAD_TYPE_YIELDABLE, p_sched, ABT_TRUE,
                       &p_sched->p_ythread);
    ABTI_CHECK_ERROR(abt_errno);
    return ABT_SUCCESS;
}

void ABTI_thread_join(ABTI_local **pp_local, ABTI_thread *p_thread)
{
    thread_join(pp_local, p_thread);
}

void ABTI_thread_free(ABTI_local *p_local, ABTI_thread *p_thread)
{
    LOG_DEBUG("[U%" PRIu64 ":E%d] freed\n", ABTI_thread_get_id(p_thread),
              ABTI_local_get_xstream_or_null(p_local)
                  ? ABTI_local_get_xstream(p_local)->rank
                  : -1);
    thread_free(p_local, p_thread, ABT_TRUE);
}

void ABTI_ythread_free_main(ABTI_local *p_local, ABTI_ythread *p_ythread)
{
    ABTI_thread *p_thread = &p_ythread->thread;
    LOG_DEBUG("[U%" PRIu64 ":E%d] main ULT freed\n",
              ABTI_thread_get_id(p_thread), p_thread->p_last_xstream->rank);
    thread_free(p_local, p_thread, ABT_FALSE);
}

void ABTI_ythread_free_root(ABTI_local *p_local, ABTI_ythread *p_ythread)
{
    thread_free(p_local, &p_ythread->thread, ABT_FALSE);
}

ABTU_noreturn void ABTI_ythread_exit(ABTI_xstream *p_local_xstream,
                                     ABTI_ythread *p_ythread)
{
    /* Set the exit request */
    ABTI_thread_set_request(&p_ythread->thread, ABTI_THREAD_REQ_TERMINATE);

    /* Terminate this ULT */
    ABTD_ythread_exit(p_local_xstream, p_ythread);
    ABTU_unreachable();
}

ABTU_ret_err int ABTI_thread_get_mig_data(ABTI_local *p_local,
                                          ABTI_thread *p_thread,
                                          ABTI_thread_mig_data **pp_mig_data)
{
    ABTI_thread_mig_data *p_mig_data =
        (ABTI_thread_mig_data *)ABTI_ktable_get(&p_thread->p_keytable,
                                                &g_thread_mig_data_key);
    if (!p_mig_data) {
        int abt_errno;
        abt_errno =
            ABTU_calloc(1, sizeof(ABTI_thread_mig_data), (void **)&p_mig_data);
        ABTI_CHECK_ERROR(abt_errno);
        abt_errno = ABTI_ktable_set(p_local, &p_thread->p_keytable,
                                    &g_thread_mig_data_key, (void *)p_mig_data);
        if (ABTI_IS_ERROR_CHECK_ENABLED && abt_errno != ABT_SUCCESS) {
            /* Failed to add p_mig_data to p_thread's keytable. */
            ABTU_free(p_mig_data);
            return abt_errno;
        }
    }
    *pp_mig_data = p_mig_data;
    return ABT_SUCCESS;
}

void ABTI_thread_print(ABTI_thread *p_thread, FILE *p_os, int indent)
{
    if (p_thread == NULL) {
        fprintf(p_os, "%*s== NULL thread ==\n", indent, "");
    } else {
        ABTI_xstream *p_xstream = p_thread->p_last_xstream;
        int xstream_rank = p_xstream ? p_xstream->rank : 0;
        const char *type, *yieldable, *state;

        if (p_thread->type & ABTI_THREAD_TYPE_MAIN) {
            type = "MAIN";
        } else if (p_thread->type & ABTI_THREAD_TYPE_MAIN_SCHED) {
            type = "MAIN_SCHED";
        } else {
            type = "USER";
        }
        if (p_thread->type & ABTI_THREAD_TYPE_YIELDABLE) {
            yieldable = "yes";
        } else {
            yieldable = "no";
        }
        switch (ABTD_atomic_acquire_load_int(&p_thread->state)) {
            case ABT_THREAD_STATE_READY:
                state = "READY";
                break;
            case ABT_THREAD_STATE_RUNNING:
                state = "RUNNING";
                break;
            case ABT_THREAD_STATE_BLOCKED:
                state = "BLOCKED";
                break;
            case ABT_THREAD_STATE_TERMINATED:
                state = "TERMINATED";
                break;
            default:
                state = "UNKNOWN";
                break;
        }

        fprintf(p_os,
                "%*s== Thread (%p) ==\n"
                "%*sid        : %" PRIu64 "\n"
                "%*stype      : %s\n"
                "%*syieldable : %s\n"
                "%*sstate     : %s\n"
                "%*slast_ES   : %p (%d)\n"
                "%*sp_arg     : %p\n"
                "%*spool      : %p\n"
                "%*srequest   : 0x%x\n"
                "%*skeytable  : %p\n",
                indent, "", (void *)p_thread, indent, "",
                ABTI_thread_get_id(p_thread), indent, "", type, indent, "",
                yieldable, indent, "", state, indent, "", (void *)p_xstream,
                xstream_rank, indent, "", p_thread->p_arg, indent, "",
                (void *)p_thread->p_pool, indent, "",
                ABTD_atomic_acquire_load_uint32(&p_thread->request), indent, "",
                ABTD_atomic_acquire_load_ptr(&p_thread->p_keytable));
    }
    fflush(p_os);
}

static ABTD_atomic_uint64 g_thread_id =
    ABTD_ATOMIC_UINT64_STATIC_INITIALIZER(0);
void ABTI_thread_reset_id(void)
{
    ABTD_atomic_release_store_uint64(&g_thread_id, 0);
}

ABT_unit_id ABTI_thread_get_id(ABTI_thread *p_thread)
{
    if (p_thread == NULL)
        return ABTI_THREAD_INIT_ID;

    if (p_thread->id == ABTI_THREAD_INIT_ID) {
        p_thread->id = thread_get_new_id();
    }
    return p_thread->id;
}

/*****************************************************************************/
/* Internal static functions                                                 */
/*****************************************************************************/

ABTU_ret_err static inline int
ythread_create(ABTI_local *p_local, ABTI_pool *p_pool,
               void (*thread_func)(void *), void *arg, ABTI_thread_attr *p_attr,
               ABTI_thread_type thread_type, ABTI_sched *p_sched,
               ABT_bool push_pool, ABTI_ythread **pp_newthread)
{
    int abt_errno;
    ABTI_ythread *p_newthread;
    ABT_thread h_newthread;
    ABTI_ktable *p_keytable = NULL;

    /* Allocate a ULT object and its stack, then create a thread context. */
    if (!p_attr) {
        abt_errno = ABTI_mem_alloc_ythread_default(p_local, &p_newthread);
        ABTI_CHECK_ERROR(abt_errno);
#ifndef ABT_CONFIG_DISABLE_MIGRATION
        thread_type |= ABTI_THREAD_TYPE_MIGRATABLE;
#endif
    } else {
        ABTI_thread_type attr_type = p_attr->thread_type;
        if (attr_type & ABTI_THREAD_TYPE_MEM_MEMPOOL_DESC_STACK) {
#ifdef ABT_CONFIG_USE_MEM_POOL
            abt_errno =
                ABTI_mem_alloc_ythread_mempool_desc_stack(p_local, p_attr,
                                                          &p_newthread);
            ABTI_CHECK_ERROR(abt_errno);
#else
            abt_errno =
                ABTI_mem_alloc_ythread_malloc_desc_stack(p_attr, &p_newthread);
#endif
            ABTI_CHECK_ERROR(abt_errno);
        } else if (attr_type & ABTI_THREAD_TYPE_MEM_MALLOC_DESC_STACK) {
            abt_errno =
                ABTI_mem_alloc_ythread_malloc_desc_stack(p_attr, &p_newthread);
            ABTI_CHECK_ERROR(abt_errno);
        } else {
            ABTI_ASSERT(attr_type & (ABTI_THREAD_TYPE_MEM_MEMPOOL_DESC |
                                     ABTI_THREAD_TYPE_MEM_MALLOC_DESC));
            /* Let's try to use mempool first since it performs better. */
            abt_errno = ABTI_mem_alloc_ythread_mempool_desc(p_local, p_attr,
                                                            &p_newthread);
            ABTI_CHECK_ERROR(abt_errno);
        }
#ifndef ABT_CONFIG_DISABLE_MIGRATION
        thread_type |= p_attr->migratable ? ABTI_THREAD_TYPE_MIGRATABLE : 0;
        if (ABTU_unlikely(p_attr->f_cb)) {
            ABTI_thread_mig_data *p_mig_data;
            abt_errno = ABTU_calloc(1, sizeof(ABTI_thread_mig_data),
                                    (void **)&p_mig_data);
            if (ABTI_IS_ERROR_CHECK_ENABLED &&
                ABTU_unlikely(abt_errno != ABT_SUCCESS)) {
                ABTI_mem_free_thread(p_local, &p_newthread->thread);
                return abt_errno;
            }
            p_mig_data->f_migration_cb = p_attr->f_cb;
            p_mig_data->p_migration_cb_arg = p_attr->p_cb_arg;
            abt_errno = ABTI_ktable_set_unsafe(p_local, &p_keytable,
                                               &g_thread_mig_data_key,
                                               (void *)p_mig_data);
            if (ABTI_IS_ERROR_CHECK_ENABLED &&
                ABTU_unlikely(abt_errno != ABT_SUCCESS)) {
                if (p_keytable)
                    ABTI_ktable_free(p_local, p_keytable);
                ABTU_free(p_mig_data);
                ABTI_mem_free_thread(p_local, &p_newthread->thread);
                return abt_errno;
            }
        }
#endif
    }

    if (thread_type & (ABTI_THREAD_TYPE_MAIN | ABTI_THREAD_TYPE_ROOT)) {
        if (p_newthread->p_stack == NULL) {
            /* We don't need to initialize the context if a thread will run on
             * OS-level threads. Invalidate the context here. */
            ABTD_ythread_context_invalidate(&p_newthread->ctx);
        } else {
            /* Create the context.  This thread is special, so dynamic promotion
             * is not supported. */
            size_t stack_size = p_newthread->stacksize;
            void *p_stack = p_newthread->p_stack;
            ABTD_ythread_context_create(NULL, stack_size, p_stack,
                                        &p_newthread->ctx);
        }
    } else {
#if ABT_CONFIG_THREAD_TYPE != ABT_THREAD_TYPE_DYNAMIC_PROMOTION
        size_t stack_size = p_newthread->stacksize;
        void *p_stack = p_newthread->p_stack;
        ABTD_ythread_context_create(NULL, stack_size, p_stack,
                                    &p_newthread->ctx);
#else
        /* The context is not fully created now. */
        ABTD_ythread_context_init(NULL, &p_newthread->ctx);
#endif
    }
    p_newthread->thread.f_thread = thread_func;
    p_newthread->thread.p_arg = arg;

    ABTD_atomic_release_store_int(&p_newthread->thread.state,
                                  ABT_THREAD_STATE_READY);
    ABTD_atomic_release_store_uint32(&p_newthread->thread.request, 0);
    p_newthread->thread.p_last_xstream = NULL;
    p_newthread->thread.p_parent = NULL;
    p_newthread->thread.p_pool = p_pool;
    p_newthread->thread.type |= thread_type;
    p_newthread->thread.id = ABTI_THREAD_INIT_ID;
    if (p_sched && !(thread_type &
                     (ABTI_THREAD_TYPE_MAIN | ABTI_THREAD_TYPE_MAIN_SCHED))) {
        /* Set a destructor for p_sched. */
        abt_errno = ABTI_ktable_set_unsafe(p_local, &p_keytable,
                                           &g_thread_sched_key, p_sched);
        if (ABTI_IS_ERROR_CHECK_ENABLED &&
            ABTU_unlikely(abt_errno != ABT_SUCCESS)) {
            if (p_keytable)
                ABTI_ktable_free(p_local, p_keytable);
            ABTI_mem_free_thread(p_local, &p_newthread->thread);
            return abt_errno;
        }
    }
    ABTD_atomic_relaxed_store_ptr(&p_newthread->thread.p_keytable, p_keytable);

#ifdef ABT_CONFIG_USE_DEBUG_LOG
    ABT_unit_id thread_id = ABTI_thread_get_id(&p_newthread->thread);
    if (thread_type & ABTI_THREAD_TYPE_MAIN) {
        LOG_DEBUG("[U%" PRIu64 "] main ULT created\n", thread_id);
    } else if (thread_type & ABTI_THREAD_TYPE_MAIN_SCHED) {
        LOG_DEBUG("[U%" PRIu64 "] main sched ULT created\n", thread_id);
    } else {
        LOG_DEBUG("[U%" PRIu64 "] created\n", thread_id);
    }
#endif

    /* Invoke a thread creation event. */
    ABTI_tool_event_thread_create(p_local, &p_newthread->thread,
                                  ABTI_local_get_xstream_or_null(p_local)
                                      ? ABTI_local_get_xstream(p_local)
                                            ->p_thread
                                      : NULL,
                                  push_pool ? p_pool : NULL);

    /* Create a wrapper unit */
    h_newthread = ABTI_ythread_get_handle(p_newthread);
    if (push_pool) {
        p_newthread->thread.unit = p_pool->u_create_from_thread(h_newthread);
        /* Add this thread to the pool */
        ABTI_pool_push(p_pool, p_newthread->thread.unit);
    } else {
        p_newthread->thread.unit = ABT_UNIT_NULL;
    }

    /* Return value */
    *pp_newthread = p_newthread;
    return ABT_SUCCESS;
}

#ifndef ABT_CONFIG_DISABLE_MIGRATION
ABTU_ret_err static int thread_migrate_to_pool(ABTI_local **pp_local,
                                               ABTI_thread *p_thread,
                                               ABTI_pool *p_pool)
{
    /* checking for cases when migration is not allowed */
    ABTI_CHECK_TRUE(!(p_thread->type &
                      (ABTI_THREAD_TYPE_MAIN | ABTI_THREAD_TYPE_MAIN_SCHED)),
                    ABT_ERR_INV_THREAD);
    ABTI_CHECK_TRUE(ABTD_atomic_acquire_load_int(&p_thread->state) !=
                        ABT_THREAD_STATE_TERMINATED,
                    ABT_ERR_INV_THREAD);

    /* checking for migration to the same pool */
    ABTI_CHECK_TRUE(p_thread->p_pool != p_pool, ABT_ERR_MIGRATION_TARGET);

    /* adding request to the thread.  p_migration_pool must be updated before
     * setting the request since the target thread would read p_migration_pool
     * after ABTI_THREAD_REQ_MIGRATE.  The update must be "atomic" (but does not
     * require acq-rel) since two threads can update the pointer value
     * simultaneously. */
    ABTI_thread_mig_data *p_mig_data;
    int abt_errno = ABTI_thread_get_mig_data(*pp_local, p_thread, &p_mig_data);
    ABTI_CHECK_ERROR(abt_errno);
    ABTD_atomic_relaxed_store_ptr(&p_mig_data->p_migration_pool,
                                  (void *)p_pool);

    ABTI_thread_set_request(p_thread, ABTI_THREAD_REQ_MIGRATE);

    /* yielding if it is the same thread */
    ABTI_xstream *p_local_xstream = ABTI_local_get_xstream_or_null(*pp_local);
    if ((!ABTI_IS_EXT_THREAD_ENABLED || p_local_xstream) &&
        p_thread == p_local_xstream->p_thread) {
        ABTI_ythread *p_ythread = ABTI_thread_get_ythread_or_null(p_thread);
        if (p_ythread) {
            ABTI_ythread_yield(&p_local_xstream, p_ythread,
                               ABT_SYNC_EVENT_TYPE_OTHER, NULL);
            *pp_local = ABTI_xstream_get_local(p_local_xstream);
        }
    }
    return ABT_SUCCESS;
}
#endif

static inline void thread_free(ABTI_local *p_local, ABTI_thread *p_thread,
                               ABT_bool free_unit)
{
    /* Invoke a thread freeing event. */
    ABTI_tool_event_thread_free(p_local, p_thread,
                                ABTI_local_get_xstream_or_null(p_local)
                                    ? ABTI_local_get_xstream(p_local)->p_thread
                                    : NULL);

    /* Free the unit */
    if (free_unit) {
        p_thread->p_pool->u_free(&p_thread->unit);
    }

    /* Free the key-value table */
    ABTI_ktable *p_ktable = ABTD_atomic_acquire_load_ptr(&p_thread->p_keytable);
    /* No parallel access to TLS is allowed. */
    ABTI_ASSERT(p_ktable != ABTI_KTABLE_LOCKED);
    if (p_ktable) {
        ABTI_ktable_free(p_local, p_ktable);
    }

    /* Free ABTI_thread (stack will also be freed) */
    ABTI_mem_free_thread(p_local, p_thread);
}

static void thread_key_destructor_stackable_sched(void *p_value)
{
    /* This destructor should be called in ABTI_ythread_free(), so it should not
     * free the thread again.  */
    ABTI_sched *p_sched = (ABTI_sched *)p_value;
    p_sched->used = ABTI_SCHED_NOT_USED;
    if (p_sched->automatic == ABT_TRUE) {
        p_sched->p_ythread = NULL;
        ABTI_sched_free(ABTI_local_get_local_uninlined(), p_sched, ABT_FALSE);
    }
}

static void thread_key_destructor_migration(void *p_value)
{
    ABTI_thread_mig_data *p_mig_data = (ABTI_thread_mig_data *)p_value;
    ABTU_free(p_mig_data);
}

static void thread_join_busywait(ABTI_thread *p_thread)
{
    while (ABTD_atomic_acquire_load_int(&p_thread->state) !=
           ABT_THREAD_STATE_TERMINATED) {
        ABTD_atomic_pause();
    }
    ABTI_tool_event_thread_join(NULL, p_thread, NULL);
}

static void thread_join_yield_ythread(ABTI_xstream **pp_local_xstream,
                                      ABTI_ythread *p_self,
                                      ABTI_ythread *p_ythread)
{
    while (ABTD_atomic_acquire_load_int(&p_ythread->thread.state) !=
           ABT_THREAD_STATE_TERMINATED) {
        ABTI_ythread_yield(pp_local_xstream, p_self,
                           ABT_SYNC_EVENT_TYPE_THREAD_JOIN, (void *)p_ythread);
    }
    ABTI_tool_event_thread_join(ABTI_xstream_get_local(*pp_local_xstream),
                                &p_ythread->thread, &p_self->thread);
}

static void thread_join_yield_task(ABTI_xstream **pp_local_xstream,
                                   ABTI_ythread *p_self, ABTI_thread *p_task)
{
    while (ABTD_atomic_acquire_load_int(&p_task->state) !=
           ABT_THREAD_STATE_TERMINATED) {
        ABTI_ythread_yield(pp_local_xstream, p_self,
                           ABT_SYNC_EVENT_TYPE_TASK_JOIN, (void *)p_task);
    }
    ABTI_tool_event_thread_join(ABTI_xstream_get_local(*pp_local_xstream),
                                p_task, &p_self->thread);
}

static inline void thread_join(ABTI_local **pp_local, ABTI_thread *p_thread)
{
    if (ABTD_atomic_acquire_load_int(&p_thread->state) ==
        ABT_THREAD_STATE_TERMINATED) {
        ABTI_tool_event_thread_join(*pp_local, p_thread,
                                    ABTI_local_get_xstream_or_null(*pp_local)
                                        ? ABTI_local_get_xstream(*pp_local)
                                              ->p_thread
                                        : NULL);
        return;
    }
    /* The main ULT cannot be joined. */
    ABTI_ASSERT(!(p_thread->type & ABTI_THREAD_TYPE_MAIN));

    ABTI_xstream *p_local_xstream = ABTI_local_get_xstream_or_null(*pp_local);
    if (ABTI_IS_EXT_THREAD_ENABLED && !p_local_xstream) {
        thread_join_busywait(p_thread);
        return;
    }

    ABTI_thread *p_self_thread = p_local_xstream->p_thread;

    ABTI_ythread *p_self = ABTI_thread_get_ythread_or_null(p_self_thread);
    if (!p_self) {
        thread_join_busywait(p_thread);
        return;
    }

    /* The target ULT should be different. */
    ABTI_ASSERT(p_thread != p_self_thread);

    ABTI_ythread *p_ythread = ABTI_thread_get_ythread_or_null(p_thread);
    if (!p_ythread) {
        thread_join_yield_task(&p_local_xstream, p_self, p_thread);
        *pp_local = ABTI_xstream_get_local(p_local_xstream);
        return;
    }

    ABT_pool_access access = p_self->thread.p_pool->access;

    if ((p_self->thread.p_pool == p_ythread->thread.p_pool) &&
        (access == ABT_POOL_ACCESS_PRIV || access == ABT_POOL_ACCESS_MPSC ||
         access == ABT_POOL_ACCESS_SPSC) &&
        (ABTD_atomic_acquire_load_int(&p_ythread->thread.state) ==
         ABT_THREAD_STATE_READY)) {

        ABTI_xstream *p_xstream = p_self->thread.p_last_xstream;

        /* If other ES is calling ABTI_ythread_set_ready(), p_ythread may not
         * have been added to the pool yet because ABTI_ythread_set_ready()
         * changes the state first followed by pushing p_ythread to the pool.
         * Therefore, we have to check whether p_ythread is in the pool, and if
         * not, we need to wait until it is added. */
        while (p_ythread->thread.p_pool->u_is_in_pool(p_ythread->thread.unit) !=
               ABT_TRUE) {
        }

        /* This is corresponding to suspension. */
        ABTI_tool_event_ythread_suspend(p_local_xstream, p_self,
                                        p_self->thread.p_parent,
                                        ABT_SYNC_EVENT_TYPE_THREAD_JOIN,
                                        (void *)p_ythread);

        /* Increase the number of blocked units.  Be sure to execute
         * ABTI_pool_inc_num_blocked before ABTI_POOL_REMOVE in order not to
         * underestimate the number of units in a pool. */
        ABTI_pool_inc_num_blocked(p_self->thread.p_pool);
        /* Remove the target ULT from the pool */
        int abt_errno =
            ABTI_pool_remove(p_ythread->thread.p_pool, p_ythread->thread.unit);
        /* This failure is fatal. */
        ABTI_ASSERT(abt_errno == ABT_SUCCESS);

        /* Set the link in the context for the target ULT.  Since p_link will be
         * referenced by p_self, this update does not require release store. */
        ABTD_atomic_relaxed_store_ythread_context_ptr(&p_ythread->ctx.p_link,
                                                      &p_self->ctx);
        /* Set the last ES */
        p_ythread->thread.p_last_xstream = p_xstream;
        ABTD_atomic_release_store_int(&p_ythread->thread.state,
                                      ABT_THREAD_STATE_RUNNING);

        /* Make the current ULT BLOCKED */
        ABTD_atomic_release_store_int(&p_self->thread.state,
                                      ABT_THREAD_STATE_BLOCKED);

        LOG_DEBUG("[U%" PRIu64 ":E%d] blocked to join U%" PRIu64 "\n",
                  ABTI_thread_get_id(&p_self->thread),
                  p_self->thread.p_last_xstream->rank,
                  ABTI_thread_get_id(&p_ythread->thread));
        LOG_DEBUG("[U%" PRIu64 ":E%d] start running\n",
                  ABTI_thread_get_id(&p_ythread->thread),
                  p_ythread->thread.p_last_xstream->rank);

        /* Switch the context */
        ABTI_ythread *p_prev =
            ABTI_ythread_context_switch_to_sibling(&p_local_xstream, p_self,
                                                   p_ythread);
        *pp_local = ABTI_xstream_get_local(p_local_xstream);
        ABTI_tool_event_thread_run(p_local_xstream, &p_self->thread,
                                   &p_prev->thread, p_self->thread.p_parent);

    } else if ((p_self->thread.p_pool != p_ythread->thread.p_pool) &&
               (access == ABT_POOL_ACCESS_PRIV ||
                access == ABT_POOL_ACCESS_SPSC)) {
        /* FIXME: once we change the suspend/resume mechanism (i.e., asking the
         * scheduler to wake up the blocked ULT), we will be able to handle all
         * access modes. */
        thread_join_yield_ythread(&p_local_xstream, p_self, p_ythread);
        *pp_local = ABTI_xstream_get_local(p_local_xstream);
        return;

    } else {
        /* Tell p_ythread that there has been a join request. */
        /* If request already has ABTI_THREAD_REQ_JOIN, p_ythread is
         * terminating. We can't block p_self in this case. */
        uint32_t req = ABTD_atomic_fetch_or_uint32(&p_ythread->thread.request,
                                                   ABTI_THREAD_REQ_JOIN);
        if (req & ABTI_THREAD_REQ_JOIN) {
            thread_join_yield_ythread(&p_local_xstream, p_self, p_ythread);
            *pp_local = ABTI_xstream_get_local(p_local_xstream);
            return;
        }

        ABTI_ythread_set_blocked(p_self);
        LOG_DEBUG("[U%" PRIu64 ":E%d] blocked to join U%" PRIu64 "\n",
                  ABTI_thread_get_id(&p_self->thread),
                  p_self->thread.p_last_xstream->rank,
                  ABTI_thread_get_id(&p_ythread->thread));

        /* Set the link in the context of the target ULT. This p_link might be
         * read by p_ythread running on another ES in parallel, so release-store
         * is needed here. */
        ABTD_atomic_release_store_ythread_context_ptr(&p_ythread->ctx.p_link,
                                                      &p_self->ctx);

        /* Suspend the current ULT */
        ABTI_ythread_suspend(&p_local_xstream, p_self,
                             ABT_SYNC_EVENT_TYPE_THREAD_JOIN,
                             (void *)p_ythread);
        *pp_local = ABTI_xstream_get_local(p_local_xstream);
    }

    /* Resume */
    /* If p_self's state is BLOCKED, the target ULT has terminated on the same
     * ES as p_self's ES and the control has come from the target ULT.
     * Otherwise, the target ULT had been migrated to a different ES, p_self
     * has been resumed by p_self's scheduler.  In the latter case, we don't
     * need to change p_self's state. */
    if (ABTD_atomic_relaxed_load_int(&p_self->thread.state) ==
        ABT_THREAD_STATE_BLOCKED) {
        ABTD_atomic_release_store_int(&p_self->thread.state,
                                      ABT_THREAD_STATE_RUNNING);
        ABTI_pool_dec_num_blocked(p_self->thread.p_pool);
        LOG_DEBUG("[U%" PRIu64 ":E%d] resume after join\n",
                  ABTI_thread_get_id(&p_self->thread),
                  p_self->thread.p_last_xstream->rank);
        ABTI_tool_event_thread_join(*pp_local, p_thread, &p_self->thread);
    } else {
        /* Use a yield-based method. */
        thread_join_yield_ythread(&p_local_xstream, p_self, p_ythread);
        *pp_local = ABTI_xstream_get_local(p_local_xstream);
        return;
    }
}

static void thread_root_func(void *arg)
{
    /* root thread is working on a special context, so it should not rely on
     * functionality that needs yield. */
    ABTI_local *p_local = ABTI_local_get_local();
    ABTI_xstream *p_local_xstream = ABTI_local_get_xstream(p_local);
    ABTI_ASSERT(ABTD_atomic_relaxed_load_int(&p_local_xstream->state) ==
                ABT_XSTREAM_STATE_RUNNING);

    ABTI_ythread *p_root_ythread = p_local_xstream->p_root_ythread;
    p_local_xstream->p_thread = &p_root_ythread->thread;
    ABTI_pool *p_root_pool = p_local_xstream->p_root_pool;

    do {
        ABT_unit unit = ABTI_pool_pop(p_root_pool);
        if (unit != ABT_UNIT_NULL) {
            ABTI_xstream *p_xstream = p_local_xstream;
            ABTI_xstream_run_unit(&p_xstream, unit, p_root_pool);
            /* The root thread must be executed on the same execution stream. */
            ABTI_ASSERT(p_xstream == p_local_xstream);
        }
    } while (ABTD_atomic_acquire_load_int(
                 &p_local_xstream->p_main_sched->p_ythread->thread.state) !=
             ABT_THREAD_STATE_TERMINATED);
    /* The main scheduler thread finishes. */

    /* Set the ES's state as TERMINATED */
    ABTD_atomic_release_store_int(&p_local_xstream->state,
                                  ABT_XSTREAM_STATE_TERMINATED);

    if (p_local_xstream->type == ABTI_XSTREAM_TYPE_PRIMARY) {
        /* Let us jump back to the main thread (then finalize Argobots) */
        ABTD_ythread_finish_context(&p_root_ythread->ctx,
                                    &gp_ABTI_global->p_main_ythread->ctx);
    }
}

static void thread_main_sched_func(void *arg)
{
    ABTI_local *p_local = ABTI_local_get_local();
    ABTI_xstream *p_local_xstream = ABTI_local_get_xstream(p_local);

    while (1) {
        /* Execute the run function of scheduler */
        ABTI_sched *p_sched = p_local_xstream->p_main_sched;
        ABTI_ASSERT(p_local_xstream->p_thread == &p_sched->p_ythread->thread);

        LOG_DEBUG("[S%" PRIu64 "] start\n", p_sched->id);
        p_sched->run(ABTI_sched_get_handle(p_sched));
        /* From here the main scheduler can have been already replaced. */
        /* The main scheduler must be executed on the same execution stream. */
        ABTI_ASSERT(p_local == ABTI_local_get_local_uninlined());
        LOG_DEBUG("[S%" PRIu64 "] end\n", p_sched->id);

        p_sched = p_local_xstream->p_main_sched;
        uint32_t request = ABTD_atomic_acquire_load_uint32(
            &p_sched->p_ythread->thread.request);

        /* If there is an exit or a cancel request, the ES terminates
         * regardless of remaining work units. */
        if (request & (ABTI_THREAD_REQ_TERMINATE | ABTI_THREAD_REQ_CANCEL))
            break;

        /* When join is requested, the ES terminates after finishing
         * execution of all work units. */
        if ((ABTD_atomic_relaxed_load_uint32(&p_sched->request) &
             ABTI_SCHED_REQ_FINISH) &&
            ABTI_sched_get_effective_size(p_local, p_sched) == 0) {
            break;
        }
    }
    /* Finish this thread and goes back to the root thread. */
}

#ifndef ABT_CONFIG_DISABLE_MIGRATION
ABTU_ret_err static int thread_migrate_to_xstream(ABTI_local **pp_local,
                                                  ABTI_thread *p_thread,
                                                  ABTI_xstream *p_xstream)
{
    /* checking for cases when migration is not allowed */
    ABTI_CHECK_TRUE(ABTD_atomic_acquire_load_int(&p_xstream->state) !=
                        ABT_XSTREAM_STATE_TERMINATED,
                    ABT_ERR_INV_XSTREAM);
    ABTI_CHECK_TRUE(!(p_thread->type &
                      (ABTI_THREAD_TYPE_MAIN | ABTI_THREAD_TYPE_MAIN_SCHED)),
                    ABT_ERR_INV_THREAD);
    ABTI_CHECK_TRUE(ABTD_atomic_acquire_load_int(&p_thread->state) !=
                        ABT_THREAD_STATE_TERMINATED,
                    ABT_ERR_INV_THREAD);

    /* We need to find the target scheduler */
    /* We check the state of the ES */
    ABTI_CHECK_TRUE(ABTD_atomic_acquire_load_int(&p_xstream->state) !=
                        ABT_XSTREAM_STATE_TERMINATED,
                    ABT_ERR_INV_XSTREAM);
    /* The migration target should be the main scheduler since it is
     * hard to guarantee the lifetime of the stackable scheduler. */
    ABTI_sched *p_sched = p_xstream->p_main_sched;

    /* We check the state of the sched */
    /* Find a pool */
    ABTI_pool *p_pool = NULL;
    int abt_errno;
    abt_errno =
        ABTI_sched_get_migration_pool(p_sched, p_thread->p_pool, &p_pool);
    ABTI_CHECK_ERROR(abt_errno);
    /* We set the migration counter to prevent the scheduler from
     * stopping */
    ABTI_pool_inc_num_migrations(p_pool);

    abt_errno = thread_migrate_to_pool(pp_local, p_thread, p_pool);
    if (ABTI_IS_ERROR_CHECK_ENABLED && abt_errno != ABT_SUCCESS) {
        ABTI_pool_dec_num_migrations(p_pool);
        return abt_errno;
    }
    return ABT_SUCCESS;
}
#endif

static inline ABT_unit_id thread_get_new_id(void)
{
    return (ABT_unit_id)ABTD_atomic_fetch_add_uint64(&g_thread_id, 1);
}

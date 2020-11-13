/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

#include "abti.h"

/* Must be in a critical section. */
ABTU_ret_err static int init_library(void);
ABTU_ret_err static int finailze_library(void);

/** @defgroup ENV Init & Finalize
 * This group is for initialization and finalization of the Argobots
 * environment.
 */

/* Global Data */
ABTI_global *gp_ABTI_global = NULL;

/* To indicate how many times ABT_init is called. */
static uint32_t g_ABTI_num_inits = 0;
/* A global lock protecting the initialization/finalization process */
static ABTI_spinlock g_ABTI_init_lock = ABTI_SPINLOCK_STATIC_INITIALIZER();
/* A flag whether Argobots has been initialized or not */
static ABTD_atomic_uint32 g_ABTI_initialized =
    ABTD_ATOMIC_UINT32_STATIC_INITIALIZER(0);

/**
 * @ingroup ENV
 * @brief   Initialize the Argobots execution environment.
 *
 * \c ABT_init() initializes the Argobots library and its execution environment.
 * It internally creates objects for the \a primary ES and the \a primary ULT.
 *
 * \c ABT_init() must be called by the primary ULT before using any other
 * Argobots functions. \c ABT_init() can be called again after
 * \c ABT_finalize() is called.
 *
 * @param[in] argc the number of arguments
 * @param[in] argv the argument vector
 * @return Error code
 * @retval ABT_SUCCESS on success
 */
int ABT_init(int argc, char **argv)
{
    ABTI_UNUSED(argc);
    ABTI_UNUSED(argv);
    /* Take a global lock protecting the initialization/finalization process. */
    ABTI_spinlock_acquire(&g_ABTI_init_lock);
    int abt_errno = init_library();
    /* Unlock a global lock */
    ABTI_spinlock_release(&g_ABTI_init_lock);
    ABTI_CHECK_ERROR(abt_errno);
    return ABT_SUCCESS;
}

/**
 * @ingroup ENV
 * @brief   Terminate the Argobots execution environment.
 *
 * \c ABT_finalize() terminates the Argobots execution environment and
 * deallocates memory internally used in Argobots. This function also contains
 * deallocation of objects for the primary ES and the primary ULT.
 *
 * \c ABT_finalize() must be called by the primary ULT. Invoking the Argobots
 * functions after \c ABT_finalize() is not allowed. To use the Argobots
 * functions after calling \c ABT_finalize(), \c ABT_init() needs to be called
 * again.
 *
 * @return Error code
 * @retval ABT_SUCCESS on success
 */
int ABT_finalize(void)
{
    /* Take a global lock protecting the initialization/finalization process. */
    ABTI_spinlock_acquire(&g_ABTI_init_lock);
    int abt_errno = finailze_library();
    /* Unlock a global lock */
    ABTI_spinlock_release(&g_ABTI_init_lock);
    ABTI_CHECK_ERROR(abt_errno);
    return ABT_SUCCESS;
}

/**
 * @ingroup ENV
 * @brief   Check whether \c ABT_init() has been called.
 *
 * \c ABT_initialized() returns \c ABT_SUCCESS if the Argobots execution
 * environment has been initialized. Otherwise, it returns
 * \c ABT_ERR_UNINITIALIZED.
 *
 * @return Error code
 * @retval ABT_SUCCESS           if the environment has been initialized.
 * @retval ABT_ERR_UNINITIALIZED if the environment has not been initialized.
 */
int ABT_initialized(void)
{
    if (ABTD_atomic_acquire_load_uint32(&g_ABTI_initialized) == 0) {
        return ABT_ERR_UNINITIALIZED;
    } else {
        return ABT_SUCCESS;
    }
}

/*****************************************************************************/
/* Internal static functions                                                 */
/*****************************************************************************/

ABTU_ret_err static int init_library(void)
{
    int abt_errno;
    /* If Argobots has already been initialized, just return */
    if (g_ABTI_num_inits++ > 0) {
        return ABT_SUCCESS;
    }

    abt_errno = ABTU_malloc(sizeof(ABTI_global), (void **)&gp_ABTI_global);
    ABTI_CHECK_ERROR(abt_errno);

    /* Initialize the system environment */
    ABTD_env_init(gp_ABTI_global);

    /* Initialize memory pool */
    ABTI_mem_init(gp_ABTI_global);

    /* Initialize IDs */
    ABTI_thread_reset_id();
    ABTI_sched_reset_id();
    ABTI_pool_reset_id();

#ifndef ABT_CONFIG_DISABLE_TOOL_INTERFACE
    /* Initialize the tool interface */
    ABTI_spinlock_clear(&gp_ABTI_global->tool_writer_lock);
    gp_ABTI_global->tool_thread_cb_f = NULL;
    gp_ABTI_global->tool_thread_user_arg = NULL;
    gp_ABTI_global->tool_task_cb_f = NULL;
    gp_ABTI_global->tool_task_user_arg = NULL;
    ABTD_atomic_relaxed_store_uint64(&gp_ABTI_global
                                          ->tool_thread_event_mask_tagged,
                                     0);
#endif

    /* Initialize the ES list */
    gp_ABTI_global->p_xstream_head = NULL;
    gp_ABTI_global->num_xstreams = 0;

    /* Initialize a spinlock */
    ABTI_spinlock_clear(&gp_ABTI_global->xstream_list_lock);

    /* Create the primary ES */
    ABTI_xstream *p_local_xstream;
    abt_errno = ABTI_xstream_create_primary(&p_local_xstream);
    ABTI_CHECK_ERROR(abt_errno);

    /* Init the ES local data */
    ABTI_local_set_xstream(p_local_xstream);

    /* Create the primary ULT, i.e., the main thread */
    ABTI_ythread *p_main_ythread;
    abt_errno =
        ABTI_ythread_create_main(ABTI_xstream_get_local(p_local_xstream),
                                 p_local_xstream, &p_main_ythread);
    /* Set as if p_local_xstream is currently running the main thread. */
    ABTD_atomic_relaxed_store_int(&p_main_ythread->thread.state,
                                  ABT_THREAD_STATE_RUNNING);
    p_main_ythread->thread.p_last_xstream = p_local_xstream;
    ABTI_CHECK_ERROR(abt_errno);
    gp_ABTI_global->p_main_ythread = p_main_ythread;
    p_local_xstream->p_thread = &p_main_ythread->thread;

    /* Start the primary ES */
    ABTI_xstream_start_primary(&p_local_xstream, p_local_xstream,
                               p_main_ythread);

    if (gp_ABTI_global->print_config == ABT_TRUE) {
        ABTI_info_print_config(stdout);
    }
    ABTD_atomic_release_store_uint32(&g_ABTI_initialized, 1);
    return ABT_SUCCESS;
}

ABTU_ret_err static int finailze_library(void)
{
    ABTI_local *p_local = ABTI_local_get_local();

    /* If Argobots is not initialized, just return */
    ABTI_CHECK_TRUE(g_ABTI_num_inits > 0, ABT_ERR_UNINITIALIZED);
    /* If Argobots is still referenced by others, just return */
    if (--g_ABTI_num_inits != 0) {
        return ABT_SUCCESS;
    }

    ABTI_xstream *p_local_xstream = ABTI_local_get_xstream_or_null(p_local);
    /* If called by an external thread, return an error. */
    ABTI_CHECK_TRUE(!ABTI_IS_EXT_THREAD_ENABLED || p_local_xstream,
                    ABT_ERR_INV_XSTREAM);

    ABTI_CHECK_TRUE_MSG(p_local_xstream->type == ABTI_XSTREAM_TYPE_PRIMARY,
                        ABT_ERR_INV_XSTREAM,
                        "ABT_finalize must be called by the primary ES.");

    ABTI_thread *p_self = p_local_xstream->p_thread;
    ABTI_CHECK_TRUE_MSG(p_self->type & ABTI_THREAD_TYPE_MAIN,
                        ABT_ERR_INV_THREAD,
                        "ABT_finalize must be called by the primary ULT.");
    ABTI_ythread *p_ythread;
    ABTI_CHECK_YIELDABLE(p_self, &p_ythread, ABT_ERR_INV_THREAD);

#ifndef ABT_CONFIG_DISABLE_TOOL_INTERFACE
    /* Turns off the tool interface */
    ABTI_tool_event_thread_update_callback(NULL, ABT_TOOL_EVENT_THREAD_NONE,
                                           NULL);
    ABTI_tool_event_task_update_callback(NULL, ABT_TOOL_EVENT_TASK_NONE, NULL);
#endif

    /* Set the orphan request for the primary ULT */
    ABTI_thread_set_request(p_self, ABTI_THREAD_REQ_ORPHAN);
    /* Finish the main scheduler of this local xstream. */
    ABTI_sched_finish(p_local_xstream->p_main_sched);
    /* p_self cannot join the main scheduler since p_self needs to be orphaned.
     * Let's wait till the main scheduler finishes.  This thread will be
     * scheduled when the main root thread finishes. */
    ABTI_ythread_yield(&p_local_xstream, p_ythread, ABT_SYNC_EVENT_TYPE_OTHER,
                       NULL);
    ABTI_ASSERT(p_local_xstream == ABTI_local_get_xstream(p_local));
    ABTI_ASSERT(p_local_xstream->p_thread == p_self);

    /* Remove the primary ULT */
    p_local_xstream->p_thread = NULL;
    ABTI_ythread_free_main(ABTI_xstream_get_local(p_local_xstream), p_ythread);

    /* Free the primary ES */
    ABTI_xstream_free(ABTI_xstream_get_local(p_local_xstream), p_local_xstream,
                      ABT_TRUE);

    /* Finalize the ES local data */
    ABTI_local_set_xstream(NULL);

    /* Free the ES array */
    ABTI_ASSERT(gp_ABTI_global->p_xstream_head == NULL);

    /* Finalize the memory pool */
    ABTI_mem_finalize(gp_ABTI_global);

    /* Restore the affinity */
    if (gp_ABTI_global->set_affinity == ABT_TRUE) {
        ABTD_affinity_finalize();
    }

    /* Free the ABTI_global structure */
    ABTU_free(gp_ABTI_global);
    gp_ABTI_global = NULL;
    ABTD_atomic_release_store_uint32(&g_ABTI_initialized, 0);
    return ABT_SUCCESS;
}

/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

#include "abti.h"

ABTU_ret_err static int info_print_thread_stacks_in_pool(FILE *fp,
                                                         ABTI_pool *p_pool);
static void info_trigger_print_all_thread_stacks(
    FILE *fp, double timeout, void (*cb_func)(ABT_bool, void *), void *arg);

/** @defgroup INFO  Information
 * This group is for getting diverse runtime information of Argobots.  The
 * routines in this group are meant for debugging and diagnosing Argobots.
 */

/**
 * @ingroup INFO
 * @brief   Get the configuration information associated with \c query_kind.
 *
 * \c ABT_info_query_config() gets the configuration information associated with
 * the given \c query_kind and writes a value to \c val.
 *
 * The behavior of \c ABT_info_query_config() depends on \c query_kind.
 * - ABT_INFO_QUERY_KIND_ENABLED_DEBUG
 *   \c val must be a pointer to a variable of the type ABT_bool.  ABT_TRUE is
 *   set to \c *val if the debug mode is enabled.  Otherwise, ABT_FALSE is set.
 * - ABT_INFO_QUERY_KIND_ENABLED_PRINT_ERRNO
 *   \c val must be a pointer to a variable of the type ABT_bool.  ABT_TRUE is
 *   set to \c *val if the Argobots library is configured to print an error
 *   number when an error happens.  Otherwise, ABT_FALSE is set.
 * - ABT_INFO_QUERY_KIND_ENABLED_LOG
 *   \c val must be a pointer to a variable of the type ABT_bool.  ABT_TRUE is
 *   set to \c *val if the Argobots library is configured to print debug
 *   messages.  Otherwise, ABT_FALSE is set.
 * - ABT_INFO_QUERY_KIND_ENABLED_VALGRIND
 *   \c val must be a pointer to a variable of the type ABT_bool.  ABT_TRUE is
 *   set to \c *val if the Argobots library is configured to be Valgrind
 *   friendly.  Otherwise, ABT_FALSE is set.
 * - ABT_INFO_QUERY_KIND_ENABLED_CHECK_ERROR
 *   \c val must be a pointer to a variable of the type ABT_bool.  ABT_FALSE is
 *   set to \c *val if the Argobots library is configured to ignore some error
 *   checks.  Otherwise, ABT_TRUE is set.
 * - ABT_INFO_QUERY_KIND_ENABLED_CHECK_POOL_PRODUCER
 *   \c val must be a pointer to a variable of the type ABT_bool.  ABT_FALSE is
 *   set to \c *val if the Argobots library is configured to ignore an access
 *   violation error regarding pool producers.  Otherwise, ABT_TRUE is set.
 * - ABT_INFO_QUERY_KIND_ENABLED_CHECK_POOL_CONSUMER
 *   \c val must be a pointer to a variable of the type ABT_bool.  ABT_FALSE is
 *   set to \c *val if the Argobots library is configured to ignore an access
 *   violation error regarding pool consumers.  Otherwise, ABT_TRUE is set.
 * - ABT_INFO_QUERY_KIND_ENABLED_PRESERVE_FPU
 *   \c val must be a pointer to a variable of the type ABT_bool.  ABT_TRUE is
 *   set to \c *val if the Argobots library is configured to save floating-point
 *   registers on user-level context switching.  Otherwise, ABT_FALSE is set.
 * - ABT_INFO_QUERY_KIND_ENABLED_THREAD_CANCEL
 *   \c val must be a pointer to a variable of the type ABT_bool.  ABT_TRUE is
 *   set to \c *val if the Argobots library is configured to enable the thread
 *   cancellation feature. Otherwise, ABT_FALSE is set.
 * - ABT_INFO_QUERY_KIND_ENABLED_TASK_CANCEL
 *   \c val must be a pointer to a variable of the type ABT_bool.  ABT_TRUE is
 *   set to \c *val if the Argobots library is configured to enable the task
 *   cancellation feature. Otherwise, ABT_FALSE is set.
 * - ABT_INFO_QUERY_KIND_ENABLED_MIGRATION
 *   \c val must be a pointer to a variable of the type ABT_bool.  ABT_TRUE is
 *   set to \c *val if the Argobots library is configured to enable the
 *   thread/task migration cancellation feature. Otherwise, ABT_FALSE is set.
 * - ABT_INFO_QUERY_KIND_ENABLED_STACKABLE_SCHED
 *   \c val must be a pointer to a variable of the type ABT_bool.  ABT_TRUE is
 *   set to \c *val if the Argobots library is configured to enable the
 *   stackable scheduler feature is supported.  Otherwise, ABT_FALSE is set.
 * - ABT_INFO_QUERY_KIND_ENABLED_EXTERNAL_THREAD
 *   \c val must be a pointer to a variable of the type ABT_bool.  ABT_TRUE is
 *   set to \c *val if the Argobots library is configured to enable the
 *   external thread feature is supported.  Otherwise, ABT_FALSE is set.
 * - ABT_INFO_QUERY_KIND_ENABLED_SCHED_SLEEP
 *   \c val must be a pointer to a variable of the type ABT_bool.  ABT_TRUE is
 *   set to \c *val if the Argobots library is configured to enable the sleep
 *   feature of predefined schedulers.  Otherwise, ABT_FALSE is set.
 * - ABT_INFO_QUERY_KIND_ENABLED_PRINT_CONFIG
 *   \c val must be a pointer to a variable of the type ABT_bool.  ABT_TRUE is
 *   set to \c *val if the Argobots library is configured to print all the
 *   configuration settings on ABT_init().  Otherwise, ABT_FALSE is set.
 * - ABT_INFO_QUERY_KIND_ENABLED_AFFINITY
 *   \c val must be a pointer to a variable of the type ABT_bool.  ABT_TRUE is
 *   set to \c *val if the Argobots library is configured to enable the affinity
 *   setting.  Otherwise, ABT_FALSE is set.
 * - ABT_INFO_QUERY_KIND_MAX_NUM_XSTREAMS
 *   \c val must be a pointer to a variable of the type unsigned int.  The
 *   maximum number of execution streams that can be created in Argobots is set
 *   to \c *val.
 * - ABT_INFO_QUERY_KIND_DEFAULT_THREAD_STACKSIZE
 *   \c val must be a pointer to a variable of the type size_t.  The default
 *   stack size of ULTs is set to \c *val.
 * - ABT_INFO_QUERY_KIND_DEFAULT_SCHED_STACKSIZE
 *   \c val must be a pointer to a variable of the type size_t.  The default
 *   stack size of ULT-type schedulers is set to \c *val.
 * - ABT_INFO_QUERY_KIND_DEFAULT_SCHED_EVENT_FREQ
 *   \c val must be a pointer to a variable of the type uint64_t.  The default
 *   event-checking frequency of predefined schedulers is set to \c *val.
 * - ABT_INFO_QUERY_KIND_DEFAULT_SCHED_SLEEP_NSEC
 *   \c val must be a pointer to a variable of the type uint64_t.  The default
 *   sleep time (nanoseconds) of predefined schedulers is set to \c *val.
 * - ABT_INFO_QUERY_KIND_ENABLED_TOOL
 *   \c val must be a pointer to a variable of the type ABT_bool.  ABT_TRUE is
 *   set to \c *val if the tool is enabled.  Otherwise, ABT_FALSE is set.
 *
 * @param[in]  query_kind  query kind
 * @param[out] val         a pointer to a result
 * @return Error code
 * @retval ABT_SUCCESS            on success
 * @retval ABT_ERR_INV_QUERY_KIND given query kind is invalid
 * @retval ABT_ERR_UNINITIALIZED  Argobots has not been initialized
 */
int ABT_info_query_config(ABT_info_query_kind query_kind, void *val)
{
    ABTI_SETUP_WITH_INIT_CHECK();

    switch (query_kind) {
        case ABT_INFO_QUERY_KIND_ENABLED_DEBUG:
            *((ABT_bool *)val) = gp_ABTI_global->use_debug;
            break;
        case ABT_INFO_QUERY_KIND_ENABLED_PRINT_ERRNO:
#ifdef ABT_CONFIG_PRINT_ABT_ERRNO
            *((ABT_bool *)val) = ABT_TRUE;
#else
            *((ABT_bool *)val) = ABT_FALSE;
#endif
            break;
        case ABT_INFO_QUERY_KIND_ENABLED_LOG:
            *((ABT_bool *)val) = gp_ABTI_global->use_logging;
            break;
        case ABT_INFO_QUERY_KIND_ENABLED_VALGRIND:
#ifdef HAVE_VALGRIND_SUPPORT
            *((ABT_bool *)val) = ABT_TRUE;
#else
            *((ABT_bool *)val) = ABT_FALSE;
#endif
            break;
        case ABT_INFO_QUERY_KIND_ENABLED_CHECK_ERROR:
#ifndef ABT_CONFIG_DISABLE_ERROR_CHECK
            *((ABT_bool *)val) = ABT_TRUE;
#else
            *((ABT_bool *)val) = ABT_FALSE;
#endif
            break;
        case ABT_INFO_QUERY_KIND_ENABLED_CHECK_POOL_PRODUCER:
            *((ABT_bool *)val) = ABT_FALSE;
            break;
        case ABT_INFO_QUERY_KIND_ENABLED_CHECK_POOL_CONSUMER:
            *((ABT_bool *)val) = ABT_FALSE;
            break;
        case ABT_INFO_QUERY_KIND_ENABLED_PRESERVE_FPU:
#ifdef ABTD_FCONTEXT_PRESERVE_FPU
            *((ABT_bool *)val) = ABT_TRUE;
#else
            *((ABT_bool *)val) = ABT_FALSE;
#endif
            break;
        case ABT_INFO_QUERY_KIND_ENABLED_THREAD_CANCEL:
#ifndef ABT_CONFIG_DISABLE_THREAD_CANCEL
            *((ABT_bool *)val) = ABT_TRUE;
#else
            *((ABT_bool *)val) = ABT_FALSE;
#endif
            break;
        case ABT_INFO_QUERY_KIND_ENABLED_TASK_CANCEL:
#ifndef ABT_CONFIG_DISABLE_TASK_CANCEL
            *((ABT_bool *)val) = ABT_TRUE;
#else
            *((ABT_bool *)val) = ABT_FALSE;
#endif
            break;
        case ABT_INFO_QUERY_KIND_ENABLED_MIGRATION:
#ifndef ABT_CONFIG_DISABLE_MIGRATION
            *((ABT_bool *)val) = ABT_TRUE;
#else
            *((ABT_bool *)val) = ABT_FALSE;
#endif
            break;
        case ABT_INFO_QUERY_KIND_ENABLED_STACKABLE_SCHED:
            *((ABT_bool *)val) = ABT_TRUE;
            break;
        case ABT_INFO_QUERY_KIND_ENABLED_EXTERNAL_THREAD:
#ifndef ABT_CONFIG_DISABLE_EXT_THREAD
            *((ABT_bool *)val) = ABT_TRUE;
#else
            *((ABT_bool *)val) = ABT_FALSE;
#endif
            break;
        case ABT_INFO_QUERY_KIND_ENABLED_SCHED_SLEEP:
#ifdef ABT_CONFIG_USE_SCHED_SLEEP
            *((ABT_bool *)val) = ABT_TRUE;
#else
            *((ABT_bool *)val) = ABT_FALSE;
#endif
            break;
        case ABT_INFO_QUERY_KIND_ENABLED_PRINT_CONFIG:
            *((ABT_bool *)val) = gp_ABTI_global->print_config;
            break;
        case ABT_INFO_QUERY_KIND_ENABLED_AFFINITY:
            *((ABT_bool *)val) = gp_ABTI_global->set_affinity;
            break;
        case ABT_INFO_QUERY_KIND_MAX_NUM_XSTREAMS:
            *((unsigned int *)val) = gp_ABTI_global->max_xstreams;
            break;
        case ABT_INFO_QUERY_KIND_DEFAULT_THREAD_STACKSIZE:
            *((size_t *)val) = gp_ABTI_global->thread_stacksize;
            break;
        case ABT_INFO_QUERY_KIND_DEFAULT_SCHED_STACKSIZE:
            *((size_t *)val) = gp_ABTI_global->sched_stacksize;
            break;
        case ABT_INFO_QUERY_KIND_DEFAULT_SCHED_EVENT_FREQ:
            *((uint64_t *)val) = gp_ABTI_global->sched_event_freq;
            break;
        case ABT_INFO_QUERY_KIND_DEFAULT_SCHED_SLEEP_NSEC:
            *((uint64_t *)val) = gp_ABTI_global->sched_sleep_nsec;
            break;
        case ABT_INFO_QUERY_KIND_ENABLED_TOOL:
#ifndef ABT_CONFIG_DISABLE_TOOL_INTERFACE
            *((ABT_bool *)val) = ABT_TRUE;
#else
            *((ABT_bool *)val) = ABT_FALSE;
#endif
            break;
        default:
            ABTI_HANDLE_ERROR(ABT_ERR_INV_QUERY_KIND);
            break;
    }
    return ABT_SUCCESS;
}

/**
 * @ingroup INFO
 * @brief   Write the configuration information to the output stream.
 *
 * \c ABT_info_print_config() writes the configuration information to the given
 * output stream \c fp.
 *
 * @param[in] fp  output stream
 * @return Error code
 * @retval ABT_SUCCESS            on success
 * @retval ABT_ERR_UNINITIALIZED  Argobots has not been initialized
 */
int ABT_info_print_config(FILE *fp)
{
    ABTI_SETUP_WITH_INIT_CHECK();

    ABTI_info_print_config(fp);
    return ABT_SUCCESS;
}

/**
 * @ingroup INFO
 * @brief   Write the information of all created ESs to the output stream.
 *
 * \c ABT_info_print_all_xstreams() writes the information of all ESs to the
 * given output stream \c fp.
 *
 * @param[in] fp  output stream
 * @return Error code
 * @retval ABT_SUCCESS            on success
 * @retval ABT_ERR_UNINITIALIZED  Argobots has not been initialized
 */
int ABT_info_print_all_xstreams(FILE *fp)
{
    ABTI_SETUP_WITH_INIT_CHECK();

    ABTI_global *p_global = gp_ABTI_global;

    ABTI_spinlock_acquire(&p_global->xstream_list_lock);

    fprintf(fp, "# of created ESs: %d\n", p_global->num_xstreams);

    ABTI_xstream *p_xstream = p_global->p_xstream_head;
    while (p_xstream) {
        ABTI_xstream_print(p_xstream, fp, 0, ABT_FALSE);
        p_xstream = p_xstream->p_next;
    }

    ABTI_spinlock_release(&p_global->xstream_list_lock);

    fflush(fp);
    return ABT_SUCCESS;
}

/**
 * @ingroup INFO
 * @brief   Write the information of the target ES to the output stream.
 *
 * \c ABT_info_print_xstream() writes the information of the target ES
 * \c xstream to the given output stream \c fp.
 *
 * @param[in] fp       output stream
 * @param[in] xstream  handle to the target ES
 * @return Error code
 * @retval ABT_SUCCESS  on success
 */
int ABT_info_print_xstream(FILE *fp, ABT_xstream xstream)
{
    ABTI_xstream *p_xstream = ABTI_xstream_get_ptr(xstream);
    ABTI_CHECK_NULL_XSTREAM_PTR(p_xstream);

    ABTI_xstream_print(p_xstream, fp, 0, ABT_FALSE);
    return ABT_SUCCESS;
}

/**
 * @ingroup INFO
 * @brief   Write the information of the target scheduler to the output stream.
 *
 * \c ABT_info_print_sched() writes the information of the target scheduler
 * \c sched to the given output stream \c fp.
 *
 * @param[in] fp     output stream
 * @param[in] sched  handle to the target scheduler
 * @return Error code
 * @retval ABT_SUCCESS  on success
 */
int ABT_info_print_sched(FILE *fp, ABT_sched sched)
{
    ABTI_sched *p_sched = ABTI_sched_get_ptr(sched);
    ABTI_CHECK_NULL_SCHED_PTR(p_sched);

    ABTI_sched_print(p_sched, fp, 0, ABT_TRUE);
    return ABT_SUCCESS;
}

/**
 * @ingroup INFO
 * @brief   Write the information of the target pool to the output stream.
 *
 * \c ABT_info_print_pool() writes the information of the target pool
 * \c pool to the given output stream \c fp.
 *
 * @param[in] fp    output stream
 * @param[in] pool  handle to the target pool
 * @return Error code
 * @retval ABT_SUCCESS  on success
 */
int ABT_info_print_pool(FILE *fp, ABT_pool pool)
{
    ABTI_pool *p_pool = ABTI_pool_get_ptr(pool);
    ABTI_CHECK_NULL_POOL_PTR(p_pool);

    ABTI_pool_print(p_pool, fp, 0);
    return ABT_SUCCESS;
}

/**
 * @ingroup INFO
 * @brief   Write the information of the target ULT to the output stream.
 *
 * \c ABT_info_print_thread() writes the information of the target ULT
 * \c thread to the given output stream \c fp.
 *
 * @param[in] fp      output stream
 * @param[in] thread  handle to the target ULT
 * @return Error code
 * @retval ABT_SUCCESS  on success
 */
int ABT_info_print_thread(FILE *fp, ABT_thread thread)
{
    ABTI_thread *p_thread = ABTI_thread_get_ptr(thread);
    ABTI_CHECK_NULL_THREAD_PTR(p_thread);

    ABTI_thread_print(p_thread, fp, 0);
    return ABT_SUCCESS;
}

/**
 * @ingroup INFO
 * @brief   Write the information of the target ULT attribute to the output
 * stream.
 *
 * \c ABT_info_print_thread_attr() writes the information of the target ULT
 * attribute \c attr to the given output stream \c fp.
 *
 * @param[in] fp    output stream
 * @param[in] attr  handle to the target ULT attribute
 * @return Error code
 * @retval ABT_SUCCESS  on success
 */
int ABT_info_print_thread_attr(FILE *fp, ABT_thread_attr attr)
{
    ABTI_thread_attr *p_attr = ABTI_thread_attr_get_ptr(attr);
    ABTI_CHECK_NULL_THREAD_ATTR_PTR(p_attr);

    ABTI_thread_attr_print(p_attr, fp, 0);
    return ABT_SUCCESS;
}

#ifdef ABT_CONFIG_USE_DOXYGEN
/**
 * @ingroup INFO
 * @brief   Write the information of the target tasklet to the output stream.
 *
 * \c ABT_info_print_task() writes the information of the target tasklet
 * \c task to the given output stream \c fp.
 *
 * @param[in] fp    output stream
 * @param[in] task  handle to the target tasklet
 * @return Error code
 * @retval ABT_SUCCESS  on success
 */
int ABT_info_print_task(FILE *fp, ABT_task task);
#endif

/**
 * @ingroup INFO
 * @brief   Dump the stack of the target thread to the output stream.
 *
 * \c ABT_info_print_thread_stack() dumps the call stack of \c thread
 * to the given output stream \c fp.
 *
 * @param[in] fp      output stream
 * @param[in] thread  handle to the target thread
 * @return Error code
 * @retval ABT_SUCCESS on success
 */
int ABT_info_print_thread_stack(FILE *fp, ABT_thread thread)
{
    ABTI_thread *p_thread = ABTI_thread_get_ptr(thread);
    ABTI_CHECK_NULL_THREAD_PTR(p_thread);
    ABTI_ythread *p_ythread;
    ABTI_CHECK_YIELDABLE(p_thread, &p_ythread, ABT_ERR_INV_THREAD);

    ABTI_ythread_print_stack(p_ythread, fp);
    return ABT_SUCCESS;
}

/**
 * @ingroup INFO
 * @brief   Dump stack information of all the threads in the target pool.
 *
 * \c ABT_info_print_thread_stacks_in_pool() dumps call stacks of all threads
 * stored in \c pool.  This function returns \c ABT_ERR_POOL if \c pool does not
 * support \c p_print_all.
 *
 * @param[in] fp    output stream
 * @param[in] pool  handle to the target pool
 * @return Error code
 * @retval ABT_SUCCESS on success
 */
int ABT_info_print_thread_stacks_in_pool(FILE *fp, ABT_pool pool)
{
    ABTI_pool *p_pool = ABTI_pool_get_ptr(pool);
    ABTI_CHECK_NULL_POOL_PTR(p_pool);

    int abt_errno = info_print_thread_stacks_in_pool(fp, p_pool);
    ABTI_CHECK_ERROR(abt_errno);
    return ABT_SUCCESS;
}

/**
 * @ingroup INFO
 * @brief   Dump stacks of threads in pools existing in Argobots.
 *
 * \c ABT_info_trigger_print_all_thread_stacks() tries to dump call stacks of
 * all threads stored in pools in the Argobots runtime. This function itself
 * does not print stacks; it immediately returns after updating a flag. Stacks
 * are printed when all execution streams stop in \c ABT_xstream_check_events().
 *
 * If some execution streams do not stop within a certain time period, one of
 * the stopped execution streams starts to print stack information. In this
 * case, this function might not work correctly and at worst causes a crash.
 * This function does not work at all if no execution stream executes
 * \c ABT_xstream_check_events().
 *
 * \c cb_func is called after completing stack dump unless it is NULL. The first
 * argument is set to \c ABT_TRUE if not all the execution streams stop within
 * \c timeout. Otherwise, \c ABT_FALSE is set. The second argument is
 * user-defined data \c arg. Since \c cb_func is not called by a thread or an
 * execution stream, \c ABT_self_...() functions in \c cb_func return undefined
 * values. Neither signal-safety nor thread-safety is required for \c cb_func.
 *
 * In Argobots, \c ABT_info_trigger_print_all_thread_stacks is exceptionally
 * signal-safe; it can be safely called in a signal handler.
 *
 * The following threads are not captured in this function:
 * - threads that are suspending (e.g., by \c ABT_thread_suspend())
 * - threads in pools that are not associated with main schedulers
 *
 * @param[in] fp       output stream
 * @param[in] timeout  timeout (second). Disabled if the value is negative.
 * @param[in] cb_func  call-back function
 * @param[in] arg      an argument passed to \c cb_func
 * @return Error code
 * @retval ABT_SUCCESS on success
 */
int ABT_info_trigger_print_all_thread_stacks(FILE *fp, double timeout,
                                             void (*cb_func)(ABT_bool, void *),
                                             void *arg)
{
    info_trigger_print_all_thread_stacks(fp, timeout, cb_func, arg);
    return ABT_SUCCESS;
}

/*****************************************************************************/
/* Private APIs                                                              */
/*****************************************************************************/

ABTU_ret_err static int print_all_thread_stacks(FILE *fp);

#define PRINT_STACK_FLAG_UNSET 0
#define PRINT_STACK_FLAG_INITIALIZE 1
#define PRINT_STACK_FLAG_WAIT 2
#define PRINT_STACK_FLAG_FINALIZE 3

static ABTD_atomic_uint32 print_stack_flag =
    ABTD_ATOMIC_UINT32_STATIC_INITIALIZER(PRINT_STACK_FLAG_UNSET);
static FILE *print_stack_fp = NULL;
static double print_stack_timeout = 0.0;
static void (*print_cb_func)(ABT_bool, void *) = NULL;
static void *print_arg = NULL;
static ABTD_atomic_uint32 print_stack_barrier =
    ABTD_ATOMIC_UINT32_STATIC_INITIALIZER(0);

void ABTI_info_check_print_all_thread_stacks(void)
{
    if (ABTD_atomic_acquire_load_uint32(&print_stack_flag) !=
        PRINT_STACK_FLAG_WAIT)
        return;

    /* Wait for the other execution streams using a barrier mechanism. */
    uint32_t self_value = ABTD_atomic_fetch_add_uint32(&print_stack_barrier, 1);
    if (self_value == 0) {
        /* This ES becomes the main ES. */
        double start_time = ABTI_get_wtime();
        ABT_bool force_print = ABT_FALSE;

        /* xstreams_lock is acquired to avoid dynamic ES creation while
         * printing data. */
        ABTI_spinlock_acquire(&gp_ABTI_global->xstream_list_lock);
        while (1) {
            if (ABTD_atomic_acquire_load_uint32(&print_stack_barrier) >=
                gp_ABTI_global->num_xstreams) {
                break;
            }
            if (print_stack_timeout >= 0.0 &&
                (ABTI_get_wtime() - start_time) >= print_stack_timeout) {
                force_print = ABT_TRUE;
                break;
            }
            ABTI_spinlock_release(&gp_ABTI_global->xstream_list_lock);
            ABTD_atomic_pause();
            ABTI_spinlock_acquire(&gp_ABTI_global->xstream_list_lock);
        }
        /* All the available ESs are (supposed to be) stopped. We *assume* that
         * no ES is calling and will call Argobots functions except this
         * function while printing stack information. */
        if (force_print) {
            fprintf(print_stack_fp,
                    "ABT_info_trigger_print_all_thread_stacks: "
                    "timeout (only %d ESs stop)\n",
                    (int)ABTD_atomic_acquire_load_uint32(&print_stack_barrier));
        }
        int abt_errno = print_all_thread_stacks(print_stack_fp);
        if (ABTI_IS_ERROR_CHECK_ENABLED && abt_errno != ABT_SUCCESS) {
            fprintf(print_stack_fp, "ABT_info_trigger_print_all_thread_stacks: "
                                    "failed because of an internal error.\n");
        }
        /* Release the lock that protects ES data. */
        ABTI_spinlock_release(&gp_ABTI_global->xstream_list_lock);
        if (print_cb_func)
            print_cb_func(force_print, print_arg);
        /* Update print_stack_flag to 3. */
        ABTD_atomic_release_store_uint32(&print_stack_flag,
                                         PRINT_STACK_FLAG_FINALIZE);
    } else {
        /* Wait for the main ES's work. */
        while (ABTD_atomic_acquire_load_uint32(&print_stack_flag) !=
               PRINT_STACK_FLAG_FINALIZE)
            ABTD_atomic_pause();
    }
    ABTI_ASSERT(ABTD_atomic_acquire_load_uint32(&print_stack_flag) ==
                PRINT_STACK_FLAG_FINALIZE);

    /* Decrement the barrier value. */
    uint32_t dec_value = ABTD_atomic_fetch_sub_uint32(&print_stack_barrier, 1);
    if (dec_value == 0) {
        /* The last execution stream resets the flag. */
        ABTD_atomic_release_store_uint32(&print_stack_flag,
                                         PRINT_STACK_FLAG_UNSET);
    }
}

void ABTI_info_print_config(FILE *fp)
{
    ABTI_global *p_global = gp_ABTI_global;

    fprintf(fp, "Argobots Configuration:\n");
    fprintf(fp, " - # of cores: %d\n", p_global->num_cores);
    fprintf(fp, " - cache line size: %u\n", ABT_CONFIG_STATIC_CACHELINE_SIZE);
    fprintf(fp, " - OS page size: %u\n", p_global->os_page_size);
    fprintf(fp, " - huge page size: %u\n", p_global->huge_page_size);
    fprintf(fp, " - max. # of ESs: %d\n", p_global->max_xstreams);
    fprintf(fp, " - cur. # of ESs: %d\n", p_global->num_xstreams);
    fprintf(fp, " - ES affinity: %s\n",
            (p_global->set_affinity == ABT_TRUE) ? "on" : "off");
    fprintf(fp, " - logging: %s\n",
            (p_global->use_logging == ABT_TRUE) ? "on" : "off");
    fprintf(fp, " - debug output: %s\n",
            (p_global->use_debug == ABT_TRUE) ? "on" : "off");
    fprintf(fp, " - key table entries: %d\n", p_global->key_table_size);
    fprintf(fp, " - ULT stack size: %u KB\n",
            (unsigned)(p_global->thread_stacksize / 1024));
    fprintf(fp, " - scheduler stack size: %u KB\n",
            (unsigned)(p_global->sched_stacksize / 1024));
    fprintf(fp, " - scheduler event check frequency: %u\n",
            p_global->sched_event_freq);

    fprintf(fp, " - timer function: "
#if defined(ABT_CONFIG_USE_CLOCK_GETTIME)
                "clock_gettime"
#elif defined(ABT_CONFIG_USE_MACH_ABSOLUTE_TIME)
                "mach_absolute_time"
#elif defined(ABT_CONFIG_USE_GETTIMEOFDAY)
                "gettimeofday"
#endif
                "\n");

#ifdef ABT_CONFIG_USE_MEM_POOL
    fprintf(fp, "Memory Pool:\n");
    fprintf(fp, " - page size for allocation: %u KB\n",
            p_global->mem_page_size / 1024);
    fprintf(fp, " - stack page size: %u KB\n", p_global->mem_sp_size / 1024);
    fprintf(fp, " - max. # of stacks per ES: %u\n", p_global->mem_max_stacks);
    switch (p_global->mem_lp_alloc) {
        case ABTI_MEM_LP_MALLOC:
            fprintf(fp, " - large page allocation: malloc\n");
            break;
        case ABTI_MEM_LP_MMAP_RP:
            fprintf(fp, " - large page allocation: mmap regular pages\n");
            break;
        case ABTI_MEM_LP_MMAP_HP_RP:
            fprintf(fp, " - large page allocation: mmap huge pages + "
                        "regular pages\n");
            break;
        case ABTI_MEM_LP_MMAP_HP_THP:
            fprintf(fp, " - large page allocation: mmap huge pages + THPs\n");
            break;
        case ABTI_MEM_LP_THP:
            fprintf(fp, " - large page allocation: THPs\n");
            break;
    }
#endif /* ABT_CONFIG_USE_MEM_POOL */

    fflush(fp);
}

/*****************************************************************************/
/* Internal static functions                                                 */
/*****************************************************************************/

struct info_print_unit_arg_t {
    FILE *fp;
    ABT_pool pool;
};

struct info_pool_set_t {
    ABT_pool *pools;
    size_t num;
    size_t len;
};

static void info_print_unit(void *arg, ABT_unit unit)
{
    /* This function may not have any side effect on unit because it is passed
     * to p_print_all. */
    struct info_print_unit_arg_t *p_arg;
    p_arg = (struct info_print_unit_arg_t *)arg;
    FILE *fp = p_arg->fp;
    ABT_pool pool = p_arg->pool;
    ABTI_pool *p_pool = ABTI_pool_get_ptr(pool);
    ABT_unit_type type = p_pool->u_get_type(unit);

    if (type == ABT_UNIT_TYPE_THREAD) {
        fprintf(fp, "=== ULT (%p) ===\n", (void *)unit);
        ABT_thread thread = p_pool->u_get_thread(unit);
        ABTI_ythread *p_ythread = ABTI_ythread_get_ptr(thread);
        ABT_unit_id thread_id = ABTI_thread_get_id(&p_ythread->thread);
        fprintf(fp,
                "id        : %" PRIu64 "\n"
                "ctx       : %p\n",
                (uint64_t)thread_id, (void *)&p_ythread->ctx);
        ABTD_ythread_print_context(p_ythread, fp, 2);
        fprintf(fp,
                "stack     : %p\n"
                "stacksize : %" PRIu64 "\n",
                p_ythread->p_stack, (uint64_t)p_ythread->stacksize);
        ABTI_ythread_print_stack(p_ythread, fp);
    } else if (type == ABT_UNIT_TYPE_TASK) {
        fprintf(fp, "=== tasklet (%p) ===\n", (void *)unit);
    } else {
        fprintf(fp, "=== unknown (%p) ===\n", (void *)unit);
    }
}

ABTU_ret_err static int info_print_thread_stacks_in_pool(FILE *fp,
                                                         ABTI_pool *p_pool)
{
    ABT_pool pool = ABTI_pool_get_handle(p_pool);

    if (!p_pool->p_print_all) {
        return ABT_ERR_POOL;
    }
    fprintf(fp, "== pool (%p) ==\n", (void *)p_pool);
    struct info_print_unit_arg_t arg;
    arg.fp = fp;
    arg.pool = pool;
    p_pool->p_print_all(pool, &arg, info_print_unit);
    return ABT_SUCCESS;
}

ABTU_ret_err static inline int
info_initialize_pool_set(struct info_pool_set_t *p_set)
{
    size_t default_len = 16;
    int abt_errno =
        ABTU_malloc(sizeof(ABT_pool) * default_len, (void **)&p_set->pools);
    ABTI_CHECK_ERROR(abt_errno);
    p_set->num = 0;
    p_set->len = default_len;
    return ABT_SUCCESS;
}

static inline void info_finalize_pool_set(struct info_pool_set_t *p_set)
{
    ABTU_free(p_set->pools);
}

ABTU_ret_err static inline int info_add_pool_set(ABT_pool pool,
                                                 struct info_pool_set_t *p_set)
{
    size_t i;
    for (i = 0; i < p_set->num; i++) {
        if (p_set->pools[i] == pool)
            return ABT_SUCCESS;
    }
    /* Add pool to p_set. */
    if (p_set->num == p_set->len) {
        size_t new_len = p_set->len * 2;
        int abt_errno =
            ABTU_realloc(sizeof(ABT_pool) * p_set->len,
                         sizeof(ABT_pool) * new_len, (void **)&p_set->pools);
        ABTI_CHECK_ERROR(abt_errno);
        p_set->len = new_len;
    }
    p_set->pools[p_set->num++] = pool;
    return ABT_SUCCESS;
}

static void info_trigger_print_all_thread_stacks(
    FILE *fp, double timeout, void (*cb_func)(ABT_bool, void *), void *arg)
{
    /* This function is signal-safe, so it may not call other functions unless
     * you really know what the called functions do. */
    if (ABTD_atomic_acquire_load_uint32(&print_stack_flag) ==
        PRINT_STACK_FLAG_UNSET) {
        if (ABTD_atomic_bool_cas_strong_uint32(&print_stack_flag,
                                               PRINT_STACK_FLAG_UNSET,
                                               PRINT_STACK_FLAG_INITIALIZE)) {
            /* Save fp and timeout. */
            print_stack_fp = fp;
            print_stack_timeout = timeout;
            print_cb_func = cb_func;
            print_arg = arg;
            /* Here print_stack_barrier must be 0. */
            ABTI_ASSERT(ABTD_atomic_acquire_load_uint32(&print_stack_barrier) ==
                        0);
            ABTD_atomic_release_store_uint32(&print_stack_flag,
                                             PRINT_STACK_FLAG_WAIT);
        }
    }
}

ABTU_ret_err static int print_all_thread_stacks(FILE *fp)
{
    int i, abt_errno;
    struct info_pool_set_t pool_set;

    abt_errno = info_initialize_pool_set(&pool_set);
    ABTI_CHECK_ERROR(abt_errno);
    ABTI_xstream *p_xstream = gp_ABTI_global->p_xstream_head;
    while (p_xstream) {
        ABTI_sched *p_main_sched = p_xstream->p_main_sched;
        fprintf(fp, "= xstream[%d] (%p) =\n", p_xstream->rank,
                (void *)p_xstream);
        fprintf(fp, "main_sched : %p\n", (void *)p_main_sched);
        if (!p_main_sched)
            continue;
        for (i = 0; i < p_main_sched->num_pools; i++) {
            ABT_pool pool = p_main_sched->pools[i];
            ABTI_ASSERT(pool != ABT_POOL_NULL);
            fprintf(fp, "  pools[%d] : %p\n", i,
                    (void *)ABTI_pool_get_ptr(pool));
            abt_errno = info_add_pool_set(pool, &pool_set);
            if (ABTI_IS_ERROR_CHECK_ENABLED && abt_errno != ABT_SUCCESS) {
                info_finalize_pool_set(&pool_set);
                ABTI_HANDLE_ERROR(abt_errno);
            }
        }
        p_xstream = p_xstream->p_next;
    }
    for (i = 0; i < pool_set.num; i++) {
        ABT_pool pool = pool_set.pools[i];
        ABTI_pool *p_pool = ABTI_pool_get_ptr(pool);
        abt_errno = info_print_thread_stacks_in_pool(fp, p_pool);
        if (abt_errno != ABT_SUCCESS)
            fprintf(fp, "  Failed to print (errno = %d).\n", abt_errno);
    }
    info_finalize_pool_set(&pool_set);
    return ABT_SUCCESS;
}

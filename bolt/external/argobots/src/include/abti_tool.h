/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

#ifndef ABTI_TOOL_H_INCLUDED
#define ABTI_TOOL_H_INCLUDED

static inline ABT_thread ABTI_ythread_get_handle(ABTI_ythread *p_thread);
static inline ABT_task ABTI_thread_get_handle(ABTI_thread *p_task);

#ifndef ABT_CONFIG_DISABLE_TOOL_INTERFACE
static inline ABTI_tool_context *
ABTI_tool_context_get_ptr(ABT_tool_context tctx)
{
#ifndef ABT_CONFIG_DISABLE_ERROR_CHECK
    ABTI_tool_context *p_tctx;
    if (tctx == ABT_TOOL_CONTEXT_NULL) {
        p_tctx = NULL;
    } else {
        p_tctx = (ABTI_tool_context *)tctx;
    }
    return p_tctx;
#else
    return (ABTI_tool_context *)tctx;
#endif
}

static inline ABT_tool_context
ABTI_tool_context_get_handle(ABTI_tool_context *p_tctx)
{
#ifndef ABT_CONFIG_DISABLE_ERROR_CHECK
    ABT_tool_context h_tctx;
    if (p_tctx == NULL) {
        h_tctx = ABT_TOOL_CONTEXT_NULL;
    } else {
        h_tctx = (ABT_tool_context)p_tctx;
    }
    return h_tctx;
#else
    return (ABT_tool_context)p_tctx;
#endif
}

#define ABTI_TOOL_EVENT_TAG_SIZE 20 /* bits */
#define ABTI_TOOL_EVENT_TAG_MASK                                               \
    ((((uint64_t)1 << (uint64_t)ABTI_TOOL_EVENT_TAG_SIZE) - 1)                 \
     << (uint64_t)(64 - 1 - ABTI_TOOL_EVENT_TAG_SIZE))
#define ABTI_TOOL_EVENT_TAG_INC                                                \
    ((uint64_t)1 << (uint64_t)(64 - 1 - ABTI_TOOL_EVENT_TAG_SIZE))
#define ABTI_TOOL_EVENT_TAG_DIRTY_BIT ((uint64_t)1 << (uint64_t)(64 - 1))

static inline void
ABTI_tool_event_thread_update_callback(ABT_tool_thread_callback_fn cb_func,
                                       uint64_t event_mask_thread,
                                       void *user_arg)
{
    /* The spinlock is needed to avoid data race between two writers. */
    ABTI_global *p_global = gp_ABTI_global;
    ABTI_spinlock_acquire(&p_global->tool_writer_lock);

    /*
     * This atomic writing process is needed to avoid data race between a reader
     * and a writer.  We need to atomically update three values (callback, event
     * mask, and user_arg) in the following cases:
     *
     * A. ES-W writes the three values while ES-R is reading the three values
     * B. ES-W1 writes and then ES-W2 writes the three values while ES-R is
     *    reading the three values
     *
     * The reader will first read the event mask and then load the other two.
     * The reader then read the event mask again and see if it is 1. the same as
     * the previous and 2. clean.  If both are satisfied, acquire-release memory
     * order guarantees that the loaded values are ones updated by the same
     * ABTI_tool_event_thread_update_callback() call, unless the tag value wraps
     * around (which does not happen practically).
     */

    uint64_t current = ABTD_atomic_acquire_load_uint64(
        &p_global->tool_thread_event_mask_tagged);
    uint64_t new_tag =
        (current + ABTI_TOOL_EVENT_TAG_INC) & ABTI_TOOL_EVENT_TAG_MASK;
    uint64_t new_mask =
        new_tag | (((event_mask_thread & ABT_TOOL_EVENT_THREAD_ALL) |
                    (current & ABT_TOOL_EVENT_TASK_ALL)) &
                   ~ABTI_TOOL_EVENT_TAG_DIRTY_BIT);
    uint64_t dirty_mask = ABTI_TOOL_EVENT_TAG_DIRTY_BIT | new_mask;

    ABTD_atomic_release_store_uint64(&p_global->tool_thread_event_mask_tagged,
                                     dirty_mask);
    p_global->tool_thread_cb_f = cb_func;
    p_global->tool_thread_user_arg = user_arg;
    ABTD_atomic_release_store_uint64(&p_global->tool_thread_event_mask_tagged,
                                     new_mask);

    ABTI_spinlock_release(&p_global->tool_writer_lock);
}

static inline void
ABTI_tool_event_task_update_callback(ABT_tool_task_callback_fn cb_func,
                                     uint64_t event_mask_task, void *user_arg)
{
    ABTI_global *p_global = gp_ABTI_global;
    /* The spinlock is needed to avoid data race between two writers. */
    ABTI_spinlock_acquire(&p_global->tool_writer_lock);

    /* This following writing process is needed to avoid data race between a
     * reader and a writer. */
    uint64_t current = ABTD_atomic_acquire_load_uint64(
        &p_global->tool_thread_event_mask_tagged);
    uint64_t new_tag =
        (current + ABTI_TOOL_EVENT_TAG_INC) & ABTI_TOOL_EVENT_TAG_MASK;
    uint64_t new_mask =
        new_tag | (((event_mask_task & ABT_TOOL_EVENT_TASK_ALL) |
                    (current & ABT_TOOL_EVENT_THREAD_ALL)) &
                   ~ABTI_TOOL_EVENT_TAG_DIRTY_BIT);
    uint64_t dirty_mask = ABTI_TOOL_EVENT_TAG_DIRTY_BIT | new_mask;

    ABTD_atomic_release_store_uint64(&p_global->tool_thread_event_mask_tagged,
                                     dirty_mask);
    p_global->tool_task_cb_f = cb_func;
    p_global->tool_task_user_arg = user_arg;
    ABTD_atomic_release_store_uint64(&p_global->tool_thread_event_mask_tagged,
                                     new_mask);

    ABTI_spinlock_release(&p_global->tool_writer_lock);
}

#endif /* !ABT_CONFIG_DISABLE_TOOL_INTERFACE */

static inline void ABTI_tool_event_thread_impl(
    ABTI_local *p_local, uint64_t event_code, ABTI_thread *p_thread,
    ABTI_thread *p_caller, ABTI_pool *p_pool, ABTI_thread *p_parent,
    ABT_sync_event_type sync_event_type, void *p_sync_object)
{
#ifdef ABT_CONFIG_DISABLE_TOOL_INTERFACE
    return;
#else
    ABTI_ythread *p_ythread = ABTI_thread_get_ythread_or_null(p_thread);
    if (!p_ythread) {
        /* Use an event code for a tasklet-type thread. */
        event_code *= ABT_TOOL_EVENT_TASK_CREATE;
    }
    ABTI_global *p_global = gp_ABTI_global;
    while (1) {
        uint64_t current_mask = ABTD_atomic_acquire_load_uint64(
            &p_global->tool_thread_event_mask_tagged);
        if (current_mask & event_code) {
            ABT_tool_thread_callback_fn cb_func_thread =
                p_global->tool_thread_cb_f;
            ABT_tool_task_callback_fn cb_func_task = p_global->tool_task_cb_f;
            void *user_arg_thread = p_global->tool_thread_user_arg;
            void *user_arg_task = p_global->tool_task_user_arg;
            /* Double check the current event mask. */
            uint64_t current_mask2 = ABTD_atomic_acquire_load_uint64(
                &p_global->tool_thread_event_mask_tagged);
            if (ABTU_unlikely(current_mask != current_mask2 ||
                              (current_mask & ABTI_TOOL_EVENT_TAG_DIRTY_BIT)))
                continue;
            ABTI_tool_context tctx;
            tctx.p_pool = p_pool;
            tctx.p_parent = p_parent;
            tctx.p_caller = p_caller;
            tctx.sync_event_type = sync_event_type;
            tctx.p_sync_object = p_sync_object;

            ABTI_xstream *p_local_xstream =
                ABTI_local_get_xstream_or_null(p_local);
            ABT_xstream h_xstream =
                p_local_xstream ? ABTI_xstream_get_handle(p_local_xstream)
                                : ABT_XSTREAM_NULL;
            if (p_ythread) {
                if (p_ythread->thread.type & ABTI_THREAD_TYPE_ROOT) {
                    /* Root thread should not be visible to users. */
                    return;
                }
                ABT_thread h_thread = ABTI_ythread_get_handle(p_ythread);
                ABT_tool_context h_tctx = ABTI_tool_context_get_handle(&tctx);
                cb_func_thread(h_thread, h_xstream, event_code, h_tctx,
                               user_arg_thread);
            } else {
                ABT_task h_task = ABTI_thread_get_handle(p_thread);
                ABT_tool_context h_tctx = ABTI_tool_context_get_handle(&tctx);
                cb_func_task(h_task, h_xstream, event_code, h_tctx,
                             user_arg_task);
            }
        }
        return;
    }
#endif /* !ABT_CONFIG_DISABLE_TOOL_INTERFACE */
}

static inline void ABTI_tool_event_thread_create_impl(ABTI_local *p_local,
                                                      ABTI_thread *p_thread,
                                                      ABTI_thread *p_caller,
                                                      ABTI_pool *p_pool)
{
    ABTI_tool_event_thread_impl(p_local, ABT_TOOL_EVENT_THREAD_CREATE, p_thread,
                                p_caller, p_pool, NULL,
                                ABT_SYNC_EVENT_TYPE_UNKNOWN, NULL);
}

static inline void ABTI_tool_event_thread_join_impl(ABTI_local *p_local,
                                                    ABTI_thread *p_thread,
                                                    ABTI_thread *p_caller)
{
    ABTI_tool_event_thread_impl(p_local, ABT_TOOL_EVENT_THREAD_JOIN, p_thread,
                                p_caller, NULL, NULL,
                                ABT_SYNC_EVENT_TYPE_UNKNOWN, NULL);
}

static inline void ABTI_tool_event_thread_free_impl(ABTI_local *p_local,
                                                    ABTI_thread *p_thread,
                                                    ABTI_thread *p_caller)
{
    ABTI_tool_event_thread_impl(p_local, ABT_TOOL_EVENT_THREAD_FREE, p_thread,
                                p_caller, NULL, NULL,
                                ABT_SYNC_EVENT_TYPE_UNKNOWN, NULL);
}

static inline void ABTI_tool_event_thread_revive_impl(ABTI_local *p_local,
                                                      ABTI_thread *p_thread,
                                                      ABTI_thread *p_caller,
                                                      ABTI_pool *p_pool)
{
    ABTI_tool_event_thread_impl(p_local, ABT_TOOL_EVENT_THREAD_REVIVE, p_thread,
                                p_caller, p_pool, NULL,
                                ABT_SYNC_EVENT_TYPE_UNKNOWN, NULL);
}

static inline void
ABTI_tool_event_thread_run_impl(ABTI_xstream *p_local_xstream,
                                ABTI_thread *p_thread, ABTI_thread *p_prev,
                                ABTI_thread *p_parent)
{
    ABTI_tool_event_thread_impl(ABTI_xstream_get_local(p_local_xstream),
                                ABT_TOOL_EVENT_THREAD_RUN, p_thread, p_prev,
                                NULL, p_parent, ABT_SYNC_EVENT_TYPE_UNKNOWN,
                                NULL);
}

static inline void
ABTI_tool_event_thread_finish_impl(ABTI_xstream *p_local_xstream,
                                   ABTI_thread *p_thread, ABTI_thread *p_parent)
{
    ABTI_tool_event_thread_impl(ABTI_xstream_get_local(p_local_xstream),
                                ABT_TOOL_EVENT_THREAD_FINISH, p_thread, NULL,
                                NULL, p_parent, ABT_SYNC_EVENT_TYPE_UNKNOWN,
                                NULL);
}

static inline void
ABTI_tool_event_thread_cancel_impl(ABTI_xstream *p_local_xstream,
                                   ABTI_thread *p_thread)
{
    ABTI_tool_event_thread_impl(ABTI_xstream_get_local(p_local_xstream),
                                ABT_TOOL_EVENT_THREAD_CANCEL, p_thread, NULL,
                                NULL, NULL, ABT_SYNC_EVENT_TYPE_UNKNOWN, NULL);
}

static inline void ABTI_tool_event_ythread_yield_impl(
    ABTI_xstream *p_local_xstream, ABTI_ythread *p_ythread,
    ABTI_thread *p_parent, ABT_sync_event_type sync_event_type, void *p_sync)
{
    if (ABTD_atomic_relaxed_load_uint32(&p_ythread->thread.request) &
        ABTI_THREAD_REQ_BLOCK) {
        ABTI_tool_event_thread_impl(ABTI_xstream_get_local(p_local_xstream),
                                    ABT_TOOL_EVENT_THREAD_SUSPEND,
                                    &p_ythread->thread, NULL,
                                    p_ythread->thread.p_pool, p_parent,
                                    sync_event_type, p_sync);

    } else {
        ABTI_tool_event_thread_impl(ABTI_xstream_get_local(p_local_xstream),
                                    ABT_TOOL_EVENT_THREAD_YIELD,
                                    &p_ythread->thread, NULL,
                                    p_ythread->thread.p_pool, p_parent,
                                    sync_event_type, p_sync);
    }
}

static inline void ABTI_tool_event_ythread_suspend_impl(
    ABTI_xstream *p_local_xstream, ABTI_ythread *p_ythread,
    ABTI_thread *p_parent, ABT_sync_event_type sync_event_type, void *p_sync)
{
    ABTI_tool_event_thread_impl(ABTI_xstream_get_local(p_local_xstream),
                                ABT_TOOL_EVENT_THREAD_SUSPEND,
                                &p_ythread->thread, NULL,
                                p_ythread->thread.p_pool, p_parent,
                                sync_event_type, p_sync);
}

static inline void ABTI_tool_event_ythread_resume_impl(ABTI_local *p_local,
                                                       ABTI_ythread *p_ythread,
                                                       ABTI_thread *p_caller)
{
    ABTI_tool_event_thread_impl(p_local, ABT_TOOL_EVENT_THREAD_RESUME,
                                &p_ythread->thread, p_caller,
                                p_ythread->thread.p_pool, NULL,
                                ABT_SYNC_EVENT_TYPE_UNKNOWN, NULL);
}

#ifndef ABT_CONFIG_DISABLE_TOOL_INTERFACE
#define ABTI_USE_TOOL_INTERFACE 1
#else
#define ABTI_USE_TOOL_INTERFACE 0
#endif

#define ABTI_tool_event_thread_create(p_local, p_thread, p_caller, p_pool)     \
    do {                                                                       \
        if (ABTI_USE_TOOL_INTERFACE) {                                         \
            ABTI_tool_event_thread_create_impl(p_local, p_thread, p_caller,    \
                                               p_pool);                        \
        }                                                                      \
    } while (0)

#define ABTI_tool_event_thread_join(p_local, p_thread, p_caller)               \
    do {                                                                       \
        if (ABTI_USE_TOOL_INTERFACE) {                                         \
            ABTI_tool_event_thread_join_impl(p_local, p_thread, p_caller);     \
        }                                                                      \
    } while (0)

#define ABTI_tool_event_thread_free(p_local, p_thread, p_caller)               \
    do {                                                                       \
        if (ABTI_USE_TOOL_INTERFACE) {                                         \
            ABTI_tool_event_thread_free_impl(p_local, p_thread, p_caller);     \
        }                                                                      \
    } while (0)

#define ABTI_tool_event_thread_revive(p_local, p_thread, p_caller, p_pool)     \
    do {                                                                       \
        if (ABTI_USE_TOOL_INTERFACE) {                                         \
            ABTI_tool_event_thread_revive_impl(p_local, p_thread, p_caller,    \
                                               p_pool);                        \
        }                                                                      \
    } while (0)

#define ABTI_tool_event_thread_run(p_local_xstream, p_thread, p_prev,          \
                                   p_parent)                                   \
    do {                                                                       \
        if (ABTI_USE_TOOL_INTERFACE) {                                         \
            ABTI_tool_event_thread_run_impl(p_local_xstream, p_thread, p_prev, \
                                            p_parent);                         \
        }                                                                      \
    } while (0)

#define ABTI_tool_event_thread_finish(p_local_xstream, p_thread, p_parent)     \
    do {                                                                       \
        if (ABTI_USE_TOOL_INTERFACE) {                                         \
            ABTI_tool_event_thread_finish_impl(p_local_xstream, p_thread,      \
                                               p_parent);                      \
        }                                                                      \
    } while (0)

#define ABTI_tool_event_thread_cancel(p_local_xstream, p_thread)               \
    do {                                                                       \
        if (ABTI_USE_TOOL_INTERFACE) {                                         \
            ABTI_tool_event_thread_cancel_impl(p_local_xstream, p_thread);     \
        }                                                                      \
    } while (0)

#define ABTI_tool_event_ythread_yield(p_local_xstream, p_ythread, p_parent,    \
                                      sync_event_type, p_sync)                 \
    do {                                                                       \
        if (ABTI_USE_TOOL_INTERFACE) {                                         \
            ABTI_tool_event_ythread_yield_impl(p_local_xstream, p_ythread,     \
                                               p_parent, sync_event_type,      \
                                               p_sync);                        \
        }                                                                      \
    } while (0)

#define ABTI_tool_event_ythread_suspend(p_local_xstream, p_ythread, p_parent,  \
                                        sync_event_type, p_sync)               \
    do {                                                                       \
        if (ABTI_USE_TOOL_INTERFACE) {                                         \
            ABTI_tool_event_ythread_suspend_impl(p_local_xstream, p_ythread,   \
                                                 p_parent, sync_event_type,    \
                                                 p_sync);                      \
        }                                                                      \
    } while (0)

#define ABTI_tool_event_ythread_resume(p_local, p_ythread, p_caller)           \
    do {                                                                       \
        if (ABTI_USE_TOOL_INTERFACE) {                                         \
            ABTI_tool_event_ythread_resume_impl(p_local, p_ythread, p_caller); \
        }                                                                      \
    } while (0)

#endif /* ABTI_TOOL_H_INCLUDED */

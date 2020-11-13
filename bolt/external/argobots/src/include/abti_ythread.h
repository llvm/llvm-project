/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

#ifndef ABTI_YTHREAD_H_INCLUDED
#define ABTI_YTHREAD_H_INCLUDED

/* Inlined functions for yieldable threads */

static inline ABTI_ythread *ABTI_ythread_get_ptr(ABT_thread thread)
{
#ifndef ABT_CONFIG_DISABLE_ERROR_CHECK
    ABTI_ythread *p_ythread;
    if (thread == ABT_THREAD_NULL) {
        p_ythread = NULL;
    } else {
        p_ythread = (ABTI_ythread *)thread;
    }
    return p_ythread;
#else
    return (ABTI_ythread *)thread;
#endif
}

static inline ABT_thread ABTI_ythread_get_handle(ABTI_ythread *p_ythread)
{
#ifndef ABT_CONFIG_DISABLE_ERROR_CHECK
    ABT_thread h_thread;
    if (p_ythread == NULL) {
        h_thread = ABT_THREAD_NULL;
    } else {
        h_thread = (ABT_thread)p_ythread;
    }
    return h_thread;
#else
    return (ABT_thread)p_ythread;
#endif
}

static inline ABTI_ythread *
ABTI_ythread_context_get_ythread(ABTD_ythread_context *p_ctx)
{
    return (ABTI_ythread *)(((char *)p_ctx) - offsetof(ABTI_ythread, ctx));
}

#if ABT_CONFIG_THREAD_TYPE == ABT_THREAD_TYPE_DYNAMIC_PROMOTION
static inline ABT_bool ABTI_ythread_is_dynamic_promoted(ABTI_ythread *p_ythread)
{
    /*
     * Create a context and switch to it. The flow of the dynamic promotion
     * thread is as follows:
     *
     * - When a ULT does not yield:
     *  ABTI_xstream_schedule_ythread : call init_and_call_fcontext
     *  init_and_call_fcontext        : jump to the stack top
     *                                : save the scheduler's context
     *                                : call the thread function
     *  thread_f                      : start thread_f
     *                                : [ULT body]
     *                                : end thread_f
     *  init_and_call_fcontext        : calculate the return address, which is
     *  [1] =>                          the original return address.
     *                                : `return`
     *  ABTI_xstream_schedule_ythread : resume the scheduler
     *
     * The ULT can return to the original scheduler by `return` if the scheduler
     * has never been resumed during the execution of the ULT, because the
     * context of the parent scheduler can be restored in a normal return
     * procedure. In this case, the context is saved only once.
     *
     * - When a ULT yields:
     *  ABTI_xstream_schedule_ythread : call init_and_call_fcontext
     *  init_and_call_fcontext        : jump to the stack top
     *                                : save the scheduler's context
     *                                : call the thread function
     *  thread_f                      : start thread_f
     *                                : [yield in ULT body]
     *  ABTD_ythread_context_dynamic_promote
     *                                : rewrite the return address to
     *                                  ABTD_ythread_terminate_ythread_no_arg
     *  jump_fcontext                 : save the ULT's context
     *                                : restore the scheduler's context
     *                                : jump to the scheduler
     *  ABTI_xstream_schedule_ythread : resume the scheduler
     *
     *            ... After a while, a scheduler resumes this ULT ...
     *
     *  jump_fcontext                 : save the scheduler's context
     *                                : restore the ULT's context
     *                                : jump to the ULT
     *  thread_f                      : [ULT body (remaining)]
     *                                : end thread_f
     *  init_and_call_fcontext        : calculate the return address, which is
     *  [2] =>                          ABTD_ythread_terminate_ythread_no_arg
     *                                : return
     *  ABTD_ythread_terminate_ythread_no_arg
     *                                : call take_fcontext
     *  take_fcontext                 : restore the scheduler's context
     *                                : jump to the scheduler
     *  ABTI_xstream_schedule_ythread : resume the scheduler
     *
     * When a ULT yields, ABTD_ythread_terminate_ythread_no_arg is set to
     * [ptr - 0x08] so that it can "jump" to the normal termination
     * function by "return" in init_and_call_fcontext. This termination
     * function calls take_fcontext, so the scheduler is resumed by user-level
     * context switch.
     *
     * For example, the stack will be as follows at [1] and [2] in the x86-64
     * case. Note that ptr points to the stack top (= p_stack + stacksize).
     *
     * In the case of [1] (no suspension):
     *  [0x12345600] : (the original instruction pointer)
     *   ...
     *  [ptr - 0x08] : the original stack pointer (i.e., 0x12345600)
     *  [ptr - 0x10] : unused (for 16-byte alignment)
     *  [ptr - xxxx] : used by thread_f
     *
     * In the case of [2] (after suspension):
     *  [ptr - 0x08] : pointing to (p_stack - 0x10)
     *  [ptr - 0x10] : the address of ABTD_ythread_terminate_ythread_no_arg
     *  [ptr - xxxx] : used by thread_f
     *
     * This technique was introduced as a "return-on-completion" thread in the
     * following paper:
     *   Lessons Learned from Analyzing Dynamic Promotion for User-Level
     *   Threading, S. Iwasaki, A. Amer, K. Taura, and P. Balaji (SC '18)
     */
    return ABTD_ythread_context_is_dynamic_promoted(&p_ythread->ctx);
}

static inline void ABTI_ythread_dynamic_promote_ythread(ABTI_ythread *p_ythread)
{
    LOG_DEBUG("[U%" PRIu64 "] dynamic-promote ULT\n",
              ABTI_thread_get_id(&p_ythread->thread));
    void *p_stack = p_ythread->p_stack;
    size_t stacksize = p_ythread->stacksize;
    void *p_stacktop = (void *)(((char *)p_stack) + stacksize);
    ABTD_ythread_context_dynamic_promote_ythread(p_stacktop);
}
#endif

static inline ABTI_ythread *ABTI_ythread_context_switch_to_sibling_internal(
    ABTI_xstream **pp_local_xstream, ABTI_ythread *p_old, ABTI_ythread *p_new,
    ABT_bool is_finish)
{
#if ABT_CONFIG_THREAD_TYPE == ABT_THREAD_TYPE_DYNAMIC_PROMOTION
    /* Dynamic promotion is unnecessary if p_old will be discarded. */
    if (!ABTI_ythread_is_dynamic_promoted(p_old)) {
        ABTI_ythread_dynamic_promote_ythread(p_old);
    }
    if (!ABTI_ythread_is_dynamic_promoted(p_new)) {
        /* p_new does not have a context, so we first need to make it. */
        ABTD_ythread_context_arm_ythread(p_new->stacksize, p_new->p_stack,
                                         &p_new->ctx);
    }
#endif
    p_new->thread.p_parent = p_old->thread.p_parent;
    if (is_finish) {
        ABTI_tool_event_thread_finish(*pp_local_xstream, &p_old->thread,
                                      p_old->thread.p_parent);
        ABTD_ythread_finish_context(&p_old->ctx, &p_new->ctx);
        ABTU_unreachable();
    } else {
        ABTD_ythread_context_switch(&p_old->ctx, &p_new->ctx);
        ABTI_local *p_local = ABTI_local_get_local_uninlined();
        ABTI_xstream *p_local_xstream = ABTI_local_get_xstream(p_local);
        *pp_local_xstream = p_local_xstream;
        ABTI_thread *p_prev = p_local_xstream->p_thread;
        p_local_xstream->p_thread = &p_old->thread;
        ABTI_ASSERT(p_prev->type & ABTI_THREAD_TYPE_YIELDABLE);
        return ABTI_thread_get_ythread(p_prev);
    }
}

static inline ABTI_ythread *ABTI_ythread_context_switch_to_parent_internal(
    ABTI_xstream **pp_local_xstream, ABTI_ythread *p_old, ABT_bool is_finish,
    ABT_sync_event_type sync_event_type, void *p_sync)
{
    ABTI_ythread *p_new = ABTI_thread_get_ythread(p_old->thread.p_parent);
#if ABT_CONFIG_THREAD_TYPE == ABT_THREAD_TYPE_DYNAMIC_PROMOTION
    /* Dynamic promotion is unnecessary if p_old will be discarded. */
    if (!is_finish && !ABTI_ythread_is_dynamic_promoted(p_old))
        ABTI_ythread_dynamic_promote_ythread(p_old);
    /* The parent's context must have been eagerly initialized. */
    ABTI_ASSERT(ABTI_ythread_is_dynamic_promoted(p_new));
#endif
    if (is_finish) {
        ABTI_tool_event_thread_finish(*pp_local_xstream, &p_old->thread,
                                      p_old->thread.p_parent);
        ABTD_ythread_finish_context(&p_old->ctx, &p_new->ctx);
        ABTU_unreachable();
    } else {
        ABTI_tool_event_ythread_yield(*pp_local_xstream, p_old,
                                      p_old->thread.p_parent, sync_event_type,
                                      p_sync);
        ABTD_ythread_context_switch(&p_old->ctx, &p_new->ctx);
        ABTI_local *p_local = ABTI_local_get_local_uninlined();
        ABTI_xstream *p_local_xstream = ABTI_local_get_xstream(p_local);
        *pp_local_xstream = p_local_xstream;
        ABTI_thread *p_prev = p_local_xstream->p_thread;
        p_local_xstream->p_thread = &p_old->thread;
        ABTI_ASSERT(p_prev->type & ABTI_THREAD_TYPE_YIELDABLE);
        /* Invoke an event of thread run. */
        ABTI_tool_event_thread_run(p_local_xstream, &p_old->thread, p_prev,
                                   p_old->thread.p_parent);
        return ABTI_thread_get_ythread(p_prev);
    }
}

static inline ABTI_ythread *ABTI_ythread_context_switch_to_child_internal(
    ABTI_xstream **pp_local_xstream, ABTI_ythread *p_old, ABTI_ythread *p_new)
{
    ABTI_xstream *p_local_xstream;
    p_new->thread.p_parent = &p_old->thread;
#if ABT_CONFIG_THREAD_TYPE == ABT_THREAD_TYPE_DYNAMIC_PROMOTION
    if (!ABTI_ythread_is_dynamic_promoted(p_old)) {
        ABTI_ythread_dynamic_promote_ythread(p_old);
    }
    if (!ABTI_ythread_is_dynamic_promoted(p_new)) {
        void *p_stacktop = ((char *)p_new->p_stack) + p_new->stacksize;
        LOG_DEBUG("[U%" PRIu64 "] run ULT (dynamic promotion)\n",
                  ABTI_thread_get_id(&p_new->thread));
        p_local_xstream = *pp_local_xstream;
        p_local_xstream->p_thread = &p_new->thread;
        /* Invoke an event of thread run. */
        ABTI_tool_event_thread_run(p_local_xstream, &p_new->thread,
                                   &p_old->thread, &p_old->thread);
        ABTD_ythread_context_make_and_call(&p_old->ctx, p_new->thread.f_thread,
                                           p_new->thread.p_arg, p_stacktop);
        /* The scheduler continues from here. If the previous thread has not
         * run dynamic promotion, ABTI_ythread_context_make_and_call took the
         * fast path. In this case, the request handling has not been done,
         * so it must be done here. */
        ABTI_local *p_local = ABTI_local_get_local_uninlined();
        p_local_xstream = ABTI_local_get_xstream(p_local);
        *pp_local_xstream = p_local_xstream;
        ABTI_thread *p_prev_thread = p_local_xstream->p_thread;
        ABTI_ASSERT(p_prev_thread->type & ABTI_THREAD_TYPE_YIELDABLE);
        ABTI_ythread *p_prev = ABTI_thread_get_ythread(p_prev_thread);
        p_local_xstream->p_thread = &p_old->thread;
        if (!ABTI_ythread_is_dynamic_promoted(p_prev)) {
            ABTI_ASSERT(p_prev == p_new);
            /* Invoke a thread-finish event of the previous thread. */
            ABTI_tool_event_thread_finish(p_local_xstream, &p_prev->thread,
                                          &p_old->thread);
            /* See ABTDI_thread_terminate for details.
             * TODO: avoid making a copy of the code. */
            ABTD_ythread_context *p_ctx = &p_prev->ctx;
            ABTD_ythread_context *p_link =
                ABTD_atomic_acquire_load_ythread_context_ptr(&p_ctx->p_link);
            if (p_link) {
                /* If p_link is set, it means that other ULT has called the
                 * join. */
                ABTI_ythread *p_joiner =
                    ABTI_ythread_context_get_ythread(p_link);
                /* The scheduler may not use a bypass mechanism, so just makes
                 * p_joiner ready. */
                ABTI_ythread_set_ready(ABTI_xstream_get_local(p_local_xstream),
                                       p_joiner);

                /* We don't need to use the atomic OR operation here because
                 * the ULT will be terminated regardless of other requests. */
                ABTD_atomic_release_store_uint32(&p_prev->thread.request,
                                                 ABTI_THREAD_REQ_TERMINATE);
            } else {
                uint32_t req =
                    ABTD_atomic_fetch_or_uint32(&p_prev->thread.request,
                                                ABTI_THREAD_REQ_JOIN |
                                                    ABTI_THREAD_REQ_TERMINATE);
                if (req & ABTI_THREAD_REQ_JOIN) {
                    /* This case means there has been a join request and the
                     * joiner has blocked.  We have to wake up the joiner ULT.
                     */
                    do {
                        p_link = ABTD_atomic_acquire_load_ythread_context_ptr(
                            &p_ctx->p_link);
                    } while (!p_link);
                    ABTI_ythread_set_ready(ABTI_xstream_get_local(
                                               p_local_xstream),
                                           ABTI_ythread_context_get_ythread(
                                               p_link));
                }
            }
        }
        return p_prev;
    }
#endif
    {
        ABTD_ythread_context_switch(&p_old->ctx, &p_new->ctx);
        ABTI_local *p_local = ABTI_local_get_local_uninlined();
        p_local_xstream = ABTI_local_get_xstream(p_local);
        *pp_local_xstream = p_local_xstream;
        ABTI_thread *p_prev = p_local_xstream->p_thread;
        p_local_xstream->p_thread = &p_old->thread;
        ABTI_ASSERT(p_prev->type & ABTI_THREAD_TYPE_YIELDABLE);
        /* p_old keeps running as a parent, so no thread-run event incurs. */
        return ABTI_thread_get_ythread(p_prev);
    }
}

/* Return the previous thread. */
static inline ABTI_ythread *
ABTI_ythread_context_switch_to_sibling(ABTI_xstream **pp_local_xstream,
                                       ABTI_ythread *p_old, ABTI_ythread *p_new)
{
    return ABTI_ythread_context_switch_to_sibling_internal(pp_local_xstream,
                                                           p_old, p_new,
                                                           ABT_FALSE);
}

static inline ABTI_ythread *ABTI_ythread_context_switch_to_parent(
    ABTI_xstream **pp_local_xstream, ABTI_ythread *p_old,
    ABT_sync_event_type sync_event_type, void *p_sync)
{
    return ABTI_ythread_context_switch_to_parent_internal(pp_local_xstream,
                                                          p_old, ABT_FALSE,
                                                          sync_event_type,
                                                          p_sync);
}

static inline ABTI_ythread *
ABTI_ythread_context_switch_to_child(ABTI_xstream **pp_local_xstream,
                                     ABTI_ythread *p_old, ABTI_ythread *p_new)
{
    return ABTI_ythread_context_switch_to_child_internal(pp_local_xstream,
                                                         p_old, p_new);
}

ABTU_noreturn static inline void
ABTI_ythread_finish_context_to_sibling(ABTI_xstream *p_local_xstream,
                                       ABTI_ythread *p_old, ABTI_ythread *p_new)
{
    ABTI_ythread_context_switch_to_sibling_internal(&p_local_xstream, p_old,
                                                    p_new, ABT_TRUE);
    ABTU_unreachable();
}

ABTU_noreturn static inline void
ABTI_ythread_finish_context_to_parent(ABTI_xstream *p_local_xstream,
                                      ABTI_ythread *p_old)
{
    ABTI_ythread_context_switch_to_parent_internal(&p_local_xstream, p_old,
                                                   ABT_TRUE,
                                                   ABT_SYNC_EVENT_TYPE_UNKNOWN,
                                                   NULL);
    ABTU_unreachable();
}

static inline void ABTI_ythread_yield(ABTI_xstream **pp_local_xstream,
                                      ABTI_ythread *p_ythread,
                                      ABT_sync_event_type sync_event_type,
                                      void *p_sync)
{
    LOG_DEBUG("[U%" PRIu64 ":E%d] yield\n",
              ABTI_thread_get_id(&p_ythread->thread),
              p_ythread->thread.p_last_xstream->rank);

    /* Change the state of current running thread */
    ABTD_atomic_release_store_int(&p_ythread->thread.state,
                                  ABT_THREAD_STATE_READY);

    /* Switch to the top scheduler */
    ABTI_ythread_context_switch_to_parent(pp_local_xstream, p_ythread,
                                          sync_event_type, p_sync);

    /* Back to the original thread */
    LOG_DEBUG("[U%" PRIu64 ":E%d] resume after yield\n",
              ABTI_thread_get_id(&p_ythread->thread),
              p_ythread->thread.p_last_xstream->rank);
}

#endif /* ABTI_YTHREAD_H_INCLUDED */

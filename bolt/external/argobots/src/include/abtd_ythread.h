/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

#ifndef ABTD_YTHREAD_H_INCLUDED
#define ABTD_YTHREAD_H_INCLUDED

#if defined(ABT_C_HAVE_VISIBILITY)
#define ABT_API_PRIVATE __attribute__((visibility("hidden")))
#else
#define ABT_API_PRIVATE
#endif

void ABTD_ythread_func_wrapper(void *p_arg);
#if ABT_CONFIG_THREAD_TYPE == ABT_THREAD_TYPE_DYNAMIC_PROMOTION
void ABTD_ythread_terminate_no_arg(void);
#endif

static inline void ABTD_ythread_context_create(ABTD_ythread_context *p_link,
                                               size_t stacksize, void *p_stack,
                                               ABTD_ythread_context *p_newctx)
{
    /* ABTD_ythread_context_make uses the top address of stack.
       Note that the parameter, p_stack, points to the bottom of stack. */
    void *p_stacktop = (void *)(((char *)p_stack) + stacksize);

    ABTD_ythread_context_make(p_newctx, p_stacktop, stacksize,
                              ABTD_ythread_func_wrapper);
    ABTD_atomic_relaxed_store_ythread_context_ptr(&p_newctx->p_link, p_link);
}

static inline void
ABTD_ythread_context_invalidate(ABTD_ythread_context *p_newctx)
{
#if ABT_CONFIG_THREAD_TYPE == ABT_THREAD_TYPE_DYNAMIC_PROMOTION
    /* p_ctx is used to check whether the context requires dynamic promotion is
     * necessary or not, so this value must not be NULL. */
    p_newctx->p_ctx = (void *)((intptr_t)0x1);
#else
    p_newctx->p_ctx = NULL;
#endif
    ABTD_atomic_relaxed_store_ythread_context_ptr(&p_newctx->p_link, NULL);
}

#if ABT_CONFIG_THREAD_TYPE == ABT_THREAD_TYPE_DYNAMIC_PROMOTION
static inline void ABTD_ythread_context_init(ABTD_ythread_context *p_link,
                                             ABTD_ythread_context *p_newctx)
{
    p_newctx->p_ctx = NULL;
    ABTD_atomic_relaxed_store_ythread_context_ptr(&p_newctx->p_link, p_link);
}

static inline void
ABTD_ythread_context_arm_ythread(size_t stacksize, void *p_stack,
                                 ABTD_ythread_context *p_newctx)
{
    /*
     * This function *arms* the dynamic promotion thread (initialized by
     * ABTD_ythread_context_init) as if it were created by
     * ABTD_ythread_context_create; this function fully creates the context
     * so that the thread can be run by ABTD_ythread_context_jump.
     *
     * ABTD_ythread_context_make uses the top address of stack.
     * Note that the parameter, p_stack, points to the bottom of stack.
     */
    void *p_stacktop = (void *)(((char *)p_stack) + stacksize);
    ABTD_ythread_context_make(p_newctx, p_stacktop, stacksize,
                              ABTD_ythread_func_wrapper);
}
#endif

static inline void ABTD_ythread_context_switch(ABTD_ythread_context *p_old,
                                               ABTD_ythread_context *p_new)
{
    ABTD_ythread_context_jump(p_old, p_new, p_new);
}

ABTU_noreturn static inline void
ABTD_ythread_finish_context(ABTD_ythread_context *p_old,
                            ABTD_ythread_context *p_new)
{
    ABTD_ythread_context_take(p_old, p_new, p_new);
}

#if ABT_CONFIG_THREAD_TYPE == ABT_THREAD_TYPE_DYNAMIC_PROMOTION
static inline void
ABTD_ythread_context_make_and_call(ABTD_ythread_context *p_old,
                                   void (*f_thread)(void *), void *p_arg,
                                   void *p_stacktop)
{
    ABTD_ythread_context_init_and_call(p_old, p_stacktop, f_thread, p_arg);
}

static inline ABT_bool
ABTD_ythread_context_is_dynamic_promoted(ABTD_ythread_context *p_ctx)
{
    /* Check if the ULT has been dynamically promoted; internally, it checks if
     * the context is NULL. */
    return p_ctx->p_ctx ? ABT_TRUE : ABT_FALSE;
}

static inline void ABTDI_ythread_context_dynamic_promote(void *p_stacktop,
                                                         void *jump_f)
{
    /* Perform dynamic promotion */
    void **p_return_address = (void **)(((char *)p_stacktop) - 0x10);
    void ***p_stack_pointer = (void ***)(((char *)p_stacktop) - 0x08);
    *p_stack_pointer = p_return_address;
    *p_return_address = jump_f;
}

static inline void
ABTD_ythread_context_dynamic_promote_ythread(void *p_stacktop)
{
    union fp_conv {
        void (*f)(void);
        void *ptr;
    } conv;
    conv.f = ABTD_ythread_terminate_no_arg;
    void *jump_f = conv.ptr;
    ABTDI_ythread_context_dynamic_promote(p_stacktop, jump_f);
}
#endif

#endif /* ABTD_YTHREAD_H_INCLUDED */

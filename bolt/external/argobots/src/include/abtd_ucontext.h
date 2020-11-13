/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

#ifndef ABTD_UCONTEXT_H_INCLUDED
#define ABTD_UCONTEXT_H_INCLUDED

static void ABTD_ucontext_wrapper(int arg1, int arg2)
{
    ABTD_ythread_context *p_self;
#if SIZEOF_VOID_P == 8
    p_self = (ABTD_ythread_context *)(((uintptr_t)((uint32_t)arg1) << 32) |
                                      ((uintptr_t)((uint32_t)arg2)));
#elif SIZEOF_VOID_P == 4
    p_self = (ABTD_ythread_context *)((uintptr_t)arg1);
#else
#error "Unknown pointer size."
#endif
    p_self->f_uctx_thread(p_self->p_uctx_arg);
    /* ABTD_ythread_context_jump or take must be called at the end of
     * f_uctx_thread, */
    ABTI_ASSERT(0);
    ABTU_unreachable();
}

static inline void ABTD_ythread_context_make(ABTD_ythread_context *p_ctx,
                                             void *sp, size_t size,
                                             void (*thread_func)(void *))
{
    int ret = getcontext(&p_ctx->uctx);
    ABTI_ASSERT(ret == 0); /* getcontext() should not return an error. */
    p_ctx->p_ctx = &p_ctx->uctx;

    /* uc_link is not used. */
    p_ctx->uctx.uc_link = NULL;
    p_ctx->uctx.uc_stack.ss_sp = (void *)(((char *)sp) - size);
    p_ctx->uctx.uc_stack.ss_size = size;
    p_ctx->f_uctx_thread = thread_func;

#if SIZEOF_VOID_P == 8
    int arg_upper = (int)(((uintptr_t)p_ctx) >> 32);
    int arg_lower = (int)(((uintptr_t)p_ctx) >> 0);
    makecontext(&p_ctx->uctx, (void (*)())ABTD_ucontext_wrapper, 2, arg_upper,
                arg_lower);
#elif SIZEOF_VOID_P == 4
    int arg = (int)((uintptr_t)p_ctx);
    makecontext(&p_ctx->uctx, (void (*)())ABTD_ucontext_wrapper, 1, arg);
#else
#error "Unknown pointer size."
#endif
}

static inline void ABTD_ythread_context_jump(ABTD_ythread_context *p_old,
                                             ABTD_ythread_context *p_new,
                                             void *arg)
{
    p_new->p_uctx_arg = arg;
    int ret = swapcontext(&p_old->uctx, &p_new->uctx);
    /* Fatal.  This out-of-stack error is not recoverable. */
    ABTI_ASSERT(ret == 0);
}

ABTU_noreturn static inline void
ABTD_ythread_context_take(ABTD_ythread_context *p_old,
                          ABTD_ythread_context *p_new, void *arg)
{
    p_new->p_uctx_arg = arg;
    int ret = setcontext(&p_new->uctx);
    ABTI_ASSERT(ret == 0); /* setcontext() should not return an error. */
    ABTU_unreachable();
}

#if ABT_CONFIG_THREAD_TYPE == ABT_THREAD_TYPE_DYNAMIC_PROMOTION
#error "ABTD_ythread_context_make_and_call is not implemented."
#endif

#endif /* ABTD_UCONTEXT_H_INCLUDED */

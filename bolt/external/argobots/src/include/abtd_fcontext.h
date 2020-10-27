/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

#ifndef ABTD_FCONTEXT_H_INCLUDED
#define ABTD_FCONTEXT_H_INCLUDED

typedef void *fcontext_t;

#if defined(ABT_C_HAVE_VISIBILITY)
#define ABT_API_PRIVATE __attribute__((visibility("hidden")))
#else
#define ABT_API_PRIVATE
#endif

fcontext_t make_fcontext(void *sp, size_t size,
                         void (*thread_func)(void *)) ABT_API_PRIVATE;
void *jump_fcontext(fcontext_t *old, fcontext_t new, void *arg) ABT_API_PRIVATE;
void *take_fcontext(fcontext_t *old, fcontext_t new, void *arg) ABT_API_PRIVATE;
#if ABT_CONFIG_THREAD_TYPE == ABT_THREAD_TYPE_DYNAMIC_PROMOTION
void init_and_call_fcontext(void *p_arg, void (*f_thread)(void *),
                            void *p_stacktop, fcontext_t *old);
#endif

static inline void ABTD_ythread_context_make(ABTD_ythread_context *p_ctx,
                                             void *sp, size_t size,
                                             void (*thread_func)(void *))
{
    p_ctx->p_ctx = make_fcontext(sp, size, thread_func);
}

static inline void ABTD_ythread_context_jump(ABTD_ythread_context *p_old,
                                             ABTD_ythread_context *p_new,
                                             void *arg)
{
    jump_fcontext(&p_old->p_ctx, p_new->p_ctx, arg);
}

ABTU_noreturn static inline void
ABTD_ythread_context_take(ABTD_ythread_context *p_old,
                          ABTD_ythread_context *p_new, void *arg)
{
    take_fcontext(&p_old->p_ctx, p_new->p_ctx, arg);
    ABTU_unreachable();
}

#if ABT_CONFIG_THREAD_TYPE == ABT_THREAD_TYPE_DYNAMIC_PROMOTION
static inline void
ABTD_ythread_context_init_and_call(ABTD_ythread_context *p_ctx, void *sp,
                                   void (*thread_func)(void *), void *arg)
{
    init_and_call_fcontext(arg, thread_func, sp, &p_ctx->p_ctx);
}
#endif

#endif /* ABTD_FCONTEXT_H_INCLUDED */

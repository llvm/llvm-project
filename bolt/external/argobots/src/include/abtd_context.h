/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

#ifndef ABTD_CONTEXT_H_INCLUDED
#define ABTD_CONTEXT_H_INCLUDED

#include "abt_config.h"

#ifndef ABT_CONFIG_USE_FCONTEXT
#ifndef _XOPEN_SOURCE
#define _XOPEN_SOURCE
#endif
#include <ucontext.h>
#endif

typedef struct ABTD_ythread_context ABTD_ythread_context;

typedef struct ABTD_ythread_context_atomic_ptr {
    ABTD_atomic_ptr val;
} ABTD_ythread_context_atomic_ptr;

static inline ABTD_ythread_context *
ABTD_atomic_relaxed_load_ythread_context_ptr(
    const ABTD_ythread_context_atomic_ptr *ptr)
{
    return (ABTD_ythread_context *)ABTD_atomic_relaxed_load_ptr(&ptr->val);
}

static inline ABTD_ythread_context *
ABTD_atomic_acquire_load_ythread_context_ptr(
    const ABTD_ythread_context_atomic_ptr *ptr)
{
    return (ABTD_ythread_context *)ABTD_atomic_acquire_load_ptr(&ptr->val);
}

static inline void ABTD_atomic_relaxed_store_ythread_context_ptr(
    ABTD_ythread_context_atomic_ptr *ptr, ABTD_ythread_context *p_ctx)
{
    ABTD_atomic_relaxed_store_ptr(&ptr->val, (void *)p_ctx);
}

static inline void ABTD_atomic_release_store_ythread_context_ptr(
    ABTD_ythread_context_atomic_ptr *ptr, ABTD_ythread_context *p_ctx)
{
    ABTD_atomic_release_store_ptr(&ptr->val, (void *)p_ctx);
}

struct ABTD_ythread_context {
    void *p_ctx;                            /* actual context of fcontext, or a
                                             * pointer to uctx */
    ABTD_ythread_context_atomic_ptr p_link; /* pointer to scheduler context */
#ifndef ABT_CONFIG_USE_FCONTEXT
    ucontext_t uctx;               /* ucontext entity pointed by p_ctx */
    void (*f_uctx_thread)(void *); /* root function called by ucontext */
    void *p_uctx_arg;              /* argument for root function */
#endif
};

static void ABTD_ythread_context_make(ABTD_ythread_context *p_ctx, void *sp,
                                      size_t size, void (*thread_func)(void *));
static void ABTD_ythread_context_jump(ABTD_ythread_context *p_old,
                                      ABTD_ythread_context *p_new, void *arg);
ABTU_noreturn static void ABTD_ythread_context_take(ABTD_ythread_context *p_old,
                                                    ABTD_ythread_context *p_new,
                                                    void *arg);
#if ABT_CONFIG_THREAD_TYPE == ABT_THREAD_TYPE_DYNAMIC_PROMOTION
static void ABTD_ythread_context_init_and_call(ABTD_ythread_context *p_ctx,
                                               void *sp,
                                               void (*thread_func)(void *),
                                               void *arg);
#endif

void ABTD_ythread_print_context(ABTI_ythread *p_ythread, FILE *p_os,
                                int indent);

#ifdef ABT_CONFIG_USE_FCONTEXT
#include "abtd_fcontext.h"
#else
#include "abtd_ucontext.h"
#endif

#endif /* ABTD_CONTEXT_H_INCLUDED */

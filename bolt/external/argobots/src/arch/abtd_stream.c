/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

#include "abti.h"

static void *xstream_context_thread_func(void *arg)
{
    ABTD_xstream_context *p_ctx = (ABTD_xstream_context *)arg;
    void *(*thread_f)(void *) = p_ctx->thread_f;
    void *p_arg = p_ctx->p_arg;
    ABTI_ASSERT(p_ctx->state == ABTD_XSTREAM_CONTEXT_STATE_RUNNING);
    while (1) {
        /* Execute a main execution stream function. */
        thread_f(p_arg);
        /* This thread has finished. */
        ABT_bool restart;
        pthread_mutex_lock(&p_ctx->state_lock);
        /* If another execution stream is waiting for this thread completion,
         * let's wake it up. */
        if (p_ctx->state == ABTD_XSTREAM_CONTEXT_STATE_REQ_JOIN) {
            pthread_cond_signal(&p_ctx->state_cond);
        }
        p_ctx->state = ABTD_XSTREAM_CONTEXT_STATE_WAITING;
        /* Wait for a request from ABTD_xstream_context_free() or
         * ABTD_xstream_context_restart().
         * The following loop is to deal with spurious wakeup. */
        do {
            pthread_cond_wait(&p_ctx->state_cond, &p_ctx->state_lock);
        } while (p_ctx->state == ABTD_XSTREAM_CONTEXT_STATE_WAITING);
        if (p_ctx->state == ABTD_XSTREAM_CONTEXT_STATE_REQ_TERMINATE) {
            /* ABTD_xstream_context_free() terminates this thread. */
            restart = ABT_FALSE;
        } else {
            /* ABTD_xstream_context_restart() restarts this thread */
            ABTI_ASSERT(p_ctx->state == ABTD_XSTREAM_CONTEXT_STATE_RUNNING ||
                        p_ctx->state == ABTD_XSTREAM_CONTEXT_STATE_REQ_JOIN);
            restart = ABT_TRUE;
        }
        pthread_mutex_unlock(&p_ctx->state_lock);
        if (!restart)
            break;
    }
    return NULL;
}

ABTU_ret_err int ABTD_xstream_context_create(void *(*f_xstream)(void *),
                                             void *p_arg,
                                             ABTD_xstream_context *p_ctx)
{
    p_ctx->thread_f = f_xstream;
    p_ctx->p_arg = p_arg;
    p_ctx->state = ABTD_XSTREAM_CONTEXT_STATE_RUNNING;
    pthread_mutex_init(&p_ctx->state_lock, NULL);
    pthread_cond_init(&p_ctx->state_cond, NULL);
    int ret = pthread_create(&p_ctx->native_thread, NULL,
                             xstream_context_thread_func, p_ctx);
    if (ret != 0) {
        HANDLE_ERROR("pthread_create");
        return ABT_ERR_XSTREAM;
    }
    return ABT_SUCCESS;
}

void ABTD_xstream_context_free(ABTD_xstream_context *p_ctx)
{
    /* Request termination */
    pthread_mutex_lock(&p_ctx->state_lock);
    ABTI_ASSERT(p_ctx->state == ABTD_XSTREAM_CONTEXT_STATE_WAITING);
    p_ctx->state = ABTD_XSTREAM_CONTEXT_STATE_REQ_TERMINATE;
    pthread_cond_signal(&p_ctx->state_cond);
    pthread_mutex_unlock(&p_ctx->state_lock);
    /* Join the target thread. */
    int ret = pthread_join(p_ctx->native_thread, NULL);
    ABTI_ASSERT(ret == 0);
    pthread_cond_destroy(&p_ctx->state_cond);
    pthread_mutex_destroy(&p_ctx->state_lock);
}

void ABTD_xstream_context_join(ABTD_xstream_context *p_ctx)
{
    /* If not finished, sleep this thread. */
    pthread_mutex_lock(&p_ctx->state_lock);
    if (p_ctx->state != ABTD_XSTREAM_CONTEXT_STATE_WAITING) {
        ABTI_ASSERT(p_ctx->state == ABTD_XSTREAM_CONTEXT_STATE_RUNNING);
        p_ctx->state = ABTD_XSTREAM_CONTEXT_STATE_REQ_JOIN;
        /* The following loop is to deal with spurious wakeup. */
        do {
            pthread_cond_wait(&p_ctx->state_cond, &p_ctx->state_lock);
        } while (p_ctx->state == ABTD_XSTREAM_CONTEXT_STATE_REQ_JOIN);
    }
    ABTI_ASSERT(p_ctx->state == ABTD_XSTREAM_CONTEXT_STATE_WAITING);
    pthread_mutex_unlock(&p_ctx->state_lock);
}

void ABTD_xstream_context_revive(ABTD_xstream_context *p_ctx)
{
    /* Request restart */
    pthread_mutex_lock(&p_ctx->state_lock);
    ABTI_ASSERT(p_ctx->state == ABTD_XSTREAM_CONTEXT_STATE_WAITING);
    p_ctx->state = ABTD_XSTREAM_CONTEXT_STATE_RUNNING;
    pthread_cond_signal(&p_ctx->state_cond);
    pthread_mutex_unlock(&p_ctx->state_lock);
}

void ABTD_xstream_context_set_self(ABTD_xstream_context *p_ctx)
{
    p_ctx->native_thread = pthread_self();
}

/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

#ifndef ABTI_COND_H_INCLUDED
#define ABTI_COND_H_INCLUDED

#include "abti_mutex.h"

/* Inlined functions for Condition Variable  */

static inline void ABTI_cond_init(ABTI_cond *p_cond)
{
    ABTI_spinlock_clear(&p_cond->lock);
    p_cond->p_waiter_mutex = NULL;
    p_cond->num_waiters = 0;
    p_cond->p_head = NULL;
    p_cond->p_tail = NULL;
}

static inline void ABTI_cond_fini(ABTI_cond *p_cond)
{
    /* The lock needs to be acquired to safely free the condition structure.
     * However, we do not have to unlock it because the entire structure is
     * freed here. */
    ABTI_spinlock_acquire(&p_cond->lock);
}

static inline ABTI_cond *ABTI_cond_get_ptr(ABT_cond cond)
{
#ifndef ABT_CONFIG_DISABLE_ERROR_CHECK
    ABTI_cond *p_cond;
    if (cond == ABT_COND_NULL) {
        p_cond = NULL;
    } else {
        p_cond = (ABTI_cond *)cond;
    }
    return p_cond;
#else
    return (ABTI_cond *)cond;
#endif
}

static inline ABT_cond ABTI_cond_get_handle(ABTI_cond *p_cond)
{
#ifndef ABT_CONFIG_DISABLE_ERROR_CHECK
    ABT_cond h_cond;
    if (p_cond == NULL) {
        h_cond = ABT_COND_NULL;
    } else {
        h_cond = (ABT_cond)p_cond;
    }
    return h_cond;
#else
    return (ABT_cond)p_cond;
#endif
}

ABTU_ret_err static inline int
ABTI_cond_wait(ABTI_local **pp_local, ABTI_cond *p_cond, ABTI_mutex *p_mutex)
{
    ABTI_ythread *p_ythread = NULL;
    ABTI_thread *p_thread;

    ABTI_xstream *p_local_xstream = ABTI_local_get_xstream_or_null(*pp_local);
    if (!ABTI_IS_EXT_THREAD_ENABLED || p_local_xstream) {
        p_thread = p_local_xstream->p_thread;
        p_ythread = ABTI_thread_get_ythread_or_null(p_thread);
    }
    if (!p_ythread) {
        /* external thread or non-yieldable thread */
        int abt_errno = ABTU_calloc(1, sizeof(ABTI_thread), (void **)&p_thread);
        ABTI_CHECK_ERROR(abt_errno);
        p_thread->type = ABTI_THREAD_TYPE_EXT;
        /* use state for synchronization */
        ABTD_atomic_relaxed_store_int(&p_thread->state,
                                      ABT_THREAD_STATE_BLOCKED);
    }

    ABTI_spinlock_acquire(&p_cond->lock);

    if (p_cond->p_waiter_mutex == NULL) {
        p_cond->p_waiter_mutex = p_mutex;
    } else {
        if (p_cond->p_waiter_mutex != p_mutex) {
            ABTI_spinlock_release(&p_cond->lock);
            if (!p_ythread)
                ABTU_free(p_thread);
            return ABT_ERR_INV_MUTEX;
        }
    }

    if (p_cond->num_waiters == 0) {
        p_thread->p_prev = p_thread;
        p_thread->p_next = p_thread;
        p_cond->p_head = p_thread;
        p_cond->p_tail = p_thread;
    } else {
        p_cond->p_tail->p_next = p_thread;
        p_cond->p_head->p_prev = p_thread;
        p_thread->p_prev = p_cond->p_tail;
        p_thread->p_next = p_cond->p_head;
        p_cond->p_tail = p_thread;
    }

    p_cond->num_waiters++;

    if (p_ythread) {
        /* Change the ULT's state to BLOCKED */
        ABTI_ythread_set_blocked(p_ythread);

        ABTI_spinlock_release(&p_cond->lock);

        /* Unlock the mutex that the calling ULT is holding */
        /* FIXME: should check if mutex was locked by the calling ULT */
        ABTI_mutex_unlock(ABTI_xstream_get_local(p_local_xstream), p_mutex);

        /* Suspend the current ULT */
        ABTI_ythread_suspend(&p_local_xstream, p_ythread,
                             ABT_SYNC_EVENT_TYPE_COND, (void *)p_cond);
        *pp_local = ABTI_xstream_get_local(p_local_xstream);
    } else {
        ABTI_spinlock_release(&p_cond->lock);
        ABTI_mutex_unlock(ABTI_xstream_get_local(p_local_xstream), p_mutex);

        /* External thread is waiting here. */
        while (ABTD_atomic_acquire_load_int(&p_thread->state) !=
               ABT_THREAD_STATE_READY)
            ;
        ABTU_free(p_thread);
    }

    /* Lock the mutex again */
    ABTI_mutex_lock(pp_local, p_mutex);
    return ABT_SUCCESS;
}

static inline void ABTI_cond_broadcast(ABTI_local *p_local, ABTI_cond *p_cond)
{
    ABTI_spinlock_acquire(&p_cond->lock);

    if (p_cond->num_waiters == 0) {
        ABTI_spinlock_release(&p_cond->lock);
        return;
    }

    /* Wake up all waiting ULTs */
    ABTI_thread *p_head = p_cond->p_head;
    ABTI_thread *p_thread = p_head;
    while (1) {
        ABTI_thread *p_next = p_thread->p_next;

        p_thread->p_prev = NULL;
        p_thread->p_next = NULL;

        ABTI_ythread *p_ythread = ABTI_thread_get_ythread_or_null(p_thread);
        if (p_ythread) {
            ABTI_ythread_set_ready(p_local, p_ythread);
        } else {
            /* When the head is an external thread */
            ABTD_atomic_release_store_int(&p_thread->state,
                                          ABT_THREAD_STATE_READY);
        }

        /* Next ULT */
        if (p_next != p_head) {
            p_thread = p_next;
        } else {
            break;
        }
    }

    p_cond->p_waiter_mutex = NULL;
    p_cond->num_waiters = 0;
    p_cond->p_head = NULL;
    p_cond->p_tail = NULL;

    ABTI_spinlock_release(&p_cond->lock);
}

#endif /* ABTI_COND_H_INCLUDED */

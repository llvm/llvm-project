/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

#include "abti.h"
#include <sys/time.h>

/** @defgroup COND Condition Variable
 * This group is for Condition Variable.
 */

/**
 * @ingroup COND
 * @brief   Create a new condition variable.
 *
 * \c ABT_cond_create() creates a new condition variable and returns its handle
 * through \c newcond.
 * If an error occurs in this routine, a non-zero error code will be returned
 * and newcond will be set to \c ABT_COND_NULL.
 *
 * @param[out] newcond  handle to a new condition variable
 * @return Error code
 * @retval ABT_SUCCESS on success
 */
int ABT_cond_create(ABT_cond *newcond)
{
    ABTI_cond *p_newcond;
    int abt_errno = ABTU_malloc(sizeof(ABTI_cond), (void **)&p_newcond);
    ABTI_CHECK_ERROR(abt_errno);

    ABTI_cond_init(p_newcond);
    /* Return value */
    *newcond = ABTI_cond_get_handle(p_newcond);
    return ABT_SUCCESS;
}

/**
 * @ingroup COND
 * @brief   Free the condition variable.
 *
 * \c ABT_cond_free() deallocates the memory used for the condition variable
 * object associated with the handle \c cond. If it is successfully processed,
 * \c cond is set to \c ABT_COND_NULL.
 *
 * @param[in,out] cond  handle to the condition variable
 * @return Error code
 * @retval ABT_SUCCESS on success
 */
int ABT_cond_free(ABT_cond *cond)
{
    ABT_cond h_cond = *cond;
    ABTI_cond *p_cond = ABTI_cond_get_ptr(h_cond);
    ABTI_CHECK_NULL_COND_PTR(p_cond);
    ABTI_CHECK_TRUE(p_cond->num_waiters == 0, ABT_ERR_COND);

    ABTI_cond_fini(p_cond);
    ABTU_free(p_cond);
    /* Return value */
    *cond = ABT_COND_NULL;
    return ABT_SUCCESS;
}

/**
 * @ingroup COND
 * @brief   Wait on the condition.
 *
 * The ULT calling \c ABT_cond_wait() waits on the condition variable until
 * it is signaled.
 * The user should call this routine while the mutex specified as \c mutex is
 * locked. The mutex will be automatically released while waiting. After signal
 * is received and the waiting ULT is awakened, the mutex will be
 * automatically locked for use by the ULT. The user is then responsible for
 * unlocking mutex when the ULT is finished with it.
 *
 * @param[in] cond   handle to the condition variable
 * @param[in] mutex  handle to the mutex
 * @return Error code
 * @retval ABT_SUCCESS on success
 */
int ABT_cond_wait(ABT_cond cond, ABT_mutex mutex)
{
    ABTI_local *p_local = ABTI_local_get_local();
    ABTI_cond *p_cond = ABTI_cond_get_ptr(cond);
    ABTI_CHECK_NULL_COND_PTR(p_cond);
    ABTI_mutex *p_mutex = ABTI_mutex_get_ptr(mutex);
    ABTI_CHECK_NULL_MUTEX_PTR(p_mutex);

    int abt_errno = ABTI_cond_wait(&p_local, p_cond, p_mutex);
    ABTI_CHECK_ERROR(abt_errno);
    return ABT_SUCCESS;
}

static inline double convert_timespec_to_sec(const struct timespec *p_ts)
{
    double secs;
    secs = ((double)p_ts->tv_sec) + 1.0e-9 * ((double)p_ts->tv_nsec);
    return secs;
}

static inline void remove_thread(ABTI_cond *p_cond, ABTI_thread *p_thread)
{
    if (p_thread->p_next == NULL)
        return;

    ABTI_spinlock_acquire(&p_cond->lock);

    if (p_thread->p_next == NULL) {
        ABTI_spinlock_release(&p_cond->lock);
        return;
    }

    /* If p_thread is still in the queue, we have to remove it. */
    p_cond->num_waiters--;
    if (p_cond->num_waiters == 0) {
        p_cond->p_waiter_mutex = NULL;
        p_cond->p_head = NULL;
        p_cond->p_tail = NULL;
    } else {
        p_thread->p_prev->p_next = p_thread->p_next;
        p_thread->p_next->p_prev = p_thread->p_prev;
        if (p_thread == p_cond->p_head) {
            p_cond->p_head = p_thread->p_next;
        } else if (p_thread == p_cond->p_tail) {
            p_cond->p_tail = p_thread->p_prev;
        }
    }

    ABTI_spinlock_release(&p_cond->lock);

    p_thread->p_prev = NULL;
    p_thread->p_next = NULL;
}

/**
 * @ingroup COND
 * @brief   Wait on the condition.
 *
 * The ULT calling \c ABT_cond_timedwait() waits on the condition variable
 * until it is signaled or the absolute time specified by \c abstime passes.
 * If system time equals or exceeds \c abstime before \c cond is signaled,
 * the error code \c ABT_ERR_COND_TIMEDOUT is returned.
 *
 * The user should call this routine while the mutex specified as \c mutex is
 * locked. The mutex will be automatically released while waiting. After signal
 * is received and the waiting ULT is awakened, the mutex will be
 * automatically locked for use by the ULT. The user is then responsible for
 * unlocking mutex when the ULT is finished with it.
 *
 * @param[in] cond     handle to the condition variable
 * @param[in] mutex    handle to the mutex
 * @param[in] abstime  absolute time for timeout
 * @return Error code
 * @retval ABT_SUCCESS            on success
 * @retval ABT_ERR_COND_TIMEDOUT  timeout
 */
int ABT_cond_timedwait(ABT_cond cond, ABT_mutex mutex,
                       const struct timespec *abstime)
{
    ABTI_local *p_local = ABTI_local_get_local();
    ABTI_cond *p_cond = ABTI_cond_get_ptr(cond);
    ABTI_CHECK_NULL_COND_PTR(p_cond);
    ABTI_mutex *p_mutex = ABTI_mutex_get_ptr(mutex);
    ABTI_CHECK_NULL_MUTEX_PTR(p_mutex);

    double tar_time = convert_timespec_to_sec(abstime);

    ABTI_thread thread;
    thread.type = ABTI_THREAD_TYPE_EXT;
    ABTD_atomic_relaxed_store_int(&thread.state, ABT_THREAD_STATE_BLOCKED);

    ABTI_spinlock_acquire(&p_cond->lock);

    if (p_cond->p_waiter_mutex == NULL) {
        p_cond->p_waiter_mutex = p_mutex;
    } else {
        if (p_cond->p_waiter_mutex != p_mutex) {
            ABTI_spinlock_release(&p_cond->lock);
            ABTI_HANDLE_ERROR(ABT_ERR_INV_MUTEX);
        }
    }

    if (p_cond->num_waiters == 0) {
        thread.p_prev = &thread;
        thread.p_next = &thread;
        p_cond->p_head = &thread;
        p_cond->p_tail = &thread;
    } else {
        p_cond->p_tail->p_next = &thread;
        p_cond->p_head->p_prev = &thread;
        thread.p_prev = p_cond->p_tail;
        thread.p_next = p_cond->p_head;
        p_cond->p_tail = &thread;
    }

    p_cond->num_waiters++;

    ABTI_spinlock_release(&p_cond->lock);

    /* Unlock the mutex that the calling ULT is holding */
    ABTI_mutex_unlock(p_local, p_mutex);

    ABTI_xstream *p_local_xstream = ABTI_local_get_xstream_or_null(p_local);
    ABTI_ythread *p_ythread = NULL;
    if (!ABTI_IS_EXT_THREAD_ENABLED || p_local_xstream) {
        p_ythread = ABTI_thread_get_ythread_or_null(p_local_xstream->p_thread);
    }
    while (ABTD_atomic_acquire_load_int(&thread.state) !=
           ABT_THREAD_STATE_READY) {
        double cur_time = ABTI_get_wtime();
        if (cur_time >= tar_time) {
            remove_thread(p_cond, &thread);
            /* Lock the mutex again */
            ABTI_mutex_lock(&p_local, p_mutex);
            return ABT_ERR_COND_TIMEDOUT;
        }
        if (p_ythread) {
            ABTI_ythread_yield(&p_local_xstream, p_ythread,
                               ABT_SYNC_EVENT_TYPE_COND, (void *)p_cond);
            p_local = ABTI_xstream_get_local(p_local_xstream);
        } else {
            ABTD_atomic_pause();
        }
    }
    /* Lock the mutex again */
    ABTI_mutex_lock(&p_local, p_mutex);
    return ABT_SUCCESS;
}

/**
 * @ingroup COND
 * @brief   Signal a condition.
 *
 * \c ABT_cond_signal() signals another ULT that is waiting on the condition
 * variable. Only one ULT is waken up by the signal and the scheduler
 * determines the ULT.
 * This routine shall have no effect if no ULTs are currently blocked on the
 * condition variable.
 *
 * @param[in] cond   handle to the condition variable
 * @return Error code
 * @retval ABT_SUCCESS on success
 */
int ABT_cond_signal(ABT_cond cond)
{
    ABTI_local *p_local = ABTI_local_get_local();
    ABTI_cond *p_cond = ABTI_cond_get_ptr(cond);
    ABTI_CHECK_NULL_COND_PTR(p_cond);

    ABTI_spinlock_acquire(&p_cond->lock);

    if (p_cond->num_waiters == 0) {
        ABTI_spinlock_release(&p_cond->lock);
        return ABT_SUCCESS;
    }

    /* Wake up the first waiting ULT */
    ABTI_thread *p_thread = p_cond->p_head;

    p_cond->num_waiters--;
    if (p_cond->num_waiters == 0) {
        p_cond->p_waiter_mutex = NULL;
        p_cond->p_head = NULL;
        p_cond->p_tail = NULL;
    } else {
        p_thread->p_prev->p_next = p_thread->p_next;
        p_thread->p_next->p_prev = p_thread->p_prev;
        p_cond->p_head = p_thread->p_next;
    }
    p_thread->p_prev = NULL;
    p_thread->p_next = NULL;

    ABTI_ythread *p_ythread = ABTI_thread_get_ythread_or_null(p_thread);
    if (p_ythread) {
        ABTI_ythread_set_ready(p_local, p_ythread);
    } else {
        /* When the head is an external thread */
        ABTD_atomic_release_store_int(&p_thread->state, ABT_THREAD_STATE_READY);
    }

    ABTI_spinlock_release(&p_cond->lock);
    return ABT_SUCCESS;
}

/**
 * @ingroup COND
 * @brief   Broadcast a condition.
 *
 * \c ABT_cond_broadcast() signals all ULTs that are waiting on the
 * condition variable.
 * This routine shall have no effect if no ULTs are currently blocked on the
 * condition variable.
 *
 * @param[in] cond   handle to the condition variable
 * @return Error code
 * @retval ABT_SUCCESS on success
 */
int ABT_cond_broadcast(ABT_cond cond)
{
    ABTI_local *p_local = ABTI_local_get_local();
    ABTI_cond *p_cond = ABTI_cond_get_ptr(cond);
    ABTI_CHECK_NULL_COND_PTR(p_cond);

    ABTI_cond_broadcast(p_local, p_cond);
    return ABT_SUCCESS;
}

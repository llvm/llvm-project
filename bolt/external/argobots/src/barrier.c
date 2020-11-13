/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

#include "abti.h"

/** @defgroup BARRIER Barrier
 * This group is for Barrier.
 */

/**
 * @ingroup BARRIER
 * @brief   Create a new barrier.
 *
 * \c ABT_barrier_create() creates a new barrier and returns its handle through
 * \c newbarrier.
 * If an error occurs in this routine, a non-zero error code will be returned
 * and \c newbarrier will be set to \c ABT_BARRIER_NULL.
 *
 * @param[in]  num_waiters  number of waiters
 * @param[out] newbarrier   handle to a new barrier
 * @return Error code
 * @retval ABT_SUCCESS on success
 */
int ABT_barrier_create(uint32_t num_waiters, ABT_barrier *newbarrier)
{
    int abt_errno;
    ABTI_barrier *p_newbarrier;

    abt_errno = ABTU_malloc(sizeof(ABTI_barrier), (void **)&p_newbarrier);
    ABTI_CHECK_ERROR(abt_errno);

    ABTI_spinlock_clear(&p_newbarrier->lock);
    p_newbarrier->num_waiters = num_waiters;
    p_newbarrier->counter = 0;
    abt_errno = ABTU_malloc(num_waiters * sizeof(ABTI_ythread *),
                            (void **)&p_newbarrier->waiters);
    if (ABTI_IS_ERROR_CHECK_ENABLED && abt_errno != ABT_SUCCESS) {
        ABTU_free(p_newbarrier);
        ABTI_HANDLE_ERROR(abt_errno);
    }
    abt_errno = ABTU_malloc(num_waiters * sizeof(ABT_unit_type),
                            (void **)&p_newbarrier->waiter_type);
    if (ABTI_IS_ERROR_CHECK_ENABLED && abt_errno != ABT_SUCCESS) {
        ABTU_free(p_newbarrier->waiters);
        ABTU_free(p_newbarrier);
        ABTI_HANDLE_ERROR(abt_errno);
    }

    /* Return value */
    *newbarrier = ABTI_barrier_get_handle(p_newbarrier);
    return ABT_SUCCESS;
}

/**
 * @ingroup BARRIER
 * @brief   Reinitialize the barrier.
 *
 * \c ABT_barrier_reinit() reinitializes the barrier \c barrier with a new
 * number of waiters \c num_waiters.  \c num_waiters can be the same as or
 * different from the one passed to \c ABT_barrier_create().
 *
 * @param[in] barrier      handle to the barrier
 * @param[in] num_waiters  number of waiters
 * @return Error code
 * @retval ABT_SUCCESS on success
 */
int ABT_barrier_reinit(ABT_barrier barrier, uint32_t num_waiters)
{
    ABTI_barrier *p_barrier = ABTI_barrier_get_ptr(barrier);
    ABTI_CHECK_NULL_BARRIER_PTR(p_barrier);
    ABTI_ASSERT(p_barrier->counter == 0);

    /* Only when num_waiters is different from p_barrier->num_waiters, we
     * change p_barrier. */
    if (num_waiters < p_barrier->num_waiters) {
        /* We can reuse waiters and waiter_type arrays */
        p_barrier->num_waiters = num_waiters;
    } else if (num_waiters > p_barrier->num_waiters) {
        /* Free existing arrays and reallocate them */
        int abt_errno;
        ABTI_ythread **new_waiters;
        ABT_unit_type *new_waiter_types;
        abt_errno = ABTU_malloc(num_waiters * sizeof(ABTI_ythread *),
                                (void **)&new_waiters);
        ABTI_CHECK_ERROR(abt_errno);
        abt_errno = ABTU_malloc(num_waiters * sizeof(ABT_unit_type),
                                (void **)&new_waiter_types);
        if (ABTI_IS_ERROR_CHECK_ENABLED && abt_errno != ABT_SUCCESS) {
            ABTU_free(new_waiters);
            ABTI_HANDLE_ERROR(abt_errno);
        }
        p_barrier->num_waiters = num_waiters;
        ABTU_free(p_barrier->waiters);
        ABTU_free(p_barrier->waiter_type);
        p_barrier->waiters = new_waiters;
        p_barrier->waiter_type = new_waiter_types;
    }
    return ABT_SUCCESS;
}

/**
 * @ingroup BARRIER
 * @brief   Free the barrier.
 *
 * \c ABT_barrier_free() deallocates the memory used for the barrier object
 * associated with the handle \c barrier. If it is successfully processed,
 * \c barrier is set to \c ABT_BARRIER_NULL.
 *
 * @param[in,out] barrier  handle to the barrier
 * @return Error code
 * @retval ABT_SUCCESS on success
 */
int ABT_barrier_free(ABT_barrier *barrier)
{
    ABT_barrier h_barrier = *barrier;
    ABTI_barrier *p_barrier = ABTI_barrier_get_ptr(h_barrier);
    ABTI_CHECK_NULL_BARRIER_PTR(p_barrier);

    ABTI_ASSERT(p_barrier->counter == 0);

    /* The lock needs to be acquired to safely free the barrier structure.
     * However, we do not have to unlock it because the entire structure is
     * freed here. */
    ABTI_spinlock_acquire(&p_barrier->lock);

    ABTU_free(p_barrier->waiters);
    ABTU_free(p_barrier->waiter_type);
    ABTU_free(p_barrier);

    /* Return value */
    *barrier = ABT_BARRIER_NULL;
    return ABT_SUCCESS;
}

/**
 * @ingroup BARRIER
 * @brief   Wait on the barrier.
 *
 * The ULT calling \c ABT_barrier_wait() waits on the barrier until all the
 * ULTs reach the barrier.
 *
 * @param[in] barrier  handle to the barrier
 * @return Error code
 * @retval ABT_SUCCESS on success
 */
int ABT_barrier_wait(ABT_barrier barrier)
{
    ABTI_local *p_local = ABTI_local_get_local();
    ABTI_barrier *p_barrier = ABTI_barrier_get_ptr(barrier);
    ABTI_CHECK_NULL_BARRIER_PTR(p_barrier);
    uint32_t pos;

    ABTI_spinlock_acquire(&p_barrier->lock);

    ABTI_ASSERT(p_barrier->counter < p_barrier->num_waiters);
    pos = p_barrier->counter++;

    /* If we do not have all the waiters yet */
    if (p_barrier->counter < p_barrier->num_waiters) {
        ABTI_ythread *p_ythread = NULL;
        ABT_unit_type type;
        ABTD_atomic_int32 ext_signal = ABTD_ATOMIC_INT32_STATIC_INITIALIZER(0);

        ABTI_xstream *p_local_xstream = ABTI_local_get_xstream_or_null(p_local);
        if (!ABTI_IS_EXT_THREAD_ENABLED || p_local_xstream) {
            p_ythread =
                ABTI_thread_get_ythread_or_null(p_local_xstream->p_thread);
        }
        if (p_ythread) {
            /* yieldable thread */
            type = ABT_UNIT_TYPE_THREAD;
        } else {
            /* external thread or non-yieldable thread */
            /* Check size if ext_signal can be stored in p_thread. */
            ABTI_STATIC_ASSERT(sizeof(ext_signal) <= sizeof(p_ythread));
            p_ythread = (ABTI_ythread *)&ext_signal;
            type = ABT_UNIT_TYPE_EXT;
        }

        /* Keep the waiter's information */
        p_barrier->waiters[pos] = p_ythread;
        p_barrier->waiter_type[pos] = type;

        if (type == ABT_UNIT_TYPE_THREAD) {
            /* Change the ULT's state to BLOCKED */
            ABTI_ythread_set_blocked(p_ythread);
        }

        ABTI_spinlock_release(&p_barrier->lock);

        if (type == ABT_UNIT_TYPE_THREAD) {
            /* Suspend the current ULT */
            ABTI_ythread_suspend(&p_local_xstream, p_ythread,
                                 ABT_SYNC_EVENT_TYPE_BARRIER,
                                 (void *)p_barrier);
        } else {
            /* External thread is waiting here polling ext_signal. */
            /* FIXME: need a better implementation */
            while (!ABTD_atomic_acquire_load_int32(&ext_signal))
                ;
        }
    } else {
        /* Signal all the waiting ULTs */
        int i;
        for (i = 0; i < p_barrier->num_waiters - 1; i++) {
            ABTI_ythread *p_ythread = p_barrier->waiters[i];
            if (p_barrier->waiter_type[i] == ABT_UNIT_TYPE_THREAD) {
                ABTI_ythread_set_ready(p_local, p_ythread);
            } else {
                /* When p_cur is an external thread */
                ABTD_atomic_int32 *p_ext_signal =
                    (ABTD_atomic_int32 *)p_ythread;
                ABTD_atomic_release_store_int32(p_ext_signal, 1);
            }

            p_barrier->waiters[i] = NULL;
        }

        /* Reset counter */
        p_barrier->counter = 0;

        ABTI_spinlock_release(&p_barrier->lock);
    }
    return ABT_SUCCESS;
}

/**
 * @ingroup BARRIER
 * @brief   Get the number of waiters for the barrier.
 *
 * \c ABT_barrier_get_num_waiters() returns the number of waiters, which was
 * passed to \c ABT_barrier_create() or \c ABT_barrier_reinit(), for the given
 * barrier \c barrier.
 *
 * @param[in]  barrier      handle to the barrier
 * @param[out] num_waiters  number of waiters
 * @return Error code
 * @retval ABT_SUCCESS on success
 */
int ABT_barrier_get_num_waiters(ABT_barrier barrier, uint32_t *num_waiters)
{
    ABTI_barrier *p_barrier = ABTI_barrier_get_ptr(barrier);
    ABTI_CHECK_NULL_BARRIER_PTR(p_barrier);

    *num_waiters = p_barrier->num_waiters;
    return ABT_SUCCESS;
}

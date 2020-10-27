/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

#include "abti.h"

/** @defgroup FUTURE Future
 * A future, an eventual, or a \a promise, is a mechanism for passing a value
 * between threads, allowing a thread to wait for a value that is set
 * asynchronously. It is used to increase concurrency in a parallel program.
 * This construction is really popular in functional programming languages,
 * in particular MultiLisp. If the programmer defines a future containing
 * an expression, the runtime system \a promises to evaluate that expression
 * concurrently. The resulting value of the expression might not be available
 * immediately, but it will be eventually computed. Therefore, futures also
 * require a synchronization interface between the program and the multiple
 * concurrent threads that may be computing portions of the code.
 *
 * In Argobots, futures are used with the purpose of synchronizing execution
 * between cooperating concurrent ULTs. There are two basic mechanisms
 * implemented, \ref EVENTUAL "eventuals" and futures.
 *
 * A \a future in Argobots has a slightly different behavior. A future is
 * created with a number of \a compartments. Each of those \a k compartments
 * will be set by contributing ULTs. Any other ULT will block on a future
 * until all the compartments have been set. In some sense, a future is
 * a multiple-buffer extension of an eventual. Eventuals and futures have
 * a different philosophy of memory management. An eventual will create and
 * destroy the memory buffer that will hold a result. In contrast, a future
 * does not create any buffer. Therefore, a future assumes each contributing
 * ULT allocates and destroys all memory buffers. When a contributing ULT
 * sets a value, it just passes a pointer to the particular memory location.
 */

/**
 * @ingroup FUTURE
 * @brief   Create a future.
 *
 * \c ABT_future_create creates a future and returns a handle to the newly
 * created future into \c newfuture. This routine allocates an array with
 * as many \c compartments as defined. Each compartment consists in a void*
 * pointer. The future has a counter to determine whether all contributions
 * have been made. This routine also creates a list of entries for all the
 * ULTs that will be blocked waiting for the future to be ready. The list
 * is initially empty. The entries in the list are set with the same order as
 * the \c ABT_future_set are terminated.
 *
 * @param[in]  compartments number of compartments in the future
 * @param[in]  cb_func      callback function to be called once the future
 *                          is ready
 * @param[out] newfuture    handle to a new future
 * @return Error code
 * @retval ABT_SUCCESS on success
 */
int ABT_future_create(uint32_t compartments, void (*cb_func)(void **arg),
                      ABT_future *newfuture)
{
    int abt_errno;
    ABTI_future *p_future;

    abt_errno = ABTU_malloc(sizeof(ABTI_future), (void **)&p_future);
    ABTI_CHECK_ERROR(abt_errno);
    ABTI_spinlock_clear(&p_future->lock);
    ABTD_atomic_relaxed_store_uint32(&p_future->counter, 0);
    p_future->compartments = compartments;
    abt_errno =
        ABTU_malloc(compartments * sizeof(void *), (void **)&p_future->array);
    if (ABTI_IS_ERROR_CHECK_ENABLED && abt_errno != ABT_SUCCESS) {
        ABTU_free(p_future);
        ABTI_HANDLE_ERROR(abt_errno);
    }
    p_future->p_callback = cb_func;
    p_future->p_head = NULL;
    p_future->p_tail = NULL;

    *newfuture = ABTI_future_get_handle(p_future);
    return ABT_SUCCESS;
}

/**
 * @ingroup FUTURE
 * @brief   Free the future object.
 *
 * \c ABT_future_free releases memory associated with the future \c future.
 * It also deallocates the array of compartments of the future. If it is
 * successfully processed, \c future is set to \c ABT_FUTURE_NULL.
 *
 * @param[in,out] future  handle to the future
 * @return Error code
 * @retval ABT_SUCCESS on success
 */
int ABT_future_free(ABT_future *future)
{
    ABTI_future *p_future = ABTI_future_get_ptr(*future);
    ABTI_CHECK_NULL_FUTURE_PTR(p_future);

    /* The lock needs to be acquired to safely free the future structure.
     * However, we do not have to unlock it because the entire structure is
     * freed here. */
    ABTI_spinlock_acquire(&p_future->lock);

    ABTU_free(p_future->array);
    ABTU_free(p_future);

    *future = ABT_FUTURE_NULL;
    return ABT_SUCCESS;
}

/**
 * @ingroup FUTURE
 * @brief   Wait on the future.
 *
 * \c ABT_future_wait blocks the caller ULT until the future \c future is
 * resolved. If the future is not ready, the ULT calling this routine
 * suspends and goes to state BLOCKED. Internally, an entry is created per
 * each blocked ULT to be awaken when the future is signaled. If the future
 * is ready, this routine returns immediately. The system keeps a list of
 * all the ULTs waiting on the future.
 *
 * @param[in] future  handle to the future
 * @return Error code
 * @retval ABT_SUCCESS on success
 */
int ABT_future_wait(ABT_future future)
{
    ABTI_local *p_local = ABTI_local_get_local();
    ABTI_future *p_future = ABTI_future_get_ptr(future);
    ABTI_CHECK_NULL_FUTURE_PTR(p_future);

    ABTI_spinlock_acquire(&p_future->lock);
    if (ABTD_atomic_relaxed_load_uint32(&p_future->counter) <
        p_future->compartments) {
        ABTI_ythread *p_ythread = NULL;
        ABTI_thread *p_thread;

        ABTI_xstream *p_local_xstream = ABTI_local_get_xstream_or_null(p_local);
        if (!ABTI_IS_EXT_THREAD_ENABLED || p_local_xstream) {
            p_thread = p_local_xstream->p_thread;
            p_ythread = ABTI_thread_get_ythread_or_null(p_thread);
        }
        if (!p_ythread) {
            /* external thread */
            int abt_errno =
                ABTU_calloc(1, sizeof(ABTI_thread), (void **)&p_thread);
            if (ABTI_IS_ERROR_CHECK_ENABLED && abt_errno != ABT_SUCCESS) {
                ABTI_spinlock_release(&p_future->lock);
                ABTI_HANDLE_ERROR(abt_errno);
            }
            p_thread->type = ABTI_THREAD_TYPE_EXT;
            /* use state for synchronization */
            ABTD_atomic_relaxed_store_int(&p_thread->state,
                                          ABT_THREAD_STATE_BLOCKED);
        }

        p_thread->p_next = NULL;
        if (p_future->p_head == NULL) {
            p_future->p_head = p_thread;
            p_future->p_tail = p_thread;
        } else {
            p_future->p_tail->p_next = p_thread;
            p_future->p_tail = p_thread;
        }

        if (p_ythread) {
            ABTI_ythread_set_blocked(p_ythread);

            ABTI_spinlock_release(&p_future->lock);

            /* Suspend the current ULT */
            ABTI_ythread_suspend(&p_local_xstream, p_ythread,
                                 ABT_SYNC_EVENT_TYPE_FUTURE, (void *)p_future);

        } else {
            ABTI_spinlock_release(&p_future->lock);

            /* External thread is waiting here. */
            while (ABTD_atomic_acquire_load_int(&p_thread->state) !=
                   ABT_THREAD_STATE_READY)
                ;
            ABTU_free(p_thread);
        }
    } else {
        ABTI_spinlock_release(&p_future->lock);
    }
    return ABT_SUCCESS;
}

/**
 * @ingroup FUTURE
 * @brief   Test whether the future is ready.
 *
 * \c ABT_future_test is a non-blocking function that tests whether the future
 * \c future is ready or not. It returns the result through \c flag.
 *
 * @param[in]  future  handle to the future
 * @param[out] flag    \c ABT_TRUE if future is ready; otherwise, \c ABT_FALSE
 * @return Error code
 * @retval ABT_SUCCESS on success
 */
int ABT_future_test(ABT_future future, ABT_bool *flag)
{
    ABTI_future *p_future = ABTI_future_get_ptr(future);
    ABTI_CHECK_NULL_FUTURE_PTR(p_future);

    uint32_t counter = ABTD_atomic_acquire_load_uint32(&p_future->counter);
    *flag = (counter == p_future->compartments) ? ABT_TRUE : ABT_FALSE;
    return ABT_SUCCESS;
}

/**
 * @ingroup FUTURE
 * @brief   Signal the future.
 *
 * \c ABT_future_set sets a value in the future's array. If all the
 * contributions have been received, this routine awakes all ULTs waiting on
 * the future \c future. In that case, all ULTs waiting on this future will
 * be ready to be scheduled. If there are contributions still missing, this
 * routine will store the pointer passed by parameter \c value and increase
 * the internal counter.
 *
 * @param[in] future  handle to the future
 * @param[in] value   pointer to the memory buffer containing the data that
 *                    will be pointed by one compartment of the future
 * @return Error code
 * @retval ABT_SUCCESS on success
 */
int ABT_future_set(ABT_future future, void *value)
{
    ABTI_local *p_local = ABTI_local_get_local();
    ABTI_future *p_future = ABTI_future_get_ptr(future);
    ABTI_CHECK_NULL_FUTURE_PTR(p_future);

    ABTI_spinlock_acquire(&p_future->lock);

    int counter = ABTD_atomic_relaxed_load_uint32(&p_future->counter);
#ifndef ABT_CONFIG_DISABLE_ERROR_CHECK
    if (counter >= p_future->compartments) {
        ABTI_spinlock_release(&p_future->lock);
        ABTI_HANDLE_ERROR(ABT_ERR_FUTURE);
    }
#endif
    p_future->array[counter] = value;
    counter++;
    ABTD_atomic_release_store_uint32(&p_future->counter, counter);

    if (counter == p_future->compartments) {
        if (p_future->p_callback != NULL)
            (*p_future->p_callback)(p_future->array);

        if (p_future->p_head == NULL) {
            ABTI_spinlock_release(&p_future->lock);
            return ABT_SUCCESS;
        }

        /* Wake up all waiting ULTs */
        ABTI_thread *p_head = p_future->p_head;
        ABTI_thread *p_thread = p_head;
        while (1) {
            ABTI_thread *p_next = p_thread->p_next;
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
            if (p_next != NULL) {
                p_thread = p_next;
            } else {
                break;
            }
        }
        p_future->p_head = NULL;
        p_future->p_tail = NULL;
    }

    ABTI_spinlock_release(&p_future->lock);
    return ABT_SUCCESS;
}

/**
 * @ingroup FUTURE
 * @brief   Reset the readiness of the target future.
 *
 * \c ABT_future_reset() resets the readiness of the target future \c future so
 * that it can be reused.  That is, it makes \c future unready irrespective of
 * its readiness.
 *
 * @param[in] future  handle to the target future
 * @return Error code
 * @retval ABT_SUCCESS on success
 */
int ABT_future_reset(ABT_future future)
{
    ABTI_future *p_future = ABTI_future_get_ptr(future);
    ABTI_CHECK_NULL_FUTURE_PTR(p_future);

    ABTI_spinlock_acquire(&p_future->lock);
    ABTD_atomic_release_store_uint32(&p_future->counter, 0);
    ABTI_spinlock_release(&p_future->lock);
    return ABT_SUCCESS;
}

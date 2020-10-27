/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

#include "abti.h"

/** @defgroup ES_BARRIER ES barrier
 * This group is for ES barrier.
 */

/**
 * @ingroup ES_BARRIER
 * @brief   Create a new ES barrier.
 *
 * \c ABT_xstream_barrier_create() creates a new ES barrier and returns its
 * handle through \c newbarrier.
 * If an error occurs in this routine, a non-zero error code will be returned
 * and \c newbarrier will be set to \c ABT_XSTREAM_BARRIER_NULL.
 *
 * @param[in]  num_waiters  number of waiters
 * @param[out] newbarrier   handle to a new ES barrier
 * @return Error code
 * @retval ABT_SUCCESS on success
 */
int ABT_xstream_barrier_create(uint32_t num_waiters,
                               ABT_xstream_barrier *newbarrier)
{
#ifdef HAVE_PTHREAD_BARRIER_INIT
    int abt_errno;
    ABTI_xstream_barrier *p_newbarrier;

    abt_errno =
        ABTU_malloc(sizeof(ABTI_xstream_barrier), (void **)&p_newbarrier);
    ABTI_CHECK_ERROR(abt_errno);

    p_newbarrier->num_waiters = num_waiters;
    abt_errno = ABTD_xstream_barrier_init(num_waiters, &p_newbarrier->bar);
    if (ABTI_IS_ERROR_CHECK_ENABLED && abt_errno != ABT_SUCCESS) {
        ABTU_free(p_newbarrier);
        ABTI_HANDLE_ERROR(abt_errno);
    }

    /* Return value */
    *newbarrier = ABTI_xstream_barrier_get_handle(p_newbarrier);
    return ABT_SUCCESS;
#else
    ABTI_HANDLE_ERROR(ABT_ERR_FEATURE_NA);
#endif
}

/**
 * @ingroup ES_BARRIER
 * @brief   Free the ES barrier.
 *
 * \c ABT_xstream_barrier_free() deallocates the memory used for the ES barrier
 * object associated with the handle \c barrier.  If it is successfully
 * processed, \c barrier is set to \c ABT_XSTREAM_BARRIER_NULL.
 *
 * @param[in,out] barrier  handle to the ES barrier
 * @return Error code
 * @retval ABT_SUCCESS on success
 */
int ABT_xstream_barrier_free(ABT_xstream_barrier *barrier)
{
#ifdef HAVE_PTHREAD_BARRIER_INIT
    ABT_xstream_barrier h_barrier = *barrier;
    ABTI_xstream_barrier *p_barrier = ABTI_xstream_barrier_get_ptr(h_barrier);
    ABTI_CHECK_NULL_XSTREAM_BARRIER_PTR(p_barrier);

    ABTD_xstream_barrier_destroy(&p_barrier->bar);
    ABTU_free(p_barrier);

    /* Return value */
    *barrier = ABT_XSTREAM_BARRIER_NULL;
    return ABT_SUCCESS;
#else
    ABTI_HANDLE_ERROR(ABT_ERR_FEATURE_NA);
#endif
}

/**
 * @ingroup ES_BARRIER
 * @brief   Wait on the barrier.
 *
 * The work unit calling \c ABT_xstream_barrier_wait() waits on the barrier and
 * blocks the entire ES until all the participants reach the barrier.
 *
 * @param[in] barrier  handle to the ES barrier
 * @return Error code
 * @retval ABT_SUCCESS on success
 */
int ABT_xstream_barrier_wait(ABT_xstream_barrier barrier)
{
#ifdef HAVE_PTHREAD_BARRIER_INIT
    ABTI_xstream_barrier *p_barrier = ABTI_xstream_barrier_get_ptr(barrier);
    ABTI_CHECK_NULL_XSTREAM_BARRIER_PTR(p_barrier);

    if (p_barrier->num_waiters > 1) {
        ABTD_xstream_barrier_wait(&p_barrier->bar);
    }
    return ABT_SUCCESS;
#else
    ABTI_HANDLE_ERROR(ABT_ERR_FEATURE_NA);
#endif
}

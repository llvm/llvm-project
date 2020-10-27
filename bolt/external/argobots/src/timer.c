/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

#include "abti.h"

ABTU_ret_err static int timer_alloc(ABTI_timer **pp_newtimer);

/** @defgroup TIMER  Timer
 * This group is for Timer.
 */

/**
 * @ingroup TIMER
 * @brief   Get elapsed wall clock time.
 *
 * \c ABT_get_wtime() returns the elapsed wall clock time in seconds
 * since an arbitrary time in the past.
 * The resolution of elapsed time is at least a unit of microsecond.
 *
 * @return Elapsed wall clock time in seconds
 */
double ABT_get_wtime(void)
{
    return ABTI_get_wtime();
}

/**
 * @ingroup TIMER
 * @brief   Create a new timer.
 *
 * \c ABT_timer_create() creates a new timer object and returns its handle
 * through \c newtimer.
 * If an error occurs in this routine, a non-zero error code will be returned
 * and \c newtimer will be set to \c ABT_TIMER_NULL.
 *
 * @param[out] newtimer  handle to a new timer
 * @return Error code
 * @retval ABT_SUCCESS on success
 */
int ABT_timer_create(ABT_timer *newtimer)
{
    ABTI_timer *p_newtimer;
    int abt_errno = timer_alloc(&p_newtimer);
    ABTI_CHECK_ERROR(abt_errno);

    *newtimer = ABTI_timer_get_handle(p_newtimer);
    return ABT_SUCCESS;
}

/**
 * @ingroup TIMER
 * @brief   Duplicate the timer.
 *
 * \c ABT_timer_dup() creates a new timer and copies the time values from the
 * timer of \c timer to the new timer. The handle of new timer will be returned
 * through \c newtimer.
 * If an error occurs in this routine, a non-zero error code will be returned
 * and \c newtimer will be set to \c ABT_TIMER_NULL.
 *
 * @param[in]  timer     handle to the timer to be duplicated
 * @param[out] newtimer  handle to a new timer
 * @return Error code
 * @retval ABT_SUCCESS on success
 */
int ABT_timer_dup(ABT_timer timer, ABT_timer *newtimer)
{
    ABTI_timer *p_timer = ABTI_timer_get_ptr(timer);
    ABTI_CHECK_NULL_TIMER_PTR(p_timer);

    ABTI_timer *p_newtimer;
    int abt_errno = timer_alloc(&p_newtimer);
    ABTI_CHECK_ERROR(abt_errno);

    memcpy(p_newtimer, p_timer, sizeof(ABTI_timer));
    *newtimer = ABTI_timer_get_handle(p_newtimer);
    return ABT_SUCCESS;
}

/**
 * @ingroup TIMER
 * @brief   Free the timer object.
 *
 * \c ABT_timer_free() deallocates the memory used for the timer object
 * associated with the handle \c timer. If it is successfully processed,
 * \c timer is set to \c ABT_TIMER_NULL.
 * Using the timer handle after calling \c ABT_timer_free() may cause
 * undefined behavior.
 *
 * @param[in,out] timer  handle to the timer
 * @return Error code
 * @retval ABT_SUCCESS on success
 */
int ABT_timer_free(ABT_timer *timer)
{
    ABTI_timer *p_timer = ABTI_timer_get_ptr(*timer);
    ABTI_CHECK_NULL_TIMER_PTR(p_timer);

    /* We use libc malloc/free for ABT_timer because ABTU_malloc/free might
     * need the initialization of Argobots if they are not the same as libc
     * malloc/free.  This is to allow ABT_timer to be used irrespective of
     * Argobots initialization. */
    free(p_timer);
    *timer = ABT_TIMER_NULL;
    return ABT_SUCCESS;
}

/**
 * @ingroup TIMER
 * @brief   Start the timer.
 *
 * \c ABT_timer_start() starts the timer and saves the time when this routine
 * is called. When this routine is called multiple times, the time of last call
 * is only kept.
 *
 * @param[in] timer  handle to the timer
 * @return Error code
 * @retval ABT_SUCCESS on success
 */
int ABT_timer_start(ABT_timer timer)
{
    ABTI_timer *p_timer = ABTI_timer_get_ptr(timer);
    ABTI_CHECK_NULL_TIMER_PTR(p_timer);

    ABTD_time_get(&p_timer->start);
    return ABT_SUCCESS;
}

/**
 * @ingroup TIMER
 * @brief   Stop the timer.
 *
 * \c ABT_timer_stop() stops the timer and saves the time when this routine
 * is called. When this routine is called multiple times, the time of last call
 * is only kept.
 *
 * @param[in] timer  handle to the timer
 * @return Error code
 * @retval ABT_SUCCESS on success
 */
int ABT_timer_stop(ABT_timer timer)
{
    ABTI_timer *p_timer = ABTI_timer_get_ptr(timer);
    ABTI_CHECK_NULL_TIMER_PTR(p_timer);

    ABTD_time_get(&p_timer->end);
    return ABT_SUCCESS;
}

/**
 * @ingroup TIMER
 * @brief   Read the elapsed time of the timer.
 *
 * \c ABT_timer_read() returns the time difference in seconds between the
 * start time of \c timer (when \c ABT_timer_start() was called) and the end
 * time of \c timer (when \c ABT_timer_stop() was called) through \c secs.
 * The resolution of elapsed time is at least a unit of microsecond.
 *
 * @param[in]  timer  handle to the timer
 * @param[out] secs   elapsed time in seconds
 * @return Error code
 * @retval ABT_SUCCESS on success
 */
int ABT_timer_read(ABT_timer timer, double *secs)
{
    ABTI_timer *p_timer = ABTI_timer_get_ptr(timer);
    ABTI_CHECK_NULL_TIMER_PTR(p_timer);

    double start = ABTD_time_read_sec(&p_timer->start);
    double end = ABTD_time_read_sec(&p_timer->end);

    *secs = end - start;
    return ABT_SUCCESS;
}

/**
 * @ingroup TIMER
 * @brief   Stop the timer and read the elapsed time of the timer.
 *
 * \c ABT_timer_stop_and_read() stops the timer and returns the time difference
 * in seconds between the start time of \c timer (when \c ABT_timer_start() was
 * called) and the end time of \c timer (when this routine was called) through
 *\c secs.
 * The resolution of elapsed time is at least a unit of microsecond.
 *
 * @param[in]  timer  handle to the timer
 * @param[out] secs   elapsed time in seconds
 * @return Error code
 * @retval ABT_SUCCESS on success
 */
int ABT_timer_stop_and_read(ABT_timer timer, double *secs)
{
    ABTI_timer *p_timer = ABTI_timer_get_ptr(timer);
    ABTI_CHECK_NULL_TIMER_PTR(p_timer);

    ABTD_time_get(&p_timer->end);
    double start = ABTD_time_read_sec(&p_timer->start);
    double end = ABTD_time_read_sec(&p_timer->end);

    *secs = end - start;
    return ABT_SUCCESS;
}

/**
 * @ingroup TIMER
 * @brief   Stop the timer and add the elapsed time of the timer.
 *
 * \c ABT_timer_stop_and_add() stops the timer and adds the time difference
 * between the start time of \c timer (when \c ABT_timer_start() was called)
 * and the end time of \c timer (when this routine was called) to \c secs.
 * That is, the elapsed time of the timer is accumulated in \c secs.
 * The resolution of elapsed time is at least a unit of microsecond.
 *
 * @param[in]     timer  handle to the timer
 * @param[in,out] secs   accumulated elapsed time in seconds
 * @return Error code
 * @retval ABT_SUCCESS on success
 */
int ABT_timer_stop_and_add(ABT_timer timer, double *secs)
{
    ABTI_timer *p_timer = ABTI_timer_get_ptr(timer);
    ABTI_CHECK_NULL_TIMER_PTR(p_timer);

    ABTD_time_get(&p_timer->end);
    double start = ABTD_time_read_sec(&p_timer->start);
    double end = ABTD_time_read_sec(&p_timer->end);

    *secs += (end - start);
    return ABT_SUCCESS;
}

/**
 * @ingroup TIMER
 * @brief   Obtain the overhead time of using ABT_timer.
 *
 * \c ABT_timer_get_overhead() returns the overhead time when measuring the
 * elapsed time with \c ABT_timer. It computes the time difference in
 * consecutive calls of \c ABT_timer_start() and \c ABT_timer_stop().
 * The resolution of overhead time is at least a unit of microsecond.
 *
 * @param[out] overhead  overhead time of ABT_timer
 * @return Error code
 * @retval ABT_SUCCESS on success
 */
int ABT_timer_get_overhead(double *overhead)
{
    int abt_errno;
    ABT_timer h_timer;
    int i;
    const int iter = 5000;
    double secs, sum = 0.0;

    abt_errno = ABT_timer_create(&h_timer);
    ABTI_CHECK_ERROR(abt_errno);

    for (i = 0; i < iter; i++) {
        ABT_timer_start(h_timer);
        ABT_timer_stop(h_timer);
        ABT_timer_read(h_timer, &secs);
        sum += secs;
    }

    abt_errno = ABT_timer_free(&h_timer);
    ABTI_CHECK_ERROR(abt_errno);

    *overhead = sum / iter;
    return ABT_SUCCESS;
}

/*****************************************************************************/
/* Internal static functions                                                 */
/*****************************************************************************/

ABTU_ret_err static int timer_alloc(ABTI_timer **pp_newtimer)
{
    /* We use libc malloc/free for ABT_timer because ABTU_malloc/free might
     * need the initialization of Argobots if they are not the same as libc
     * malloc/free.  This is to allow ABT_timer to be used irrespective of
     * Argobots initialization. */
    ABTI_timer *p_newtimer = (ABTI_timer *)malloc(sizeof(ABTI_timer));
    ABTI_CHECK_TRUE(p_newtimer != NULL, ABT_ERR_MEM);

    *pp_newtimer = p_newtimer;
    return ABT_SUCCESS;
}

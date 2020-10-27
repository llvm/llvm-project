/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

#include "abti.h"

/** @defgroup RWLOCK Readers Writer Lock
 * A Readers writer lock allows concurrent access for readers and exclusionary
 * access for writers.
 */

/**
 * @ingroup RWLOCK
 * @brief Create a new rwlock
 * \c ABT_rwlock_create creates a new rwlock object and returns its handle
 * through \c newrwlock. If an error occurs in this routine, a non-zero error
 * code will be returned and \c newrwlock will be set to \c ABT_RWLOCK_NULL.
 *
 * @param[out] newrwlock  handle to a new rwlock
 * @return Error code
 * @retval ABT_SUCCESS on success
 */
int ABT_rwlock_create(ABT_rwlock *newrwlock)
{
    int abt_errno;
    ABTI_rwlock *p_newrwlock;

    abt_errno = ABTU_malloc(sizeof(ABTI_rwlock), (void **)&p_newrwlock);
    ABTI_CHECK_ERROR(abt_errno);

    ABTI_CHECK_TRUE(p_newrwlock != NULL, ABT_ERR_MEM);
    abt_errno = ABTI_mutex_init(&p_newrwlock->mutex);
    if (ABTI_IS_ERROR_CHECK_ENABLED && abt_errno != ABT_SUCCESS) {
        ABTU_free(p_newrwlock);
        ABTI_HANDLE_ERROR(abt_errno);
    }
    ABTI_cond_init(&p_newrwlock->cond);
    p_newrwlock->reader_count = 0;
    p_newrwlock->write_flag = 0;

    /* Return value */
    *newrwlock = ABTI_rwlock_get_handle(p_newrwlock);
    return ABT_SUCCESS;
}

/**
 * @ingroup RWLOCK
 * @brief   Free the rwlock object.
 *
 * \c ABT_rwlock_free deallocates the memory used for the rwlock object
 * associated with the handle \c rwlock. If it is successfully processed,
 * \c rwlock is set to \c ABT_RWLOCK_NULL.
 *
 * Using the rwlock handle after calling \c ABT_rwlock_free may cause
 * undefined behavior.
 *
 * @param[in,out] rwlock  handle to the rwlock
 * @return Error code
 * @retval ABT_SUCCESS on success
 */
int ABT_rwlock_free(ABT_rwlock *rwlock)
{
    ABT_rwlock h_rwlock = *rwlock;
    ABTI_rwlock *p_rwlock = ABTI_rwlock_get_ptr(h_rwlock);
    ABTI_CHECK_NULL_RWLOCK_PTR(p_rwlock);

    ABTI_mutex_fini(&p_rwlock->mutex);
    ABTI_cond_fini(&p_rwlock->cond);
    ABTU_free(p_rwlock);

    /* Return value */
    *rwlock = ABT_RWLOCK_NULL;
    return ABT_SUCCESS;
}

/**
 * @ingroup RWLOCK
 * @brief   Lock the rwlock as a reader.
 *
 * \c ABT_rwlock_rdlock locks the rwlock \c rwlock. If this routine successfully
 * returns, the caller ULT acquires the rwlock. If the rwlock has been locked
 * by a writer, the caller ULT will be blocked until the rwlock becomes
 * available. rwlocks may be acquired by any number of readers concurrently.
 * When the caller ULT is blocked, the context is switched to the scheduler
 * of the associated ES to make progress of other work units.
 *
 * @param[in] rwlock  handle to the rwlock
 * @return Error code
 * @retval ABT_SUCCESS on success
 */
int ABT_rwlock_rdlock(ABT_rwlock rwlock)
{
    ABTI_local *p_local = ABTI_local_get_local();
    ABTI_rwlock *p_rwlock = ABTI_rwlock_get_ptr(rwlock);
    ABTI_CHECK_NULL_RWLOCK_PTR(p_rwlock);

    ABTI_mutex_lock(&p_local, &p_rwlock->mutex);
    int abt_errno = ABT_SUCCESS;
    while (p_rwlock->write_flag && abt_errno == ABT_SUCCESS) {
        abt_errno = ABTI_cond_wait(&p_local, &p_rwlock->cond, &p_rwlock->mutex);
    }
    if (abt_errno == ABT_SUCCESS) {
        p_rwlock->reader_count++;
    }
    ABTI_mutex_unlock(p_local, &p_rwlock->mutex);
    ABTI_CHECK_ERROR(abt_errno);
    return ABT_SUCCESS;
}

/**
 * @ingroup RWLOCK
 * @brief   Lock the rwlock as a writer.
 *
 * \c ABT_rwlock_wrlock locks the rwlock \c rwlock. If this routine successfully
 * returns, the caller ULT acquires the rwlock. If the rwlock has been locked
 * by a reader or a writer, the caller ULT will be blocked until the rwlock
 * becomes available. When the caller ULT is blocked, the context is switched
 * to the scheduler of the associated ES to make progress of other work units.
 *
 * @param[in] rwlock  handle to the rwlock
 * @return Error code
 * @retval ABT_SUCCESS on success
 */
int ABT_rwlock_wrlock(ABT_rwlock rwlock)
{
    ABTI_local *p_local = ABTI_local_get_local();
    ABTI_rwlock *p_rwlock = ABTI_rwlock_get_ptr(rwlock);
    ABTI_CHECK_NULL_RWLOCK_PTR(p_rwlock);

    ABTI_mutex_lock(&p_local, &p_rwlock->mutex);
    int abt_errno = ABT_SUCCESS;
    while ((p_rwlock->write_flag || p_rwlock->reader_count) &&
           abt_errno == ABT_SUCCESS) {
        abt_errno = ABTI_cond_wait(&p_local, &p_rwlock->cond, &p_rwlock->mutex);
    }
    if (abt_errno == ABT_SUCCESS) {
        p_rwlock->write_flag = 1;
    }
    ABTI_mutex_unlock(p_local, &p_rwlock->mutex);
    ABTI_CHECK_ERROR(abt_errno);
    return ABT_SUCCESS;
}

/**
 * @ingroup RWLOCK
 * @brief Unlock the rwlock
 *
 * \c ABT_rwlock_unlock unlocks the rwlock \c rwlock.
 * If the caller ULT locked the rwlock, this routine unlocks the rwlock.
 * However, if the caller ULT did not lock the rwlock, this routine may result
 * in undefined behavior.
 *
 * @param[in] rwlock  handle to the rwlock
 * @return Error code
 * @retval ABT_SUCCESS on success
 */
int ABT_rwlock_unlock(ABT_rwlock rwlock)
{
    ABTI_local *p_local = ABTI_local_get_local();
    ABTI_rwlock *p_rwlock = ABTI_rwlock_get_ptr(rwlock);
    ABTI_CHECK_NULL_RWLOCK_PTR(p_rwlock);

    ABTI_mutex_lock(&p_local, &p_rwlock->mutex);
    if (p_rwlock->write_flag) {
        p_rwlock->write_flag = 0;
    } else {
        p_rwlock->reader_count--;
    }
    ABTI_cond_broadcast(p_local, &p_rwlock->cond);
    ABTI_mutex_unlock(p_local, &p_rwlock->mutex);
    return ABT_SUCCESS;
}

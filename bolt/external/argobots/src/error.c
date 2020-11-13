/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

#include "abti.h"

/** @defgroup ERROR Error
 * This group is for error classes.
 */

/**
 * @ingroup ERROR
 * @brief   Get the string of error code and its length.
 *
 * \c ABT_error_get_str returns the string of given error code through
 * \c str and its length in bytes via \c len. If \c str is NULL, only \c len
 * is returned. If \c str is not NULL, it should have enough space to save
 * \c len bytes of characters. If \c len is NULL, \c len is ignored.
 *
 * @param[in]  err    error code
 * @param[out] str    error string
 * @param[out] len    the length of string in bytes
 * @return Error code
 * @retval ABT_SUCCESS on success
 */
int ABT_error_get_str(int err, char *str, size_t *len)
{
    static const char *err_str[] = { "ABT_SUCCESS",
                                     "ABT_ERR_UNINITIALIZED",
                                     "ABT_ERR_MEM",
                                     "ABT_ERR_OTHER",
                                     "ABT_ERR_INV_XSTREAM",
                                     "ABT_ERR_INV_XSTREAM_RANK",
                                     "ABT_ERR_INV_XSTREAM_BARRIER",
                                     "ABT_ERR_INV_SCHED",
                                     "ABT_ERR_INV_SCHED_KIND",
                                     "ABT_ERR_INV_SCHED_PREDEF",
                                     "ABT_ERR_INV_SCHED_TYPE",
                                     "ABT_ERR_INV_SCHED_CONFIG",
                                     "ABT_ERR_INV_POOL",
                                     "ABT_ERR_INV_POOL_KIND",
                                     "ABT_ERR_INV_POOL_ACCESS",
                                     "ABT_ERR_INV_UNIT",
                                     "ABT_ERR_INV_THREAD",
                                     "ABT_ERR_INV_THREAD_ATTR",
                                     "ABT_ERR_INV_TASK",
                                     "ABT_ERR_INV_KEY",
                                     "ABT_ERR_INV_MUTEX",
                                     "ABT_ERR_INV_MUTEX_ATTR",
                                     "ABT_ERR_INV_COND",
                                     "ABT_ERR_INV_RWLOCK",
                                     "ABT_ERR_INV_EVENTUAL",
                                     "ABT_ERR_INV_FUTURE",
                                     "ABT_ERR_INV_BARRIER",
                                     "ABT_ERR_INV_TIMER",
                                     "ABT_ERR_INV_QUERY_KIND",
                                     "ABT_ERR_XSTREAM",
                                     "ABT_ERR_XSTREAM_STATE",
                                     "ABT_ERR_XSTREAM_BARRIER",
                                     "ABT_ERR_SCHED",
                                     "ABT_ERR_SCHED_CONFIG",
                                     "ABT_ERR_POOL",
                                     "ABT_ERR_UNIT",
                                     "ABT_ERR_THREAD",
                                     "ABT_ERR_TASK",
                                     "ABT_ERR_KEY",
                                     "ABT_ERR_MUTEX",
                                     "ABT_ERR_MUTEX_LOCKED",
                                     "ABT_ERR_COND",
                                     "ABT_ERR_COND_TIMEDOUT",
                                     "ABT_ERR_RWLOCK",
                                     "ABT_ERR_EVENTUAL",
                                     "ABT_ERR_FUTURE",
                                     "ABT_ERR_BARRIER",
                                     "ABT_ERR_TIMER",
                                     "ABT_ERR_EVENT",
                                     "ABT_ERR_MIGRATION_TARGET",
                                     "ABT_ERR_MIGRATION_NA",
                                     "ABT_ERR_MISSING_JOIN",
                                     "ABT_ERR_FEATURE_NA" };

    ABTI_CHECK_TRUE(err >= ABT_SUCCESS && err <= ABT_ERR_FEATURE_NA,
                    ABT_ERR_OTHER);
    if (str)
        strcpy(str, err_str[err]);
    if (len)
        *len = strlen(err_str[err]);
    return ABT_SUCCESS;
}

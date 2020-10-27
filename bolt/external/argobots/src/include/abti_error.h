/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

#ifndef ABTI_ERROR_H_INCLUDED
#define ABTI_ERROR_H_INCLUDED

#include <assert.h>
#include <abt_config.h>

#define ABTI_ASSERT(cond)                                                      \
    do {                                                                       \
        if (ABTI_IS_ERROR_CHECK_ENABLED) {                                     \
            assert(cond);                                                      \
        }                                                                      \
    } while (0)

#define ABTI_STATIC_ASSERT(cond)                                               \
    do {                                                                       \
        ((void)sizeof(char[2 * !!(cond)-1]));                                  \
    } while (0)

#ifdef ABT_CONFIG_PRINT_ABT_ERRNO
#define ABTI_IS_PRINT_ABT_ERRNO_ENABLED 1
#else
#define ABTI_IS_PRINT_ABT_ERRNO_ENABLED 0
#endif

#define HANDLE_WARNING(msg)                                                    \
    do {                                                                       \
        if (ABTI_IS_PRINT_ABT_ERRNO_ENABLED) {                                 \
            fprintf(stderr, "[%s:%d] %s\n", __FILE__, __LINE__, msg);          \
        }                                                                      \
    } while (0)

#define HANDLE_ERROR(msg)                                                      \
    do {                                                                       \
        if (ABTI_IS_PRINT_ABT_ERRNO_ENABLED) {                                 \
            fprintf(stderr, "[%s:%d] %s\n", __FILE__, __LINE__, msg);          \
        }                                                                      \
    } while (0)

#define HANDLE_ERROR_WITH_CODE(msg, n)                                         \
    do {                                                                       \
        if (ABTI_IS_PRINT_ABT_ERRNO_ENABLED) {                                 \
            fprintf(stderr, "[%s:%d] %s: %d\n", __FILE__, __LINE__, msg, n);   \
        }                                                                      \
    } while (0)

#define HANDLE_ERROR_FUNC_WITH_CODE(n)                                         \
    do {                                                                       \
        if (ABTI_IS_PRINT_ABT_ERRNO_ENABLED) {                                 \
            fprintf(stderr, "[%s:%d] %s: %d\n", __FILE__, __LINE__, __func__,  \
                    n);                                                        \
        }                                                                      \
    } while (0)

#define ABTI_SETUP_WITH_INIT_CHECK()                                           \
    do {                                                                       \
        if (ABTI_IS_ERROR_CHECK_ENABLED &&                                     \
            ABTU_unlikely(gp_ABTI_global == NULL)) {                           \
            HANDLE_ERROR_FUNC_WITH_CODE(ABT_ERR_UNINITIALIZED);                \
            return ABT_ERR_UNINITIALIZED;                                      \
        }                                                                      \
    } while (0)

#define ABTI_SETUP_LOCAL_XSTREAM(pp_local_xstream)                             \
    do {                                                                       \
        ABTI_xstream *p_local_xstream_tmp =                                    \
            ABTI_local_get_xstream_or_null(ABTI_local_get_local());            \
        if (ABTI_IS_EXT_THREAD_ENABLED &&                                      \
            ABTU_unlikely(p_local_xstream_tmp == NULL)) {                      \
            HANDLE_ERROR_FUNC_WITH_CODE(ABT_ERR_INV_XSTREAM);                  \
            return ABT_ERR_INV_XSTREAM;                                        \
        }                                                                      \
        ABTI_xstream **pp_local_xstream_tmp = (pp_local_xstream);              \
        if (pp_local_xstream_tmp) {                                            \
            *pp_local_xstream_tmp = p_local_xstream_tmp;                       \
        }                                                                      \
    } while (0)

#define ABTI_SETUP_LOCAL_YTHREAD(pp_local_xstream, pp_ythread)                 \
    do {                                                                       \
        ABTI_xstream *p_local_xstream_tmp =                                    \
            ABTI_local_get_xstream_or_null(ABTI_local_get_local());            \
        if (ABTI_IS_ERROR_CHECK_ENABLED && ABTI_IS_EXT_THREAD_ENABLED &&       \
            ABTU_unlikely(p_local_xstream_tmp == NULL)) {                      \
            HANDLE_ERROR_FUNC_WITH_CODE(ABT_ERR_INV_XSTREAM);                  \
            return ABT_ERR_INV_XSTREAM;                                        \
        }                                                                      \
        ABTI_xstream **pp_local_xstream_tmp = (pp_local_xstream);              \
        if (pp_local_xstream_tmp) {                                            \
            *pp_local_xstream_tmp = p_local_xstream_tmp;                       \
        }                                                                      \
        ABTI_thread *p_thread_tmp = p_local_xstream_tmp->p_thread;             \
        if (ABTI_IS_ERROR_CHECK_ENABLED &&                                     \
            ABTU_unlikely(                                                     \
                !(p_thread_tmp->type & ABTI_THREAD_TYPE_YIELDABLE))) {         \
            HANDLE_ERROR_FUNC_WITH_CODE(ABT_ERR_INV_THREAD);                   \
            return ABT_ERR_INV_THREAD;                                         \
        }                                                                      \
        ABTI_ythread **pp_ythread_tmp = (pp_ythread);                          \
        if (pp_ythread_tmp) {                                                  \
            *pp_ythread_tmp = ABTI_thread_get_ythread(p_thread_tmp);           \
        }                                                                      \
    } while (0)

#define ABTI_SETUP_LOCAL_XSTREAM_WITH_INIT_CHECK(pp_local_xstream)             \
    do {                                                                       \
        ABTI_SETUP_WITH_INIT_CHECK();                                          \
        ABTI_SETUP_LOCAL_XSTREAM(pp_local_xstream);                            \
    } while (0)

#define ABTI_SETUP_LOCAL_YTHREAD_WITH_INIT_CHECK(pp_local_xstream, pp_ythread) \
    do {                                                                       \
        ABTI_SETUP_WITH_INIT_CHECK();                                          \
        ABTI_SETUP_LOCAL_YTHREAD(pp_local_xstream, pp_ythread);                \
    } while (0)

#define ABTI_HANDLE_ERROR(n)                                                   \
    do {                                                                       \
        HANDLE_ERROR_FUNC_WITH_CODE(n);                                        \
        return n;                                                              \
    } while (0)

#define ABTI_CHECK_ERROR(abt_errno)                                            \
    do {                                                                       \
        int abt_errno_ = (abt_errno);                                          \
        if (ABTI_IS_ERROR_CHECK_ENABLED &&                                     \
            ABTU_unlikely(abt_errno_ != ABT_SUCCESS)) {                        \
            HANDLE_ERROR_FUNC_WITH_CODE(abt_errno_);                           \
            return abt_errno_;                                                 \
        }                                                                      \
    } while (0)

#define ABTI_CHECK_TRUE(cond, abt_errno)                                       \
    do {                                                                       \
        if (ABTI_IS_ERROR_CHECK_ENABLED && ABTU_unlikely(!(cond))) {           \
            int abt_errno_ = (abt_errno);                                      \
            HANDLE_ERROR_FUNC_WITH_CODE(abt_errno_);                           \
            return abt_errno_;                                                 \
        }                                                                      \
    } while (0)

#define ABTI_CHECK_YIELDABLE(p_thread, pp_ythread, abt_errno)                  \
    do {                                                                       \
        ABTI_thread *p_tmp = (p_thread);                                       \
        if (ABTI_IS_ERROR_CHECK_ENABLED &&                                     \
            ABTU_unlikely(!(p_tmp->type & ABTI_THREAD_TYPE_YIELDABLE))) {      \
            int abt_errno_ = (abt_errno);                                      \
            HANDLE_ERROR_FUNC_WITH_CODE(abt_errno_);                           \
            return abt_errno_;                                                 \
        }                                                                      \
        *(pp_ythread) = ABTI_thread_get_ythread(p_tmp);                        \
    } while (0)

#define ABTI_CHECK_TRUE_MSG(cond, abt_errno, msg)                              \
    do {                                                                       \
        if (ABTI_IS_ERROR_CHECK_ENABLED && ABTU_unlikely(!(cond))) {           \
            HANDLE_ERROR(msg);                                                 \
            return (abt_errno);                                                \
        }                                                                      \
    } while (0)

#define ABTI_CHECK_NULL_XSTREAM_PTR(p)                                         \
    do {                                                                       \
        if (ABTI_IS_ERROR_CHECK_ENABLED &&                                     \
            ABTU_unlikely(p == (ABTI_xstream *)NULL)) {                        \
            HANDLE_ERROR_FUNC_WITH_CODE(ABT_ERR_INV_XSTREAM);                  \
            return ABT_ERR_INV_XSTREAM;                                        \
        }                                                                      \
    } while (0)

#define ABTI_CHECK_NULL_POOL_PTR(p)                                            \
    do {                                                                       \
        if (ABTI_IS_ERROR_CHECK_ENABLED &&                                     \
            ABTU_unlikely(p == (ABTI_pool *)NULL)) {                           \
            HANDLE_ERROR_FUNC_WITH_CODE(ABT_ERR_INV_POOL);                     \
            return ABT_ERR_INV_POOL;                                           \
        }                                                                      \
    } while (0)

#define ABTI_CHECK_NULL_SCHED_PTR(p)                                           \
    do {                                                                       \
        if (ABTI_IS_ERROR_CHECK_ENABLED &&                                     \
            ABTU_unlikely(p == (ABTI_sched *)NULL)) {                          \
            HANDLE_ERROR_FUNC_WITH_CODE(ABT_ERR_INV_SCHED);                    \
            return ABT_ERR_INV_SCHED;                                          \
        }                                                                      \
    } while (0)

#define ABTI_CHECK_NULL_THREAD_PTR(p)                                          \
    do {                                                                       \
        if (ABTI_IS_ERROR_CHECK_ENABLED &&                                     \
            ABTU_unlikely(p == (ABTI_thread *)NULL)) {                         \
            HANDLE_ERROR_FUNC_WITH_CODE(ABT_ERR_INV_THREAD);                   \
            return ABT_ERR_INV_THREAD;                                         \
        }                                                                      \
    } while (0)

#define ABTI_CHECK_NULL_YTHREAD_PTR(p)                                         \
    do {                                                                       \
        if (ABTI_IS_ERROR_CHECK_ENABLED &&                                     \
            ABTU_unlikely(p == (ABTI_ythread *)NULL)) {                        \
            HANDLE_ERROR_FUNC_WITH_CODE(ABT_ERR_INV_THREAD);                   \
            return ABT_ERR_INV_THREAD;                                         \
        }                                                                      \
    } while (0)

#define ABTI_CHECK_NULL_THREAD_ATTR_PTR(p)                                     \
    do {                                                                       \
        if (ABTI_IS_ERROR_CHECK_ENABLED &&                                     \
            ABTU_unlikely(p == (ABTI_thread_attr *)NULL)) {                    \
            HANDLE_ERROR_FUNC_WITH_CODE(ABT_ERR_INV_THREAD_ATTR);              \
            return ABT_ERR_INV_THREAD_ATTR;                                    \
        }                                                                      \
    } while (0)

#define ABTI_CHECK_NULL_TASK_PTR(p)                                            \
    do {                                                                       \
        if (ABTI_IS_ERROR_CHECK_ENABLED &&                                     \
            ABTU_unlikely(p == (ABTI_thread *)NULL)) {                         \
            HANDLE_ERROR_FUNC_WITH_CODE(ABT_ERR_INV_TASK);                     \
            return ABT_ERR_INV_TASK;                                           \
        }                                                                      \
    } while (0)

#define ABTI_CHECK_NULL_KEY_PTR(p)                                             \
    do {                                                                       \
        if (ABTI_IS_ERROR_CHECK_ENABLED &&                                     \
            ABTU_unlikely(p == (ABTI_key *)NULL)) {                            \
            HANDLE_ERROR_FUNC_WITH_CODE(ABT_ERR_INV_KEY);                      \
            return ABT_ERR_INV_KEY;                                            \
        }                                                                      \
    } while (0)

#define ABTI_CHECK_NULL_MUTEX_PTR(p)                                           \
    do {                                                                       \
        if (ABTI_IS_ERROR_CHECK_ENABLED &&                                     \
            ABTU_unlikely(p == (ABTI_mutex *)NULL)) {                          \
            HANDLE_ERROR_FUNC_WITH_CODE(ABT_ERR_INV_MUTEX);                    \
            return ABT_ERR_INV_MUTEX;                                          \
        }                                                                      \
    } while (0)

#define ABTI_CHECK_NULL_MUTEX_ATTR_PTR(p)                                      \
    do {                                                                       \
        if (ABTI_IS_ERROR_CHECK_ENABLED &&                                     \
            ABTU_unlikely(p == (ABTI_mutex_attr *)NULL)) {                     \
            HANDLE_ERROR_FUNC_WITH_CODE(ABT_ERR_INV_MUTEX_ATTR);               \
            return ABT_ERR_INV_MUTEX_ATTR;                                     \
        }                                                                      \
    } while (0)

#define ABTI_CHECK_NULL_COND_PTR(p)                                            \
    do {                                                                       \
        if (ABTI_IS_ERROR_CHECK_ENABLED &&                                     \
            ABTU_unlikely(p == (ABTI_cond *)NULL)) {                           \
            HANDLE_ERROR_FUNC_WITH_CODE(ABT_ERR_INV_COND);                     \
            return ABT_ERR_INV_COND;                                           \
        }                                                                      \
    } while (0)

#define ABTI_CHECK_NULL_RWLOCK_PTR(p)                                          \
    do {                                                                       \
        if (ABTI_IS_ERROR_CHECK_ENABLED &&                                     \
            ABTU_unlikely(p == (ABTI_rwlock *)NULL)) {                         \
            HANDLE_ERROR_FUNC_WITH_CODE(ABT_ERR_INV_RWLOCK);                   \
            return ABT_ERR_INV_RWLOCK;                                         \
        }                                                                      \
    } while (0)

#define ABTI_CHECK_NULL_FUTURE_PTR(p)                                          \
    do {                                                                       \
        if (ABTI_IS_ERROR_CHECK_ENABLED &&                                     \
            ABTU_unlikely(p == (ABTI_future *)NULL)) {                         \
            HANDLE_ERROR_FUNC_WITH_CODE(ABT_ERR_INV_FUTURE);                   \
            return ABT_ERR_INV_FUTURE;                                         \
        }                                                                      \
    } while (0)

#define ABTI_CHECK_NULL_EVENTUAL_PTR(p)                                        \
    do {                                                                       \
        if (ABTI_IS_ERROR_CHECK_ENABLED &&                                     \
            ABTU_unlikely(p == (ABTI_eventual *)NULL)) {                       \
            HANDLE_ERROR_FUNC_WITH_CODE(ABT_ERR_INV_EVENTUAL);                 \
            return ABT_ERR_INV_EVENTUAL;                                       \
        }                                                                      \
    } while (0)

#define ABTI_CHECK_NULL_BARRIER_PTR(p)                                         \
    do {                                                                       \
        if (ABTI_IS_ERROR_CHECK_ENABLED &&                                     \
            ABTU_unlikely(p == (ABTI_barrier *)NULL)) {                        \
            HANDLE_ERROR_FUNC_WITH_CODE(ABT_ERR_INV_BARRIER);                  \
            return ABT_ERR_INV_BARRIER;                                        \
        }                                                                      \
    } while (0)

#define ABTI_CHECK_NULL_XSTREAM_BARRIER_PTR(p)                                 \
    do {                                                                       \
        if (ABTI_IS_ERROR_CHECK_ENABLED &&                                     \
            ABTU_unlikely(p == (ABTI_xstream_barrier *)NULL)) {                \
            HANDLE_ERROR_FUNC_WITH_CODE(ABT_ERR_INV_XSTREAM_BARRIER);          \
            return ABT_ERR_INV_XSTREAM_BARRIER;                                \
        }                                                                      \
    } while (0)

#define ABTI_CHECK_NULL_TIMER_PTR(p)                                           \
    do {                                                                       \
        if (ABTI_IS_ERROR_CHECK_ENABLED &&                                     \
            ABTU_unlikely(p == (ABTI_timer *)NULL)) {                          \
            HANDLE_ERROR_FUNC_WITH_CODE(ABT_ERR_INV_TIMER);                    \
            return ABT_ERR_INV_TIMER;                                          \
        }                                                                      \
    } while (0)

#define ABTI_CHECK_NULL_TOOL_CONTEXT_PTR(p)                                    \
    do {                                                                       \
        if (ABTI_IS_ERROR_CHECK_ENABLED &&                                     \
            ABTU_unlikely(p == (ABTI_tool_context *)NULL)) {                   \
            HANDLE_ERROR_FUNC_WITH_CODE(ABT_ERR_INV_TOOL_CONTEXT);             \
            return ABT_ERR_INV_TOOL_CONTEXT;                                   \
        }                                                                      \
    } while (0)

#endif /* ABTI_ERROR_H_INCLUDED */

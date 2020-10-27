/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

#ifndef ABTI_THREAD_H_INCLUDED
#define ABTI_THREAD_H_INCLUDED

static inline ABTI_thread *ABTI_thread_get_ptr(ABT_thread thread)
{
#ifndef ABT_CONFIG_DISABLE_ERROR_CHECK
    ABTI_thread *p_thread;
    if (thread == ABT_THREAD_NULL) {
        p_thread = NULL;
    } else {
        p_thread = (ABTI_thread *)thread;
    }
    return p_thread;
#else
    return (ABTI_thread *)thread;
#endif
}

static inline ABT_thread ABTI_thread_get_handle(ABTI_thread *p_thread)
{
#ifndef ABT_CONFIG_DISABLE_ERROR_CHECK
    ABT_thread h_thread;
    if (p_thread == NULL) {
        h_thread = ABT_THREAD_NULL;
    } else {
        h_thread = (ABT_thread)p_thread;
    }
    return h_thread;
#else
    return (ABT_thread)p_thread;
#endif
}

/* Inlined functions for User-level Thread (ULT) */

static inline ABT_unit_type ABTI_thread_type_get_type(ABTI_thread_type type)
{
    if (type & ABTI_THREAD_TYPE_YIELDABLE) {
        return ABT_UNIT_TYPE_THREAD;
    } else if (type == ABTI_THREAD_TYPE_EXT) {
        return ABT_UNIT_TYPE_EXT;
    } else {
        return ABT_UNIT_TYPE_TASK;
    }
}

static inline ABTI_ythread *ABTI_thread_get_ythread(ABTI_thread *p_thread)
{
    ABTI_STATIC_ASSERT(offsetof(ABTI_ythread, thread) == 0);
    return (ABTI_ythread *)p_thread;
}

static inline ABTI_ythread *
ABTI_thread_get_ythread_or_null(ABTI_thread *p_thread)
{
    if (p_thread->type & ABTI_THREAD_TYPE_YIELDABLE) {
        return ABTI_thread_get_ythread(p_thread);
    } else {
        return NULL;
    }
}

static inline void ABTI_thread_set_request(ABTI_thread *p_thread, uint32_t req)
{
    ABTD_atomic_fetch_or_uint32(&p_thread->request, req);
}

static inline void ABTI_thread_unset_request(ABTI_thread *p_thread,
                                             uint32_t req)
{
    ABTD_atomic_fetch_and_uint32(&p_thread->request, ~req);
}

#endif /* ABTI_THREAD_H_INCLUDED */

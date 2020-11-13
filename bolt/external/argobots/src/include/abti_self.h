/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

#ifndef ABTI_SELF_H_INCLUDED
#define ABTI_SELF_H_INCLUDED

static inline ABTI_native_thread_id
ABTI_self_get_native_thread_id(ABTI_local *p_local)
{
    ABTI_xstream *p_local_xstream = ABTI_local_get_xstream_or_null(p_local);
    /* This is when an external thread called this routine. */
    if (ABTI_IS_EXT_THREAD_ENABLED && p_local_xstream == NULL) {
        /* A pointer to a thread local variable can distinguish all external
         * threads and execution streams. */
        return (ABTI_native_thread_id)ABTI_local_get_local_ptr();
    }
    return (ABTI_native_thread_id)p_local_xstream;
}

static inline ABTI_thread_id ABTI_self_get_thread_id(ABTI_local *p_local)
{
    ABTI_xstream *p_local_xstream = ABTI_local_get_xstream_or_null(p_local);
    /* This is when an external thread called this routine. */
    if (ABTI_IS_EXT_THREAD_ENABLED && p_local_xstream == NULL) {
        /* A pointer to a thread local variable is unique to an external thread
         * and its value is different from pointers to ULTs and tasks. */
        return (ABTI_thread_id)ABTI_local_get_local_ptr();
    }
    return (ABTI_thread_id)p_local_xstream->p_thread;
}

#endif /* ABTI_SELF_H_INCLUDED */

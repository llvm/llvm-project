/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

#ifndef ABTI_XSTREAM_H_INCLUDED
#define ABTI_XSTREAM_H_INCLUDED

/* Inlined functions for Execution Stream (ES) */

static inline ABTI_xstream *ABTI_xstream_get_ptr(ABT_xstream xstream)
{
#ifndef ABT_CONFIG_DISABLE_ERROR_CHECK
    ABTI_xstream *p_xstream;
    if (xstream == ABT_XSTREAM_NULL) {
        p_xstream = NULL;
    } else {
        p_xstream = (ABTI_xstream *)xstream;
    }
    return p_xstream;
#else
    return (ABTI_xstream *)xstream;
#endif
}

static inline ABT_xstream ABTI_xstream_get_handle(ABTI_xstream *p_xstream)
{
#ifndef ABT_CONFIG_DISABLE_ERROR_CHECK
    ABT_xstream h_xstream;
    if (p_xstream == NULL) {
        h_xstream = ABT_XSTREAM_NULL;
    } else {
        h_xstream = (ABT_xstream)p_xstream;
    }
    return h_xstream;
#else
    return (ABT_xstream)p_xstream;
#endif
}

/* Get the first pool of the main scheduler. */
static inline ABTI_pool *ABTI_xstream_get_main_pool(ABTI_xstream *p_xstream)
{
    ABT_pool pool = p_xstream->p_main_sched->pools[0];
    return ABTI_pool_get_ptr(pool);
}

static inline void ABTI_xstream_terminate_thread(ABTI_local *p_local,
                                                 ABTI_thread *p_thread)
{
    LOG_DEBUG("[U%" PRIu64 ":E%d] terminated\n", ABTI_thread_get_id(p_thread),
              p_thread->p_last_xstream->rank);
    if (!(p_thread->type & ABTI_THREAD_TYPE_NAMED)) {
        ABTD_atomic_release_store_int(&p_thread->state,
                                      ABT_THREAD_STATE_TERMINATED);
        ABTI_thread_free(p_local, p_thread);
    } else {
        /* NOTE: We set the ULT's state as TERMINATED after checking refcount
         * because the ULT can be freed on a different ES.  In other words, we
         * must not access any field of p_thead after changing the state to
         * TERMINATED. */
        ABTD_atomic_release_store_int(&p_thread->state,
                                      ABT_THREAD_STATE_TERMINATED);
    }
}

static inline ABTI_local *ABTI_xstream_get_local(ABTI_xstream *p_xstream)
{
    return (ABTI_local *)p_xstream;
}

#endif /* ABTI_XSTREAM_H_INCLUDED */

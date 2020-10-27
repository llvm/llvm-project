/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

#ifndef ABTI_FUTURE_H_INCLUDED
#define ABTI_FUTURE_H_INCLUDED

/* Inlined functions for Future */

static inline ABTI_future *ABTI_future_get_ptr(ABT_future future)
{
#ifndef ABT_CONFIG_DISABLE_ERROR_CHECK
    ABTI_future *p_future;
    if (future == ABT_FUTURE_NULL) {
        p_future = NULL;
    } else {
        p_future = (ABTI_future *)future;
    }
    return p_future;
#else
    return (ABTI_future *)future;
#endif
}

static inline ABT_future ABTI_future_get_handle(ABTI_future *p_future)
{
#ifndef ABT_CONFIG_DISABLE_ERROR_CHECK
    ABT_future h_future;
    if (p_future == NULL) {
        h_future = ABT_FUTURE_NULL;
    } else {
        h_future = (ABT_future)p_future;
    }
    return h_future;
#else
    return (ABT_future)p_future;
#endif
}

#endif /* ABTI_FUTURE_H_INCLUDED */

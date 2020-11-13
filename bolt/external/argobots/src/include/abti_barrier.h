/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

#ifndef ABTI_BARRIER_H_INCLUDED
#define ABTI_BARRIER_H_INCLUDED

/* Inlined functions for Barrier */

/* Barrier */
static inline ABTI_barrier *ABTI_barrier_get_ptr(ABT_barrier barrier)
{
#ifndef ABT_CONFIG_DISABLE_ERROR_CHECK
    ABTI_barrier *p_barrier;
    if (barrier == ABT_BARRIER_NULL) {
        p_barrier = NULL;
    } else {
        p_barrier = (ABTI_barrier *)barrier;
    }
    return p_barrier;
#else
    return (ABTI_barrier *)barrier;
#endif
}

static inline ABT_barrier ABTI_barrier_get_handle(ABTI_barrier *p_barrier)
{
#ifndef ABT_CONFIG_DISABLE_ERROR_CHECK
    ABT_barrier h_barrier;
    if (p_barrier == NULL) {
        h_barrier = ABT_BARRIER_NULL;
    } else {
        h_barrier = (ABT_barrier)p_barrier;
    }
    return h_barrier;
#else
    return (ABT_barrier)p_barrier;
#endif
}

#endif /* ABTI_BARRIER_H_INCLUDED */

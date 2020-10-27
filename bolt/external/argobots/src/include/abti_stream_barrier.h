/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

#ifndef ABTI_XSTREAM_BARRIER_H_INCLUDED
#define ABTI_XSTREAM_BARRIER_H_INCLUDED

#ifdef HAVE_PTHREAD_BARRIER_INIT
static inline ABTI_xstream_barrier *
ABTI_xstream_barrier_get_ptr(ABT_xstream_barrier barrier)
{
#ifndef ABT_CONFIG_DISABLE_ERROR_CHECK
    ABTI_xstream_barrier *p_barrier;
    if (barrier == ABT_XSTREAM_BARRIER_NULL) {
        p_barrier = NULL;
    } else {
        p_barrier = (ABTI_xstream_barrier *)barrier;
    }
    return p_barrier;
#else
    return (ABTI_xstream_barrier *)barrier;
#endif
}

static inline ABT_xstream_barrier
ABTI_xstream_barrier_get_handle(ABTI_xstream_barrier *p_barrier)
{
#ifndef ABT_CONFIG_DISABLE_ERROR_CHECK
    ABT_xstream_barrier h_barrier;
    if (p_barrier == NULL) {
        h_barrier = ABT_XSTREAM_BARRIER_NULL;
    } else {
        h_barrier = (ABT_xstream_barrier)p_barrier;
    }
    return h_barrier;
#else
    return (ABT_xstream_barrier)p_barrier;
#endif
}
#endif

#endif /* ABTI_XSTREAM_BARRIER_H_INCLUDED */

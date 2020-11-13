/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

#ifndef ABTD_STREAM_H_INCLUDED
#define ABTD_STREAM_H_INCLUDED

#ifdef HAVE_PTHREAD_BARRIER_INIT
ABTU_ret_err static inline int
ABTD_xstream_barrier_init(uint32_t num_waiters, ABTD_xstream_barrier *p_barrier)
{
    int ret = pthread_barrier_init(p_barrier, NULL, num_waiters);
    return (ret == 0) ? ABT_SUCCESS : ABT_ERR_XSTREAM_BARRIER;
}

static inline void ABTD_xstream_barrier_destroy(ABTD_xstream_barrier *p_barrier)
{
    int ret = pthread_barrier_destroy(p_barrier);
    ABTI_ASSERT(ret == 0);
}

static inline void ABTD_xstream_barrier_wait(ABTD_xstream_barrier *p_barrier)
{
    int ret = pthread_barrier_wait(p_barrier);
    ABTI_ASSERT(ret == PTHREAD_BARRIER_SERIAL_THREAD || ret == 0);
}
#endif

#endif /* ABTD_STREAM_H_INCLUDED */

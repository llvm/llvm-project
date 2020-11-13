/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

#ifndef ABTI_SCHED_H_INCLUDED
#define ABTI_SCHED_H_INCLUDED

/* Inlined functions for Scheduler */

static inline ABTI_sched *ABTI_sched_get_ptr(ABT_sched sched)
{
#ifndef ABT_CONFIG_DISABLE_ERROR_CHECK
    ABTI_sched *p_sched;
    if (sched == ABT_SCHED_NULL) {
        p_sched = NULL;
    } else {
        p_sched = (ABTI_sched *)sched;
    }
    return p_sched;
#else
    return (ABTI_sched *)sched;
#endif
}

static inline ABT_sched ABTI_sched_get_handle(ABTI_sched *p_sched)
{
#ifndef ABT_CONFIG_DISABLE_ERROR_CHECK
    ABT_sched h_sched;
    if (p_sched == NULL) {
        h_sched = ABT_SCHED_NULL;
    } else {
        h_sched = (ABT_sched)p_sched;
    }
    return h_sched;
#else
    return (ABT_sched)p_sched;
#endif
}

/* Set `used` of p_sched to NOT_USED and free p_sched if its `automatic` is
 * ABT_TRUE, which means it is safe to free p_sched inside the runtime. */
static inline void ABTI_sched_discard_and_free(ABTI_local *p_local,
                                               ABTI_sched *p_sched,
                                               ABT_bool force_free)
{
    p_sched->used = ABTI_SCHED_NOT_USED;
    if (p_sched->automatic == ABT_TRUE || force_free) {
        ABTI_sched_free(p_local, p_sched, force_free);
    } else {
        /* Threads should be discarded here. */
        ABTI_thread_free(p_local, &p_sched->p_ythread->thread);
        p_sched->p_ythread = NULL;
    }
}

static inline void ABTI_sched_set_request(ABTI_sched *p_sched, uint32_t req)
{
    ABTD_atomic_fetch_or_uint32(&p_sched->request, req);
}

static inline void ABTI_sched_unset_request(ABTI_sched *p_sched, uint32_t req)
{
    ABTD_atomic_fetch_and_uint32(&p_sched->request, ~req);
}

#ifdef ABT_CONFIG_USE_SCHED_SLEEP
#define CNT_DECL(c) int c
#define CNT_INIT(c, v) c = v
#define CNT_INC(c) c++
#define SCHED_SLEEP(c, t)                                                      \
    if (c == 0)                                                                \
    nanosleep(&(t), NULL)
#else
#define CNT_DECL(c)
#define CNT_INIT(c, v)
#define CNT_INC(c)
#define SCHED_SLEEP(c, t)
#endif

#endif /* ABTI_SCHED_H_INCLUDED */

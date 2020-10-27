/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include "abt.h"
#include "abttest.h"

#define DEFAULT_NUM_XSTREAMS 4
#define DEFAULT_NUM_THREADS 4

static ABT_mutex mutex = ABT_MUTEX_NULL;
static ABT_cond cond = ABT_COND_NULL;
static int g_counter = 0;

void cond_test(void *arg)
{
    int ret;
    struct timespec ts;
    struct timeval tv;
    int eid;
    ABT_unit_id tid;

    ret = ABT_xstream_self_rank(&eid);
    ATS_ERROR(ret, "ABT_xstream_self_rank");
    ret = ABT_thread_self_id(&tid);
    ATS_ERROR(ret, "ABT_thread_self_id");

    ret = gettimeofday(&tv, NULL);
    assert(!ret);

    ts.tv_sec = tv.tv_sec;
    ts.tv_nsec = tv.tv_usec * 1000;
    ts.tv_sec += 1;

    ret = ABT_mutex_lock(mutex);
    ATS_ERROR(ret, "ABT_mutex_lock");

    ATS_printf(1, "[U%d:E%d] blocked\n", (int)tid, eid);
    ret = ABT_cond_timedwait(cond, mutex, &ts);
    if (ret == ABT_ERR_COND_TIMEDOUT) {
        g_counter++;
        ATS_printf(1, "[U%d:E%d] cond timed out\n", (int)tid, eid);
        ret = ABT_mutex_unlock(mutex);
        ATS_ERROR(ret, "ABT_mutex_unlock");
        ABT_thread_exit();
    }
    ATS_ERROR(ret, "ABT_cond_timedwait");
    ATS_printf(1, "[U%d:E%d] cond waken up\n", (int)tid, eid);

    g_counter++;
    ret = ABT_mutex_unlock(mutex);
    ATS_ERROR(ret, "ABT_mutex_unlock");
}

int main(int argc, char *argv[])
{
    ABT_xstream *xstreams;
    ABT_pool *pools;
    ABT_thread *threads;
    int num_xstreams;
    int num_threads;
    int ret, i, pidx = 0;
    int eid;
    ABT_unit_id tid;

    /* Initialize */
    ATS_read_args(argc, argv);
    if (argc < 2) {
        num_xstreams = DEFAULT_NUM_XSTREAMS;
        num_threads = DEFAULT_NUM_THREADS;
    } else {
        num_xstreams = ATS_get_arg_val(ATS_ARG_N_ES);
        num_threads = ATS_get_arg_val(ATS_ARG_N_ULT);
    }
    ATS_init(argc, argv, num_xstreams);

    ATS_printf(1, "# of ESs : %d\n", num_xstreams);
    ATS_printf(1, "# of ULTs: %d\n", num_threads);

    ret = ABT_xstream_self_rank(&eid);
    ATS_ERROR(ret, "ABT_xstream_self_rank");
    ret = ABT_thread_self_id(&tid);
    ATS_ERROR(ret, "ABT_thread_self_id");

    xstreams = (ABT_xstream *)malloc(num_xstreams * sizeof(ABT_xstream));
    pools = (ABT_pool *)malloc(num_xstreams * sizeof(ABT_pool));
    threads = (ABT_thread *)malloc(num_threads * sizeof(ABT_thread));

    /* Create a mutex */
    ret = ABT_mutex_create(&mutex);
    ATS_ERROR(ret, "ABT_mutex_create");

    /* Create a condition variable */
    ret = ABT_cond_create(&cond);
    ATS_ERROR(ret, "ABT_cond_create");

    /* Create ESs */
    ret = ABT_xstream_self(&xstreams[0]);
    ATS_ERROR(ret, "ABT_xstream_self");
    for (i = 1; i < num_xstreams; i++) {
        ret = ABT_xstream_create(ABT_SCHED_NULL, &xstreams[i]);
        ATS_ERROR(ret, "ABT_xstream_create");
    }

    /* Get the pools */
    for (i = 0; i < num_xstreams; i++) {
        ret = ABT_xstream_get_main_pools(xstreams[i], 1, &pools[i]);
        ATS_ERROR(ret, "ABT_xstream_get_main_pools");
    }

    /* Create ULTs */
    for (i = 0; i < num_threads; i++) {
        ret = ABT_thread_create(pools[pidx], cond_test, NULL,
                                ABT_THREAD_ATTR_NULL, &threads[i]);
        ATS_ERROR(ret, "ABT_thread_create");
        pidx = (pidx + 1) % num_xstreams;
    }

    ABT_thread_yield();

    ret = ABT_mutex_lock(mutex);
    ATS_ERROR(ret, "ABT_mutex_lock");

    ret = ABT_cond_broadcast(cond);
    ATS_ERROR(ret, "ABT_cond_broadcast");
    ATS_printf(1, "[U%d:E%d] cond_broadcast\n", (int)tid, eid);

    ret = ABT_mutex_unlock(mutex);
    ATS_ERROR(ret, "ABT_mutex_unlock");

    /* Join and free ULTs */
    for (i = 0; i < num_threads; i++) {
        ret = ABT_thread_free(&threads[i]);
        ATS_ERROR(ret, "ABT_thread_free");
    }

    /* Join and free ESs */
    for (i = 1; i < num_xstreams; i++) {
        ret = ABT_xstream_join(xstreams[i]);
        ATS_ERROR(ret, "ABT_xstream_join");
        ret = ABT_xstream_free(&xstreams[i]);
        ATS_ERROR(ret, "ABT_xstream_free");
    }

    /* Free the mutex */
    ret = ABT_mutex_free(&mutex);
    ATS_ERROR(ret, "ABT_mutex_free");

    /* Free the condition variables */
    ret = ABT_cond_free(&cond);
    ATS_ERROR(ret, "ABT_cond_free");

    /* Validation */
    int expected = num_threads;
    if (g_counter != expected) {
        printf("g_counter = %d (expected: %d)\n", g_counter, expected);
    }

    /* Finalize */
    ret = ATS_finalize(g_counter != expected);

    free(threads);
    free(pools);
    free(xstreams);

    return ret;
}

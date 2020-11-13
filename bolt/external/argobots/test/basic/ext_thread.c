/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include "abt.h"
#include "abttest.h"

/* This code tests that a pthread can call Argobots APIs without any problems.
 * When a pthread calls Arogobots APIs, it might get an error code as a return
 * value, but there should be no problems like segmentation fault. */

#define BUF_SIZE 10

ABT_mutex g_mutex = ABT_MUTEX_NULL;
ABT_cond g_cond = ABT_COND_NULL;
ABT_eventual g_eventual[2] = { ABT_EVENTUAL_NULL, ABT_EVENTUAL_NULL };
ABT_future g_future[2] = { ABT_FUTURE_NULL, ABT_FUTURE_NULL };
int g_counter = 0;
volatile int g_threads = 0;

void task_func(void *arg)
{
    ABT_xstream xstream;
    int rank, ret;
    int my_id = (int)(size_t)arg;

    ret = ABT_xstream_self(&xstream);
    ATS_ERROR(ret, "ABT_xstream_self");
    ret = ABT_xstream_self_rank(&rank);
    ATS_ERROR(ret, "ABT_xstream_self_rank");

    ATS_printf(1, "TASK %d: running on ES %d\n", my_id, rank);
}

void thread_func(void *arg)
{
    ABT_xstream xstream;
    int rank, ret;
    int my_id = (int)(size_t)arg;

    ret = ABT_xstream_self(&xstream);
    ATS_ERROR(ret, "ABT_xstream_self");
    ret = ABT_xstream_self_rank(&rank);
    ATS_ERROR(ret, "ABT_xstream_self_rank");

    ABT_mutex_lock(g_mutex);
    g_counter++;
    ABT_mutex_unlock(g_mutex);
    ATS_printf(1, "ULT %d: mutex passed\n", my_id);

    /* ULT are waiting, and pthread will broadcast the signal. */
    ABT_mutex_lock(g_mutex);
    g_threads++;
    ABT_cond_wait(g_cond, g_mutex);
    ABT_mutex_unlock(g_mutex);
    ATS_printf(1, "ULT %d: cond #1 passed\n", my_id);

    /* ULT 1 and pthread are waiting, and ULT 0 broadcasts. */
    if (my_id == 0) {
        while (g_threads < 2)
            ABT_thread_yield();
        ABT_mutex_lock(g_mutex);
        ABT_cond_broadcast(g_cond);
        ABT_mutex_unlock(g_mutex);
    } else {
        ABT_mutex_lock(g_mutex);
        g_threads++;
        ABT_cond_wait(g_cond, g_mutex);
        ABT_mutex_unlock(g_mutex);
    }
    ATS_printf(1, "ULT %d: cond #2 passed\n", my_id);

    /* Test eventual */
    void *evt_result;
    char *buf_evt = (char *)malloc(BUF_SIZE);
    ABT_eventual_wait(g_eventual[0], &evt_result);
    ATS_printf(1, "ULT %d: eventual #1 passed\n", my_id);
    if (my_id == 0) {
        ABT_eventual_wait(g_eventual[1], &evt_result);
    } else {
        ABT_eventual_set(g_eventual[1], buf_evt, BUF_SIZE);
    }
    ATS_printf(1, "ULT %d: eventual #2 passed\n", my_id);
    free(buf_evt);

    /* Test future */
    char *buf_fut = (char *)malloc(BUF_SIZE);
    if (my_id == 0) {
        ABT_future_set(g_future[0], (void *)buf_fut);
    } else {
        ABT_future_wait(g_future[0]);
    }
    ATS_printf(1, "ULT %d: future #1 passed\n", my_id);
    ABT_future_wait(g_future[1]);
    ATS_printf(1, "ULT %d: future #2 passed\n", my_id);
    free(buf_fut);

    ATS_printf(1, "ULT %d running on ES %d\n", my_id, rank);
}

void *pthread_test(void *arg)
{
    ABT_xstream xstreams[2];
    ABT_pool pools[2];
    ABT_thread threads[2];
    ABT_task tasks[2];
    size_t i;
    int ret;

    /* Execution Streams */
    for (i = 0; i < 2; i++) {
        ret = ABT_xstream_create(ABT_SCHED_NULL, &xstreams[i]);
        ATS_ERROR(ret, "ABT_xstream_create");
    }

    /* Create synchronization objects */
    ret = ABT_mutex_create(&g_mutex);
    ATS_ERROR(ret, "ABT_mutex_create");
    ret = ABT_cond_create(&g_cond);
    ATS_ERROR(ret, "ABT_cond_create");
    for (i = 0; i < 2; i++) {
        ret = ABT_eventual_create(BUF_SIZE, &g_eventual[i]);
        ATS_ERROR(ret, "ABT_eventual_create");
        ret = ABT_future_create(1, NULL, &g_future[i]);
        ATS_ERROR(ret, "ABT_future_create");
    }

    /* Create ULTs and tasklets */
    for (i = 0; i < 2; i++) {
        ret = ABT_xstream_get_main_pools(xstreams[i], 1, &pools[i]);
        ATS_ERROR(ret, "ABT_xstream_get_main_pools");

        ret = ABT_thread_create(pools[i], thread_func, (void *)i,
                                ABT_THREAD_ATTR_NULL, &threads[i]);
        ATS_ERROR(ret, "ABT_thread_create");

        ret = ABT_task_create(pools[i], task_func, (void *)i, &tasks[i]);
        ATS_ERROR(ret, "ABT_thread_create");
    }

    /* Test mutex */
    ABT_mutex_lock(g_mutex);
    g_counter++;
    ABT_mutex_unlock(g_mutex);
    ATS_printf(1, "pthread: mutex passed\n");

    /* Test condition variable */
    /* Wake up other ULTs that are waiting on the condition variable */
    while (g_threads < 2)
        ;
    ABT_mutex_lock(g_mutex);
    g_threads = 0;
    ABT_cond_broadcast(g_cond);
    ABT_mutex_unlock(g_mutex);
    ATS_printf(1, "pthread: cond #1 passed\n");

    /* Wait on the condition variable */
    ABT_mutex_lock(g_mutex);
    g_threads++;
    ABT_cond_wait(g_cond, g_mutex);
    ABT_mutex_unlock(g_mutex);
    ATS_printf(1, "pthread: cond #2 passed\n");

    /* Test eventual */
    void *evt_result;
    char *buf_evt = (char *)malloc(BUF_SIZE);
    ABT_eventual_set(g_eventual[0], buf_evt, BUF_SIZE);
    ATS_printf(1, "pthread: eventual #1 passed\n");
    ABT_eventual_wait(g_eventual[1], &evt_result);
    ATS_printf(1, "pthread: eventual #2 passed\n");
    free(buf_evt);

    /* Test future */
    char *buf_fut = (char *)malloc(BUF_SIZE);
    ABT_future_wait(g_future[0]);
    ATS_printf(1, "pthread: future #1 passed\n");
    ABT_future_set(g_future[1], (void *)buf_fut);
    ATS_printf(1, "pthread: future #2 passed\n");
    free(buf_fut);

    /* Join */
    for (i = 0; i < 2; i++) {
        ret = ABT_thread_join(threads[i]);
        ATS_ERROR(ret, "ABT_thread_join");
    }
    for (i = 0; i < 2; i++) {
        ret = ABT_xstream_join(xstreams[i]);
        ATS_ERROR(ret, "ABT_xstream_join");
    }

    /* Free */
    ret = ABT_mutex_free(&g_mutex);
    ATS_ERROR(ret, "ABT_mutex_free");
    ret = ABT_cond_free(&g_cond);
    ATS_ERROR(ret, "ABT_cond_free");
    for (i = 0; i < 2; i++) {
        ret = ABT_eventual_free(&g_eventual[i]);
        ATS_ERROR(ret, "ABT_eventual_free");
        ret = ABT_future_free(&g_future[i]);
        ATS_ERROR(ret, "ABT_future_free");

        ret = ABT_thread_free(&threads[i]);
        ATS_ERROR(ret, "ABT_thread_free");
        ret = ABT_task_free(&tasks[i]);
        ATS_ERROR(ret, "ABT_task_free");
        ret = ABT_xstream_free(&xstreams[i]);
        ATS_ERROR(ret, "ABT_xstream_free");
    }

    /* Validation */
    ATS_printf(1, "g_counter=%d\n", g_counter);
    assert(g_counter == 3);

    return NULL;
}

int main(int argc, char *argv[])
{
    pthread_t pthread;
    int ret;

    /* Initialize */
    ATS_read_args(argc, argv);
    ATS_init(argc, argv, 1);

    /* Create a pthread */
    ret = pthread_create(&pthread, NULL, pthread_test, NULL);
    assert(ret == 0);

    /* Join the pthread */
    ret = pthread_join(pthread, NULL);
    assert(ret == 0);

    /* Finalize */
    return ATS_finalize(0);
}

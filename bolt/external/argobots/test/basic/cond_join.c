/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

#include <stdio.h>
#include <stdlib.h>
#include "abt.h"
#include "abttest.h"

#define NUM_THREADS 2
#define NUM_XSTREAMS NUM_THREADS

typedef struct {
    ABT_mutex mutex;
    ABT_cond cond;
    int count;
    volatile int curcount;
    volatile int generation;
} barrier_t;

typedef struct {
    int eid;
    int uid;
    int tid;
    barrier_t *barrier;
} thread_arg_t;

barrier_t *barrier_create(int count)
{
    int ret;
    barrier_t *barrier = (barrier_t *)malloc(sizeof(barrier_t));

    ret = ABT_mutex_create(&barrier->mutex);
    ATS_ERROR(ret, "ABT_mutex_create");

    ret = ABT_cond_create(&barrier->cond);
    ATS_ERROR(ret, "ABT_cond_create");

    barrier->count = count;
    barrier->curcount = 0;
    barrier->generation = 0;

    return barrier;
}

void barrier_free(barrier_t *barrier)
{
    int ret;

    ret = ABT_mutex_free(&barrier->mutex);
    ATS_ERROR(ret, "ABT_mutex_free");

    ret = ABT_cond_free(&barrier->cond);
    ATS_ERROR(ret, "ABT_cond_free");

    free(barrier);
}

void thread_wait(thread_arg_t *my_arg)
{
    int generation;
    barrier_t *barrier = my_arg->barrier;

    ABT_mutex_lock(barrier->mutex);
    if ((barrier->curcount + 1) == barrier->count) {
        barrier->generation++;
        barrier->curcount = 0;

        ATS_printf(3, "<S%d:TH%d> T%d broadcast-1\n", my_arg->eid, my_arg->uid,
                   my_arg->tid);
        ABT_cond_broadcast(barrier->cond);
        ABT_mutex_unlock(barrier->mutex);
        ATS_printf(3, "<S%d:TH%d> T%d broadcast-2\n", my_arg->eid, my_arg->uid,
                   my_arg->tid);
        return;
    }
    barrier->curcount++;
    generation = barrier->generation;
    do {
        ATS_printf(3, "<S%d:TH%d> T%d wait-1\n", my_arg->eid, my_arg->uid,
                   my_arg->tid);
        ABT_cond_wait(barrier->cond, barrier->mutex);
        ATS_printf(3, "<S%d:TH%d> T%d wait-2\n", my_arg->eid, my_arg->uid,
                   my_arg->tid);
    } while (generation == barrier->generation);
    ABT_mutex_unlock(barrier->mutex);
    ATS_printf(3, "<S%d:TH%d> T%d wait-3\n", my_arg->eid, my_arg->uid,
               my_arg->tid);
}

void cond_test(void *arg)
{
    thread_arg_t *my_arg = (thread_arg_t *)arg;
    int i;

    ABT_unit_id tid;
    ABT_thread thread;
    ABT_thread_self(&thread);
    ABT_thread_get_id(thread, &tid);
    my_arg->uid = tid;
    ABT_xstream_self_rank(&my_arg->eid);

    for (i = 0; i < 2000; i++) {
        thread_wait(my_arg);
    }

    ABT_thread_exit();
}

void thread_work(void *arg)
{
    ABT_xstream xstreams[NUM_XSTREAMS];
    ABT_pool pools[NUM_XSTREAMS];
    ABT_thread threads[NUM_THREADS];
    thread_arg_t args[NUM_THREADS];
    barrier_t *barrier;
    int i, t, ret;
    int iter = (int)(size_t)arg;

    for (t = 0; t < iter; t++) {
        ATS_printf(2, "t=%d\n", t);
        barrier = barrier_create(NUM_THREADS);

        for (i = 0; i < NUM_THREADS; i++) {
            args[i].tid = i;
            args[i].barrier = barrier;
        }

        /* Create Execution Streams */
        ret = ABT_xstream_self(&xstreams[0]);
        ATS_ERROR(ret, "ABT_xstream_self");
        for (i = 1; i < NUM_XSTREAMS; i++) {
            ret = ABT_xstream_create(ABT_SCHED_NULL, &xstreams[i]);
            ATS_ERROR(ret, "ABT_xstream_create");
        }

        /* Get the first pool of each ES */
        for (i = 0; i < NUM_XSTREAMS; i++) {
            ret = ABT_xstream_get_main_pools(xstreams[i], 1, &pools[i]);
            ATS_ERROR(ret, "ABT_xstream_get_main_pools");
        }

        /* Create ULTs */
        for (i = 0; i < NUM_THREADS; i++) {
            ret = ABT_thread_create(pools[i % NUM_XSTREAMS], cond_test,
                                    (void *)&args[i], ABT_THREAD_ATTR_NULL,
                                    &threads[i]);
            ATS_ERROR(ret, "ABT_thread_create");
        }

        for (i = 0; i < NUM_THREADS; i++) {
            ret = ABT_thread_join(threads[i]);
            ATS_ERROR(ret, "ABT_thread_join");
            ret = ABT_thread_free(&threads[i]);
            ATS_ERROR(ret, "ABT_thread_free");
        }

        barrier_free(barrier);

        /* Join and free Execution Streams */
        for (i = 1; i < NUM_XSTREAMS; i++) {
            ret = ABT_xstream_join(xstreams[i]);
            ATS_ERROR(ret, "ABT_xstream_join");
            ret = ABT_xstream_free(&xstreams[i]);
            ATS_ERROR(ret, "ABT_xstream_free");
        }
    }
}

int main(int argc, char *argv[])
{
    ABT_xstream xstream;
    ABT_pool pool;
    ABT_thread thread;
    int ret;
    int iter = 5;

    if (argc > 1)
        iter = atoi(argv[1]);

    /* Initialize */
    ATS_read_args(argc, argv);

    ATS_init(argc, argv, NUM_XSTREAMS);

    ATS_printf(1, "# of ES   : %d\n", NUM_XSTREAMS);
    ATS_printf(1, "# of ULT  : %d\n", NUM_THREADS);
    ATS_printf(1, "iterations: %d\n", iter);

    ret = ABT_xstream_self(&xstream);
    ATS_ERROR(ret, "ABT_xstream_self");

    ret = ABT_xstream_get_main_pools(xstream, 1, &pool);
    ATS_ERROR(ret, "ABT_xstream_get_main_pools");

    ret = ABT_thread_create(pool, thread_work, (void *)(size_t)iter,
                            ABT_THREAD_ATTR_NULL, &thread);
    ATS_ERROR(ret, "ABT_thread_create");

    ret = ABT_thread_join(thread);
    ATS_ERROR(ret, "ABT_thread_join");
    ret = ABT_thread_free(&thread);
    ATS_ERROR(ret, "ABT_thread_free");

    /* Finalize */
    return ATS_finalize(0);
}

/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

/*
 * In this test, normal ULTs will schedule other ULTs.  This can mimic a
 * stackable scheduler without creating a scheduler.  Termination condition and
 * event handling must be taken care of by a user.
 */

#include <stdio.h>
#include <stdlib.h>
#include "abt.h"
#include "abttest.h"

int num_xstreams;
int num_threads;
#define DEFAULT_NUM_XSTREAMS 4
#define DEFAULT_NUM_THREADS 4
#define NUM_LEAF_THREADS 8
ABT_pool *pools;

volatile int atomic_val = 0;
#define ATOMIC_VAL_SUM (num_threads * num_xstreams * NUM_LEAF_THREADS)

void leaf_func(void *arg)
{
    /* Meaningless yield to check the robustness. */
    int ret = ABT_thread_yield();
    ATS_ERROR(ret, "ABT_thread_yield");
    ATS_atomic_fetch_add(&atomic_val, 1);
}

void root_func(void *arg)
{
    /* root_func creates threads and schedule them. */
    int ret, i;
    for (i = 0; i < NUM_LEAF_THREADS; i++) {
        ret = ABT_thread_create(pools[i % num_xstreams], leaf_func,
                                (void *)(size_t)i, ABT_THREAD_ATTR_NULL, NULL);
        ATS_ERROR(ret, "ABT_thread_create");
    }
    while (ATS_atomic_load(&atomic_val) != ATOMIC_VAL_SUM) {
        int rank = 0;
        ret = ABT_xstream_self_rank(&rank);
        ATS_ERROR(ret, "ABT_xstream_self_rank");

        /* Get a unit and schedule it. */
        ABT_unit unit;
        ret = ABT_pool_pop(pools[rank], &unit);
        ATS_ERROR(ret, "ABT_pool_pop");

        if (unit != ABT_UNIT_NULL) {
            ret = ABT_xstream_run_unit(unit, pools[rank]);
            ATS_ERROR(ret, "ABT_xstream_run_unit");
        } else {
            ret = ABT_thread_yield();
            ATS_ERROR(ret, "ABT_thread_yield");
        }
    }
}

int main(int argc, char *argv[])
{
    int i, j;
    int ret;
    num_xstreams = DEFAULT_NUM_XSTREAMS;
    num_threads = DEFAULT_NUM_THREADS;
    if (argc > 1)
        num_xstreams = atoi(argv[1]);
    assert(num_xstreams >= 0);
    if (argc > 2)
        num_threads = atoi(argv[2]);
    assert(num_threads >= 0);

    ABT_xstream *xstreams;
    xstreams = (ABT_xstream *)malloc(sizeof(ABT_xstream) * num_xstreams);
    pools = (ABT_pool *)malloc(sizeof(ABT_pool) * num_xstreams);

    /* Initialize */
    ATS_read_args(argc, argv);
    ATS_init(argc, argv, num_xstreams);

    /* Create Execution Streams */
    ret = ABT_xstream_self(&xstreams[0]);
    ATS_ERROR(ret, "ABT_xstream_self");
    for (i = 1; i < num_xstreams; i++) {
        ret = ABT_xstream_create(ABT_SCHED_NULL, &xstreams[i]);
        ATS_ERROR(ret, "ABT_xstream_create");
    }

    /* Get the pools attached to an execution stream */
    for (i = 0; i < num_xstreams; i++) {
        ret = ABT_xstream_get_main_pools(xstreams[i], 1, pools + i);
        ATS_ERROR(ret, "ABT_xstream_get_main_pools");
    }

    /* Create threads */
    ABT_thread *threads =
        (ABT_thread *)malloc(sizeof(ABT_thread) * num_xstreams * num_threads);
    for (i = 0; i < num_xstreams; i++) {
        for (j = 0; j < num_threads; j++) {
            size_t tid = i * num_threads + j + 1;
            ret = ABT_thread_create(pools[i], root_func, (void *)tid,
                                    ABT_THREAD_ATTR_NULL,
                                    &threads[i * num_threads + j]);
            ATS_ERROR(ret, "ABT_thread_create");
        }
    }

    for (i = 0; i < num_xstreams * num_threads; i++) {
        ret = ABT_thread_free(&threads[i]);
        ATS_ERROR(ret, "ABT_thread_free");
    }
    free(threads);

    /* Join Execution Streams */
    for (i = 1; i < num_xstreams; i++) {
        ret = ABT_xstream_join(xstreams[i]);
        ATS_ERROR(ret, "ABT_xstream_join");
    }

    /* Free Execution Streams */
    for (i = 1; i < num_xstreams; i++) {
        ret = ABT_xstream_free(&xstreams[i]);
        ATS_ERROR(ret, "ABT_xstream_free");
    }

    /* Finalize */
    ret = ATS_finalize(0);

    free(pools);
    free(xstreams);

    return ret;
}

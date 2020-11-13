/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

#include <stdio.h>
#include <stdlib.h>
#include "abt.h"
#include "abttest.h"

#define DEFAULT_NUM_XSTREAMS 4
#define DEFAULT_NUM_THREADS 4

int value = 0;

void thread_func(void *arg)
{
    int i;
    int ret;
    ABT_thread thread;
    size_t my_id = (size_t)arg;
    ATS_printf(1, "[TH%lu]: Hello, world!\n", my_id);

    ABT_thread_self(&thread);
    for (i = 0; i < 3; i++) {
        ATS_printf(1, "[TH%lu]: request migration ...\n", my_id);
        ret = ABT_thread_migrate(thread);
        if (ret != ABT_ERR_MIGRATION_NA)
            ATS_ERROR(ret, "ABT_thread_migrate");
    }

    __sync_fetch_and_add(&value, 1);
}

int main(int argc, char *argv[])
{
    int i, j;
    int ret;
    int num_xstreams = DEFAULT_NUM_XSTREAMS;
    int num_threads = DEFAULT_NUM_THREADS;
    if (argc > 1)
        num_xstreams = atoi(argv[1]);
    assert(num_xstreams >= 0);
    if (argc > 2)
        num_threads = atoi(argv[2]);
    assert(num_threads >= 0);

    ABT_xstream *xstreams;
    ABT_thread **threads;
    xstreams = (ABT_xstream *)malloc(sizeof(ABT_xstream) * num_xstreams);
    threads = (ABT_thread **)malloc(sizeof(ABT_thread *) * num_xstreams);
    for (i = 0; i < num_xstreams; i++) {
        threads[i] = (ABT_thread *)malloc(sizeof(ABT_thread) * num_threads);
    }

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
    ABT_pool *pools;
    pools = (ABT_pool *)malloc(sizeof(ABT_pool) * num_xstreams);
    for (i = 0; i < num_xstreams; i++) {
        ret = ABT_xstream_get_main_pools(xstreams[i], 1, pools + i);
        ATS_ERROR(ret, "ABT_xstream_get_main_pools");
    }

    /* Create threads */
    for (i = 0; i < num_xstreams; i++) {
        for (j = 0; j < num_threads; j++) {
            size_t tid = i * num_threads + j + 1;
            ret = ABT_thread_create(pools[i], thread_func, (void *)tid,
                                    ABT_THREAD_ATTR_NULL, &threads[i][j]);
            ATS_ERROR(ret, "ABT_thread_create");
        }
    }

    /* Join and free threads */
    for (i = 0; i < num_xstreams; i++) {
        for (j = 0; j < num_threads; j++) {
            ret = ABT_thread_free(&threads[i][j]);
            ATS_ERROR(ret, "ABT_thread_free");
        }
    }

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

    if (value != num_xstreams * num_threads)
        ATS_ERROR(ABT_ERR_OTHER, "wrong value");

    /* Finalize */
    ret = ATS_finalize(0);

    for (i = 0; i < num_xstreams; i++) {
        free(threads[i]);
    }
    free(threads);
    free(pools);
    free(xstreams);

    return ret;
}

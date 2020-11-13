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

int num_threads = DEFAULT_NUM_THREADS;

void thread_func(void *arg)
{
    size_t my_id = (size_t)arg;
    ATS_printf(1, "[TH%lu]: Hello, world!\n", my_id);
}

void thread_create(void *arg)
{
    int i, ret;
    size_t my_id = (size_t)arg;
    ABT_thread my_thread;
    ABT_pool my_pool;

    ret = ABT_thread_self(&my_thread);
    ATS_ERROR(ret, "ABT_thread_self");
    ret = ABT_thread_get_last_pool(my_thread, &my_pool);
    ATS_ERROR(ret, "ABT_thread_get_last_pool");

    /* Create threads */
    for (i = 0; i < num_threads; i++) {
        size_t tid = 100 * my_id + i;
        ret = ABT_thread_create(my_pool, thread_func, (void *)tid,
                                ABT_THREAD_ATTR_NULL, NULL);
        ATS_ERROR(ret, "ABT_thread_create");
    }

    ATS_printf(1, "[TH%lu]: created %d threads\n", my_id, num_threads);
}

int main(int argc, char *argv[])
{
    int i;
    int ret;
    int num_xstreams = DEFAULT_NUM_XSTREAMS;
    if (argc > 1)
        num_xstreams = atoi(argv[1]);
    assert(num_xstreams >= 0);
    if (argc > 2)
        num_threads = atoi(argv[2]);
    assert(num_threads >= 0);

    ABT_xstream *xstreams;
    xstreams = (ABT_xstream *)malloc(sizeof(ABT_xstream) * num_xstreams);

    ABT_pool *pools;
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

    /* Create one thread for each ES */
    for (i = 0; i < num_xstreams; i++) {
        size_t tid = i + 1;
        ret = ABT_thread_create(pools[i], thread_create, (void *)tid,
                                ABT_THREAD_ATTR_NULL, NULL);
        ATS_ERROR(ret, "ABT_thread_create");
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

    /* Finalize */
    ret = ATS_finalize(0);

    free(pools);
    free(xstreams);

    return ret;
}

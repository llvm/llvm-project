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

void thread_func(void *arg)
{
    size_t my_id = (size_t)arg;
    ATS_printf(1, "[TH%lu]: Hello, world!\n", my_id);
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
    xstreams = (ABT_xstream *)malloc(sizeof(ABT_xstream) * num_xstreams);

    ABT_sched *scheds;
    scheds = (ABT_sched *)malloc(sizeof(ABT_sched) * num_xstreams);

    ABT_pool *pools;
    pools = (ABT_pool *)malloc(sizeof(ABT_pool) * num_xstreams);

    /* Initialize */
    ATS_read_args(argc, argv);
    ATS_init(argc, argv, num_xstreams);

    /* Create schedulers */
    for (i = 0; i < num_xstreams; i++) {
        ret = ABT_sched_create_basic(ABT_SCHED_BASIC_WAIT, 0, NULL,
                                     ABT_SCHED_CONFIG_NULL, &scheds[i]);
        ATS_ERROR(ret, "ABT_sched_create_basic");
    }

    /* Create Execution Streams */
    ret = ABT_xstream_self(&xstreams[0]);
    ATS_ERROR(ret, "ABT_xstream_self");
    ret = ABT_xstream_set_main_sched(xstreams[0], scheds[0]);
    ATS_ERROR(ret, "ABT_xstream_set_main_sched");
    for (i = 1; i < num_xstreams; i++) {
        ret = ABT_xstream_create(scheds[i], &xstreams[i]);
        ATS_ERROR(ret, "ABT_xstream_create");
    }

    /* Get the pools attached to an execution stream */
    for (i = 0; i < num_xstreams; i++) {
        ret = ABT_xstream_get_main_pools(xstreams[i], 1, pools + i);
        ATS_ERROR(ret, "ABT_xstream_get_main_pools");
    }

    /* Create ULTs */
    for (i = 0; i < num_xstreams; i++) {
        for (j = 0; j < num_threads; j++) {
            size_t tid = i * num_threads + j + 1;
            ret = ABT_thread_create(pools[i], thread_func, (void *)tid,
                                    ABT_THREAD_ATTR_NULL, NULL);
            ATS_ERROR(ret, "ABT_thread_create");
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

    /* Finalize */
    ret = ATS_finalize(0);

    free(pools);
    free(scheds);
    free(xstreams);

    return ret;
}

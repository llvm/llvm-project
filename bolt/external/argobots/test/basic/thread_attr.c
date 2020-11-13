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
    ABT_thread thread;
    ABT_thread_attr attr;
    size_t stacksize1, stacksize2;
    int ret;

    ABT_thread_self(&thread);
    ret = ABT_thread_get_attr(thread, &attr);
    ATS_ERROR(ret, "ABT_thread_get_attr");

    ret = ABT_thread_attr_get_stacksize(attr, &stacksize1);
    ATS_ERROR(ret, "ABT_thread_attr_get_stacksize");
    ret = ABT_thread_get_stacksize(thread, &stacksize2);
    ATS_ERROR(ret, "ABT_thread_get_stacksize");
    assert(stacksize1 == stacksize2);
    ATS_printf(1, "[TH%lu]: stacksize=%lu\n", my_id, stacksize1);

    ret = ABT_thread_attr_free(&attr);
    ATS_ERROR(ret, "ABT_thread_attr_free");
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

    ABT_thread_attr attr;
    ABT_xstream *xstreams;
    xstreams = (ABT_xstream *)malloc(sizeof(ABT_xstream) * num_xstreams);

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

    /* ULT attribute */
    ret = ABT_thread_attr_create(&attr);
    ATS_ERROR(ret, "ABT_thread_attr_create");
    ABT_thread_attr_set_stacksize(attr, 8192);

    /* Create threads */
    for (i = 0; i < num_xstreams; i++) {
        for (j = 0; j < num_threads; j++) {
            size_t tid = i * num_threads + j + 1;
            ret = ABT_thread_create(pools[i], thread_func, (void *)tid,
                                    (tid % 2 ? attr : ABT_THREAD_ATTR_NULL),
                                    NULL);
            ATS_ERROR(ret, "ABT_thread_create");
        }
    }

    /* Free the attribute */
    ret = ABT_thread_attr_free(&attr);
    ATS_ERROR(ret, "ABT_thread_attr_free");

    thread_func((void *)0);

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

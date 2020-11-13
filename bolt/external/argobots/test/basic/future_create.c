/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

#include <stdio.h>
#include <stdlib.h>
#include "abt.h"
#include "abttest.h"

#define DEFAULT_NUM_XSTREAMS 2
#define DEFAULT_NUM_THREADS 3
#define BUFFER_SIZE 10

ABT_future myfuture;
ABT_future myfuture2;

/* Total number of threads (num_xstreams * num_threads) */
int num_xstreams = DEFAULT_NUM_XSTREAMS;
int num_threads = DEFAULT_NUM_THREADS;
int total_num_threads;

void future_wait(void *args)
{
    int th = (int)(intptr_t)args;
    ATS_printf(1, "Thread %d is waiting for future\n", th);
    ABT_future_wait(myfuture);
    ATS_printf(1, "Thread %d returns from future_wait\n", th);
}

void future_set(void *args)
{
    int th = (int)(intptr_t)args;
    ABT_future_set(myfuture, (void *)(intptr_t)th);
    ATS_printf(1, "Thread %d signals future\n", th);
}

void future_cb(void **args)
{
    int i;
    int total = 0;
    for (i = 0; i < total_num_threads; i++) {
        total += (int)(intptr_t)args[i];
    }
    if (total_num_threads * (total_num_threads - 1) / 2 != total) {
        ATS_ERROR(ABT_ERR_OTHER, "Wrong value!");
    }

    ABT_future_set(myfuture2, NULL);
    ATS_printf(1, "Callback signals future\n");
}

int main(int argc, char *argv[])
{
    int i, j, r;
    int ret;
    if (argc > 1)
        num_xstreams = atoi(argv[1]);
    assert(num_xstreams >= 0);
    if (argc > 2)
        num_threads = atoi(argv[2]);
    assert(num_threads >= 0);
    total_num_threads = num_threads * num_xstreams;

    /* init and thread creation */
    ATS_read_args(argc, argv);
    ATS_init(argc, argv, num_xstreams);

    /* Create Execution Streams */
    ABT_xstream *xstreams =
        (ABT_xstream *)malloc(num_xstreams * sizeof(ABT_xstream));
    ret = ABT_xstream_self(&xstreams[0]);
    ATS_ERROR(ret, "ABT_xstream_self");
    for (i = 1; i < num_xstreams; i++) {
        ret = ABT_xstream_create(ABT_SCHED_NULL, &xstreams[i]);
        ATS_ERROR(ret, "ABT_xstream_create");
    }

    /* Get the pools attached to an execution stream */
    ABT_pool *pools = (ABT_pool *)malloc(num_xstreams * sizeof(ABT_pool));
    for (i = 0; i < num_xstreams; i++) {
        ret = ABT_xstream_get_main_pools(xstreams[i], 1, pools + i);
        ATS_ERROR(ret, "ABT_xstream_get_main_pools");
    }

    ret = ABT_future_create(total_num_threads, future_cb, &myfuture);
    ATS_ERROR(ret, "ABT_future_create");
    ret = ABT_future_create(1, NULL, &myfuture2);
    ATS_ERROR(ret, "ABT_future_create");

    for (r = 0; r < 5; r++) {
        ret = ABT_future_reset(myfuture);
        ATS_ERROR(ret, "ABT_future_reset");
        ret = ABT_future_reset(myfuture2);
        ATS_ERROR(ret, "ABT_future_reset");
        for (i = 0; i < num_xstreams; i++) {
            for (j = 0; j < num_threads; j++) {
                int idx = i * num_threads + j;
                ret = ABT_thread_create(pools[i], future_wait,
                                        (void *)(intptr_t)(idx +
                                                           total_num_threads),
                                        ABT_THREAD_ATTR_NULL, NULL);
                ATS_ERROR(ret, "ABT_thread_create");
                ret = ABT_thread_create(pools[i], future_set,
                                        (void *)(intptr_t)idx,
                                        ABT_THREAD_ATTR_NULL, NULL);
                ATS_ERROR(ret, "ABT_thread_create");
            }
        }
        ATS_printf(1, "Thread main is waiting for future2\n");
        ABT_future_wait(myfuture2);
        ATS_printf(1, "Thread main returns from future2\n");
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

    /* Release futures */
    ret = ABT_future_free(&myfuture);
    ATS_ERROR(ret, "ABT_future_free");
    ret = ABT_future_free(&myfuture2);
    ATS_ERROR(ret, "ABT_future_free");

    /* Finalize */
    ret = ATS_finalize(0);

    free(xstreams);
    free(pools);

    return ret;
}

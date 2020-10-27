/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

#include <stdio.h>
#include <stdlib.h>
#include "abt.h"

#define NUM_XSTREAMS 4
#define NUM_THREADS (NUM_XSTREAMS * 2)

void thread_hello(void *arg)
{
    int rank, tid = (int)(size_t)arg;
    ABT_xstream_self_rank(&rank);
    printf("[U%d:E%d] Hello, world!\n", tid, rank);
}

int main(int argc, char *argv[])
{
    ABT_xstream xstreams[NUM_XSTREAMS];
    ABT_sched scheds[NUM_XSTREAMS];
    ABT_pool shared_pool;
    ABT_thread threads[NUM_THREADS];
    int i;

    ABT_init(argc, argv);

    /* Create a shared pool */
    ABT_pool_create_basic(ABT_POOL_FIFO, ABT_POOL_ACCESS_MPMC, ABT_TRUE,
                          &shared_pool);

    /* Create schedulers */
    for (i = 0; i < NUM_XSTREAMS; i++) {
        ABT_sched_create_basic(ABT_SCHED_DEFAULT, 1, &shared_pool,
                               ABT_SCHED_CONFIG_NULL, &scheds[i]);
    }

    /* Create ESs */
    ABT_xstream_self(&xstreams[0]);
    ABT_xstream_set_main_sched(xstreams[0], scheds[0]);
    for (i = 1; i < NUM_XSTREAMS; i++) {
        ABT_xstream_create(scheds[i], &xstreams[i]);
    }

    /* Create ULTs */
    for (i = 0; i < NUM_THREADS; i++) {
        size_t tid = (size_t)i;
        ABT_thread_create(shared_pool, thread_hello, (void *)tid,
                          ABT_THREAD_ATTR_NULL, &threads[i]);
    }

    /* Join & Free */
    for (i = 0; i < NUM_THREADS; i++) {
        ABT_thread_join(threads[i]);
        ABT_thread_free(&threads[i]);
    }
    for (i = 1; i < NUM_XSTREAMS; i++) {
        ABT_xstream_join(xstreams[i]);
        ABT_xstream_free(&xstreams[i]);
    }

    ABT_finalize();

    return 0;
}

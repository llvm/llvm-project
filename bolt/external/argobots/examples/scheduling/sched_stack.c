/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

#include <stdio.h>
#include <stdlib.h>
#include "abt.h"

#define NUM_XSTREAMS 2
#define NUM_TASKS 10

void task_hello(void *arg)
{
    int rank, tid = (int)(size_t)arg;
    ABT_xstream_self_rank(&rank);
    printf("  [T%d:E%d] Hello, world!\n", tid, rank);
}

void add_sched(void *arg)
{
    int rank, tid = (int)(size_t)arg;
    ABT_xstream_self_rank(&rank);
    ABT_sched sched;
    ABT_pool cur_pool, tar_pool;
    ABT_thread cur_thread;
    int i;

    /* Create a new scheduler */
    printf("[U%d:E%d] Create a scheduler and tasks\n", tid, rank);
    ABT_sched_create_basic(ABT_SCHED_BASIC, 0, NULL, ABT_SCHED_CONFIG_NULL,
                           &sched);

    /* Create tasks */
    ABT_sched_get_pools(sched, 1, 0, &tar_pool);
    for (i = 0; i < NUM_TASKS; i++) {
        ABT_task_create(tar_pool, task_hello, (void *)(size_t)i, NULL);
    }

    /* Push the scheduler to the current pool */
    printf("[U%d:E%d] Push the scheduler\n", tid, rank);
    ABT_thread_self(&cur_thread);
    ABT_thread_get_last_pool(cur_thread, &cur_pool);
    ABT_pool_add_sched(cur_pool, sched);
}

int main(int argc, char *argv[])
{
    ABT_xstream xstreams[NUM_XSTREAMS];
    ABT_pool pools[NUM_XSTREAMS];
    ABT_thread threads[NUM_XSTREAMS];
    int i;

    ABT_init(argc, argv);

    /* Create ESs */
    ABT_xstream_self(&xstreams[0]);
    for (i = 1; i < NUM_XSTREAMS; i++) {
        ABT_xstream_create(ABT_SCHED_NULL, &xstreams[i]);
    }

    /* Get the first pool associated with each ES */
    for (i = 0; i < NUM_XSTREAMS; i++) {
        ABT_xstream_get_main_pools(xstreams[i], 1, &pools[i]);
    }

    /* Create ULTs */
    for (i = 0; i < NUM_XSTREAMS; i++) {
        ABT_thread_create(pools[i], add_sched, (void *)(size_t)i,
                          ABT_THREAD_ATTR_NULL, &threads[i]);
    }

    /* Join & Free */
    for (i = 0; i < NUM_XSTREAMS; i++) {
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

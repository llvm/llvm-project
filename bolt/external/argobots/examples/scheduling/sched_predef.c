/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

#include <stdio.h>
#include <stdlib.h>
#include "abt.h"

#define NUM_XSTREAMS 4

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
    int num_pools[NUM_XSTREAMS];
    ABT_pool *pools[NUM_XSTREAMS];
    int i, k;

    ABT_init(argc, argv);

    /* Create schedulers */
    for (i = 0; i < NUM_XSTREAMS; i++) {
        ABT_sched_predef predef;
        predef = (i % 2) ? ABT_SCHED_BASIC : ABT_SCHED_PRIO;
        ABT_sched_create_basic(predef, 0, NULL, ABT_SCHED_CONFIG_NULL,
                               &scheds[i]);
    }

    /* Create ESs */
    ABT_xstream_self(&xstreams[0]);
    ABT_xstream_set_main_sched(xstreams[0], scheds[0]);
    for (i = 1; i < NUM_XSTREAMS; i++) {
        ABT_xstream_create(scheds[i], &xstreams[i]);
    }

    /* Get the pools associated with each scheduler */
    for (i = 0; i < NUM_XSTREAMS; i++) {
        ABT_sched_get_num_pools(scheds[i], &num_pools[i]);
        pools[i] = (ABT_pool *)malloc(num_pools[i] * sizeof(ABT_pool));
        ABT_sched_get_pools(scheds[i], num_pools[i], 0, pools[i]);
    }

    /* Create ULTs */
    for (i = 0; i < NUM_XSTREAMS; i++) {
        for (k = num_pools[i] - 1; k >= 0; k--) {
            size_t tid = (i + 1) * 10 + k;
            ABT_thread_create(pools[i][k], thread_hello, (void *)tid,
                              ABT_THREAD_ATTR_NULL, NULL);
        }
    }

    /* Join & Free */
    for (i = 1; i < NUM_XSTREAMS; i++) {
        ABT_xstream_join(xstreams[i]);
        ABT_xstream_free(&xstreams[i]);
    }

    for (i = 0; i < NUM_XSTREAMS; i++) {
        free(pools[i]);
    }

    ABT_finalize();

    return 0;
}

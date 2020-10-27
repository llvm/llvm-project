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

static int num_threads = DEFAULT_NUM_THREADS;

static void thread_hello(void *arg)
{
    ATS_UNUSED(arg);
    int rank;
    ABT_thread self;
    ABT_unit_id id;

    ABT_xstream_self_rank(&rank);
    ABT_thread_self(&self);
    ABT_thread_get_id(self, &id);

    ATS_printf(1, "[U%lu:E%d] Hello, world!\n", id, rank);
}

static void thread_func(void *arg)
{
    int i = (int)(size_t)arg;
    int rank, ret;
    ABT_xstream xstream;
    ABT_thread self;
    ABT_unit_id id;
    ABT_sched sched;
    ABT_pool pool;
    ABT_thread *threads;

    ret = ABT_xstream_self(&xstream);
    ATS_ERROR(ret, "ABT_xstream_self");
    ABT_xstream_get_rank(xstream, &rank);

    ret = ABT_thread_self(&self);
    ATS_ERROR(ret, "ABT_thread_self");
    ABT_thread_get_id(self, &id);

    ATS_printf(1, "[U%lu:E%d] change the main scheduler\n", id, rank);

    /* Create a new scheduler */
    /* NOTE: Since we use ABT_sched_create_basic, the new scheduler and its
     * associated pools will be freed automatically when it is not used anymore
     * or the associated ES is terminated. */
    ret = ABT_sched_create_basic(ABT_SCHED_PRIO, 0, NULL, ABT_SCHED_CONFIG_NULL,
                                 &sched);
    ATS_ERROR(ret, "ABT_sched_create_basic");

    /* Change the main scheduler */
    ret = ABT_xstream_set_main_sched(xstream, sched);
    ATS_ERROR(ret, "ABT_xstream_set_main_sched");

    /* Create ULTs for the new scheduler */
    ret = ABT_xstream_get_main_pools(xstream, 1, &pool);
    ATS_ERROR(ret, "ABT_xstream_get_main_pools");
    threads = (ABT_thread *)malloc(sizeof(ABT_thread) * num_threads);
    for (i = 0; i < num_threads; i++) {
        ret = ABT_thread_create(pool, thread_hello, NULL, ABT_THREAD_ATTR_NULL,
                                &threads[i]);
        ATS_ERROR(ret, "ABT_thread_create");
    }
    for (i = 0; i < num_threads; i++) {
        ret = ABT_thread_free(&threads[i]);
        ATS_ERROR(ret, "ABT_thread_free");
    }
    free(threads);
}

int main(int argc, char *argv[])
{
    size_t i;
    int ret;
    int num_xstreams = DEFAULT_NUM_XSTREAMS;
    ABT_xstream *xstreams;
    ABT_pool *pools;
    ABT_thread *threads;

    /* Initialize */
    ATS_read_args(argc, argv);
    if (argc > 1) {
        num_xstreams = ATS_get_arg_val(ATS_ARG_N_ES);
        num_threads = ATS_get_arg_val(ATS_ARG_N_ULT);
    }
    ATS_init(argc, argv, num_xstreams);

    ATS_printf(1, "num_xstreams=%d num_threads=%d\n", num_xstreams,
               num_threads);

    xstreams = (ABT_xstream *)malloc(sizeof(ABT_xstream) * num_xstreams);
    pools = (ABT_pool *)malloc(sizeof(ABT_pool) * num_xstreams);
    threads = (ABT_thread *)malloc(sizeof(ABT_thread) * num_xstreams);

    /* Create Execution Streams */
    ret = ABT_xstream_self(&xstreams[0]);
    ATS_ERROR(ret, "ABT_xstream_self");
    for (i = 1; i < num_xstreams; i++) {
        ret = ABT_xstream_create(ABT_SCHED_NULL, &xstreams[i]);
        ATS_ERROR(ret, "ABT_xstream_create");

        /* Get the first associated pool */
        ret = ABT_xstream_get_main_pools(xstreams[i], 1, pools + i);
        ATS_ERROR(ret, "ABT_xstream_get_main_pools");
    }

    /* Create ULTs */
    for (i = 1; i < num_xstreams; i++) {
        ret = ABT_thread_create(pools[i], thread_func, (void *)i,
                                ABT_THREAD_ATTR_NULL, &threads[i]);
        ATS_ERROR(ret, "ABT_thread_create");
    }

    thread_func((void *)0);

    /* Join and free all ULTs */
    for (i = 1; i < num_xstreams; i++) {
        ret = ABT_thread_free(&threads[i]);
        ATS_ERROR(ret, "ABT_thread_free");
    }

    /* Join and free ESs */
    for (i = 1; i < num_xstreams; i++) {
        ret = ABT_xstream_join(xstreams[i]);
        ATS_ERROR(ret, "ABT_xstream_join");
        ret = ABT_xstream_free(&xstreams[i]);
        ATS_ERROR(ret, "ABT_xstream_free");
    }

    /* Finalize */
    ret = ATS_finalize(0);

    free(threads);
    free(pools);
    free(xstreams);

    return ret;
}

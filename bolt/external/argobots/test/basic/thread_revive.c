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
    int rank, ret;
    ABT_thread self;
    ABT_unit_id id;

    assert((size_t)arg == 1);

    ret = ABT_xstream_self_rank(&rank);
    ATS_ERROR(ret, "ABT_xstream_self_rank");
    ret = ABT_thread_self(&self);
    ATS_ERROR(ret, "ABT_thread_self");
    ret = ABT_thread_get_id(self, &id);
    ATS_ERROR(ret, "ABT_thread_get_id");

    ATS_printf(1, "[U%lu:E%u]: Hello, world!\n", id, rank);
}

void thread_func2(void *arg)
{
    int rank, ret;
    ABT_thread self;
    ABT_unit_id id;

    assert((size_t)arg == 2);

    ret = ABT_xstream_self_rank(&rank);
    ATS_ERROR(ret, "ABT_xstream_self_rank");
    ret = ABT_thread_self(&self);
    ATS_ERROR(ret, "ABT_thread_self");
    ret = ABT_thread_get_id(self, &id);
    ATS_ERROR(ret, "ABT_thread_get_id");

    ATS_printf(1, "[U%lu:E%u]: Good-bye, world!\n", id, rank);
}

void thread_create(void *arg)
{
    int rank, i, ret;
    ABT_thread self;
    ABT_unit_id id;
    ABT_pool my_pool;
    ABT_thread *threads;

    assert((size_t)arg == 0);

    ret = ABT_xstream_self_rank(&rank);
    ATS_ERROR(ret, "ABT_xstream_self_rank");
    ret = ABT_thread_self(&self);
    ATS_ERROR(ret, "ABT_thread_self");
    ret = ABT_thread_get_id(self, &id);
    ATS_ERROR(ret, "ABT_thread_get_id");
    ret = ABT_thread_get_last_pool(self, &my_pool);
    ATS_ERROR(ret, "ABT_thread_get_last_pool");

    threads = (ABT_thread *)malloc(num_threads * sizeof(ABT_thread));

    /* Create ULTs */
    for (i = 0; i < num_threads; i++) {
        ret = ABT_thread_create(my_pool, thread_func, (void *)1,
                                ABT_THREAD_ATTR_NULL, &threads[i]);
        ATS_ERROR(ret, "ABT_thread_create");
    }
    ATS_printf(1, "[U%lu:E%u]: created %d ULTs\n", id, rank, num_threads);

    /* Join ULTs */
    for (i = 0; i < num_threads; i++) {
        ret = ABT_thread_join(threads[i]);
        ATS_ERROR(ret, "ABT_thread_join");
    }
    ATS_printf(1, "[U%lu:E%u]: joined %d ULTs\n", id, rank, num_threads);

    /* Revive ULTs with a different function */
    for (i = 0; i < num_threads; i++) {
        ret = ABT_thread_revive(my_pool, thread_func2, (void *)2, &threads[i]);
        ATS_ERROR(ret, "ABT_thread_revive");
    }
    ATS_printf(1, "[U%lu:E%u]: revived %d ULTs\n", id, rank, num_threads);

    /* Join and free ULTs */
    for (i = 0; i < num_threads; i++) {
        ret = ABT_thread_free(&threads[i]);
        ATS_ERROR(ret, "ABT_thread_free");
    }
    ATS_printf(1, "[U%lu:E%u]: freed %d ULTs\n", id, rank, num_threads);

    free(threads);
}

int main(int argc, char *argv[])
{
    int i, ret;
    ABT_xstream *xstreams;
    ABT_pool *pools;
    ABT_thread *threads;
    int num_xstreams = DEFAULT_NUM_XSTREAMS;

    /* Initialize */
    ATS_read_args(argc, argv);
    if (argc >= 2) {
        num_xstreams = ATS_get_arg_val(ATS_ARG_N_ES);
        num_threads = ATS_get_arg_val(ATS_ARG_N_ULT);
    }
    ATS_init(argc, argv, num_xstreams);

    ATS_printf(1, "# of ESs    : %d\n", num_xstreams);
    ATS_printf(1, "# of ULTs/ES: %d\n", num_threads);

    xstreams = (ABT_xstream *)malloc(num_xstreams * sizeof(ABT_xstream));
    pools = (ABT_pool *)malloc(num_xstreams * sizeof(ABT_pool));
    threads = (ABT_thread *)malloc(num_xstreams * sizeof(ABT_thread));

    /* Create Execution Streams */
    ret = ABT_xstream_self(&xstreams[0]);
    ATS_ERROR(ret, "ABT_xstream_self");
    for (i = 1; i < num_xstreams; i++) {
        ret = ABT_xstream_create(ABT_SCHED_NULL, &xstreams[i]);
        ATS_ERROR(ret, "ABT_xstream_create");
    }

    /* Get the first pool of each ES */
    for (i = 0; i < num_xstreams; i++) {
        ret = ABT_xstream_get_main_pools(xstreams[i], 1, &pools[i]);
        ATS_ERROR(ret, "ABT_xstream_get_main_pools");
    }

    /* Create one ULT for each ES */
    for (i = 1; i < num_xstreams; i++) {
        ret = ABT_thread_create(pools[i], thread_create, (void *)0,
                                ABT_THREAD_ATTR_NULL, &threads[i]);
        ATS_ERROR(ret, "ABT_thread_create");
    }

    thread_create((void *)0);

    /* Join and free ULTs */
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

    free(xstreams);
    free(pools);
    free(threads);

    return ret;
}

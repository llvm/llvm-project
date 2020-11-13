/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

#include <stdio.h>
#include <stdlib.h>
#include "abt.h"
#include "abttest.h"

#define DEFAULT_NUM_XSTREAMS 4
#define DEFAULT_NUM_ITER 10

static int num_iter = DEFAULT_NUM_ITER;
static ABT_xstream_barrier barrier = ABT_XSTREAM_BARRIER_NULL;
/* Neither volatile nor atomic operation is necessary for value, but PGI-20.1 is
 * broken so we need a workaround to make PGI pass this test.
 * See https://github.com/pmodels/argobots/issues/213 for details. */
volatile int value;

void test_xstream_barrier(void *arg)
{
    int rank = (int)(intptr_t)arg;
    int i, ret;

    for (i = 0; i < num_iter; i++) {
        if (rank == 0)
            ATS_atomic_store(&value, i);
        ret = ABT_xstream_barrier_wait(barrier);
        ATS_ERROR(ret, "ABT_xstream_barrier_wait");

        assert(ATS_atomic_load(&value) == i);
        ret = ABT_xstream_barrier_wait(barrier);
        ATS_ERROR(ret, "ABT_xstream_barrier_wait");
    }
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
        num_iter = ATS_get_arg_val(ATS_ARG_N_ITER);
    }
    ATS_init(argc, argv, num_xstreams);

    ATS_printf(1, "# of ESs       : %d\n", num_xstreams);
    ATS_printf(1, "# of iterations: %d\n", num_iter);

    /* Create an ES barrier */
    ret = ABT_xstream_barrier_create(num_xstreams, &barrier);
    ATS_ERROR(ret, "ABT_xstream_barrier_create");

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
        ret = ABT_thread_create(pools[i], test_xstream_barrier,
                                (void *)(intptr_t)i, ABT_THREAD_ATTR_NULL,
                                &threads[i]);
        ATS_ERROR(ret, "ABT_thread_create");
    }

    test_xstream_barrier((void *)0);

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

    /* Free the ES barrier */
    ret = ABT_xstream_barrier_free(&barrier);
    ATS_ERROR(ret, "ABT_xstream_barrier_free");

    /* Finalize */
    ret = ATS_finalize(0);

    free(xstreams);
    free(pools);
    free(threads);

    return ret;
}

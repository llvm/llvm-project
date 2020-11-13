/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

#include <stdio.h>
#include <stdlib.h>
#include "abt.h"
#include "abttest.h"

int N;
int *values;
ABT_barrier *row_barrier;
ABT_barrier *col_barrier;
ABT_barrier global_barrier;

void run(void *args)
{
    int *pos = (int *)args;
    int i, j;
    i = pos[0];
    j = pos[1];

    assert(values[i] == i);
    ABT_barrier_wait(row_barrier[i]);
    if (!j)
        values[i] = i + N;

    ABT_barrier_wait(row_barrier[i]);
    assert(values[i] == i + N);

    ABT_barrier_wait(global_barrier);
    ABT_barrier_wait(global_barrier);
    if (!i)
        values[j] = j + 2 * N;

    ABT_barrier_wait(col_barrier[j]);
    assert(values[j] == j + 2 * N);
}

int main(int argc, char *argv[])
{
    ABT_xstream *xstreams;
    ABT_pool *pools;
    ABT_thread *threads;
    int num_xstreams;
    int es = 0;
    int *args;
    int i, j, iter, t;
    int ret;

    /* Initialize */
    ATS_read_args(argc, argv);
    if (argc < 2) {
        num_xstreams = 4;
        N = 10;
        iter = 100;
    } else {
        num_xstreams = ATS_get_arg_val(ATS_ARG_N_ES);
        N = ATS_get_arg_val(ATS_ARG_N_ULT);
        iter = ATS_get_arg_val(ATS_ARG_N_ITER);
    }
    ATS_init(argc, argv, num_xstreams);

    ATS_printf(1, "# of ESs : %d\n", num_xstreams);
    ATS_printf(1, "# of ULTs: %d\n", N * N);
    ATS_printf(1, "# of iter: %d\n", iter);

    xstreams = (ABT_xstream *)malloc(num_xstreams * sizeof(ABT_xstream));
    pools = (ABT_pool *)malloc(num_xstreams * sizeof(ABT_pool));
    threads = (ABT_thread *)malloc(N * N * sizeof(ABT_thread));

    values = (int *)malloc(N * sizeof(int));
    row_barrier = (ABT_barrier *)malloc(N * sizeof(ABT_barrier));
    col_barrier = (ABT_barrier *)malloc(N * sizeof(ABT_barrier));

    /* Create the values and barriers */
    for (i = 0; i < N; i++) {
        ret = ABT_barrier_create((size_t)N, &row_barrier[i]);
        ATS_ERROR(ret, "ABT_barrier_create");
        ret = ABT_barrier_create((size_t)N, &col_barrier[i]);
        ATS_ERROR(ret, "ABT_barrier_create");
    }
    ret = ABT_barrier_create((size_t)N * N, &global_barrier);
    ATS_ERROR(ret, "ABT_barrier_create");

    args = (int *)malloc(2 * N * N * sizeof(int));

    /* Create ESs */
    for (t = 0; t < iter; t++) {
        ATS_printf(1, "iter=%d\n", t);
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

        /* Create ULTs */
        for (i = 0; i < N; i++) {
            values[i] = i;
        }
        for (i = 0; i < N; i++) {
            for (j = 0; j < N; j++) {
                args[2 * (i * N + j)] = i;
                args[2 * (i * N + j) + 1] = j;
                ret = ABT_thread_create(pools[es], run,
                                        (void *)&args[2 * (i * N + j)],
                                        ABT_THREAD_ATTR_NULL,
                                        &threads[i * N + j]);
                ATS_ERROR(ret, "ABT_thread_create");
                es = (es + 1) % num_xstreams;
            }
        }

        /* Join and free ULTs */
        for (i = 0; i < N; i++) {
            for (j = 0; j < N; j++) {
                ret = ABT_thread_free(&threads[i * N + j]);
                ATS_ERROR(ret, "ABT_thread_free");
            }
        }

        /* Join ESs */
        for (i = 1; i < num_xstreams; i++) {
            ret = ABT_xstream_join(xstreams[i]);
            ATS_ERROR(ret, "ABT_xstream_join");
        }

        /* Free ESs */
        for (i = 1; i < num_xstreams; i++) {
            ret = ABT_xstream_free(&xstreams[i]);
            ATS_ERROR(ret, "ABT_xstream_free");
        }
    }

    /* Free the barriers */
    for (i = 0; i < N; i++) {
        ret = ABT_barrier_free(&row_barrier[i]);
        ATS_ERROR(ret, "ABT_barrier_create");
        ret = ABT_barrier_free(&col_barrier[i]);
        ATS_ERROR(ret, "ABT_barrier_create");
    }
    ret = ABT_barrier_free(&global_barrier);
    ATS_ERROR(ret, "ABT_barrier_free");

    /* Finalize */
    ret = ATS_finalize(0);

    free(xstreams);
    free(pools);
    free(threads);
    free(values);
    free(row_barrier);
    free(col_barrier);
    free(args);

    return ret;
}

/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

#include <stdio.h>
#include <stdlib.h>
#include "abt.h"
#include "abttest.h"

#define DEFAULT_NUM_XSTREAMS 4
#define DEFAULT_NUM_THREADS 8
#define NUM_TLS 8
#define NUM_STEPS 128

static ABT_barrier g_barrier;
static ABT_key tls[NUM_TLS];
static int num_threads;

static void thread_f(void *arg)
{
    int i, ret;

    ret = ABT_barrier_wait(g_barrier);
    ATS_ERROR(ret, "ABT_barrier_wait");
    /* Check the value. */
    for (i = 0; i < 3; i++) {
        void *check;
        ret = ABT_key_get(tls[i], &check);
        ATS_ERROR(ret, "ABT_key_get");
        assert(check == (void *)(intptr_t)i);
        ret = ABT_key_set(tls[i], (void *)(intptr_t)(i * 2));
        ATS_ERROR(ret, "ABT_key_set");
    }
    ret = ABT_barrier_wait(g_barrier);
    ATS_ERROR(ret, "ABT_barrier_wait");
    ret = ABT_barrier_wait(g_barrier);
    ATS_ERROR(ret, "ABT_barrier_wait");
    /* Check if there is no data race. */
    for (i = 3; i < NUM_TLS; i++) {
        void *check;
        ret = ABT_key_get(tls[i], &check);
        ATS_ERROR(ret, "ABT_key_get");
        assert(check == NULL || check == (void *)(intptr_t)i);
        ret = ABT_key_set(tls[i], (void *)(intptr_t)(i * 2));
        ATS_ERROR(ret, "ABT_key_set");
    }
}

int main(int argc, char *argv[])
{
    int num_xstreams;
    ABT_xstream *xstreams;
    ABT_thread *threads;
    ABT_pool *pools;
    int i, j, step, ret;

    /* Initialize */
    ATS_read_args(argc, argv);
    if (argc < 2) {
        num_xstreams = DEFAULT_NUM_XSTREAMS;
        num_threads = DEFAULT_NUM_THREADS;
    } else {
        num_xstreams = ATS_get_arg_val(ATS_ARG_N_ES);
        num_threads = ATS_get_arg_val(ATS_ARG_N_ULT);
    }
    ATS_init(argc, argv, num_xstreams);

    ATS_printf(1, "# of ESs    : %d\n", num_xstreams);
    ATS_printf(1, "# of ULTs/ES: %d\n", num_threads);
    for (i = 0; i < NUM_TLS; i++) {
        tls[i] = ABT_KEY_NULL;
    }

    xstreams = (ABT_xstream *)malloc(num_xstreams * sizeof(ABT_xstream));
    pools = (ABT_pool *)malloc(num_xstreams * sizeof(ABT_pool));
    threads = (ABT_thread *)malloc(num_threads * sizeof(ABT_thread));

    /* Create ULT-specific data keys */
    assert(NUM_TLS >= 4);
    for (i = 0; i < NUM_TLS; i++) {
        ret = ABT_key_create(NULL, &tls[i]);
        ATS_ERROR(ret, "ABT_key_create");
    }

    /* Create Execution Streams */
    ret = ABT_xstream_self(&xstreams[0]);
    ATS_ERROR(ret, "ABT_xstream_self");
    for (i = 1; i < num_xstreams; i++) {
        ret = ABT_xstream_create(ABT_SCHED_NULL, &xstreams[i]);
        ATS_ERROR(ret, "ABT_xstream_create");
    }

    /* Get the pools attached to each ES */
    for (i = 0; i < num_xstreams; i++) {
        ret = ABT_xstream_get_main_pools(xstreams[i], 1, &pools[i]);
        ATS_ERROR(ret, "ABT_xstream_get_main_pools");
    }

    ret = ABT_barrier_create(num_threads, &g_barrier);
    ATS_ERROR(ret, "ABT_barrier_create");
    for (step = 0; step < NUM_STEPS; step++) {
        /* Create one ULT for each ES */
        for (i = 1; i < num_threads; i++) {
            ret = ABT_thread_create(pools[i % num_xstreams], thread_f, NULL,
                                    ABT_THREAD_ATTR_NULL, &threads[i]);
            ATS_ERROR(ret, "ABT_thread_create");
            /* Access that TLS. */
            for (j = 0; j < 3; j++) {
                ret = ABT_thread_set_specific(threads[i], tls[j],
                                              (void *)(intptr_t)j);
                ATS_ERROR(ret, "ABT_thread_set_specific");
            }
        }
        ret = ABT_barrier_wait(g_barrier);
        ATS_ERROR(ret, "ABT_barrier_wait");
        ret = ABT_barrier_wait(g_barrier);
        ATS_ERROR(ret, "ABT_barrier_wait");
        for (i = 1; i < num_threads; i++) {
            for (j = 0; j < 3; j++) {
                void *check;
                ret = ABT_thread_get_specific(threads[i], tls[j], &check);
                ATS_ERROR(ret, "ABT_thread_get_specific");
                assert(check == (void *)(intptr_t)(j * 2));
            }
        }
        ret = ABT_barrier_wait(g_barrier);
        ATS_ERROR(ret, "ABT_barrier_wait");
        for (i = 1; i < num_threads; i++) {
            for (j = 3; j < NUM_TLS; j++) {
                if (j % 2 == 0) {
                    ret = ABT_thread_set_specific(threads[i], tls[j],
                                                  (void *)(intptr_t)j);
                    ATS_ERROR(ret, "ABT_thread_set_specific");
                } else {
                    void *check;
                    ret = ABT_thread_get_specific(threads[i], tls[j], &check);
                    ATS_ERROR(ret, "ABT_thread_get_specific");
                    assert(check == NULL || check == (void *)(intptr_t)(j * 2));
                }
            }
        }
        /* Join ULTs */
        for (i = 1; i < num_threads; i++) {
            ret = ABT_thread_free(&threads[i]);
            ATS_ERROR(ret, "ABT_thread_free");
            assert(threads[i] == ABT_THREAD_NULL);
        }
    }
    ret = ABT_barrier_free(&g_barrier);
    ATS_ERROR(ret, "ABT_barrier_free");

    /* Join and free Execution Streams */
    for (i = 1; i < num_xstreams; i++) {
        ret = ABT_xstream_join(xstreams[i]);
        ATS_ERROR(ret, "ABT_xstream_join");
        ret = ABT_xstream_free(&xstreams[i]);
        ATS_ERROR(ret, "ABT_xstream_free");
        assert(xstreams[i] == ABT_XSTREAM_NULL);
    }

    /* Delete keys */
    for (i = 0; i < NUM_TLS; i++) {
        ret = ABT_key_free(&tls[i]);
        ATS_ERROR(ret, "ABT_key_free");
        assert(tls[i] == ABT_KEY_NULL);
    }

    /* Finalize */
    ret = ATS_finalize(0);

    free(xstreams);
    free(pools);
    free(threads);

    return ret;
}

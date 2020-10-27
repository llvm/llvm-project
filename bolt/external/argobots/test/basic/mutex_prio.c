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
#define DEFAULT_NUM_ITER 100

int g_counter = 0;
int g_iter = DEFAULT_NUM_ITER;

typedef struct thread_arg {
    int id;
    ABT_mutex mutex;
} thread_arg_t;

void thread_func(void *arg)
{
    int i;
    int rank;
    ABT_unit_id id;
    ABT_thread self;
    thread_arg_t *t_arg = (thread_arg_t *)arg;

    ABT_xstream_self_rank(&rank);
    ABT_thread_self(&self);
    ABT_thread_get_id(self, &id);
    t_arg->id = (int)id;

    ABT_thread_yield();

    ATS_printf(1, "[U%d:E%d] START\n", t_arg->id, rank);
    for (i = 0; i < g_iter; i++) {
        if (i & 1) {
            ABT_mutex_lock(t_arg->mutex);
        } else {
            ABT_mutex_lock_low(t_arg->mutex);
        }

        g_counter++;

        if (i & 1) {
            ABT_mutex_unlock(t_arg->mutex);
        } else {
            ABT_mutex_unlock_se(t_arg->mutex);
        }
    }
    ATS_printf(1, "[U%d:E%d] END\n", t_arg->id, rank);

    ABT_thread_yield();
}

int main(int argc, char *argv[])
{
    int i, j;
    int ret, expected;
    int num_xstreams = DEFAULT_NUM_XSTREAMS;
    int num_threads = DEFAULT_NUM_THREADS;

    /* Initialize */
    ATS_read_args(argc, argv);
    if (argc >= 2) {
        num_xstreams = ATS_get_arg_val(ATS_ARG_N_ES);
        num_threads = ATS_get_arg_val(ATS_ARG_N_ULT);
        g_iter = ATS_get_arg_val(ATS_ARG_N_ITER);
    }
    ATS_init(argc, argv, num_xstreams);

    ATS_printf(2, "# of ESs : %d\n", num_xstreams);
    ATS_printf(1, "# of ULTs: %d\n", num_threads);
    ATS_printf(1, "# of iter: %d\n", g_iter);

    ABT_mutex mutex;
    ABT_xstream *xstreams;
    ABT_thread **threads;
    thread_arg_t **args;
    xstreams = (ABT_xstream *)malloc(sizeof(ABT_xstream) * num_xstreams);
    assert(xstreams != NULL);
    threads = (ABT_thread **)malloc(sizeof(ABT_thread *) * num_xstreams);
    assert(threads != NULL);
    args = (thread_arg_t **)malloc(sizeof(thread_arg_t *) * num_xstreams);
    assert(args != NULL);
    for (i = 0; i < num_xstreams; i++) {
        threads[i] = (ABT_thread *)malloc(sizeof(ABT_thread) * num_threads);
        args[i] = (thread_arg_t *)malloc(sizeof(thread_arg_t) * num_threads);
    }

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

    /* Create a mutex */
    ret = ABT_mutex_create(&mutex);
    ATS_ERROR(ret, "ABT_mutex_create");

    /* Create ULTs */
    for (i = 0; i < num_xstreams; i++) {
        for (j = 0; j < num_threads; j++) {
            int tid = i * num_threads + j + 1;
            args[i][j].id = tid;
            args[i][j].mutex = mutex;
            ret = ABT_thread_create(pools[i], thread_func, (void *)&args[i][j],
                                    ABT_THREAD_ATTR_NULL, &threads[i][j]);
            ATS_ERROR(ret, "ABT_thread_create");
        }
    }

    /* Join and free ULTs */
    for (i = 0; i < num_xstreams; i++) {
        for (j = 0; j < num_threads; j++) {
            ret = ABT_thread_free(&threads[i][j]);
            ATS_ERROR(ret, "ABT_thread_free");
        }
    }

    /* Join Execution Streams */
    for (i = 1; i < num_xstreams; i++) {
        ret = ABT_xstream_join(xstreams[i]);
        ATS_ERROR(ret, "ABT_xstream_join");
    }

    /* Free the mutex */
    ret = ABT_mutex_free(&mutex);
    ATS_ERROR(ret, "ABT_mutex_free");

    /* Free Execution Streams */
    for (i = 1; i < num_xstreams; i++) {
        ret = ABT_xstream_free(&xstreams[i]);
        ATS_ERROR(ret, "ABT_xstream_free");
    }

    /* Validation */
    expected = num_xstreams * num_threads * g_iter;
    if (g_counter != expected) {
        printf("g_counter = %d vs. expected = %d\n", g_counter, expected);
    }

    /* Finalize */
    ret = ATS_finalize(g_counter != expected);

    for (i = 0; i < num_xstreams; i++) {
        free(threads[i]);
        free(args[i]);
    }
    free(threads);
    free(args);
    free(xstreams);
    free(pools);

    return ret;
}

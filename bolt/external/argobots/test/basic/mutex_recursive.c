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
#define RECURSIVE_DEPTH 5

static ABT_mutex g_mutex = ABT_MUTEX_NULL;
static int g_counter = 0;

typedef struct thread_arg {
    int id;
    int depth;
} thread_arg_t;

static void thread_func(void *arg)
{
    thread_arg_t *t_arg = (thread_arg_t *)arg;

    if (t_arg->depth > 0) {
        t_arg->depth--;
        ABT_mutex_lock(g_mutex);
        {
            g_counter++;
            ATS_printf(1, "[U%d] g_counter=%d\n", t_arg->id, g_counter);
            thread_func(arg);
        }
        ABT_mutex_unlock(g_mutex);
    }
}

int main(int argc, char *argv[])
{
    ABT_mutex_attr mattr;
    ABT_xstream *xstreams;
    ABT_pool *pools;
    ABT_thread **threads;
    thread_arg_t **args;

    int i, j;
    int ret, expected;
    int num_xstreams, num_threads;

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

    xstreams = (ABT_xstream *)malloc(sizeof(ABT_xstream) * num_xstreams);
    pools = (ABT_pool *)malloc(sizeof(ABT_pool) * num_xstreams);
    threads = (ABT_thread **)malloc(sizeof(ABT_thread *) * num_xstreams);
    args = (thread_arg_t **)malloc(sizeof(thread_arg_t *) * num_xstreams);
    assert(xstreams && pools && threads && args);

    for (i = 0; i < num_xstreams; i++) {
        threads[i] = (ABT_thread *)malloc(sizeof(ABT_thread) * num_threads);
        args[i] = (thread_arg_t *)malloc(sizeof(thread_arg_t) * num_threads);
        assert(threads[i] && args[i]);
    }

    /* Create a mutex with the recursive property */
    ret = ABT_mutex_attr_create(&mattr);
    ATS_ERROR(ret, "ABT_mutex_attr_create");

    ret = ABT_mutex_attr_set_recursive(mattr, ABT_TRUE);
    ATS_ERROR(ret, "ABT_mutex_attr_set_recurvise");

    ret = ABT_mutex_create_with_attr(mattr, &g_mutex);
    ATS_ERROR(ret, "ABT_mutex_create_with_attr");

    ret = ABT_mutex_attr_free(&mattr);
    ATS_ERROR(ret, "ABT_mutex_attr_free");

    /* Create Execution Streams */
    ret = ABT_xstream_self(&xstreams[0]);
    ATS_ERROR(ret, "ABT_xstream_self");
    for (i = 1; i < num_xstreams; i++) {
        ret = ABT_xstream_create(ABT_SCHED_NULL, &xstreams[i]);
        ATS_ERROR(ret, "ABT_xstream_create");
    }

    /* Get the main pool associated with each ES */
    for (i = 0; i < num_xstreams; i++) {
        ret = ABT_xstream_get_main_pools(xstreams[i], 1, pools + i);
        ATS_ERROR(ret, "ABT_xstream_get_main_pools");
    }

    /* Create ULTs */
    for (i = 0; i < num_xstreams; i++) {
        for (j = 0; j < num_threads; j++) {
            int tid = i * num_threads + j + 1;
            args[i][j].id = tid;
            args[i][j].depth = RECURSIVE_DEPTH;
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

    /* Join and free ESs */
    for (i = 1; i < num_xstreams; i++) {
        ret = ABT_xstream_join(xstreams[i]);
        ATS_ERROR(ret, "ABT_xstream_join");
        ret = ABT_xstream_free(&xstreams[i]);
        ATS_ERROR(ret, "ABT_xstream_free");
    }

    /* Free the mutex */
    ret = ABT_mutex_free(&g_mutex);
    ATS_ERROR(ret, "ABT_mutex_free");

    /* Validation */
    expected = num_xstreams * num_threads * RECURSIVE_DEPTH;
    if (g_counter != expected) {
        printf("g_counter = %d\n", g_counter);
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

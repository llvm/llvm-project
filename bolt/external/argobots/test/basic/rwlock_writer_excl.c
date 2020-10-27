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
#define DEFAULT_NUM_TESTS 10
#define DEFAULT_NUM_TEST_ITERS 100

static int g_counter = 0;

typedef struct test_arg test_arg_t;
typedef struct thread_arg thread_arg_t;

struct thread_arg {
    int id;
    test_arg_t *targ;
};

struct test_arg {
    int num_xstreams;
    int num_threads;
    int iters;
    ABT_rwlock rwlock;
    ABT_pool *pools;
    ABT_thread **threads; /* malloc'd by caller, set by test */
    thread_arg_t **args;  /* malloc'd by caller, set by test */
};

static void thread_func(void *arg)
{
    thread_arg_t *t_arg = (thread_arg_t *)arg;
    int i;
    int ret;

    for (i = 0; i < t_arg->targ->iters; i++) {
        ABT_thread_yield();
        if (t_arg->id % 2) {
            ret = ABT_rwlock_rdlock(t_arg->targ->rwlock);
            ATS_ERROR(ret, "ABT_rwlock_rdlock");
            ATS_printf(1, "[TH%d] read\n", t_arg->id);
        } else {
            ret = ABT_rwlock_wrlock(t_arg->targ->rwlock);
            ATS_ERROR(ret, "ABT_rwlock_wrlock");
            g_counter++;
            ATS_printf(1, "[TH%d] increased\n", t_arg->id);
        }
        ret = ABT_rwlock_unlock(t_arg->targ->rwlock);
        ATS_ERROR(ret, "ABT_rwlock_unlock");
    }
}

int run_test(test_arg_t *targ)
{
    int expected = ((targ->num_xstreams * targ->num_threads) / 2) * targ->iters;
    int i, j;
    int ret;

    g_counter = 0;

    /* Create ULTs */
    for (i = 0; i < targ->num_xstreams; i++) {
        for (j = 0; j < targ->num_threads; j++) {
            int tid = i * targ->num_threads + j + 1;
            targ->args[i][j].id = tid;
            targ->args[i][j].targ = targ;
            ret = ABT_thread_create(targ->pools[i], thread_func,
                                    (void *)&targ->args[i][j],
                                    ABT_THREAD_ATTR_NULL, &targ->threads[i][j]);
            ATS_ERROR(ret, "ABT_thread_create");
        }
    }

    /* Join and free ULTs */
    for (i = 0; i < targ->num_xstreams; i++) {
        for (j = 0; j < targ->num_threads; j++) {
            ret = ABT_thread_free(&targ->threads[i][j]);
            ATS_ERROR(ret, "ABT_thread_free");
        }
    }

    /* Validation */
    if (g_counter != expected) {
        fprintf(stderr,
                "g_counter = %d, expected = %d (%d xstreams, %d threads)\n",
                g_counter, expected, targ->num_xstreams, targ->num_threads);
        return 1;
    } else {
        return 0;
    }
}

int main(int argc, char *argv[])
{
    int i;
    int ret, test_ret;
    int num_tests = DEFAULT_NUM_TESTS;
    test_arg_t targ;
    targ.num_xstreams = DEFAULT_NUM_XSTREAMS;
    targ.num_threads = DEFAULT_NUM_THREADS;
    targ.iters = DEFAULT_NUM_TEST_ITERS;
    if (argc > 1)
        targ.num_xstreams = atoi(argv[1]);
    assert(targ.num_xstreams >= 0);
    if (argc > 2)
        targ.num_threads = atoi(argv[2]);
    assert(targ.num_threads >= 0);
    if (argc > 3)
        num_tests = atoi(argv[3]);
    assert(num_tests > 0);
    if (argc > 4)
        targ.iters = atoi(argv[4]);
    assert(targ.iters > 0);

    ABT_xstream *xstreams;
    xstreams = (ABT_xstream *)malloc(sizeof(ABT_xstream) * targ.num_xstreams);
    assert(xstreams != NULL);
    targ.threads =
        (ABT_thread **)malloc(sizeof(ABT_thread *) * targ.num_xstreams);
    assert(targ.threads != NULL);
    targ.args =
        (thread_arg_t **)malloc(sizeof(thread_arg_t *) * targ.num_xstreams);
    assert(targ.args != NULL);
    for (i = 0; i < targ.num_xstreams; i++) {
        targ.threads[i] =
            (ABT_thread *)malloc(sizeof(ABT_thread) * targ.num_threads);
        targ.args[i] =
            (thread_arg_t *)malloc(sizeof(thread_arg_t) * targ.num_threads);
    }

    /* Initialize */
    ATS_read_args(argc, argv);
    ATS_init(argc, argv, targ.num_xstreams);

    /* Create Execution Streams */
    ret = ABT_xstream_self(&xstreams[0]);
    ATS_ERROR(ret, "ABT_xstream_self");
    for (i = 1; i < targ.num_xstreams; i++) {
        ret = ABT_xstream_create(ABT_SCHED_NULL, &xstreams[i]);
        ATS_ERROR(ret, "ABT_xstream_create");
    }

    /* Get the pools attached to an execution stream */
    targ.pools = (ABT_pool *)malloc(sizeof(ABT_pool) * targ.num_xstreams);
    for (i = 0; i < targ.num_xstreams; i++) {
        ret = ABT_xstream_get_main_pools(xstreams[i], 1, targ.pools + i);
        ATS_ERROR(ret, "ABT_xstream_get_main_pools");
    }

    /* Create a rwlock */
    ret = ABT_rwlock_create(&targ.rwlock);
    ATS_ERROR(ret, "ABT_rwlock_create");

    /* Execute tests */
    test_ret = 0;
    for (i = 0; i < num_tests && test_ret == 0; i++) {
        test_ret = run_test(&targ);
    }

    /* Join Execution Streams */
    for (i = 1; i < targ.num_xstreams; i++) {
        ret = ABT_xstream_join(xstreams[i]);
        ATS_ERROR(ret, "ABT_xstream_join");
    }

    /* Free the rwlock */
    ret = ABT_rwlock_free(&targ.rwlock);
    ATS_ERROR(ret, "ABT_rwlock_free");

    /* Free Execution Streams */
    for (i = 1; i < targ.num_xstreams; i++) {
        ret = ABT_xstream_free(&xstreams[i]);
        ATS_ERROR(ret, "ABT_xstream_free");
    }

    /* Finalize */
    ret = ATS_finalize(test_ret);

    for (i = 0; i < targ.num_xstreams; i++) {
        free(targ.threads[i]);
        free(targ.args[i]);
    }
    free(targ.threads);
    free(targ.args);
    free(xstreams);
    free(targ.pools);

    return ret;
}

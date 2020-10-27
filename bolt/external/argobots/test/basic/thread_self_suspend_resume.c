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
#define DEFAULT_NUM_ITER 10

static ABT_pool *pools;
static int num_xstreams;
static int num_threads;
static int num_iter;
static ABT_thread **threads;
static int **values;
static ABT_barrier barrier;

typedef struct {
    int eid;
    int tid;
} thread_arg_t;

void thread_test_suspend(void *arg)
{
    thread_arg_t *my_arg = (thread_arg_t *)arg;
    int eid = my_arg->eid;
    int tid = my_arg->tid;
    free(arg);
    int i, ret;

    ATS_printf(1, "ULT[%d][%d] start\n", eid, tid);
    for (i = 0; i < num_iter; i++) {
        ret = ABT_self_suspend();
        ATS_ERROR(ret, "ABT_self_suspend");
        values[eid][tid]++;
    }
    ATS_printf(1, "ULT[%d][%d] end\n", eid, tid);
}

void thread_test_resume(void *arg)
{
    int i, k, tar, ret;
    int eid = (int)(size_t)arg;
    thread_arg_t *t_arg;
    ABT_thread_state state;

    /* Create ULTs */
    ATS_printf(1, "ULT%d: create %d ULTs\n", eid, num_threads);
    for (i = 0; i < num_threads; i++) {
        t_arg = (thread_arg_t *)malloc(sizeof(thread_arg_t));
        t_arg->eid = eid;
        t_arg->tid = i;

        ret = ABT_thread_create(pools[eid], thread_test_suspend, (void *)t_arg,
                                ABT_THREAD_ATTR_NULL, &threads[eid][i]);
        ATS_ERROR(ret, "ABT_thread_create");
    }

    ABT_barrier_wait(barrier);

    for (i = 0; i < num_iter; i++) {
        /* We rotate ES IDs to wake up ULTs on a different ES each time */
        tar = (eid + i) % num_xstreams;
        for (k = 0; k < num_threads; k++) {
            while (1) {
                ret = ABT_thread_get_state(threads[tar][k], &state);
                ATS_ERROR(ret, "ABT_thread_get_state");
                if (state == ABT_THREAD_STATE_BLOCKED) {
                    ret = ABT_thread_resume(threads[tar][k]);
                    ATS_ERROR(ret, "ABT_thread_resume");
                    break;
                } else {
                    ret = ABT_thread_yield();
                    ATS_ERROR(ret, "ABT_thread_yield");
                }
            }
        }

        ABT_barrier_wait(barrier);
    }

    /* Join & free ULTs */
    for (i = 0; i < num_threads; i++) {
        ret = ABT_thread_free(&threads[eid][i]);
        ATS_ERROR(ret, "ABT_thread_free");
    }
}

int main(int argc, char *argv[])
{
    ABT_xstream *xstreams;
    ABT_thread *main_threads;
    int i, k;
    int ret;

    /* Initialize */
    ATS_read_args(argc, argv);
    if (argc < 2) {
        num_xstreams = DEFAULT_NUM_XSTREAMS;
        num_threads = DEFAULT_NUM_THREADS;
        num_iter = DEFAULT_NUM_ITER;
    } else {
        num_xstreams = ATS_get_arg_val(ATS_ARG_N_ES);
        num_threads = ATS_get_arg_val(ATS_ARG_N_ULT);
        num_iter = ATS_get_arg_val(ATS_ARG_N_ITER);
    }
    ATS_init(argc, argv, num_xstreams);

    ATS_printf(1, "# of ESs    : %d\n", num_xstreams);
    ATS_printf(1, "# of ULTs/ES: %d\n", num_threads);
    ATS_printf(1, "# of iter.  : %d\n", num_iter);

    xstreams = (ABT_xstream *)malloc(num_xstreams * sizeof(ABT_xstream));
    pools = (ABT_pool *)malloc(num_xstreams * sizeof(ABT_pool));
    main_threads = (ABT_thread *)malloc(num_xstreams * sizeof(ABT_thread));
    threads = (ABT_thread **)malloc(num_xstreams * sizeof(ABT_thread *));
    values = (int **)malloc(num_xstreams * sizeof(int *));
    for (i = 0; i < num_xstreams; i++) {
        threads[i] = (ABT_thread *)malloc(num_threads * sizeof(ABT_thread));
        values[i] = (int *)calloc(num_threads, sizeof(int));
    }

    /* Create a barrier */
    ret = ABT_barrier_create((size_t)num_xstreams, &barrier);
    ATS_ERROR(ret, "ABT_barrier_create");

    /* Create ESs */
    ret = ABT_xstream_self(&xstreams[0]);
    ATS_ERROR(ret, "ABT_xstream_self");
    for (i = 1; i < num_xstreams; i++) {
        ret = ABT_xstream_create(ABT_SCHED_NULL, &xstreams[i]);
        ATS_ERROR(ret, "ABT_xstream_create");
    }

    /* Get the pool associated with each ES */
    for (i = 0; i < num_xstreams; i++) {
        ret = ABT_xstream_get_main_pools(xstreams[i], 1, &pools[i]);
        ATS_ERROR(ret, "ABT_xstream_get_main_pools");
    }

    /* Create one ULT for each ES */
    ret = ABT_thread_self(&main_threads[0]);
    ATS_ERROR(ret, "ABT_thread_self");
    for (i = 1; i < num_xstreams; i++) {
        ret = ABT_thread_create(pools[i], thread_test_resume, (void *)(size_t)i,
                                ABT_THREAD_ATTR_NULL, &main_threads[i]);
        ATS_ERROR(ret, "ABT_thread_create");
    }

    thread_test_resume((void *)0);

    for (i = 1; i < num_xstreams; i++) {
        ret = ABT_thread_free(&main_threads[i]);
        ATS_ERROR(ret, "ABT_thread_free");
    }

    /* Join Execution Streams */
    for (i = 1; i < num_xstreams; i++) {
        ret = ABT_xstream_join(xstreams[i]);
        ATS_ERROR(ret, "ABT_xstream_join");
    }

    /* Free Execution Streams */
    for (i = 1; i < num_xstreams; i++) {
        ret = ABT_xstream_free(&xstreams[i]);
        ATS_ERROR(ret, "ABT_xstream_free");
    }

    /* Free the barrier */
    ret = ABT_barrier_free(&barrier);
    ATS_ERROR(ret, "ABT_barrier_free");

    /* Finalize */
    ret = ATS_finalize(0);

    for (i = 0; i < num_xstreams; i++) {
        for (k = 0; k < num_threads; k++) {
            assert(values[i][k] == num_iter);
        }
    }

    for (i = 0; i < num_xstreams; i++) {
        free(threads[i]);
        free(values[i]);
    }
    free(threads);
    free(values);
    free(main_threads);
    free(pools);
    free(xstreams);

    return ret;
}

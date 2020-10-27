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
#define DEFAULT_NUM_TASKS 4
#define DEFAULT_NUM_ITER 10

void thread_func(void *arg)
{
    ABT_thread self;
    int ret;
    void *ret_arg = NULL;

    ret = ABT_thread_self(&self);
    ATS_ERROR(ret, "ABT_thread_self");

    ret = ABT_thread_get_arg(self, &ret_arg);
    ATS_ERROR(ret, "ABT_thread_get_arg");
    assert(ret_arg == arg);

    ret_arg = NULL;
    ret = ABT_self_get_arg(&ret_arg);
    ATS_ERROR(ret, "ABT_self_get_arg");
    assert(ret_arg == arg);
}

void task_func(void *arg)
{
    ABT_task self;
    int ret;
    void *ret_arg = NULL;

    ret = ABT_task_self(&self);
    ATS_ERROR(ret, "ABT_task_self");

    ret = ABT_task_get_arg(self, &ret_arg);
    ATS_ERROR(ret, "ABT_task_get_arg");
    assert(ret_arg == arg);

    ret_arg = NULL;
    ret = ABT_self_get_arg(&ret_arg);
    ATS_ERROR(ret, "ABT_self_get_arg");
    assert(ret_arg == arg);
}

int main(int argc, char *argv[])
{
    int num_xstreams, num_threads, num_tasks;
    ABT_xstream *xstreams;
    ABT_pool *pools;
    int i, k, ret;
    size_t tid;
    void *ret_arg;

    /* Initialize */
    ATS_read_args(argc, argv);
    if (argc < 2) {
        num_xstreams = DEFAULT_NUM_XSTREAMS;
        num_threads = DEFAULT_NUM_THREADS;
        num_tasks = DEFAULT_NUM_TASKS;
    } else {
        num_xstreams = ATS_get_arg_val(ATS_ARG_N_ES);
        num_threads = ATS_get_arg_val(ATS_ARG_N_ULT);
        num_tasks = ATS_get_arg_val(ATS_ARG_N_TASK);
    }
    ATS_init(argc, argv, num_xstreams);

    ATS_printf(1, "# of ESs     : %d\n", num_xstreams);
    ATS_printf(1, "# of ULTs/ES : %d\n", num_threads);
    ATS_printf(1, "# of tasks/ES: %d\n", num_tasks);

    xstreams = (ABT_xstream *)malloc(num_xstreams * sizeof(ABT_xstream));
    pools = (ABT_pool *)malloc(num_xstreams * sizeof(ABT_pool));

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

    /* Create ULTs and tasklets for each ES */
    for (i = 0; i < num_xstreams; i++) {
        for (k = 0; k < num_threads; k++) {
            tid = i * num_threads + k;
            ret = ABT_thread_create(pools[i], thread_func, (void *)tid,
                                    ABT_THREAD_ATTR_NULL, NULL);
            ATS_ERROR(ret, "ABT_thread_create");
        }

        for (k = 0; k < num_tasks; k++) {
            tid = i * num_tasks + k;
            ret = ABT_task_create(pools[i], task_func, (void *)tid, NULL);
        }
    }

    ret = ABT_self_get_arg(&ret_arg);
    ATS_ERROR(ret, "ABT_self_get_arg");
    assert(ret_arg == NULL);

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

    return ret;
}

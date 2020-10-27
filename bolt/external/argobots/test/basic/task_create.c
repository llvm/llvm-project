/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

#include <stdio.h>
#include <stdlib.h>
#include "abt.h"
#include "abttest.h"

#define DEFAULT_NUM_XSTREAMS 2
#define DEFAULT_NUM_TASKS 4

typedef struct {
    size_t num;
    unsigned long long result;
} task_arg_t;

void task_func1(void *arg)
{
    size_t i;
    size_t num = (size_t)arg;
    unsigned long long result = 1;
    for (i = 2; i <= num; i++) {
        result += i;
    }
    ATS_printf(1, "task_func1: num=%lu result=%llu\n", num, result);
}

void task_func2(void *arg)
{
    size_t i;
    task_arg_t *my_arg = (task_arg_t *)arg;
    unsigned long long result = 1;
    for (i = 2; i <= my_arg->num; i++) {
        result += i;
    }
    my_arg->result = result;
}

int main(int argc, char *argv[])
{
    int i, ret;
    int num_xstreams = DEFAULT_NUM_XSTREAMS;
    int num_tasks = DEFAULT_NUM_TASKS;
    if (argc > 1)
        num_xstreams = atoi(argv[1]);
    assert(num_xstreams >= 0);
    if (argc > 2)
        num_tasks = atoi(argv[2]);
    assert(num_tasks >= 0);

    ABT_xstream *xstreams;
    ABT_pool *pools;
    ABT_task *tasks;
    task_arg_t *args;
    xstreams = (ABT_xstream *)malloc(sizeof(ABT_xstream) * num_xstreams);
    pools = (ABT_pool *)malloc(sizeof(ABT_pool) * num_xstreams);
    tasks = (ABT_task *)malloc(sizeof(ABT_task) * num_tasks);
    args = (task_arg_t *)malloc(sizeof(task_arg_t) * num_tasks);

    /* Initialize */
    ATS_read_args(argc, argv);
    ATS_init(argc, argv, num_xstreams);

    /* Create Execution Streams */
    ret = ABT_xstream_self(&xstreams[0]);
    ATS_ERROR(ret, "ABT_xstream_self");
    for (i = 1; i < num_xstreams; i++) {
        ret = ABT_xstream_create(ABT_SCHED_NULL, &xstreams[i]);
        ATS_ERROR(ret, "ABT_xstream_create");
    }

    /* Get the pools attached to an execution stream */
    for (i = 0; i < num_xstreams; i++) {
        ret = ABT_xstream_get_main_pools(xstreams[i], 1, pools + i);
        ATS_ERROR(ret, "ABT_xstream_get_main_pools");
    }

    /* Create tasks with task_func1 */
    for (i = 0; i < num_tasks; i++) {
        size_t num = 100 + i;
        ret = ABT_task_create(pools[i % num_xstreams], task_func1, (void *)num,
                              NULL);
        ATS_ERROR(ret, "ABT_task_create");
    }

    /* Create tasks with task_func2 */
    for (i = 0; i < num_tasks; i++) {
        args[i].num = 100 + i;
        ret = ABT_task_create(pools[i % num_xstreams], task_func2,
                              (void *)&args[i], &tasks[i]);
        ATS_ERROR(ret, "ABT_task_create");
    }

    /* Results of task_funcs2 */
    for (i = 0; i < num_tasks; i++) {
        ABT_task_state state;
        do {
            ABT_task_get_state(tasks[i], &state);
            ABT_thread_yield();
        } while (state != ABT_TASK_STATE_TERMINATED);

        ATS_printf(1, "task_func2: num=%lu result=%llu\n", args[i].num,
                   args[i].result);

        /* Free named tasks */
        ret = ABT_task_free(&tasks[i]);
        ATS_ERROR(ret, "ABT_task_free");
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

    /* Finalize */
    ret = ATS_finalize(0);

    free(args);
    free(tasks);
    free(pools);
    free(xstreams);

    return ret;
}

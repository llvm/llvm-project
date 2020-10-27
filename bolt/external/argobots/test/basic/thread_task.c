/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "abt.h"
#include "abttest.h"

#define DEFAULT_NUM_XSTREAMS 4
#define DEFAULT_NUM_THREADS 4
#define DEFAULT_NUM_TASKS 4

typedef struct {
    size_t num;
    unsigned long long result;
} task_arg_t;

typedef struct thread_arg {
    int id;
    int num_threads;
    ABT_thread *threads;
} thread_arg_t;

ABT_thread pick_one(ABT_thread *threads, int num_threads, unsigned *seed,
                    ABT_thread caller)
{
    int k, i, ret;
    ABT_bool is_same;
    ABT_thread next;
    ABT_thread_state state = ABT_THREAD_STATE_TERMINATED;

    for (k = 0; k < num_threads; k++) {
        i = rand_r(seed) % num_threads;
        next = threads[i];
        ret = ABT_thread_equal(next, caller, &is_same);
        ATS_ERROR(ret, "ABT_thread_equal");
        if (is_same == ABT_TRUE)
            continue;

        if (next != ABT_THREAD_NULL) {
            ret = ABT_thread_get_state(next, &state);
            ATS_ERROR(ret, "ABT_thread_get_state");
            if (state != ABT_THREAD_STATE_TERMINATED) {
                return next;
            }
        }
    }

    return ABT_THREAD_NULL;
}

void thread_func(void *arg)
{
    ABT_thread my_handle, next;
    ABT_thread_state my_state;
    int ret;

    ret = ABT_thread_self(&my_handle);
    ATS_ERROR(ret, "ABT_thread_self");
    ret = ABT_thread_get_state(my_handle, &my_state);
    ATS_ERROR(ret, "ABT_thread_get_state");
    if (my_state != ABT_THREAD_STATE_RUNNING) {
        fprintf(stderr, "ERROR: not in the RUNNUNG state\n");
        exit(-1);
    }

    thread_arg_t *t_arg = (thread_arg_t *)arg;
    unsigned seed = time(NULL);

    ATS_printf(1, "[TH%d]: before yield\n", t_arg->id);

    next = pick_one(t_arg->threads, t_arg->num_threads, &seed, my_handle);
    if (next != ABT_THREAD_NULL) {
        ret = ABT_thread_yield_to(next);
        ATS_ERROR(ret, "ABT_thread_yield_to");
    }

    ATS_printf(1, "[TH%d]: doing something ...\n", t_arg->id);

    next = pick_one(t_arg->threads, t_arg->num_threads, &seed, my_handle);
    if (next != ABT_THREAD_NULL) {
        ret = ABT_thread_yield_to(next);
        ATS_ERROR(ret, "ABT_thread_yield_to");
    }

    ATS_printf(1, "[TH%d]: after yield\n", t_arg->id);

    ABT_bool is_check_error = ABT_FALSE;
    ret = ABT_info_query_config(ABT_INFO_QUERY_KIND_ENABLED_CHECK_ERROR,
                                (void *)&is_check_error);
    ATS_ERROR(ret, "ABT_info_query_config");
    if (is_check_error) {
        ABT_task task;
        ret = ABT_task_self(&task);
        assert(ret == ABT_ERR_INV_TASK);
        if (task != ABT_TASK_NULL) {
            fprintf(stderr, "ERROR: should not be a tasklet\n");
            exit(-1);
        }
    }
}

void task_func1(void *arg)
{
    ABT_task my_handle;
    ABT_task_state my_state;
    int ret;

    ret = ABT_task_self(&my_handle);
    ATS_ERROR(ret, "ABT_task_self");
    ret = ABT_task_get_state(my_handle, &my_state);
    ATS_ERROR(ret, "ABT_task_get_state");
    if (my_state != ABT_TASK_STATE_RUNNING) {
        fprintf(stderr, "ERROR: not in the RUNNUNG state\n");
        exit(-1);
    }

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
    ABT_thread thread;
    int ret;

    ABT_bool is_check_error = ABT_FALSE;
    ret = ABT_info_query_config(ABT_INFO_QUERY_KIND_ENABLED_CHECK_ERROR,
                                (void *)&is_check_error);
    ATS_ERROR(ret, "ABT_info_query_config");
    if (is_check_error) {
        ret = ABT_thread_self(&thread);
        assert(ret == ABT_ERR_INV_THREAD);
        if (thread != ABT_THREAD_NULL) {
            fprintf(stderr, "ERROR: should not be a ULT\n");
            exit(-1);
        }
    }

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
    int i, j, ret;
    int num_xstreams = DEFAULT_NUM_XSTREAMS;
    int num_threads = DEFAULT_NUM_THREADS;
    int num_tasks = DEFAULT_NUM_TASKS;
    if (argc > 1)
        num_xstreams = atoi(argv[1]);
    assert(num_xstreams >= 0);
    if (argc > 2)
        num_threads = atoi(argv[2]);
    assert(num_threads >= 0);
    if (argc > 3)
        num_tasks = atoi(argv[3]);
    assert(num_tasks >= 0);

    ABT_xstream *xstreams;
    ABT_thread **threads;
    thread_arg_t **thread_args;
    ABT_task *tasks;
    task_arg_t *task_args;

    xstreams = (ABT_xstream *)malloc(sizeof(ABT_xstream) * num_xstreams);
    threads = (ABT_thread **)malloc(sizeof(ABT_thread *) * num_xstreams);
    thread_args =
        (thread_arg_t **)malloc(sizeof(thread_arg_t *) * num_xstreams);
    for (i = 0; i < num_xstreams; i++) {
        threads[i] = (ABT_thread *)malloc(sizeof(ABT_thread) * num_threads);
        for (j = 0; j < num_threads; j++) {
            threads[i][j] = ABT_THREAD_NULL;
        }
        thread_args[i] =
            (thread_arg_t *)malloc(sizeof(thread_arg_t) * num_threads);
    }
    tasks = (ABT_task *)malloc(sizeof(ABT_task) * num_tasks);
    task_args = (task_arg_t *)malloc(sizeof(task_arg_t) * num_tasks);

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
    ABT_pool *pools;
    pools = (ABT_pool *)malloc(sizeof(ABT_pool) * num_xstreams);
    for (i = 0; i < num_xstreams; i++) {
        ret = ABT_xstream_get_main_pools(xstreams[i], 1, pools + i);
        ATS_ERROR(ret, "ABT_xstream_get_main_pools");
    }

    /* Create threads */
    for (i = 0; i < num_xstreams; i++) {
        for (j = 0; j < num_threads; j++) {
            int tid = i * num_threads + j + 1;
            thread_args[i][j].id = tid;
            thread_args[i][j].num_threads = num_threads;
            thread_args[i][j].threads = &threads[i][0];
            ret = ABT_thread_create(pools[i], thread_func,
                                    (void *)&thread_args[i][j],
                                    ABT_THREAD_ATTR_NULL, &threads[i][j]);
            ATS_ERROR(ret, "ABT_thread_create");
        }
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
        task_args[i].num = 100 + i;
        ret = ABT_task_create(pools[i % num_xstreams], task_func2,
                              (void *)&task_args[i], &tasks[i]);
        ATS_ERROR(ret, "ABT_task_create");
    }

    /* Results of task_funcs2 */
    for (i = 0; i < num_tasks; i++) {
        ABT_task_state state;
        do {
            ABT_task_get_state(tasks[i], &state);
            ABT_thread_yield();
        } while (state != ABT_TASK_STATE_TERMINATED);

        ATS_printf(1, "task_func2: num=%lu result=%llu\n", task_args[i].num,
                   task_args[i].result);

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
    for (i = 0; i < num_xstreams; i++) {
        for (j = 0; j < num_threads; j++) {
            ret = ABT_thread_free(&threads[i][j]);
            ATS_ERROR(ret, "ABT_thread_free");
        }

        if (i == 0)
            continue;

        ret = ABT_xstream_free(&xstreams[i]);
        ATS_ERROR(ret, "ABT_xstream_free");
    }

    /* Finalize */
    ret = ATS_finalize(0);

    for (i = 0; i < num_xstreams; i++) {
        free(thread_args[i]);
        free(threads[i]);
    }
    free(thread_args);
    free(threads);
    free(task_args);
    free(tasks);
    free(pools);
    free(xstreams);

    return ret;
}

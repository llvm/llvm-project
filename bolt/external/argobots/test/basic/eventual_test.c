/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

#include <stdio.h>
#include <stdlib.h>
#include "abt.h"
#include "abttest.h"

#define DEFAULT_NUM_XSTREAMS 4
#define DEFAULT_NUM_TASKS 4
#define DEFAULT_NUM_THREADS (DEFAULT_NUM_TASKS * 2)
#define DEFAULT_NUM_ITER 100

static int num_xstreams;
static int num_threads;
static int num_tasks;
static int num_iter;
static ABT_pool *pools;

#define OP_WAIT 0
#define OP_SET 1

typedef struct {
    int eid;
    int tid;
    ABT_eventual ev;
    int nbytes;
    int op_type;
} arg_t;

void thread_func(void *arg)
{
    int ret;
    arg_t *my_arg = (arg_t *)arg;

    ATS_printf(3, "[U%d:E%d] %s\n", my_arg->tid, my_arg->eid,
               my_arg->op_type == OP_WAIT ? "wait" : "set");

    if (my_arg->op_type == OP_WAIT) {
        if (my_arg->nbytes == 0) {
            ret = ABT_eventual_wait(my_arg->ev, NULL);
        } else {
            void *val;
            ret = ABT_eventual_wait(my_arg->ev, &val);
            assert(*(int *)val == 1);
        }
        ATS_ERROR(ret, "ABT_eventual_wait");

    } else if (my_arg->op_type == OP_SET) {
        if (my_arg->nbytes == 0) {
            ret = ABT_eventual_set(my_arg->ev, NULL, 0);
        } else {
            int val = 1;
            ret = ABT_eventual_set(my_arg->ev, &val, sizeof(int));
        }
        ATS_ERROR(ret, "ABT_eventual_set");
    }

    ATS_printf(3, "[U%d:E%d] done\n", my_arg->tid, my_arg->eid);
}

void task_func(void *arg)
{
    int ret;
    arg_t *my_arg = (arg_t *)arg;
    assert(my_arg->op_type == OP_SET);

    if (my_arg->nbytes == 0) {
        ret = ABT_eventual_set(my_arg->ev, NULL, 0);
    } else {
        int val = 1;
        ret = ABT_eventual_set(my_arg->ev, &val, sizeof(int));
    }
    ATS_ERROR(ret, "ABT_eventual_set");
}

void eventual_test(void *arg)
{
    int i, t, ret;
    size_t pid = (size_t)arg;
    int eid = (int)pid;

    ABT_thread *threads;
    arg_t *thread_args;
    int *nbytes;
    ABT_eventual *evs1;

    ABT_thread *waiters;
    arg_t *waiter_args;
    ABT_task *tasks;
    arg_t *task_args;
    ABT_eventual *evs2;

    ATS_printf(1, "[M%u] start\n", (unsigned)(size_t)arg);

    assert(num_tasks * 2 == num_threads);

    threads = (ABT_thread *)malloc(num_threads * sizeof(ABT_thread));
    thread_args = (arg_t *)malloc(num_threads * sizeof(arg_t));
    nbytes = (int *)malloc(num_tasks * sizeof(int));
    evs1 = (ABT_eventual *)malloc(num_tasks * sizeof(ABT_eventual));
    assert(threads && thread_args && nbytes && evs1);

    waiters = (ABT_thread *)malloc(num_tasks * sizeof(ABT_thread));
    waiter_args = (arg_t *)malloc(num_tasks * sizeof(arg_t));
    tasks = (ABT_task *)malloc(num_tasks * sizeof(ABT_task));
    task_args = (arg_t *)malloc(num_tasks * sizeof(arg_t));
    evs2 = (ABT_eventual *)malloc(num_tasks * sizeof(ABT_eventual));
    assert(waiters && waiter_args && tasks && task_args && evs2);

    for (t = 0; t < num_tasks; t++) {
        nbytes[t] = (t & 1) ? sizeof(int) : 0;

        ret = ABT_eventual_create(nbytes[t], &evs1[t]);
        ATS_ERROR(ret, "ABT_eventual_create");

        ret = ABT_eventual_create(nbytes[t], &evs2[t]);
        ATS_ERROR(ret, "ABT_eventual_create");
    }

    for (i = 0; i < num_iter; i++) {
        ATS_printf(2, "[M%u] iter=%d\n", (unsigned)(size_t)arg, i);

        /* Use eventual between ULTs */
        for (t = 0; t < num_threads; t += 2) {
            thread_args[t].eid = eid;
            thread_args[t].tid = t;
            thread_args[t].ev = evs1[t / 2];
            thread_args[t].nbytes = nbytes[t / 2];
            thread_args[t].op_type = OP_WAIT;
            ret = ABT_thread_create(pools[pid], thread_func, &thread_args[t],
                                    ABT_THREAD_ATTR_NULL, &threads[t]);
            ATS_ERROR(ret, "ABT_thread_create");
            pid = (pid + 1) % num_xstreams;

            thread_args[t + 1].eid = eid;
            thread_args[t + 1].tid = t + 1;
            thread_args[t + 1].ev = evs1[t / 2];
            thread_args[t + 1].nbytes = nbytes[t / 2];
            thread_args[t + 1].op_type = OP_SET;
            ret =
                ABT_thread_create(pools[pid], thread_func, &thread_args[t + 1],
                                  ABT_THREAD_ATTR_NULL, &threads[t + 1]);
            ATS_ERROR(ret, "ABT_thread_create");
            pid = (pid + 1) % num_xstreams;
        }

        /* Use eventual between ULT and tasklet */
        for (t = 0; t < num_tasks; t++) {
            waiter_args[t].ev = evs2[t];
            waiter_args[t].nbytes = nbytes[t];
            waiter_args[t].op_type = OP_WAIT;
            ret = ABT_thread_create(pools[pid], thread_func, &waiter_args[t],
                                    ABT_THREAD_ATTR_NULL, &waiters[t]);
            ATS_ERROR(ret, "ABT_thread_create");
            pid = (pid + 1) % num_xstreams;

            task_args[t].ev = evs2[t];
            task_args[t].nbytes = nbytes[t];
            task_args[t].op_type = OP_SET;
            ret = ABT_task_create(pools[pid], task_func, &task_args[t],
                                  &tasks[t]);
            ATS_ERROR(ret, "ABT_task_create");
            pid = (pid + 1) % num_xstreams;
        }

        for (t = 0; t < num_threads; t++) {
            ret = ABT_thread_free(&threads[t]);
            ATS_ERROR(ret, "ABT_thread_free");
        }

        for (t = 0; t < num_tasks; t++) {
            ret = ABT_thread_free(&waiters[t]);
            ATS_ERROR(ret, "ABT_thread_free");
            ret = ABT_task_free(&tasks[t]);
            ATS_ERROR(ret, "ABT_task_free");
        }

        for (t = 0; t < num_tasks; t++) {
            ret = ABT_eventual_reset(evs1[t]);
            ATS_ERROR(ret, "ABT_eventual_reset");
            ret = ABT_eventual_reset(evs2[t]);
            ATS_ERROR(ret, "ABT_eventual_reset");
        }
    }

    for (t = 0; t < num_tasks; t++) {
        ret = ABT_eventual_free(&evs1[t]);
        ATS_ERROR(ret, "ABT_eventual_free");
        ret = ABT_eventual_free(&evs2[t]);
        ATS_ERROR(ret, "ABT_eventual_free");
    }

    free(threads);
    free(thread_args);
    free(nbytes);
    free(evs1);

    free(waiters);
    free(waiter_args);
    free(tasks);
    free(task_args);
    free(evs2);

    ATS_printf(1, "[M%u] end\n", (unsigned)(size_t)arg);
}

int main(int argc, char *argv[])
{
    ABT_xstream *xstreams;
    ABT_thread *main_threads;
    int i, ret;

    /* Initialize */
    ATS_read_args(argc, argv);
    if (argc < 2) {
        num_xstreams = DEFAULT_NUM_XSTREAMS;
        num_threads = DEFAULT_NUM_THREADS;
        num_tasks = DEFAULT_NUM_TASKS;
        num_iter = DEFAULT_NUM_ITER;
    } else {
        num_xstreams = ATS_get_arg_val(ATS_ARG_N_ES);
        num_tasks = ATS_get_arg_val(ATS_ARG_N_TASK);
        num_threads = num_tasks * 2;
        num_iter = ATS_get_arg_val(ATS_ARG_N_ITER);
    }
    ATS_init(argc, argv, num_xstreams);

    ATS_printf(1, "# of ESs        : %d\n", num_xstreams);
    ATS_printf(1, "# of ULTs/ES    : %d\n", num_threads + num_tasks);
    ATS_printf(1, "# of tasklets/ES: %d\n", num_tasks);
    ATS_printf(1, "# of iter       : %d\n", num_iter);

    xstreams = (ABT_xstream *)malloc(num_xstreams * sizeof(ABT_xstream));
    pools = (ABT_pool *)malloc(num_xstreams * sizeof(ABT_pool));
    main_threads = (ABT_thread *)malloc(num_xstreams * sizeof(ABT_thread));

    /* Create Execution Streams */
    ret = ABT_xstream_self(&xstreams[0]);
    ATS_ERROR(ret, "ABT_xstream_self");
    for (i = 1; i < num_xstreams; i++) {
        ret = ABT_xstream_create(ABT_SCHED_NULL, &xstreams[i]);
        ATS_ERROR(ret, "ABT_xstream_create");
    }

    /* Get the main pool of each ES */
    for (i = 0; i < num_xstreams; i++) {
        ret = ABT_xstream_get_main_pools(xstreams[i], 1, &pools[i]);
        ATS_ERROR(ret, "ABT_xstream_get_main_pools");
    }

    /* Create a main ULT for each ES */
    for (i = 1; i < num_xstreams; i++) {
        ret = ABT_thread_create(pools[i], eventual_test, (void *)(size_t)i,
                                ABT_THREAD_ATTR_NULL, &main_threads[i]);
        ATS_ERROR(ret, "ABT_thread_create");
    }

    eventual_test((void *)0);

    /* Join main ULTs */
    for (i = 1; i < num_xstreams; i++) {
        ret = ABT_thread_free(&main_threads[i]);
        ATS_ERROR(ret, "ABT_thread_free");
    }

    /* Join and free Execution Streams */
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
    free(main_threads);

    return ret;
}

/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

#include <stdio.h>
#include <stdlib.h>
#include "abt.h"
#include "abttest.h"

enum {
    T_CREATE_COLD = 0,
    T_FREE_COLD,
    T_CREATE_FREE_COLD,
    T_CREATE,
    T_FREE,
    T_CREATE_FREE,
    T_CREATE_UNNAMED,
    T_LAST
};
static char *t_names[] = {
    "create (cold)", "free (cold)", "create/free (cold)", "create",
    "free",          "create/free", "create (unnamed)",
};

enum {
    T_ALL_CREATE_FREE_COLD = 0,
    T_ALL_CREATE_FREE,
    T_ALL_CREATE_UNNAMED,
    T_ALL_LAST
};
static char *t_all_names[] = {
    "all create/free (cold)",
    "all create/free",
    "all create (unnamed)",
};

static int iter;
static int num_xstreams;
static int num_tasks;

static ABT_xstream_barrier g_xbarrier = ABT_XSTREAM_BARRIER_NULL;

static ABT_xstream *g_xstreams;
static ABT_pool *g_pools;
static ABT_task **g_tasks;

static uint64_t (*t_times)[T_LAST];
static uint64_t t_all[T_ALL_LAST];

void task_func(void *arg)
{
    ATS_UNUSED(arg);
}

void task_test(void *arg)
{
    int eid = (int)(size_t)arg;
    ABT_pool my_pool = g_pools[eid];
    ABT_task *my_tasks = g_tasks[eid];
    uint64_t *my_times = t_times[eid];
    uint64_t t_all_start, t_start, t_time;
    int i, t;

    ATS_printf(1, "[E%d] main ULT: start\n", eid);

    /*************************************************************************/
    /* tasklet: create/join (cold) */
    ABT_xstream_barrier_wait(g_xbarrier);
    if (eid == 0)
        t_all_start = ATS_get_cycles();

    t_start = ATS_get_cycles();
    for (i = 0; i < num_tasks; i++) {
        ABT_task_create(my_pool, task_func, NULL, &my_tasks[i]);
    }
    my_times[T_CREATE_COLD] = ATS_get_cycles() - t_start;

    t_start = ATS_get_cycles();
    for (i = 0; i < num_tasks; i++) {
        ABT_task_free(&my_tasks[i]);
    }
    my_times[T_FREE_COLD] = ATS_get_cycles() - t_start;

    ABT_xstream_barrier_wait(g_xbarrier);
    if (eid == 0) {
        /* execution time for all ESs */
        t_all[T_ALL_CREATE_FREE_COLD] = ATS_get_cycles() - t_all_start;
    }
    my_times[T_CREATE_COLD] /= num_tasks;
    my_times[T_FREE_COLD] /= num_tasks;
    my_times[T_CREATE_FREE_COLD] =
        my_times[T_CREATE_COLD] + my_times[T_FREE_COLD];
    /*************************************************************************/

    /*************************************************************************/
    /* tasklet: create/join */
    /* measure the time for individual operation */
    ABT_xstream_barrier_wait(g_xbarrier);
    for (i = 0; i < iter; i++) {
        t_start = ATS_get_cycles();
        for (t = 0; t < num_tasks; t++) {
            ABT_task_create(my_pool, task_func, NULL, &my_tasks[t]);
        }
        my_times[T_CREATE] += (ATS_get_cycles() - t_start);

        t_start = ATS_get_cycles();
        for (t = 0; t < num_tasks; t++) {
            ABT_task_free(&my_tasks[t]);
        }
        my_times[T_FREE] += (ATS_get_cycles() - t_start);
    }
    my_times[T_CREATE] /= (iter * num_tasks);
    my_times[T_FREE] /= (iter * num_tasks);
    my_times[T_CREATE_FREE] = my_times[T_CREATE] + my_times[T_FREE];

    /* measure tasklet create/free time */
    ABT_xstream_barrier_wait(g_xbarrier);
    if (eid == 0)
        t_all_start = ATS_get_cycles();
    t_start = ATS_get_cycles();
    for (i = 0; i < iter; i++) {
        for (t = 0; t < num_tasks; t++) {
            ABT_task_create(my_pool, task_func, NULL, &my_tasks[t]);
        }

        for (t = 0; t < num_tasks; t++) {
            ABT_task_free(&my_tasks[t]);
        }
    }
    my_times[T_CREATE_FREE] = ATS_get_cycles() - t_start;
    ABT_xstream_barrier_wait(g_xbarrier);
    if (eid == 0) {
        /* execution time for all ESs */
        t_time = ATS_get_cycles() - t_all_start;
        t_all[T_ALL_CREATE_FREE] = t_time / iter;
    }
    my_times[T_CREATE_FREE] /= (iter * num_tasks);

    /* measure tasklet create (unnamed) time */
    ABT_xstream_barrier_wait(g_xbarrier);
    if (eid == 0)
        t_all_start = ATS_get_cycles();
    t_start = ATS_get_cycles();
    for (i = 0; i < iter; i++) {
        for (t = 0; t < num_tasks; t++) {
            ABT_task_create(my_pool, task_func, NULL, NULL);
        }
        while (1) {
            ABT_thread_yield();
            size_t size;
            ABT_pool_get_size(my_pool, &size);
            if (size == 0)
                break;
        }
    }
    my_times[T_CREATE_UNNAMED] = ATS_get_cycles() - t_start;
    ABT_xstream_barrier_wait(g_xbarrier);
    if (eid == 0) {
        /* execution time for all ESs */
        t_time = ATS_get_cycles() - t_all_start;
        t_all[T_ALL_CREATE_UNNAMED] = t_time / iter;
    }
    my_times[T_CREATE_UNNAMED] /= (iter * num_tasks);
    /*************************************************************************/
}

int main(int argc, char *argv[])
{
    int i, t;
    uint64_t t_avg[T_LAST];
    uint64_t t_min[T_LAST];
    uint64_t t_max[T_LAST];

    /* read command-line arguments */
    ATS_read_args(argc, argv);
    num_xstreams = ATS_get_arg_val(ATS_ARG_N_ES);
    num_tasks = ATS_get_arg_val(ATS_ARG_N_TASK);
    iter = ATS_get_arg_val(ATS_ARG_N_ITER);

    /* initialize */
    ATS_init(argc, argv, num_xstreams);

    for (i = 0; i < T_LAST; i++) {
        t_avg[i] = 0;
        t_min[i] = UINTMAX_MAX;
        t_max[i] = 0;
    }
    for (i = 0; i < T_ALL_LAST; i++) {
        t_all[i] = 0;
    }

    g_xstreams = (ABT_xstream *)malloc(num_xstreams * sizeof(ABT_xstream));
    g_pools = (ABT_pool *)malloc(num_xstreams * sizeof(ABT_pool));
    g_tasks = (ABT_task **)malloc(num_xstreams * sizeof(ABT_task *));
    for (i = 0; i < num_xstreams; i++) {
        g_tasks[i] = (ABT_task *)malloc(num_tasks * sizeof(ABT_task));
    }
    t_times =
        (uint64_t(*)[T_LAST])calloc(num_xstreams, sizeof(uint64_t) * T_LAST);

    /* create a global barrier */
    ABT_xstream_barrier_create(num_xstreams, &g_xbarrier);

    /* create pools */
    for (i = 0; i < num_xstreams; i++) {
        ABT_pool_create_basic(ABT_POOL_FIFO, ABT_POOL_ACCESS_PRIV, ABT_TRUE,
                              &g_pools[i]);
    }

    /* create a main ULT for each ES */
    for (i = 1; i < num_xstreams; i++) {
        /* The ULT will run all test cases */
        ABT_thread_create(g_pools[i], task_test, (void *)(size_t)i,
                          ABT_THREAD_ATTR_NULL, NULL);
    }

    /* create ESs with a new default scheduler */
    ABT_xstream_self(&g_xstreams[0]);
    ABT_xstream_set_main_sched_basic(g_xstreams[0], ABT_SCHED_DEFAULT, 1,
                                     &g_pools[0]);
    for (i = 1; i < num_xstreams; i++) {
        ABT_xstream_create_basic(ABT_SCHED_DEFAULT, 1, &g_pools[i],
                                 ABT_SCHED_CONFIG_NULL, &g_xstreams[i]);
    }

    /* execute task_test() using the primary ULT */
    task_test((void *)0);

    /* join and free ESs */
    for (i = 1; i < num_xstreams; i++) {
        ABT_xstream_join(g_xstreams[i]);
        ABT_xstream_free(&g_xstreams[i]);
    }
    ABT_xstream_barrier_free(&g_xbarrier);

    /* find min, max, avg of each case */
    for (i = 0; i < num_xstreams; i++) {
        for (t = 0; t < T_LAST; t++) {
            if (t_times[i][t] < t_min[t])
                t_min[t] = t_times[i][t];
            if (t_times[i][t] > t_max[t])
                t_max[t] = t_times[i][t];
            t_avg[t] += t_times[i][t];
        }
    }
    for (t = 0; t < T_LAST; t++) {
        t_avg[t] = t_avg[t] / num_xstreams;
    }

    /* finalize */
    ATS_finalize(0);

    /* output */
    int line_size = 59;
    ATS_print_line(stdout, '-', line_size);
    printf("# of ESs            : %d\n", num_xstreams);
    printf("# of tasklets per ES: %d\n", num_tasks);
    ATS_print_line(stdout, '-', line_size);
    printf("Avg. execution time (in seconds, %d times)\n", iter);
    ATS_print_line(stdout, '-', line_size);
    printf("%-23s %11s %11s %11s\n", "operation", "avg", "min", "max");
    ATS_print_line(stdout, '-', line_size);
    for (i = 0; i < T_LAST; i++) {
        printf("%-22s  %11" PRIu64 " %11" PRIu64 " %11" PRIu64 "\n", t_names[i],
               t_avg[i], t_min[i], t_max[i]);
    }
    ATS_print_line(stdout, '-', line_size);
    for (i = 0; i < T_ALL_LAST; i++) {
        printf("%-22s  %11" PRIu64 "\n", t_all_names[i], t_all[i]);
    }
    ATS_print_line(stdout, '-', line_size);

    free(g_xstreams);
    free(g_pools);
    for (i = 0; i < num_xstreams; i++) {
        free(g_tasks[i]);
    }
    free(g_tasks);
    free(t_times);

    return EXIT_SUCCESS;
}

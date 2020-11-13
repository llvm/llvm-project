/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

#include <stdio.h>
#include <stdlib.h>
#include "abt.h"
#include "abttest.h"

//#define TEST_MIGRATE_TO
#define USE_JOIN_MANY
#ifdef USE_JOIN_MANY
#define ABT_THREAD_JOIN_MANY(n, tl) ABT_thread_join_many(n, tl)
#else
#define ABT_THREAD_JOIN_MANY(n, tl)
#endif

enum {
    T_CREATE_COLD = 0,
    T_JOIN_COLD,
    T_CREATE_JOIN_COLD,
    T_CREATE,
    T_JOIN,
    T_CREATE_JOIN,
    T_CREATE_UNNAMED,
    T_YIELD,
    T_YIELD_TO,
#ifdef TEST_MIGRATE_TO
    T_MIGRATE_TO_XSTREAM,
#endif
    T_LAST
};
static char *t_names[] = { "create (cold)",
                           "join (cold)",
                           "create/join (cold)",
                           "create",
                           "join",
                           "create/join",
                           "create (unnamed)",
                           "yield",
                           "yield_to",
#ifdef TEST_MIGRATE_TO
                           "migrate_to_xstream"
#endif
};

enum {
    T_ALL_CREATE_JOIN_COLD = 0,
    T_ALL_CREATE_JOIN,
    T_ALL_CREATE_UNNAMED,
    T_ALL_YIELD,
    T_ALL_YIELD_TO,
#ifdef TEST_MIGRATE_TO
    T_ALL_MIGRATE_TO_XSTREAM,
#endif
    T_ALL_LAST
};
static char *t_all_names[] = { "all create/join (cold)", "all create/join",
                               "all create (unnamed)",   "all yield",
                               "all yield_to",
#ifdef TEST_MIGRATE_TO
                               "all migrate_to_xstream"
#endif
};

typedef struct {
    int eid;
    int tid;
} arg_t;

static int iter;
static int num_xstreams;
static int num_threads;

static ABT_xstream_barrier g_xbarrier = ABT_XSTREAM_BARRIER_NULL;

static ABT_xstream *g_xstreams;
static ABT_pool *g_pools;
static ABT_thread **g_threads;

static uint64_t (*t_times)[T_LAST];
static uint64_t t_all[T_ALL_LAST];

void thread_func(void *arg)
{
    ATS_UNUSED(arg);
}

void thread_func_yield(void *arg)
{
    ATS_UNUSED(arg);
    int i;
    for (i = 0; i < iter; i++) {
        ABT_thread_yield();
    }
}

void thread_func_yield_to(void *arg)
{
    arg_t *my_arg = (arg_t *)arg;
    int eid = my_arg->eid;
    int tid = my_arg->tid;
    int nid = (tid + 1) % num_threads;
    ABT_thread next = g_threads[eid][nid];
    int i;

    for (i = 0; i < iter; i++) {
        ABT_thread_yield_to(next);
    }
    ABT_thread_yield();
}

#ifdef TEST_MIGRATE_TO
void thread_func_migrate_to_xstream(void *arg)
{
    arg_t *my_arg = (arg_t *)arg;
    int eid = my_arg->eid;
    ABT_xstream cur_xstream, tar_xstream;
    ABT_thread self;
    int i, next;

    ABT_thread_self(&self);

    for (i = 0; i < iter; i++) {
        next = (eid + 1) % num_xstreams;
        tar_xstream = g_xstreams[next];

        ABT_thread_migrate_to_xstream(self, tar_xstream);
        while (1) {
            ABT_bool flag;
            ABT_xstream_self(&cur_xstream);
            ABT_xstream_equal(cur_xstream, tar_xstream, &flag);
            if (flag == ABT_TRUE) {
                break;
            }
            ABT_thread_yield();
        }
        eid = next;
    }
}
#endif

void thread_test(void *arg)
{
    int eid = (int)(size_t)arg;
    ABT_pool my_pool = g_pools[eid];
    ABT_thread *my_threads = g_threads[eid];
    uint64_t *my_times = t_times[eid];

    uint64_t t_all_start, t_start, t_time;
    int i, t;
    arg_t *args;

    ATS_printf(1, "[E%d] main ULT: start\n", eid);

    /*************************************************************************/
    /* ULT: create/join (cold) */
    ABT_xstream_barrier_wait(g_xbarrier);
    if (eid == 0)
        t_all_start = ATS_get_cycles();

    t_start = ATS_get_cycles();
    for (i = 0; i < num_threads; i++) {
        ABT_thread_create(my_pool, thread_func, NULL, ABT_THREAD_ATTR_NULL,
                          &my_threads[i]);
    }
    my_times[T_CREATE_COLD] = ATS_get_cycles() - t_start;

    t_start = ATS_get_cycles();
    ABT_THREAD_JOIN_MANY(num_threads, my_threads);
    for (i = 0; i < num_threads; i++) {
        ABT_thread_free(&my_threads[i]);
    }
    my_times[T_JOIN_COLD] = ATS_get_cycles() - t_start;

    ABT_xstream_barrier_wait(g_xbarrier);
    if (eid == 0) {
        /* execution time for all ESs */
        t_all[T_ALL_CREATE_JOIN_COLD] = ATS_get_cycles() - t_all_start;
    }
    my_times[T_CREATE_COLD] /= num_threads;
    my_times[T_JOIN_COLD] /= num_threads;
    my_times[T_CREATE_JOIN_COLD] =
        my_times[T_CREATE_COLD] + my_times[T_JOIN_COLD];
    /*************************************************************************/

    /*************************************************************************/
    /* ULT: create/join */
    /* measure the time for individual operation */
    ABT_xstream_barrier_wait(g_xbarrier);
    for (i = 0; i < iter; i++) {
        t_start = ATS_get_cycles();
        for (t = 0; t < num_threads; t++) {
            ABT_thread_create(my_pool, thread_func, NULL, ABT_THREAD_ATTR_NULL,
                              &my_threads[t]);
        }
        my_times[T_CREATE] += (ATS_get_cycles() - t_start);

        t_start = ATS_get_cycles();
        ABT_THREAD_JOIN_MANY(num_threads, my_threads);
        for (t = 0; t < num_threads; t++) {
            ABT_thread_free(&my_threads[t]);
        }
        my_times[T_JOIN] += (ATS_get_cycles() - t_start);
    }
    my_times[T_CREATE] /= (iter * num_threads);
    my_times[T_JOIN] /= (iter * num_threads);

    /* measure the time for create/join operations */
    ABT_xstream_barrier_wait(g_xbarrier);
    if (eid == 0)
        t_all_start = ATS_get_cycles();
    t_start = ATS_get_cycles();
    for (i = 0; i < iter; i++) {
        for (t = 0; t < num_threads; t++) {
            ABT_thread_create(my_pool, thread_func, NULL, ABT_THREAD_ATTR_NULL,
                              &my_threads[t]);
        }

        ABT_THREAD_JOIN_MANY(num_threads, my_threads);
        for (t = 0; t < num_threads; t++) {
            ABT_thread_free(&my_threads[t]);
        }
    }
    my_times[T_CREATE_JOIN] = ATS_get_cycles() - t_start;
    ABT_xstream_barrier_wait(g_xbarrier);
    if (eid == 0) {
        /* execution time for all ESs */
        t_time = ATS_get_cycles() - t_all_start;
        t_all[T_ALL_CREATE_JOIN] = t_time / iter;
    }
    my_times[T_CREATE_JOIN] /= (iter * num_threads);

    /* measure the time for create (unnamed) operations */
    ABT_xstream_barrier_wait(g_xbarrier);
    if (eid == 0)
        t_all_start = ATS_get_cycles();
    t_start = ATS_get_cycles();
    for (i = 0; i < iter; i++) {
        for (t = 0; t < num_threads; t++) {
            ABT_thread_create(my_pool, thread_func, NULL, ABT_THREAD_ATTR_NULL,
                              NULL);
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
    my_times[T_CREATE_UNNAMED] /= (iter * num_threads);
    /*************************************************************************/

    /*************************************************************************/
    /* ULT: yield */
    /* cache warm-up */
    for (i = 0; i < num_threads; i++) {
        ABT_thread_create(my_pool, thread_func_yield, NULL,
                          ABT_THREAD_ATTR_NULL, &my_threads[i]);
    }
    ABT_THREAD_JOIN_MANY(num_threads, my_threads);
    for (i = 0; i < num_threads; i++) {
        ABT_thread_free(&my_threads[i]);
    }

    /* measure the time */
    ABT_xstream_barrier_wait(g_xbarrier);
    if (eid == 0)
        t_all_start = ATS_get_cycles();
    t_start = ATS_get_cycles();
    for (i = 0; i < num_threads; i++) {
        ABT_thread_create(my_pool, thread_func_yield, NULL,
                          ABT_THREAD_ATTR_NULL, &my_threads[i]);
    }
    ABT_THREAD_JOIN_MANY(num_threads, my_threads);
    for (i = 0; i < num_threads; i++) {
        ABT_thread_free(&my_threads[i]);
    }
    my_times[T_YIELD] = ATS_get_cycles() - t_start;
    ABT_xstream_barrier_wait(g_xbarrier);
    if (eid == 0) {
        /* execution time for all ESs */
        t_time = ATS_get_cycles() - t_all_start;
        t_all[T_ALL_YIELD] = t_time / iter;
    }
    if (my_times[T_YIELD] > my_times[T_CREATE_JOIN]) {
        my_times[T_YIELD] -= my_times[T_CREATE_JOIN];
    } else {
        my_times[T_YIELD] = 0;
    }
    my_times[T_YIELD] /= (iter * num_threads);
    /*************************************************************************/

    /*************************************************************************/
    /* ULT: yield_to */
    /* cache warm-up */
    args = (arg_t *)malloc(num_threads * sizeof(arg_t));
    for (i = 0; i < num_threads; i++) {
        args[i].eid = eid;
        args[i].tid = i;
        ABT_thread_create(my_pool, thread_func_yield_to, (void *)&args[i],
                          ABT_THREAD_ATTR_NULL, &my_threads[i]);
    }
    ABT_THREAD_JOIN_MANY(num_threads, my_threads);
    for (i = 0; i < num_threads; i++) {
        ABT_thread_free(&my_threads[i]);
    }
    free(args);

    /* measure the time */
    args = (arg_t *)malloc(num_threads * sizeof(arg_t));
    ABT_xstream_barrier_wait(g_xbarrier);
    if (eid == 0)
        t_all_start = ATS_get_cycles();
    t_start = ATS_get_cycles();
    for (i = 0; i < num_threads; i++) {
        args[i].eid = eid;
        args[i].tid = i;
        ABT_thread_create(my_pool, thread_func_yield_to, (void *)&args[i],
                          ABT_THREAD_ATTR_NULL, &my_threads[i]);
    }
    ABT_THREAD_JOIN_MANY(num_threads, my_threads);
    for (i = 0; i < num_threads; i++) {
        ABT_thread_free(&my_threads[i]);
    }
    my_times[T_YIELD_TO] = ATS_get_cycles() - t_start;
    ABT_xstream_barrier_wait(g_xbarrier);
    if (eid == 0) {
        /* execution time for all ESs */
        t_time = ATS_get_cycles() - t_all_start;
        t_all[T_ALL_YIELD_TO] = t_time / iter;
    }
    if (my_times[T_YIELD_TO] > my_times[T_CREATE_JOIN]) {
        my_times[T_YIELD_TO] -= my_times[T_CREATE_JOIN];
    } else {
        my_times[T_YIELD_TO] = 0;
    }
    my_times[T_YIELD_TO] /= (iter * num_threads);
    free(args);
    /*************************************************************************/

#ifdef TEST_MIGRATE_TO
    /*************************************************************************/
    /* ULT: migrate_to_xstream */
    args = (arg_t *)malloc(num_threads * sizeof(arg_t));
    ABT_xstream_barrier_wait(g_xbarrier);
    if (eid == 0)
        t_all_start = ATS_get_cycles();
    t_start = ATS_get_cycles();
    for (i = 0; i < num_threads; i++) {
        args[i].eid = eid;
        args[i].tid = i;
        ABT_thread_create(my_pool, thread_func_migrate_to_xstream,
                          (void *)&args[i], ABT_THREAD_ATTR_NULL,
                          &my_threads[i]);
    }
    ABT_THREAD_JOIN_MANY(num_threads, my_threads);
    for (i = 0; i < num_threads; i++) {
        ABT_thread_free(&my_threads[i]);
    }
    my_times[T_MIGRATE_TO_XSTREAM] = ATS_get_cycles() - t_start;
    ABT_xstream_barrier_wait(g_xbarrier);
    if (eid == 0) {
        /* execution time for all ESs */
        t_time = ATS_get_cycles() - t_all_start;
        t_all[T_ALL_MIGRATE_TO_XSTREAM] = t_time / iter;
    }
    if (my_times[T_MIGRATE_TO_XSTREAM] > my_times[T_CREATE_JOIN]) {
        my_times[T_MIGRATE_TO_XSTREAM] -= my_times[T_CREATE_JOIN];
    } else {
        my_times[T_MIGRATE_TO_XSTREAM] = 0;
    }
    my_times[T_MIGRATE_TO_XSTREAM] /= (iter * num_threads);
    free(args);
    /*************************************************************************/
#endif
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
    num_threads = ATS_get_arg_val(ATS_ARG_N_ULT);
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
    g_threads = (ABT_thread **)malloc(num_xstreams * sizeof(ABT_thread *));
    for (i = 0; i < num_xstreams; i++) {
        g_threads[i] = (ABT_thread *)malloc(num_threads * sizeof(ABT_thread));
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
        ABT_thread_create(g_pools[i], thread_test, (void *)(size_t)i,
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

    /* execute thread_test() using the primary ULT */
    thread_test((void *)0);

    /* join and free */
    for (i = 1; i < num_xstreams; i++) {
        ABT_xstream_join(g_xstreams[i]);
        ABT_xstream_free(&g_xstreams[i]);
    }
    ABT_xstream_barrier_free(&g_xbarrier);

    /* find min, max, and avg of each case */
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
    int line_size = 56;
    ATS_print_line(stdout, '-', line_size);
    printf("%s\n", "Argobots");
    ATS_print_line(stdout, '-', line_size);
    printf("# of ESs        : %d\n", num_xstreams);
    printf("# of ULTs per ES: %d\n", num_threads);
    ATS_print_line(stdout, '-', line_size);
    printf("Avg. execution time (in seconds, %d times)\n", iter);
    ATS_print_line(stdout, '-', line_size);
    printf("%-20s %11s %11s %11s\n", "operation", "avg", "min", "max");
    ATS_print_line(stdout, '-', line_size);
    for (i = 0; i < T_LAST; i++) {
        printf("%-19s  %11" PRIu64 " %11" PRIu64 " %11" PRIu64 "\n", t_names[i],
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
        free(g_threads[i]);
    }
    free(g_threads);
    free(t_times);

    return EXIT_SUCCESS;
}

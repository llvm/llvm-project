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
    T_CREATE_JOIN = 0,
    T_CREATE_UNNAMED,
    T_YIELD_OVERHEAD,
    T_YIELD_ALL,
    T_YIELD,
    T_YIELD_TO_OVERHEAD,
    T_YIELD_TO_ALL,
    T_YIELD_TO,
#ifdef TEST_MIGRATE_TO
    T_MIGRATE_TO_XSTREAM,
#endif
    T_LAST
};
static char *t_names[] = { "create/join",
                           "create (unnamed)",
                           "yield_overhead",
                           "yield_all",
                           "yield",
                           "yield_to_overhead",
                           "yield_to_all",
                           "yield_to",
#ifdef TEST_MIGRATE_TO
                           "migrate_to_xstream"
#endif
};

typedef struct {
    int eid;
    int tid;
} arg_t;

static int iter;
static int num_xstreams;
static int num_threads;

static ABT_xstream *g_xstreams;
static ABT_pool *g_pools;
static ABT_thread **g_threads;

#ifdef USE_TIME
static double t_times[T_LAST];
#else
static uint64_t t_times[T_LAST];
#endif

void thread_func(void *arg)
{
    ATS_UNUSED(arg);
}

void thread_func_yield_overhead(void *arg)
{
    ATS_UNUSED(arg);
    int i;
    for (i = 0; i < iter; i++) {
    }
}

void thread_func_yield(void *arg)
{
    ATS_UNUSED(arg);
    int i;
    for (i = 0; i < iter; i++) {
        ABT_thread_yield();
    }
}

void thread_func_yield_to_overhead(void *arg)
{
    arg_t *my_arg = (arg_t *)arg;
    int eid = my_arg->eid;
    int tid = my_arg->tid;
    int nid = (tid + 1) % num_threads;
    ABT_thread next = g_threads[eid][nid];
    ATS_UNUSED(next);
    int i;

    for (i = 0; i < iter; i++) {
    }
    ABT_thread_yield();
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

void test_create_join(void *arg)
{
    int eid = (int)(size_t)arg;
    ABT_pool my_pool = g_pools[eid];
    ABT_thread *my_threads = g_threads[eid];
    int i, t;

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
}

void test_create_unnamed(void *arg)
{
    int eid = (int)(size_t)arg;
    ABT_pool my_pool = g_pools[eid];
    int i, t;

    for (i = 0; i < iter; i++) {
        for (t = 0; t < num_threads; t++) {
            ABT_thread_create(my_pool, thread_func, NULL, ABT_THREAD_ATTR_NULL,
                              NULL);
        }
        ABT_thread_yield();
    }
}

void test_yield_overhead(void *arg)
{
    int eid = (int)(size_t)arg;
    ABT_pool my_pool = g_pools[eid];
    ABT_thread *my_threads = g_threads[eid];
    int i;

    for (i = 0; i < num_threads; i++) {
        ABT_thread_create(my_pool, thread_func_yield_overhead, NULL,
                          ABT_THREAD_ATTR_NULL, &my_threads[i]);
    }
    ABT_THREAD_JOIN_MANY(num_threads, my_threads);
    for (i = 0; i < num_threads; i++) {
        ABT_thread_free(&my_threads[i]);
    }
}

void test_yield(void *arg)
{
    int eid = (int)(size_t)arg;
    ABT_pool my_pool = g_pools[eid];
    ABT_thread *my_threads = g_threads[eid];
    int i;

    for (i = 0; i < num_threads; i++) {
        ABT_thread_create(my_pool, thread_func_yield, NULL,
                          ABT_THREAD_ATTR_NULL, &my_threads[i]);
    }
    ABT_THREAD_JOIN_MANY(num_threads, my_threads);
    for (i = 0; i < num_threads; i++) {
        ABT_thread_free(&my_threads[i]);
    }
}

void test_yield_to_overhead(void *arg)
{
    int eid = (int)(size_t)arg;
    ABT_pool my_pool = g_pools[eid];
    ABT_thread *my_threads = g_threads[eid];
    int i;

    arg_t *args = (arg_t *)malloc(num_threads * sizeof(arg_t));
    for (i = 0; i < num_threads; i++) {
        args[i].eid = eid;
        args[i].tid = i;
        ABT_thread_create(my_pool, thread_func_yield_to_overhead,
                          (void *)&args[i], ABT_THREAD_ATTR_NULL,
                          &my_threads[i]);
    }
    ABT_THREAD_JOIN_MANY(num_threads, my_threads);
    for (i = 0; i < num_threads; i++) {
        ABT_thread_free(&my_threads[i]);
    }
    free(args);
}

void test_yield_to(void *arg)
{
    int eid = (int)(size_t)arg;
    ABT_pool my_pool = g_pools[eid];
    ABT_thread *my_threads = g_threads[eid];
    int i;

    arg_t *args = (arg_t *)malloc(num_threads * sizeof(arg_t));
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
}

#ifdef TEST_MIGRATE_TO
void test_migrate_to_xstream(void *arg)
{
    int eid = (int)(size_t)arg;
    ABT_pool my_pool = g_pools[eid];
    ABT_thread *my_threads = g_threads[eid];
    int i;

    arg_t *args = (arg_t *)malloc(num_threads * sizeof(arg_t));
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
    free(args);
}
#endif

int main(int argc, char *argv[])
{
    ABT_pool(*all_pools)[2];
    ABT_sched *scheds;
    ABT_thread *top_threads;
    size_t i, t;
    uint64_t t_start;

    /* read command-line arguments */
    ATS_read_args(argc, argv);
    num_xstreams = ATS_get_arg_val(ATS_ARG_N_ES);
    num_threads = ATS_get_arg_val(ATS_ARG_N_ULT);
    iter = ATS_get_arg_val(ATS_ARG_N_ITER);

    /* initialize */
    ATS_init(argc, argv, num_xstreams);

    for (i = 0; i < T_LAST; i++) {
        t_times[i] = 0;
    }

    g_xstreams = (ABT_xstream *)malloc(num_xstreams * sizeof(ABT_xstream));
    g_pools = (ABT_pool *)malloc(num_xstreams * sizeof(ABT_pool));
    g_threads = (ABT_thread **)malloc(num_xstreams * sizeof(ABT_thread *));
    for (i = 0; i < num_xstreams; i++) {
        g_threads[i] = (ABT_thread *)malloc(num_threads * sizeof(ABT_thread));
    }
    all_pools = (ABT_pool(*)[2])malloc(num_xstreams * sizeof(ABT_pool) * 2);
    scheds = (ABT_sched *)malloc(num_xstreams * sizeof(ABT_sched));
    top_threads = (ABT_thread *)malloc(num_xstreams * sizeof(ABT_thread));

    /* create pools and schedulers */
    for (i = 0; i < num_xstreams; i++) {
        ABT_pool_create_basic(ABT_POOL_FIFO, ABT_POOL_ACCESS_MPSC, ABT_TRUE,
                              &all_pools[i][0]);
        ABT_pool_create_basic(ABT_POOL_FIFO, ABT_POOL_ACCESS_PRIV, ABT_TRUE,
                              &all_pools[i][1]);
        g_pools[i] = all_pools[i][1];

        ABT_sched_create_basic(ABT_SCHED_DEFAULT, 2, all_pools[i],
                               ABT_SCHED_CONFIG_NULL, &scheds[i]);
    }

    /* create ESs */
    ABT_xstream_self(&g_xstreams[0]);
    ABT_xstream_set_main_sched(g_xstreams[0], scheds[0]);
    for (i = 1; i < num_xstreams; i++) {
        ABT_xstream_create(scheds[i], &g_xstreams[i]);
    }

    /* benchmarking */
    for (t = 0; t < T_LAST; t++) {
        void (*test_fn)(void *);

        if (t == T_YIELD) {
            if (t_times[T_YIELD_ALL] > t_times[T_YIELD_OVERHEAD]) {
                t_times[t] = t_times[T_YIELD_ALL] - t_times[T_YIELD_OVERHEAD];
            } else {
                t_times[t] = 0;
            }
            continue;
        } else if (t == T_YIELD_TO) {
            if (t_times[T_YIELD_TO_ALL] > t_times[T_YIELD_TO_OVERHEAD]) {
                t_times[t] =
                    t_times[T_YIELD_TO_ALL] - t_times[T_YIELD_TO_OVERHEAD];
            } else {
                t_times[t] = 0;
            }
            continue;
        }

        switch (t) {
            case T_CREATE_JOIN:
                test_fn = test_create_join;
                break;
            case T_CREATE_UNNAMED:
                test_fn = test_create_unnamed;
                break;
            case T_YIELD_OVERHEAD:
                test_fn = test_yield_overhead;
                break;
            case T_YIELD_ALL:
                test_fn = test_yield;
                break;
            case T_YIELD_TO_OVERHEAD:
                test_fn = test_yield_to_overhead;
                break;
            case T_YIELD_TO_ALL:
                test_fn = test_yield_to;
                break;
#ifdef TEST_MIGRATE_TO
            case T_MIGRATE_TO_XSTREAM:
                test_fn = test_migrate_to_xstream;
                break;
#endif
            default:
                assert(0);
        }

        /* warm-up */
        for (i = 0; i < num_xstreams; i++) {
            ABT_thread_create(all_pools[i][0], test_fn, (void *)i,
                              ABT_THREAD_ATTR_NULL, &top_threads[i]);
        }
        for (i = 0; i < num_xstreams; i++) {
            ABT_thread_free(&top_threads[i]);
        }

        /* measurement */
#ifdef USE_TIME
        t_start = ABT_get_wtime();
#else
        t_start = ATS_get_cycles();
#endif
        for (i = 0; i < num_xstreams; i++) {
            ABT_thread_create(all_pools[i][0], test_fn, (void *)i,
                              ABT_THREAD_ATTR_NULL, &top_threads[i]);
        }
        for (i = 0; i < num_xstreams; i++) {
            ABT_thread_free(&top_threads[i]);
        }
#ifdef USE_TIME
        t_times[t] = ABT_get_wtime() - t_start;
#else
        t_times[t] = ATS_get_cycles() - t_start;
#endif
    }

    /* join and free */
    for (i = 1; i < num_xstreams; i++) {
        ABT_xstream_join(g_xstreams[i]);
        ABT_xstream_free(&g_xstreams[i]);
    }

    /* finalize */
    ATS_finalize(0);

    /* compute the execution time for one iteration */
    for (i = 0; i < T_LAST; i++) {
        t_times[i] = t_times[i] / iter / num_threads;
    }

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
    printf("%-20s %-s\n", "operation", "time");
    ATS_print_line(stdout, '-', line_size);
    for (i = 0; i < T_LAST; i++) {
#ifdef USE_TIME
        printf("%-19s  %.9lf\n", t_names[i], t_times[i]);
#else
        printf("%-19s  %11" PRIu64 "\n", t_names[i], t_times[i]);
#endif
    }
    ATS_print_line(stdout, '-', line_size);

    free(g_xstreams);
    free(g_pools);
    for (i = 0; i < num_xstreams; i++) {
        free(g_threads[i]);
    }
    free(g_threads);
    free(all_pools);
    free(scheds);
    free(top_threads);

    return EXIT_SUCCESS;
}

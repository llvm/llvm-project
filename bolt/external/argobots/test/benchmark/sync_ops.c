/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

#include <stdio.h>
#include <stdlib.h>
#include "abt.h"
#include "abttest.h"

enum {
    T_MUTEX_CREATE_COLD = 0,
    T_MUTEX_FREE_COLD,
    T_MUTEX_CREATE_FREE_COLD,
    T_MUTEX_CREATE,
    T_MUTEX_FREE,
    T_MUTEX_CREATE_FREE,
    T_MUTEX_LOCK_UNLOCK,
    T_MUTEX_LOCK_UNLOCK_ALL,
    T_LAST
};
static char *t_names[] = {
    "mutex: create (cold)",
    "mutex: free (cold)",
    "mutex: create/free (cold)",
    "mutex: create",
    "mutex: free",
    "mutex: create/free",
    "mutex: lock/unlock",
    "mutex: lock/unlock (all)",
};

typedef struct {
    int eid; /* ES id */
    int test_kind;
} launch_t;

typedef struct {
    int eid;
    int tid;
} arg_t;

static int iter;
static int num_xstreams;
static int num_threads;

static ABT_barrier g_barrier = ABT_BARRIER_NULL;
static ABT_mutex g_mutex = ABT_MUTEX_NULL;

static double t_overhead = 0.0;
static double t_timers[T_LAST];

void mutex_lock_unlock(void *arg)
{
    arg_t *my_arg = (arg_t *)arg;
    int eid = my_arg->eid;
    int tid = my_arg->tid;

    ABT_timer timer;
    double t_time;
    int i;

    if (eid == 0 && tid == 0) {
        ABT_timer_create(&timer);
    }

    /* barrier */
    ABT_barrier_wait(g_barrier);

    /* start timer */
    if (eid == 0 && tid == 0)
        ABT_timer_start(timer);

    /* measure mutex lock/unlock time */
    for (i = 0; i < iter; i++) {
        ABT_mutex_lock(g_mutex);
        ABT_mutex_unlock(g_mutex);
    }

    /* barrier */
    ABT_barrier_wait(g_barrier);

    /* stop timer */
    if (eid == 0 && tid == 0) {
        ABT_timer_stop_and_read(timer, &t_time);
        t_timers[T_MUTEX_LOCK_UNLOCK] = (t_time - t_overhead) / iter;
        ABT_timer_free(&timer);
    }
}

void launch_test(void *arg)
{
    launch_t *my_arg = (launch_t *)arg;
    int eid = my_arg->eid;
    int test_kind = my_arg->test_kind;

    ABT_xstream xstream;
    ABT_pool pool;
    ABT_thread *threads;
    void (*test_fn)(void *);
    int i;

    ATS_printf(1, "[E%d] main ULT: start\n", eid);

    switch (test_kind) {
        case T_MUTEX_LOCK_UNLOCK:
            test_fn = mutex_lock_unlock;
            break;
        default:
            fprintf(stderr, "Unknown test kind!\n");
            exit(EXIT_FAILURE);
    }

    threads = (ABT_thread *)malloc(num_threads * sizeof(ABT_thread));
    arg_t *args = (arg_t *)malloc(num_threads * sizeof(arg_t));

    ABT_xstream_self(&xstream);
    ABT_xstream_get_main_pools(xstream, 1, &pool);

    for (i = 0; i < num_threads; i++) {
        args[i].eid = eid;
        args[i].tid = i;
        ABT_thread_create(pool, test_fn, (void *)&args[i], ABT_THREAD_ATTR_NULL,
                          &threads[i]);
    }
    for (i = 0; i < num_threads; i++) {
        ABT_thread_join(threads[i]);
        ABT_thread_free(&threads[i]);
    }

    free(threads);
    free(args);
}

int main(int argc, char *argv[])
{
    ABT_xstream *xstreams;
    ABT_pool *pools;
    ABT_thread *threads;
    ABT_mutex *mutexes;
    ABT_timer timer;
    launch_t *largs;
    double t_time;
    int i;

    /* read command-line arguments */
    ATS_read_args(argc, argv);
    num_xstreams = ATS_get_arg_val(ATS_ARG_N_ES);
    num_threads = ATS_get_arg_val(ATS_ARG_N_ULT);
    iter = ATS_get_arg_val(ATS_ARG_N_ITER);

    /* initialize */
    ATS_init(argc, argv, num_xstreams);

    /* create a timer */
    ABT_timer_create(&timer);
    ABT_timer_start(timer);
    ABT_timer_stop(timer);
    ABT_timer_get_overhead(&t_overhead);
    for (i = 0; i < T_LAST; i++)
        t_timers[i] = 0.0;

    xstreams = (ABT_xstream *)malloc(num_xstreams * sizeof(ABT_xstream));
    pools = (ABT_pool *)malloc(num_xstreams * sizeof(ABT_pool));
    threads = (ABT_thread *)malloc(num_xstreams * sizeof(ABT_thread));
    mutexes = (ABT_mutex *)malloc(iter * sizeof(ABT_mutex));

    /* mutex create (cold) time */
    ABT_timer_start(timer);
    for (i = 0; i < iter; i++) {
        ABT_mutex_create(&mutexes[i]);
    }
    ABT_timer_stop_and_read(timer, &t_time);
    t_timers[T_MUTEX_CREATE_COLD] = (t_time - t_overhead) / iter;

    /* mutex free (cold) time */
    ABT_timer_start(timer);
    for (i = 0; i < iter; i++) {
        ABT_mutex_free(&mutexes[i]);
    }
    ABT_timer_stop_and_read(timer, &t_time);
    t_timers[T_MUTEX_FREE_COLD] = (t_time - t_overhead) / iter;

    /* mutex create/free (cold) time */
    t_timers[T_MUTEX_CREATE_FREE_COLD] =
        t_timers[T_MUTEX_CREATE_COLD] + t_timers[T_MUTEX_FREE_COLD];

    /* mutex create time */
    ABT_timer_start(timer);
    for (i = 0; i < iter; i++) {
        ABT_mutex_create(&mutexes[i]);
    }
    ABT_timer_stop_and_read(timer, &t_time);
    t_timers[T_MUTEX_CREATE] = (t_time - t_overhead) / iter;

    /* mutex free time */
    ABT_timer_start(timer);
    for (i = 0; i < iter; i++) {
        ABT_mutex_free(&mutexes[i]);
    }
    ABT_timer_stop_and_read(timer, &t_time);
    t_timers[T_MUTEX_FREE] = (t_time - t_overhead) / iter;

    /* mutex create/free time */
    t_timers[T_MUTEX_CREATE_FREE] =
        t_timers[T_MUTEX_CREATE] + t_timers[T_MUTEX_FREE];

    /* mutex lock/unlock time */
    ABT_timer_start(timer);

    largs = (launch_t *)malloc(num_xstreams * sizeof(launch_t));
    ABT_barrier_create(num_xstreams * num_threads, &g_barrier);
    ABT_mutex_create(&g_mutex);

    ABT_xstream_self(&xstreams[0]);
    for (i = 1; i < num_xstreams; i++) {
        ABT_xstream_create(ABT_SCHED_NULL, &xstreams[i]);
    }
    for (i = 1; i < num_xstreams; i++) {
        ABT_xstream_get_main_pools(xstreams[i], 1, &pools[i]);
        largs[i].eid = i;
        largs[i].test_kind = T_MUTEX_LOCK_UNLOCK;
        ABT_thread_create(pools[i], launch_test, (void *)&largs[i],
                          ABT_THREAD_ATTR_NULL, NULL);
    }

    largs[0].eid = 0;
    largs[0].test_kind = T_MUTEX_LOCK_UNLOCK;
    launch_test((void *)&largs[0]);

    for (i = 1; i < num_xstreams; i++) {
        ABT_xstream_join(xstreams[i]);
        ABT_xstream_free(&xstreams[i]);
    }
    ABT_barrier_free(&g_barrier);
    ABT_mutex_free(&g_mutex);
    free(largs);

    ABT_timer_stop_and_read(timer, &t_time);
    t_timers[T_MUTEX_LOCK_UNLOCK_ALL] = (t_time - t_overhead) / iter;

    /* finalize */
    ABT_timer_free(&timer);
    ATS_finalize(0);

    /* output */
    int line_size = 45;
    ATS_print_line(stdout, '-', line_size);
    printf("# of ESs        : %d\n", num_xstreams);
    printf("# of ULTs per ES: %d\n", num_threads);
    ATS_print_line(stdout, '-', line_size);
    printf("Avg. execution time (in seconds, %d times)\n", iter);
    ATS_print_line(stdout, '-', line_size);
    for (i = 0; i < T_LAST; i++) {
        printf("%-25s  %.9f\n", t_names[i], t_timers[i]);
    }
    ATS_print_line(stdout, '-', line_size);

    free(xstreams);
    free(pools);
    free(threads);
    free(mutexes);

    return EXIT_SUCCESS;
}

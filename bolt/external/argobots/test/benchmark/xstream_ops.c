/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

#include <stdio.h>
#include <stdlib.h>
#include "abt.h"
#include "abttest.h"

enum {
    T_CREATE_JOIN = 0,
    T_CREATE_JOIN_FREE,
    T_CREATE_JOIN_WITH_ULT,
    T_CREATE_JOIN_FREE_WITH_ULT,
    T_LAST
};

static char *t_names[] = { "ES: create/join without creating a ULT",
                           "ES: create/join/free without creating a ULT",
                           "ES: create/join with creating a ULT",
                           "ES: create/join/free with creating a ULT" };

static double t_times[T_LAST];

void thread_func(void *arg)
{
    ATS_UNUSED(arg);
}

int main(int argc, char *argv[])
{
    ABT_xstream *xstreams;
    ABT_pool *pools;
    ABT_timer timer;

    int i, t;
    int num_xstreams, iter;
    double t_overhead;

    /* read command-line arguments */
    ATS_read_args(argc, argv);
    num_xstreams = ATS_get_arg_val(ATS_ARG_N_ES);
    iter = ATS_get_arg_val(ATS_ARG_N_ITER);
    /* initialize */
    ATS_init(argc, argv, num_xstreams);

    /* create a timer */
    ABT_timer_create(&timer);
    ABT_timer_start(timer);
    ABT_timer_stop(timer);
    ABT_timer_get_overhead(&t_overhead);
    for (i = 0; i < T_LAST; i++)
        t_times[i] = 0.0;

    xstreams = (ABT_xstream *)malloc(num_xstreams * sizeof(ABT_xstream));
    pools = (ABT_pool *)malloc(num_xstreams * sizeof(ABT_pool));

    /* cache warm-up */
    for (t = 0; t < num_xstreams; t++) {
        ABT_xstream_create(ABT_SCHED_NULL, &xstreams[t]);
        ABT_xstream_get_main_pools(xstreams[t], 1, &pools[t]);
        ABT_thread_create(pools[t], thread_func, NULL, ABT_THREAD_ATTR_NULL,
                          NULL);
    }
    for (t = 0; t < num_xstreams; t++) {
        ABT_xstream_join(xstreams[t]);
        ABT_xstream_free(&xstreams[t]);
    }

    /* measure create/join time without creating a ULT */
    for (i = 0; i < iter; i++) {
        ABT_timer_start(timer);
        for (t = 0; t < num_xstreams; t++) {
            ABT_xstream_create(ABT_SCHED_NULL, &xstreams[t]);
        }
        for (t = 0; t < num_xstreams; t++) {
            ABT_xstream_join(xstreams[t]);
        }
        ABT_timer_stop_and_add(timer, &t_times[T_CREATE_JOIN]);

        for (t = 0; t < num_xstreams; t++) {
            ABT_xstream_free(&xstreams[t]);
        }
    }
    t_times[T_CREATE_JOIN] /= iter;
    t_times[T_CREATE_JOIN] -= t_overhead;

    /* measure create/join/free time without creating a ULT */
    for (i = 0; i < iter; i++) {
        ABT_timer_start(timer);
        for (t = 0; t < num_xstreams; t++) {
            ABT_xstream_create(ABT_SCHED_NULL, &xstreams[t]);
        }
        for (t = 0; t < num_xstreams; t++) {
            ABT_xstream_join(xstreams[t]);
            ABT_xstream_free(&xstreams[t]);
        }
        ABT_timer_stop_and_add(timer, &t_times[T_CREATE_JOIN_FREE]);
    }
    t_times[T_CREATE_JOIN_FREE] /= iter;
    t_times[T_CREATE_JOIN_FREE] -= t_overhead;

    /* measure create/join time with creating a ULT */
    for (i = 0; i < iter; i++) {
        ABT_timer_start(timer);
        for (t = 0; t < num_xstreams; t++) {
            ABT_xstream_create(ABT_SCHED_NULL, &xstreams[t]);
            ABT_xstream_get_main_pools(xstreams[t], 1, &pools[t]);
            ABT_thread_create(pools[t], thread_func, NULL, ABT_THREAD_ATTR_NULL,
                              NULL);
        }
        for (t = 0; t < num_xstreams; t++) {
            ABT_xstream_join(xstreams[t]);
        }
        ABT_timer_stop_and_add(timer, &t_times[T_CREATE_JOIN_WITH_ULT]);

        for (t = 0; t < num_xstreams; t++) {
            ABT_xstream_free(&xstreams[t]);
        }
    }
    t_times[T_CREATE_JOIN_WITH_ULT] /= iter;
    t_times[T_CREATE_JOIN_WITH_ULT] -= t_overhead;

    /* measure create/join/free time with creating a ULT */
    for (i = 0; i < iter; i++) {
        ABT_timer_start(timer);
        for (t = 0; t < num_xstreams; t++) {
            ABT_xstream_create(ABT_SCHED_NULL, &xstreams[t]);
            ABT_xstream_get_main_pools(xstreams[t], 1, &pools[t]);
            ABT_thread_create(pools[t], thread_func, NULL, ABT_THREAD_ATTR_NULL,
                              NULL);
        }
        for (t = 0; t < num_xstreams; t++) {
            ABT_xstream_join(xstreams[t]);
            ABT_xstream_free(&xstreams[t]);
        }
        ABT_timer_stop_and_add(timer, &t_times[T_CREATE_JOIN_FREE_WITH_ULT]);
    }
    t_times[T_CREATE_JOIN_FREE_WITH_ULT] /= iter;
    t_times[T_CREATE_JOIN_FREE_WITH_ULT] -= t_overhead;

    /* calculate the average for one ES */
    for (t = 0; t < T_LAST; t++) {
        t_times[t] = t_times[t] / num_xstreams;
    }

    /* finalize */
    ABT_timer_free(&timer);
    ATS_finalize(0);

    /* output */
    int line_size = 56;
    ATS_print_line(stdout, '-', line_size);
    printf("# of ESs: %d\n", num_xstreams);
    ATS_print_line(stdout, '-', line_size);
    printf("Avg. execution time (in seconds, %d times)\n", iter);
    ATS_print_line(stdout, '-', line_size);
    for (i = 0; i < T_LAST; i++) {
        printf("%-43s  %.9f\n", t_names[i], t_times[i]);
    }
    ATS_print_line(stdout, '-', line_size);

    free(xstreams);
    free(pools);

    return EXIT_SUCCESS;
}

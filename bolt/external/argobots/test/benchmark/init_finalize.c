/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

#include <stdio.h>
#include <stdlib.h>
#include "abt.h"
#include "abttest.h"

enum {
    T_INIT_FINALIZE_COLD = 0,
    T_INIT,
    T_FINALIZE,
    T_INIT_FINALIZE,
    T_INIT_FINALIZE_WITH_WORK,
    T_LAST
};

static char *t_names[] = { "init/finalize (cold)", "init", "finalize",
                           "init/finalize", "init/finalize with work" };

static double t_timers[T_LAST];

void thread_func(void *arg)
{
    ATS_UNUSED(arg);
}

int main(int argc, char *argv[])
{
    ABT_xstream xstream;
    ABT_pool pool;
    ABT_timer timer;
    int i, iter;
    double t_overhead;

    ATS_read_args(argc, argv);
    iter = ATS_get_arg_val(ATS_ARG_N_ITER);

    ABT_timer_create(&timer);
    ABT_timer_start(timer);
    ABT_timer_stop(timer);
    ABT_timer_get_overhead(&t_overhead);
    for (i = 0; i < T_LAST; i++)
        t_timers[i] = 0.0;

    /* measure init/finalize time (cold) */
    ABT_timer_start(timer);
    ATS_init(argc, argv, 2);
    ATS_finalize(0);
    ABT_timer_stop_and_read(timer, &t_timers[T_INIT_FINALIZE_COLD]);
    t_timers[T_INIT_FINALIZE_COLD] -= t_overhead;

    /* measure init/finalize time */
    for (i = 0; i < iter; i++) {
        ABT_timer_start(timer);
        ABT_init(argc, argv);
        ABT_timer_stop_and_add(timer, &t_timers[T_INIT]);

        ABT_timer_start(timer);
        ABT_finalize();
        ABT_timer_stop_and_add(timer, &t_timers[T_FINALIZE]);
    }
    t_timers[T_INIT] /= iter;
    t_timers[T_INIT] -= t_overhead;
    t_timers[T_FINALIZE] /= iter;
    t_timers[T_FINALIZE] -= t_overhead;
    t_timers[T_INIT_FINALIZE] = t_timers[T_INIT] + t_timers[T_FINALIZE];

    /* measure time of init/finalize with work */
    for (i = 0; i < iter; i++) {
        ABT_timer_start(timer);
        ABT_init(argc, argv);
        ABT_timer_stop_and_add(timer, &t_timers[T_INIT_FINALIZE_WITH_WORK]);

        ABT_xstream_create(ABT_SCHED_NULL, &xstream);
        ABT_xstream_get_main_pools(xstream, 1, &pool);
        ABT_thread_create(pool, thread_func, NULL, ABT_THREAD_ATTR_NULL, NULL);
        ABT_xstream_join(xstream);
        ABT_xstream_free(&xstream);

        ABT_timer_start(timer);
        ABT_finalize();
        ABT_timer_stop_and_add(timer, &t_timers[T_INIT_FINALIZE_WITH_WORK]);
    }
    t_timers[T_INIT_FINALIZE_WITH_WORK] /= iter;
    t_timers[T_INIT_FINALIZE_WITH_WORK] -= t_overhead;

    /* output */
    int line_size = 45;
    ATS_print_line(stdout, '-', line_size);
    printf("Avg. execution time (in seconds, %d times)\n", iter);
    ATS_print_line(stdout, '-', line_size);
    for (i = 0; i < T_LAST; i++) {
        printf("%-23s  %.9f\n", t_names[i], t_timers[i]);
    }
    ATS_print_line(stdout, '-', line_size);

    ABT_timer_free(&timer);

    return EXIT_SUCCESS;
}

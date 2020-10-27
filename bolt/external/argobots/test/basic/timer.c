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

ABT_timer timer = ABT_TIMER_NULL;
int num_threads = DEFAULT_NUM_THREADS;

void thread_func(void *arg)
{
    ATS_UNUSED(arg);
    /* Do nothing */
}

void thread_create(void *arg)
{
    int i, ret;
    size_t my_id = (size_t)arg;
    ABT_thread my_thread;
    ABT_pool my_pool;
    ABT_timer my_timer;
    double t_start = 0.0;
    double t_create = 0.0;

    ret = ABT_timer_dup(timer, &my_timer);
    ATS_ERROR(ret, "ABT_timer_dup");

    ret = ABT_thread_self(&my_thread);
    ATS_ERROR(ret, "ABT_thread_self");
    ret = ABT_thread_get_last_pool(my_thread, &my_pool);
    ATS_ERROR(ret, "ABT_thread_get_last_pool");

    ABT_timer_stop_and_read(my_timer, &t_start);

    /* Create ULTs */
    for (i = 0; i < num_threads; i++) {
        ABT_timer_start(my_timer);
        size_t tid = 100 * my_id + i;
        ret = ABT_thread_create(my_pool, thread_func, (void *)tid,
                                ABT_THREAD_ATTR_NULL, NULL);
        ATS_ERROR(ret, "ABT_thread_create");
        ABT_timer_stop_and_add(my_timer, &t_create);
    }

    ATS_printf(1, "[T%d] start: %.9f, %d ULTs creation time: %.9f\n",
               (int)my_id, t_start, num_threads, t_create);
    ret = ABT_timer_free(&my_timer);
    ATS_ERROR(ret, "ABT_timer_free");
}

int main(int argc, char *argv[])
{
    int i, ret;
    int num_xstreams = DEFAULT_NUM_XSTREAMS;
    if (argc > 1)
        num_xstreams = atoi(argv[1]);
    assert(num_xstreams >= 0);
    if (argc > 2)
        num_threads = atoi(argv[2]);
    assert(num_threads >= 0);

    double t_init = 0.0;
    double t_exec = 0.0;
    double t_fini = 0.0;
    double t_overhead = 0.0;

    ABT_xstream *xstreams;
    ABT_pool *pools;

    /* ABT_timer can be created regardless of Argobots initialization */
    ret = ABT_timer_create(&timer);
    ATS_ERROR(ret, "ABT_timer_create");

    xstreams = (ABT_xstream *)malloc(sizeof(ABT_xstream) * num_xstreams);
    pools = (ABT_pool *)malloc(sizeof(ABT_pool) * num_xstreams);

    /* Initialize */
    t_init = ABT_get_wtime();
    ATS_read_args(argc, argv);
    ATS_init(argc, argv, num_xstreams);
    t_init = (ABT_get_wtime() - t_init);

    ABT_timer_start(timer);

    /* Create Execution Streams */
    ret = ABT_xstream_self(&xstreams[0]);
    ATS_ERROR(ret, "ABT_xstream_self");
    for (i = 1; i < num_xstreams; i++) {
        ret = ABT_xstream_create(ABT_SCHED_NULL, &xstreams[i]);
        ATS_ERROR(ret, "ABT_xstream_create");
    }

    /* Get the first pool attached to each ES */
    for (i = 0; i < num_xstreams; i++) {
        ret = ABT_xstream_get_main_pools(xstreams[i], 1, pools + i);
        ATS_ERROR(ret, "ABT_xstream_get_main_pools");
    }

    /* Create one ULT for each ES */
    for (i = 0; i < num_xstreams; i++) {
        size_t tid = i + 1;
        ret = ABT_thread_create(pools[i], thread_create, (void *)tid,
                                ABT_THREAD_ATTR_NULL, NULL);
        ATS_ERROR(ret, "ABT_thread_create");
    }

    /* Join and free ESs */
    for (i = 1; i < num_xstreams; i++) {
        ret = ABT_xstream_join(xstreams[i]);
        ATS_ERROR(ret, "ABT_xstream_join");
        ret = ABT_xstream_free(&xstreams[i]);
        ATS_ERROR(ret, "ABT_xstream_free");
    }

    ABT_timer_stop_and_read(timer, &t_exec);

    /* Finalize */
    ABT_timer_start(timer);
    ret = ATS_finalize(0);
    ABT_timer_stop(timer);
    ABT_timer_read(timer, &t_fini);

    ABT_timer_free(&timer);
    free(xstreams);
    free(pools);

    /* Get the timer overhead */
    ABT_timer_get_overhead(&t_overhead);

    ATS_printf(1, "# of ESs      : %d\n", num_xstreams);
    ATS_printf(1, "# of ULTs/ES  : %d\n", num_threads);
    ATS_printf(1, "Timer overhead: %.9f sec\n", t_overhead);
    ATS_printf(1, "Init. time    : %.9f sec\n", t_init);
    ATS_printf(1, "Exec. time    : %.9f sec (w/o overhead: %.9f sec)\n", t_exec,
               t_exec - t_overhead);
    ATS_printf(1, "Fini. time    : %.9f sec (w/o overhead: %.9f sec)\n", t_fini,
               t_fini - t_overhead);

    return ret;
}

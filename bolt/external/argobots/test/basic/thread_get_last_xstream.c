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

int num_threads = DEFAULT_NUM_THREADS;

void thread_func(void *arg)
{
    int ret;
    ABT_thread self_thread;
    ABT_xstream self_xstream, last_xstream;

    ret = ABT_xstream_self(&self_xstream);
    ATS_ERROR(ret, "ABT_xstream_self");

    ret = ABT_thread_self(&self_thread);
    ATS_ERROR(ret, "ABT_thread_self");
    ret = ABT_thread_get_last_xstream(self_thread, &last_xstream);
    ATS_ERROR(ret, "ABT_thread_get_last_xstream");

    assert(self_xstream == last_xstream);
}

void task_func(void *arg)
{
    int ret;
    ABT_task self_task;
    ABT_xstream self_xstream, last_xstream;

    ret = ABT_xstream_self(&self_xstream);
    ATS_ERROR(ret, "ABT_xstream_self");

    ret = ABT_task_self(&self_task);
    ATS_ERROR(ret, "ABT_task_self");
    ret = ABT_task_get_xstream(self_task, &last_xstream);
    ATS_ERROR(ret, "ABT_task_get_xstream");

    assert(self_xstream == last_xstream);
}

int main(int argc, char *argv[])
{
    int i;
    int ret;
    int num_xstreams = DEFAULT_NUM_XSTREAMS;
    if (argc > 1)
        num_xstreams = atoi(argv[1]);
    assert(num_xstreams >= 0);
    if (argc > 2)
        num_threads = atoi(argv[2]);
    assert(num_threads >= 0);

    ABT_xstream *xstreams =
        (ABT_xstream *)malloc(sizeof(ABT_xstream) * num_xstreams);
    ABT_thread *threads =
        (ABT_thread *)malloc(sizeof(ABT_thread) * num_xstreams);
    ABT_thread *tasks = (ABT_task *)malloc(sizeof(ABT_task) * num_xstreams);

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

    /* Create one ULT for each ES */
    for (i = 0; i < num_xstreams; i++) {
        ret = ABT_thread_create_on_xstream(xstreams[i], thread_func, NULL,
                                           ABT_THREAD_ATTR_NULL, &threads[i]);
        ATS_ERROR(ret, "ABT_thread_create_on_xstream");
    }

    /* Create one tasklets for each ES */
    for (i = 0; i < num_xstreams; i++) {
        ret =
            ABT_task_create_on_xstream(xstreams[i], task_func, NULL, &tasks[i]);
        ATS_ERROR(ret, "ABT_task_create_on_xstream");
    }

    /* Join and free ULTs */
    for (i = 0; i < num_xstreams; i++) {
        ret = ABT_thread_join(threads[i]);
        ATS_ERROR(ret, "ABT_thread_join");

        ABT_xstream last_xstream;
        ret = ABT_thread_get_last_xstream(threads[i], &last_xstream);
        ATS_ERROR(ret, "ABT_thread_get_last_xstream");
        /* last_xstream must be the same as xstreams[i] */
        assert(last_xstream == xstreams[i]);

        ret = ABT_thread_free(&threads[i]);
        ATS_ERROR(ret, "ABT_thread_free");
    }

    /* Join and free tasks */
    for (i = 0; i < num_xstreams; i++) {
        ret = ABT_task_join(tasks[i]);
        ATS_ERROR(ret, "ABT_task_join");

        ABT_xstream last_xstream;
        ret = ABT_task_get_xstream(tasks[i], &last_xstream);
        ATS_ERROR(ret, "ABT_task_get_xstream");
        /* last_xstream must be the same as xstreams[i] */
        assert(last_xstream == xstreams[i]);

        ret = ABT_task_free(&tasks[i]);
        ATS_ERROR(ret, "ABT_task_free");
    }

    /* Join and free Execution Streams */
    for (i = 1; i < num_xstreams; i++) {
        ret = ABT_xstream_join(xstreams[i]);
        ATS_ERROR(ret, "ABT_xstream_join");
        ret = ABT_xstream_free(&xstreams[i]);
        ATS_ERROR(ret, "ABT_xstream_free");
    }

    free(xstreams);
    free(threads);
    free(tasks);

    /* Finalize */
    ret = ATS_finalize(0);

    return ret;
}

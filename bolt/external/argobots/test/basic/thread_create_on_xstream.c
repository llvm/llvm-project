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
    size_t my_id = (size_t)arg;
    ATS_printf(1, "[TH%lu]: Hello, world!\n", my_id);
}

void thread_create(void *arg)
{
    int i, ret;
    size_t my_id = (size_t)arg;
    ABT_xstream my_xstream;

    ret = ABT_xstream_self(&my_xstream);
    ATS_ERROR(ret, "ABT_xstream_self");

    /* Create ULTs */
    for (i = 0; i < num_threads; i++) {
        size_t tid = 100 * my_id + i;
        ret = ABT_thread_create_on_xstream(my_xstream, thread_func, (void *)tid,
                                           ABT_THREAD_ATTR_NULL, NULL);
        ATS_ERROR(ret, "ABT_thread_create_on_xstream");
    }

    ATS_printf(1, "[TH%lu]: created %d ULTs\n", my_id, num_threads);
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

    ABT_xstream *xstreams;
    ABT_thread *threads;

    xstreams = (ABT_xstream *)malloc(sizeof(ABT_xstream) * num_xstreams);
    threads = (ABT_thread *)malloc(sizeof(ABT_thread) * num_xstreams);

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
        size_t tid = i + 1;
        ret = ABT_thread_create_on_xstream(xstreams[i], thread_create,
                                           (void *)tid, ABT_THREAD_ATTR_NULL,
                                           &threads[i]);
        ATS_ERROR(ret, "ABT_thread_create_on_xstream");
    }

    /* Join and free ULTs */
    for (i = 0; i < num_xstreams; i++) {
        ret = ABT_thread_join(threads[i]);
        ATS_ERROR(ret, "ABT_thread_join");
        ret = ABT_thread_free(&threads[i]);
        ATS_ERROR(ret, "ABT_thread_free");
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

    /* Finalize */
    ret = ATS_finalize(0);

    return ret;
}

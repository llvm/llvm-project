/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include "abt.h"
#include "abttest.h"

/* This code tests that a pthread can free threads created locally,
 * and vice versa. */

ABT_xstream xstreams[2];
ABT_pool pools[2];
ABT_thread threads[2];

void thread_func(void *arg)
{
    /* Do nothing. */
}

void join_threads()
{
    size_t i;
    int ret;
    /* Join */
    for (i = 0; i < 2; i++) {
        ret = ABT_thread_join(threads[i]);
        ATS_ERROR(ret, "ABT_thread_join");
    }
    /* Free */
    for (i = 0; i < 2; i++) {
        ret = ABT_thread_free(&threads[i]);
        ATS_ERROR(ret, "ABT_thread_free");
    }
}

void create_threads()
{
    size_t i;
    int ret;
    /* Create */
    for (i = 0; i < 2; i++) {
        ret = ABT_thread_create(pools[i], thread_func, (void *)i,
                                ABT_THREAD_ATTR_NULL, &threads[i]);
        ATS_ERROR(ret, "ABT_thread_create");
    }
}

void *pthread_join_threads(void *arg)
{
    join_threads();
    return NULL;
}

void *pthread_create_threads(void *arg)
{
    create_threads();
    return NULL;
}

int main(int argc, char *argv[])
{
    pthread_t pthread;
    int ret;
    size_t i;

    /* Initialize */
    ATS_init(argc, argv, 2);

    /* Set up execution streams and pools. */
    for (i = 0; i < 2; i++) {
        ret = ABT_xstream_create(ABT_SCHED_NULL, &xstreams[i]);
        ATS_ERROR(ret, "ABT_xstream_create");
        ret = ABT_xstream_get_main_pools(xstreams[i], 1, &pools[i]);
        ATS_ERROR(ret, "ABT_xstream_get_main_pools");
    }

    /* Create threads locally, and join threads by pthread */
    create_threads();
    ret = pthread_create(&pthread, NULL, pthread_join_threads, NULL);
    assert(ret == 0);
    ret = pthread_join(pthread, NULL);
    assert(ret == 0);

    /* Create threads by pthread, and join threads locally */
    ret = pthread_create(&pthread, NULL, pthread_create_threads, NULL);
    assert(ret == 0);
    ret = pthread_join(pthread, NULL);
    assert(ret == 0);
    join_threads();

    /* Join and free Execution Streams */
    for (i = 0; i < 2; i++) {
        ret = ABT_xstream_join(xstreams[i]);
        ATS_ERROR(ret, "ABT_xstream_join");
        ret = ABT_xstream_free(&xstreams[i]);
        ATS_ERROR(ret, "ABT_xstream_free");
    }

    /* Finalize */
    return ATS_finalize(0);
}

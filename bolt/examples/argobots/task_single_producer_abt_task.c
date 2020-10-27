/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */

/*
 * See LICENSE.txt in top-level directory.
 */

/*  This code creates all tasks from the main ES but using as many pools as
 *  xstreams and they are executed by all the xstreams. It mimics one producer
 *  all consumers system
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <abt.h>
#include <sys/time.h>
#include <unistd.h>

#define NUM_TASKS       5000000
#define NUM_XSTREAMS    4
#define NUM_REPS        1

void vector_scal(void *arguments)
{
    float *a;
    a = (float *)arguments;
    *a = *a * 0.9f;
}

int main(int argc, char *argv[])
{
    int i, j;
    int ntasks;
    int num_xstreams;
    int num_pools;
    ABT_xstream *xstreams;
    ABT_task *tasks;
    ABT_pool *pools;
    struct timeval start, end;
    char *str, *endptr;
    float *a;

    num_xstreams = argc > 1 ? atoi(argv[1]) : NUM_XSTREAMS;
    if (argc > 2) {
        str = argv[2];
    }
    ntasks = argc > 2 ? strtoll(str, &endptr, 10) : NUM_TASKS;
    if (ntasks < num_xstreams) {
        ntasks = num_xstreams;
    }
    num_pools = argc > 5 ? atoi(argv[5]) : num_xstreams;

    a = malloc(sizeof(float) * ntasks);

    for (i = 0; i < ntasks; i++) {
        a[i] = i + 100.0f;
    }

    xstreams = (ABT_xstream *)malloc(sizeof(ABT_xstream) * num_xstreams);
    tasks = (ABT_task *)malloc(sizeof(ABT_task) * num_xstreams);
    pools = (ABT_pool *)malloc(sizeof(ABT_pool) * num_pools);

    /* initialization */
    ABT_init(argc, argv);

    /* shared pool creation */
    for (i = 0; i < num_pools; i++) {
        ABT_pool_create_basic(ABT_POOL_FIFO, ABT_POOL_ACCESS_MPMC, ABT_TRUE,
                              &pools[i]);
    }
    /* ES creation */
    ABT_xstream_self(&xstreams[0]);
    ABT_xstream_set_main_sched_basic(xstreams[0], ABT_SCHED_DEFAULT,
                                     1, &pools[0]);
    for (i = 1; i < num_xstreams; i++) {
        ABT_xstream_create_basic(ABT_SCHED_DEFAULT, 1, &pools[i % num_pools],
                                 ABT_SCHED_CONFIG_NULL, &xstreams[i]);
        ABT_xstream_start(xstreams[i]);
    }
    /* Work here */
    gettimeofday(&start, NULL);
    for (j = 0; j < ntasks; j++) {
        ABT_task_create(pools[j % num_pools], vector_scal, (void *)&a[j], NULL);
    }

    gettimeofday(&end, NULL);
    double time2 = (end.tv_sec * 1000000 + end.tv_usec)
        - (start.tv_sec * 1000000 + start.tv_usec);
    ABT_thread_yield();
    for (i = 0; i < num_pools; i++) {
        size_t size;
        do {
            ABT_pool_get_size(pools[i], &size);
        } while (size != 0);
    }

    gettimeofday(&end, NULL);
    double time = (end.tv_sec * 1000000 + end.tv_usec)
                - (start.tv_sec * 1000000 + start.tv_usec);
    printf("nxstreams: %d\nntasks %d\nTime(s): %f Creation Time(s): %f\n",
           num_xstreams, ntasks, time / 1000000.0, time2 / 1000000.0);

    for (i = 0; i < ntasks; i++) {
        if (a[i] != (i + 100.0f) * 0.9f) {
            printf("error: a[%d]\n", i);
        }
    }

    /* join ESs */
    for (i = 1; i < num_xstreams; i++) {
        ABT_xstream_join(xstreams[i]);
        ABT_xstream_free(&xstreams[i]);
    }

    free(tasks);
    free(xstreams);

    return EXIT_SUCCESS;
}

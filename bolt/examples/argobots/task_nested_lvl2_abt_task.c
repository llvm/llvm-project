/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */

/*
 * See LICENSE.txt in top-level directory.
 */

/*  This code creates all tasks from the main ES but using as many pools as
 *  xstreams and they are executed by all the xstreams. This code mimics the 1
 *  producers all consumers system.
*/


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <abt.h>
#include <sys/time.h>
#include <unistd.h>

#define NUM_TASKS       50000
#define NUM_XSTREAMS    4
#define NUM_REPS        1
#define LEVELS          2

int num_pools;
int num_xstreams;
int o = 0;
ABT_pool *g_pools;
int lvl;

void vector_scal(void *arguments)
{
    float *a;
    a = (float *)arguments;
    *a = *a * 0.9f;
}

void na(void *arguments)
{
    o++;
}

void prevector_scal2(void *arguments)
{
    int rank;
    ABT_xstream_self_rank(&rank);
    ABT_task_create(g_pools[rank], vector_scal, arguments, NULL);
    ABT_task_create(g_pools[rank], na, arguments, NULL);
}

void prevector_scal(void *arguments)
{
    int rank;
    ABT_xstream_self_rank(&rank);
    ABT_task_create(g_pools[rank], prevector_scal2, arguments, NULL);
    ABT_task_create(g_pools[rank], prevector_scal2, arguments, NULL);
}


int main(int argc, char *argv[])
{
    int i, j;
    int ntasks;
    ABT_xstream *xstreams;
    ABT_task *tasks;
    struct timeval start, end, end2;
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
    num_pools = argc > 3 ? atoi(argv[3]) : num_xstreams;
    lvl = (argc > 4) ? atoi(argv[4]) : LEVELS;

    a = malloc(sizeof(float) * ntasks);

    for (i = 0; i < ntasks; i++) {
        a[i] = i + 100.0f;
    }

    xstreams = (ABT_xstream *)malloc(sizeof(ABT_xstream) * num_xstreams);
    tasks = (ABT_task *)malloc(sizeof(ABT_task) * num_xstreams);
    g_pools = (ABT_pool *)malloc(sizeof(ABT_pool) * num_pools);

    /* initialization */
    ABT_init(argc, argv);

    /* shared pool creation */
    for (i = 0; i < num_pools; i++) {
        ABT_pool_create_basic(ABT_POOL_FIFO, ABT_POOL_ACCESS_MPMC, ABT_TRUE,
                              &g_pools[i]);
    }
    /* ES creation */
    ABT_xstream_self(&xstreams[0]);
    ABT_xstream_set_main_sched_basic(xstreams[0], ABT_SCHED_DEFAULT,
                                     1, &g_pools[0]);

    for (i = 1; i < num_xstreams; i++) {
        ABT_xstream_create_basic(ABT_SCHED_DEFAULT, 1, &g_pools[i % num_pools],
                                 ABT_SCHED_CONFIG_NULL, &xstreams[i]);
        ABT_xstream_start(xstreams[i]);
    }
    /* Work here */
    gettimeofday(&start, NULL);
    for (j = 0; j < ntasks; j++) {
        ABT_task_create_on_xstream(xstreams[j % num_xstreams], prevector_scal,
                                   (void *)&a[j], NULL);
    }

    gettimeofday(&end2, NULL);
    ABT_thread_yield();
    for (i = 1; i < num_xstreams; i++) {
        size_t size;
        do {
            ABT_pool_get_size(g_pools[i], &size);
        } while (size != 0);

    }

    gettimeofday(&end, NULL);
    double time = (end.tv_sec * 1000000 + end.tv_usec)
                - (start.tv_sec * 1000000 + start.tv_usec);
    double time2 = (end2.tv_sec * 1000000 + end2.tv_usec)
                 - (start.tv_sec * 1000000 + start.tv_usec);

    printf("nxstreams: %d\nntasks %d\nTime(s): %f\n",
           num_xstreams, ntasks, time / 1000000.0);
    printf("o=%d ans it should be %d\n", o, ntasks);
    printf("Creation time= %f\n", time2 / 1000000.0);

    /* join ESs */
    for (i = 1; i < num_xstreams; i++) {
        ABT_xstream_join(xstreams[i]);
        ABT_xstream_free(&xstreams[i]);
    }
    //TODO: it only works for 1 rep
    for (i = 0; i < ntasks; i++) {
        if (a[i] != (i + 100.0f) * 0.9f) {
            printf("error: a[%d]\n", i);
        }
    }
    ABT_finalize();

    free(tasks);
    free(xstreams);

    return EXIT_SUCCESS;
}

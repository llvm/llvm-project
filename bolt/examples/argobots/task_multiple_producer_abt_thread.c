/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */

/*
 * See LICENSE.txt in top-level directory.
 */

/*  This code creates one task for each argobots xstream and each task creates
 *  a number of tasks. This version uses as many pools as execution streams are
 *  created. This number of tasks is the division between number of tasks
 *  required and number of streams. This code mimics the all producer all
 *  consumers system.
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

ABT_pool *g_pools;

typedef struct {
    float *ptr;
    float value;
    int start;
    int end;
    int id;
} vector_scal_task_args_t;

void task_function(void *args)
{
    float *a;
    a = (float *)args;
    *a = *a * 0.9f;
}

void task_creator(void *args)
{
    int i;
    vector_scal_task_args_t *arg;
    arg = (vector_scal_task_args_t *)args;
    for (i = arg->start; i < arg->end; i++) {
        ABT_thread_create(g_pools[arg->id], task_function, (void *)&arg->ptr[i],
                          ABT_THREAD_ATTR_NULL, NULL);
    }
}

int main(int argc, char *argv[])
{
    int i, j;
    int ntasks;
    int start, end;
    int num_xstreams;
    ABT_xstream *xstreams;
    vector_scal_task_args_t *args;
    struct timeval t_start, t_end, t_end2;
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

    printf("# of ESs: %d\n", num_xstreams);

    xstreams = (ABT_xstream *)malloc(sizeof(ABT_xstream) * num_xstreams);
    args = (vector_scal_task_args_t *)malloc(sizeof(vector_scal_task_args_t)
                                             * num_xstreams);
    g_pools = (ABT_pool *)malloc(sizeof(ABT_pool) * num_xstreams);

    a = malloc(sizeof(float) * ntasks);

    for (i = 0; i < ntasks; i++) {
        a[i] = i + 100.0f;
    }

    /* initialization */
    ABT_init(argc, argv);

    for (i = 0; i < num_xstreams; i++) {
        ABT_pool_create_basic(ABT_POOL_FIFO, ABT_POOL_ACCESS_MPMC, ABT_TRUE,
                              &g_pools[i]);
    }

    /* ES creation */
    ABT_xstream_self(&xstreams[0]);
    ABT_xstream_set_main_sched_basic(xstreams[0], ABT_SCHED_DEFAULT,
                                     1, &g_pools[0]);

    for (i = 1; i < num_xstreams; i++) {
        ABT_xstream_create_basic(ABT_SCHED_DEFAULT, 1, &g_pools[i],
                                 ABT_SCHED_CONFIG_NULL, &xstreams[i]);
        ABT_xstream_start(xstreams[i]);
    }

    /* Work here */
    start = end = 0;
    int bloc = ntasks / num_xstreams;
    int rest = ntasks % num_xstreams;
    gettimeofday(&t_start, NULL);
    for (j = 0; j < num_xstreams; j++) {
        start = end;
        end = start + bloc;
        if (j < rest) {
            end++;
        }
        args[j].ptr = a;
        args[j].value = 0.9f;
        args[j].start = start;
        args[j].end = end;
        args[j].id = j;
        ABT_thread_create_on_xstream(xstreams[j], task_creator,
                                     (void *)&args[j], ABT_THREAD_ATTR_NULL,
                                     NULL);
    }
    gettimeofday(&t_end2, NULL);

    for (i = 0; i < num_xstreams; i++) {
        size_t size;
        do {
            ABT_thread_yield();
            ABT_pool_get_size(g_pools[i], &size);
        } while (size != 0);
    }

    gettimeofday(&t_end, NULL);

    for (i = 0; i < ntasks; i++) {
        if (a[i] != (i + 100.0f) * 0.9f) {
            printf("error: a[%d]\n", i);
        }
    }

    double time = (t_end.tv_sec * 1000000 + t_end.tv_usec)
                - (t_start.tv_sec * 1000000 + t_start.tv_usec);
    double time2 = (t_end2.tv_sec * 1000000 + t_end2.tv_usec)
                 - (t_start.tv_sec * 1000000 + t_start.tv_usec);

    printf("nxstreams: %d\nntasks %d\nTime(s): %f\n",
           num_xstreams, ntasks, time / 1000000.0);
    /* join ESs */
    for (i = 1; i < num_xstreams; i++) {
        ABT_xstream_join(xstreams[i]);
        ABT_xstream_free(&xstreams[i]);
    }
    printf("Creation time=%f\n", time2 / 1000000.0);
    ABT_finalize();

    free(xstreams);

    return EXIT_SUCCESS;
}

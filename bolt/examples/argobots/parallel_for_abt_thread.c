/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */

/*
 * See LICENSE.txt in top-level directory.
 */

/*  parallel_for_abt_thread.c code mimics the parallel for OpenMP directive.
 *  It creates as many ESs as user requires, and tasks are created and assigned
 *  by static blocks to each ES.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <abt.h>
#include <math.h>
#include <sys/time.h>

#define NUM_ELEMS       5017600     /* 2GB */
#define NUM_XSTREAMS    4
#define NUM_REPS        1

ABT_pool *g_pools;

typedef struct {
    float *ptr;
    float value;
    int start;
    int end;
} vector_scal_args_t;


void vector_scal(void *arguments)
{
    int i;
    vector_scal_args_t *arg;
    arg = (vector_scal_args_t *)arguments;
    int mystart = arg->start;
    int myend = arg->end;
    float value = arg->value;
    float *ptr = arg->ptr;
    for (i = mystart; i < myend; i++) {
        ptr[i] *= value;
    }
}


int main(int argc, char *argv[])
{
    int i, j;
    int ntasks;
    int num_xstreams;
    char *str, *endptr;
    ABT_xstream *xstreams;
    vector_scal_args_t *args;
    struct timeval t_start, t_end;
    float *a;
    ABT_thread *threads;
    num_xstreams = argc > 1 ? atoi(argv[1]) : NUM_XSTREAMS;
    if (argc > 2) {
        str = argv[2];
    }
    ntasks = argc > 2 ? strtoll(str, &endptr, 10) : NUM_ELEMS;
    g_pools = (ABT_pool *)malloc(sizeof(ABT_pool) * num_xstreams);
    threads = (ABT_thread *)malloc(sizeof(ABT_thread) * num_xstreams);

    a = malloc(sizeof(float) * ntasks);
    for (i = 0; i < ntasks; i++) {
        a[i] = i * 1.0f;
    }

    xstreams = (ABT_xstream *)malloc(sizeof(ABT_xstream) * num_xstreams);
    args = (vector_scal_args_t *)malloc(sizeof(vector_scal_args_t)
                                        * num_xstreams);

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

    gettimeofday(&t_start, NULL);

    /* Each task is created on the xstream which is going to execute it */

    int bloc = ntasks / (num_xstreams);
    int rest = ntasks % (num_xstreams);
    int start = 0;
    int end = 0;
    for (j = 0; j < num_xstreams; j++) {
        start = end;
        int inc = (j < rest) ? 1 : 0;
        end += bloc + inc;
        args[j].start = start;
        args[j].end = end;
        args[j].value = 0.9f;
        args[j].ptr = a;
        ABT_thread_create_on_xstream(xstreams[j], vector_scal,
                                     (void *)&args[j], ABT_THREAD_ATTR_NULL,
                                     &threads[j]);
    }

    ABT_thread_yield();

    for (i = 0; i < num_xstreams; i++) {
        ABT_thread_free(&threads[i]);
    }

    gettimeofday(&t_end, NULL);
    double time = (t_end.tv_sec * 1000000 + t_end.tv_usec) -
        (t_start.tv_sec * 1000000 + t_start.tv_usec);

    for (i = 1; i < num_xstreams; i++) {
        ABT_xstream_join(xstreams[i]);
    }

    for (i = 1; i < num_xstreams; i++) {
        ABT_xstream_free(&xstreams[i]);
    }
    printf("%d %d %f\n", num_xstreams, ntasks, time / 1000000.0);

    ABT_finalize();
    free(xstreams);
    for (i = 0; i < ntasks; i++) {
        if (a[i] != i * 0.9f) {
            printf("%f\n", a[i]);
            return EXIT_FAILURE;
        }
    }

    return EXIT_SUCCESS;
}

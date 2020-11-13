/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */

/*
 * See LICENSE.txt in top-level directory.
 */

/*  This code mimics the parallel for OpenMP directive in nested loops.
 *  It creates as many streams as user requires and threads  are created and
 *  assigned by static blocs to each stream for the outer loop.
 *  For the inner loop, as many task as the user requires are created.
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
    int rank;
    int it;
    int start;
    int end;
} vector_scal_args_t;

typedef struct {
    float *ptr;
    float value;
    int nxstreams;
    int it;
    int start;
    int end;
} vector_scal_task_args_t;

void vector_scal(void *arguments)
{
    int i, rank;
    vector_scal_args_t *arg;
    arg = (vector_scal_args_t *)arguments;
    ABT_xstream_self_rank(&rank);
    rank = arg->rank;
    int mystart = arg->start;
    int myend = arg->end;
    int it = arg->it;
    int base = rank * it;
    float value = arg->value;
    float *ptr = arg->ptr;
    for (i = mystart; i < myend; i++) {
        ptr[base + i] *= value;
    }
}

void vector_scal_launch(void *arguments)
{
    int i, it, j, num_ults, rank, mystart, myend, p;
    ABT_task *tasks;
    ABT_xstream xstream;
    ABT_xstream_self(&xstream);
    vector_scal_task_args_t *arg;
    arg = (vector_scal_task_args_t *) arguments;
    vector_scal_args_t *args;
    it = arg->it;
    num_ults = arg->nxstreams;
    mystart = arg->start;
    myend = arg->end;
    int current = 0;
    args = (vector_scal_args_t *)malloc(sizeof(vector_scal_args_t)
                                        * num_ults);
    tasks = (ABT_task *)malloc(sizeof(ABT_task) * num_ults);
    /* ES creation */
    int bloc = it / (num_ults);
    int rest = it % (num_ults);
    int start = 0;
    int end = 0;
    ABT_xstream_self_rank(&rank);
    for (i = mystart; i < myend; i++) {
        for (j = 0; j < num_ults; j++) {
            start = end;
            int inc = (j < rest) ? 1 : 0;
            end += bloc + inc;
            args[j].start = start;
            args[j].end = end;
            args[j].value = arg->value;
            args[j].ptr = arg->ptr;
            args[j].it = it;
            args[j].rank = rank;

            ABT_task_create(g_pools[rank], vector_scal,
                            (void *)&args[j], &tasks[j]);
        }
        current++;
        for (p = 0; p < num_ults; p++) {
            ABT_task_free(&tasks[p]);
        }
    }
    ABT_thread_yield();
}


int main(int argc, char *argv[])
{
    int i, j;
    int ntasks;
    int num_xstreams;
    char *str, *endptr;
    ABT_xstream *xstreams;
    vector_scal_task_args_t *args;
    struct timeval t_start, t_end;
    struct timeval t_start2;
    double time, time_join;
    float *a;
    int it;
    int inner_xstreams;

    num_xstreams = argc > 1 ? atoi(argv[1]) : NUM_XSTREAMS;
    if (argc > 2) {
        str = argv[2];
    }

    ntasks = argc > 2 ? strtoll(str, &endptr, 10) : NUM_ELEMS;
    it = ceil(sqrt(ntasks));
    ntasks = it * it;
    inner_xstreams = argc > 3 ? atoi(argv[3]) : NUM_XSTREAMS;

    g_pools = (ABT_pool *)malloc(sizeof(ABT_pool) * num_xstreams);

    a = malloc(sizeof(float) * ntasks);
    for (i = 0; i < ntasks; i++) {
        a[i] = i * 1.0f;
    }

    xstreams = (ABT_xstream *) malloc(sizeof(ABT_xstream) * num_xstreams);
    args = (vector_scal_task_args_t *) malloc(sizeof(vector_scal_task_args_t)
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
    }

    gettimeofday(&t_start, NULL);

    /* Each task is created on the xstream which is going to execute it */

    int bloc = it / (num_xstreams);
    int rest = it % (num_xstreams);
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
        args[j].it = it;
        args[j].nxstreams = inner_xstreams;

        ABT_thread_create_on_xstream(xstreams[j], vector_scal_launch,
                                     (void *)&args[j], ABT_THREAD_ATTR_NULL,
                                     NULL);
    }

    ABT_thread_yield();

    gettimeofday(&t_start2, NULL);
    for (i = 1; i < num_xstreams; i++) {
        size_t size;
        do {
            ABT_pool_get_size(g_pools[i], &size);
        } while (size != 0);
    }

    gettimeofday(&t_end, NULL);
    time = (t_end.tv_sec * 1000000 + t_end.tv_usec) -
           (t_start.tv_sec * 1000000 + t_start.tv_usec);
    time_join = (t_end.tv_sec * 1000000 + t_end.tv_usec) -
                (t_start2.tv_sec * 1000000 + t_start2.tv_usec);

    for (i = 1; i < num_xstreams; i++) {
        ABT_xstream_join(xstreams[i]);
        ABT_xstream_free(&xstreams[i]);
    }

    printf("%d %d %d %f %f\n",
           num_xstreams, inner_xstreams, ntasks, time / 1000000.0,
           time_join / 1000000.0);

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

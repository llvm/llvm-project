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
#include <assert.h>

#define NUM_ELEMS       5017600     /* 2GB */
#define NUM_XSTREAMS    4
#define NUM_REPS        1

ABT_pool *g_pools;

typedef struct {
    int start;
    int end;
} vector_scal_args_t;

typedef struct {
    int nxstreams;
    int it;
    int start;
    int end;
} vector_scal_task_args_t;

void exe_random(void *arguments)
{
    int i, k;
    vector_scal_args_t *arg;
    arg = (vector_scal_args_t *) arguments;
    int mystart = arg->start;
    int myend = arg->end;
    for (i = mystart; i < myend; i++) {
        int random = rand() % 10000;
        int kk = 0;
        for (k = 0; k < random; k++)
            kk++;
        assert(kk == random);
    }
}

void random_launch(void *arguments)
{
    int i, it, j, num_ults, rank, mystart, myend, p;
#ifdef PROFTIME
    struct timeval t_start, t_end;
    struct timeval t_start2, t_end2;
    double time, time2;
#endif
    ABT_task *tasks;
    vector_scal_task_args_t *arg;
    arg = (vector_scal_task_args_t *) arguments;
    vector_scal_args_t *args;
    it = arg->it;
    num_ults = arg->nxstreams;
    mystart = arg->start;
    myend = arg->end;
    int current = 0;
    
    args = (vector_scal_args_t *) malloc(sizeof(vector_scal_args_t)
                                         * num_ults);
    tasks = (ABT_task *)malloc(sizeof(ABT_task) * num_ults);
    
    int bloc = it / (num_ults);
    int rest = it % (num_ults);
    int start = 0;
    int end = 0;
    ABT_xstream_self_rank(&rank);
#ifdef PROFTIME
    gettimeofday(&t_start, NULL);
#endif
    for (i = mystart; i < myend; i++) {
        for (j = 0; j < num_ults; j++) {
            start = end;
            int inc = (j < rest) ? 1 : 0;
            end += bloc + inc;
            args[j].start = start;
            args[j].end = end;

#ifdef PROFTIME
            gettimeofday(&t_start2, NULL);
#endif
            ABT_task_create(g_pools[rank], exe_random,
                            (void *)&args[j], &tasks[j]);
#ifdef PROFTIME
            gettimeofday(&t_end2, NULL);
            time2 = (t_end2.tv_sec * 1000000 + t_end2.tv_usec) -
                (t_start2.tv_sec * 1000000 + t_start2.tv_usec);
            printf("Inner_task_creation_time %f\n", (time2 / 1000000.0));
#endif
        }
        current++;
#ifdef PROFTIME
        gettimeofday(&t_start2, NULL);
#endif
        for (p = 0; p < num_ults; p++) {
            ABT_task_free(&tasks[p]);
        }
#ifdef PROFTIME
        gettimeofday(&t_end2, NULL);
        time2 = (t_end2.tv_sec * 1000000 + t_end2.tv_usec) -
            (t_start2.tv_sec * 1000000 + t_start2.tv_usec);
        printf("Inner_join_time %f\n", (time2 / 1000000.0));
#endif
    }
#ifdef PROFTIME
    gettimeofday(&t_end, NULL);
    time = (t_end.tv_sec * 1000000 + t_end.tv_usec) -
        (t_start.tv_sec * 1000000 + t_start.tv_usec);
    printf("ult_time %f\n", (time2 / 1000000.0));
#endif
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
#ifdef PROFTIME
    struct timeval t_start2, t_end2;
    double time2;
#endif
    int it;
    int inner_xstreams;
    srand(1983);
    num_xstreams = argc > 1 ? atoi(argv[1]) : NUM_XSTREAMS;
    if (argc > 2) {
        str = argv[2];
    }
    ntasks = argc > 2 ? strtoll(str, &endptr, 10) : NUM_ELEMS;
    it = ceil(sqrt(ntasks));
    ntasks = it * it;
    inner_xstreams = argc > 3 ? atoi(argv[3]) : NUM_XSTREAMS;
    g_pools = (ABT_pool *)malloc(sizeof(ABT_pool) * num_xstreams);

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
        ABT_xstream_start(xstreams[i]);
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
        args[j].it = it;
        args[j].nxstreams = inner_xstreams;
#ifdef PROFTIME
        gettimeofday(&t_start2, NULL);
#endif
        ABT_thread_create_on_xstream(xstreams[j], random_launch,
                                     (void *)&args[j], ABT_THREAD_ATTR_NULL,
                                     NULL);
#ifdef PROFTIME
        gettimeofday(&t_end2, NULL);
        time2 = (t_end2.tv_sec * 1000000 + t_end2.tv_usec) -
            (t_start2.tv_sec * 1000000 + t_start2.tv_usec);
        printf("ULT creation time %f\n", time2 / 1000000.0);
#endif
    }

    ABT_thread_yield();

#ifdef PROFTIME
    gettimeofday(&t_start2, NULL);
#endif
    for (i = 0; i < num_xstreams; i++) {
        size_t size;
        while (1) {
            ABT_pool_get_size(g_pools[i], &size);
            if (size == 0) break;
            ABT_thread_yield();
        }
    }
#ifdef PROFTIME
    gettimeofday(&t_end2, NULL);
    time2 = (t_end2.tv_sec * 1000000 + t_end2.tv_usec) -
        (t_start2.tv_sec * 1000000 + t_start2.tv_usec);
    printf("Join time %f\n", time2 / 1000000.0);

#endif

    gettimeofday(&t_end, NULL);
    double time = (t_end.tv_sec * 1000000 + t_end.tv_usec) -
        (t_start.tv_sec * 1000000 + t_start.tv_usec);



    for (i = 1; i < num_xstreams; i++) {
        ABT_xstream_join(xstreams[i]);
        ABT_xstream_free(&xstreams[i]);
    }
    printf("%d %d %d %f\n",
           num_xstreams, inner_xstreams, ntasks, time / 1000000.0);

    ABT_finalize();
    free(xstreams);

    return EXIT_SUCCESS;
}

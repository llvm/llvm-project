/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */

/*
 * See LICENSE.txt in top-level directory.
 */

/*  This code mimics the parallel for OpenMP directive in nested loops.
 *  It creates as many streams as user requires and threads are created and
 *  assigned by static blocs to each stream for the outer loop.
 *  For the inner loop, as many threads as the user requires are created.
 */


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <abt.h>
#include <math.h>
#include <sys/time.h>
#include <unistd.h>

#define NUM_XSTREAMS    36
#define NUM         1000
#define NUM_REPS    1

int in[NUM][NUM];
int out[NUM][NUM];

/* Vector initialization */
void init(void)
{
    int i, j;
    for (i = 0; i < NUM; i++) {
        for (j = 0; j < NUM; j++) {
            in[i][j] = 1;
            out[i][j] = 0;
        }
    }
}

int comp(int v)
{
    int i;
    double ret = 0.0;
    for (i = 0; i < 100; i++) {
        ret += sqrt(cos((double)v) * sin((double)v));
    }
    return (int)ret;
}

void check(void)
{
    int i, j;
    for (i = 0; i < NUM; i++) {
        for (j = 0; j < NUM; j++) {
            int expected = comp(in[i][j]);
            if (out[i][j] != expected) {
                printf("out[%d][%d]=%d expected=%d\n", i, j, out[i][j], expected);
                return;
            }
        }
    }
    printf("Verification: SUCCESS\n");
}


static ABT_pool *g_pools;

typedef struct {
    int start;
    int end;
    int x;
} vector_scal_args_t;

typedef struct {
    int nxstreams;
    int it;
    int start;
    int end;
} vector_scal_task_args_t;

void vector_scal(void *arguments)
{
    int j;
    vector_scal_args_t *arg;
    arg = (vector_scal_args_t *)arguments;

    int mystart = arg->start;
    int myend = arg->end;
    int x = arg->x;

    for (j = mystart; j < myend; j++) {
        out[x][j] = comp(in[x][j]);
    }
}

void vector_scal_launch(void *arguments)
{
    int i, it, j, num_ults, rank, mystart, myend, p;
    ABT_thread *threads;
    ABT_xstream xstream;
    ABT_xstream_self(&xstream);
    vector_scal_task_args_t *arg;
    arg = (vector_scal_task_args_t *) arguments;
    vector_scal_args_t *args;
    it = arg->it;
    num_ults = arg->nxstreams;
    mystart = arg->start;
    myend = arg->end;

    args = (vector_scal_args_t *)malloc(sizeof(vector_scal_args_t)
                                        * num_ults);

    threads = (ABT_thread *)malloc(sizeof(ABT_thread) * num_ults);

    int bloc = it / (num_ults);
    int rest = it % (num_ults);
    ABT_xstream_self_rank(&rank);
    for (i = mystart; i < myend; i++) {
        int start = 0;
        int end = 0;
        for (j = 0; j < num_ults; j++) {
            start = end;
            int inc = (j < rest) ? 1 : 0;
            end += bloc + inc;
            args[j].start = start;
            args[j].end = end;
            args[j].x = i;

            if (j > 0) {
                ABT_thread_create(g_pools[rank], vector_scal,
                                  (void *)&args[j], ABT_THREAD_ATTR_NULL,
                                  &threads[j]);
            }
        }
        vector_scal((void *)&args[0]);
        for (p = 1; p < num_ults; p++) {
            ABT_thread_free(&threads[p]);
        }
    }

    free(threads);
    free(args);
}


int main(int argc, char *argv[])
{
    int i, j, r;
    int num_xstreams;
    char *str, *endptr;
    ABT_xstream *xstreams;
    ABT_thread *threads;
    vector_scal_task_args_t *args;
    int inner_xstreams;
    double *time, avg_time = 0.0;

    num_xstreams = (argc > 1) ? atoi(argv[1]) : NUM_XSTREAMS;
    inner_xstreams = (argc > 2) ? atoi(argv[2]) : NUM_XSTREAMS;
    int rep = (argc > 3) ? atoi(argv[3]) : NUM_REPS;
    time = (double *)malloc(sizeof(double) * rep);

    init();

    g_pools = (ABT_pool *)malloc(sizeof(ABT_pool) * num_xstreams);
    xstreams = (ABT_xstream *)malloc(sizeof(ABT_xstream) * num_xstreams);
    threads = (ABT_thread *)malloc(sizeof(ABT_thread) * num_xstreams);
    args = (vector_scal_task_args_t *)malloc(sizeof(vector_scal_task_args_t)
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

    /* Each task is created on the xstream which is going to execute it */

    for (r = 0; r < rep; r++) {
        time[r] = ABT_get_wtime();

        int bloc = NUM / (num_xstreams);
        int rest = NUM % (num_xstreams);
        int start = 0;
        int end = 0;

        for (j = 0; j < num_xstreams; j++) {
            start = end;
            int inc = (j < rest) ? 1 : 0;
            end += bloc + inc;
            args[j].start = start;
            args[j].end = end;
            args[j].it = NUM;
            args[j].nxstreams = inner_xstreams;
            if (j > 0) {
                ABT_thread_create(g_pools[j], vector_scal_launch,
                                  (void *)&args[j], ABT_THREAD_ATTR_NULL,
                                  &threads[j]);
            }
        }
        vector_scal_launch((void *)&args[0]);

        for (j = 1; j < num_xstreams; j++) {
            ABT_thread_free(&threads[j]);
        }

        time[r] = ABT_get_wtime() - time[r];
        avg_time += time[r];
    }
    avg_time /= rep;
    printf("%d %d %f\n", num_xstreams, inner_xstreams, avg_time);
    check();

    for (i = 1; i < num_xstreams; i++) {
        ABT_xstream_join(xstreams[i]);
        ABT_xstream_free(&xstreams[i]);
    }

    ABT_finalize();

    free(g_pools);
    free(xstreams);
    free(threads);
    free(args);
    free(time);

    return EXIT_SUCCESS;
}

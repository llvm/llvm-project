/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

/*
 * This example performs DAXPY in parallel.
 *   y = a * x + y  where a is scalar and x and y are vectors.
 * The following adopts a typical divide-and-conquer strategy.  Each thread
 * decomposes a vector until its vector length becomes less than a cut-off
 * threshold.
 *
 * The input parameters affect the following performance metrics.
 *
 * - Thread granularity
 *   If a cut-off threshold (-c CUTOFF) is large, each thread performs larger.
 *   computation, making the granularity larger.  On most platforms, thread
 *   fork-join overheads of Argobots are about a few microseconds, so the
 *   threading overheads get almost negligible if granularity is 1 ms or longer.
 *
 *   Advanced profiling (-p 2) gives more accurate thread granularity, but it is
 *   costlier.
 *
 * - Non-main scheduling ratio
 *   If a cut-off threshold (-c CUTOFF) is sufficiently large, computation time
 *   gets dominant over scheduling time (e.g., the non-main scheduling ratio
 *   exceeds 95%).  If the cut-off value is too large, however, some schedulers
 *   become idle because of load imbalance and lower the non-main scheduling
 *   ratio.  In general, users should to keep this number as high as possible
 *   (e.g., 95%).
 *
 * This example also shows that the overall execution time is affected by the
 * profiling mode (-p PROF_MODE).
 */

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <unistd.h>
#include <stdarg.h>
#include <abt.h>

#include "abtx_prof.h"

#define DEFAULT_NUM_XSTREAMS 4
#define DEFAULT_N (8 * 1024 * 1024)
#define DEFAULT_CUTOFF 1024
#define NUM_REPEATS 4

ABT_pool *pools;

typedef struct {
    int n;
    double a;
    double *x, *y;
    int cutoff;
} daxpy_arg_t;

void daxpy(void *arg)
{
    int n = ((daxpy_arg_t *)arg)->n;
    double a = ((daxpy_arg_t *)arg)->a;
    double *x = ((daxpy_arg_t *)arg)->x;
    double *y = ((daxpy_arg_t *)arg)->y;
    int cutoff = ((daxpy_arg_t *)arg)->cutoff;

    if (n <= cutoff) {
        /* Serialize the computation */
        int i;
        for (i = 0; i < n; i++) {
            y[i] = a * x[i] + y[i];
        }
    } else {
        daxpy_arg_t child1_arg = { n / 2, a, x, y, cutoff };
        daxpy_arg_t child2_arg = { n - n / 2, a, x + n / 2, y + n / 2, cutoff };

        int rank;
        ABT_xstream_self_rank(&rank);
        ABT_pool target_pool = pools[rank];
        ABT_thread child1;
        /* Calculate daxpy([0 : n/2]). */
        ABT_thread_create(target_pool, daxpy, &child1_arg, ABT_THREAD_ATTR_NULL,
                          &child1);
        /* Calculate daxpy([n/2 : n]).
         * We do not need to create another ULT. */
        daxpy(&child2_arg);
        ABT_thread_free(&child1);
    }
}

int main(int argc, char **argv)
{
    /* Read arguments. */
    int num_xstreams = DEFAULT_NUM_XSTREAMS;
    int n = DEFAULT_N;
    int cutoff = DEFAULT_CUTOFF;
    int i, j;
    int prof_mode = 1;
    while (1) {
        int opt = getopt(argc, argv, "he:n:c:p:");
        if (opt == -1)
            break;
        switch (opt) {
            case 'e':
                num_xstreams = atoi(optarg);
                break;
            case 'n':
                n = atoi(optarg);
                break;
            case 'c':
                cutoff = atoi(optarg);
                break;
            case 'p':
                prof_mode = atoi(optarg);
                break;
            case 'h':
            default:
                printf("Usage: ./daxpy [-e NUM_XSTREAMS] [-n N] [-c CUTOFF] "
                       "[-p PROF_MODE]\n"
                       "PROF_MODE = 0 : disable ABTX profiler\n"
                       "            1 : enable ABTX profiler (basic)\n"
                       "            2 : enable ABTX profiler (advanced)\n");
                return -1;
        }
    }

    /* Allocate memory. */
    ABT_xstream *xstreams =
        (ABT_xstream *)malloc(sizeof(ABT_xstream) * num_xstreams);
    pools = (ABT_pool *)malloc(sizeof(ABT_pool) * num_xstreams);
    ABT_sched *scheds = (ABT_sched *)malloc(sizeof(ABT_sched) * num_xstreams);

    /* Initialize Argobots. */
    ABT_init(argc, argv);

    /* Create pools. */
    for (i = 0; i < num_xstreams; i++) {
        ABT_pool_create_basic(ABT_POOL_FIFO, ABT_POOL_ACCESS_MPMC, ABT_TRUE,
                              &pools[i]);
    }

    /* Create schedulers. */
    for (i = 0; i < num_xstreams; i++) {
        ABT_pool *tmp = (ABT_pool *)malloc(sizeof(ABT_pool) * num_xstreams);
        for (j = 0; j < num_xstreams; j++) {
            tmp[j] = pools[(i + j) % num_xstreams];
        }
        ABT_sched_create_basic(ABT_SCHED_DEFAULT, num_xstreams, tmp,
                               ABT_SCHED_CONFIG_NULL, &scheds[i]);
        free(tmp);
    }

    /* Set up a primary execution stream. */
    ABT_xstream_self(&xstreams[0]);
    ABT_xstream_set_main_sched(xstreams[0], scheds[0]);

    /* Create secondary execution streams. */
    for (i = 1; i < num_xstreams; i++) {
        ABT_xstream_create(scheds[i], &xstreams[i]);
    }

    /* Setup vectors. */
    double *x = (double *)malloc(n * sizeof(double));
    double *y = (double *)calloc(n, sizeof(double));
    const double a = 2.0;
    for (i = 0; i < n; i++) {
        x[i] = ((double)i) / n;
    }

    int prof_init = 0;
    ABTX_prof_context prof_context;
    /* Initialize the profiler. */
    prof_init = ABTX_prof_init(&prof_context);

    for (i = 0; i < NUM_REPEATS; i++) {
        if (prof_init == ABT_SUCCESS && prof_mode == 1) {
            ABTX_prof_start(prof_context, ABTX_PROF_MODE_BASIC);
        } else if (prof_init == ABT_SUCCESS && prof_mode == 2) {
            ABTX_prof_start(prof_context, ABTX_PROF_MODE_DETAILED);
        }
        double start_time = ABT_get_wtime();
        daxpy_arg_t arg = { n, a, x, y, cutoff };
        daxpy(&arg);
        double end_time = ABT_get_wtime();

        if (prof_init == ABT_SUCCESS && (prof_mode == 1 || prof_mode == 2)) {
            ABTX_prof_stop(prof_context);
        }

        printf("##############################\n");
        printf("[%d] elapsed time = %f [s]\n", i, end_time - start_time);
        if (prof_init == ABT_SUCCESS && (prof_mode == 1 || prof_mode == 2)) {
            ABTX_prof_print(prof_context, stdout,
                            ABTX_PRINT_MODE_SUMMARY | ABTX_PRINT_MODE_FANCY);
        }
    }

    /* Check the results (only 100 points). */
    int ret = 0;
    for (i = 0; i < 100; i++) {
        int idx = (int)(n / 100.0 * i);
        double ans = x[idx] * a * NUM_REPEATS;
        if (y[idx] != ans) {
            printf("y[%d] = %f (ans: %f)\n", idx, y[idx], ans);
            ret = -1;
            break;
        }
    }

    /* Free vectors. */
    free(x);
    free(y);

    /* Join secondary execution streams. */
    for (i = 1; i < num_xstreams; i++) {
        ABT_xstream_join(xstreams[i]);
        ABT_xstream_free(&xstreams[i]);
    }

    /* Finalize the profiler. */
    if (prof_init == ABT_SUCCESS) {
        ABTX_prof_finalize(prof_context);
    }

    /* Finalize Argobots. */
    ABT_finalize();

    /* Free allocated memory. */
    free(xstreams);
    free(pools);
    free(scheds);

    return ret;
}

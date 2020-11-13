/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */

/*
 * See LICENSE.txt in top-level directory.
 */

/* Nested Pragma omp parallel for directive evaluation
 * Output: avg time
 */

#include <assert.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

#define NUM_ELEMS       5017600     /* 2GB */
#define NUM_REPS        1

int main(int argc, char *argv[])
{
    int i, j, r, nthreads;
    double *time, avg_time = 0.0;

    #pragma omp parallel
    {
        #pragma omp master
        {
            nthreads = omp_get_num_threads();
        }
    }
    int n = (argc > 1) ? atoi(argv[1]) : NUM_ELEMS;
    int in_th = (argc > 2) ? atoi(argv[2]) : nthreads;
    int rep = (argc > 3) ? atoi(argv[3]) : 3;
    int it = ceil(sqrt((double)n));
    srand(1983);

    n = it * it;
    time = (double *)malloc(sizeof(double) * rep);
    for (r = 0; r < rep; r++) {
        time[r] = omp_get_wtime();
        #pragma omp parallel for
        for (j = 0; j < it; j++) {
            omp_set_num_threads(in_th);
            #pragma omp parallel for
            for (i = 0; i < it; i++) {
                int random = rand() % 10000;
                volatile int kk = 0;
                int k;
                for (k = 0; k < random; k++)
                    kk++;
                assert(kk == random);
            }
        }
        time[r] = omp_get_wtime() - time[r];
        avg_time += time[r];
    }

    avg_time /= rep;
    printf("%d %d %d %f\n", nthreads, in_th, n, avg_time);

    free(time);

    return EXIT_SUCCESS;
}

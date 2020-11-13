/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */

/*
 * See LICENSE.txt in top-level directory.
 */

/* Nested Pragma omp parallel for directives evaluation
 * Output: avg time
 */

#include <assert.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>

#define NUM         1000
#define NUM_REPS    10

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

void petsc_voodoo(int x)
{
    int j;

    #pragma omp parallel for
    for (j = 0; j < NUM; j++) {
        out[x][j] = comp(in[x][j]);
    }
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
    int in_th = (argc > 1) ? atoi(argv[1]) : nthreads;
    int rep = (argc > 2) ? atoi(argv[2]) : NUM_REPS;
    time = (double *)malloc(sizeof(double) * rep);
    init();
    for (r = 0; r < rep; r++) {
        time[r] = omp_get_wtime();

        #pragma omp parallel for
        for (i = 0; i < NUM; i++) {
            omp_set_num_threads(in_th);
            petsc_voodoo(i);
        }
        time[r] = omp_get_wtime() - time[r];
        avg_time += time[r];
    }
    avg_time /= rep;
    printf("%d %d %f\n", nthreads, in_th, avg_time);
    check();

    free(time);

    return EXIT_SUCCESS;
}

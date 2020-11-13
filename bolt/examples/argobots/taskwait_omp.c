/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */

/*
 * See LICENSE.txt in top-level directory.
 */

/*
 * A bunch of n tasks (1st arg) are created by a single thread.
 * Each task creates two tasks more and executes a taskwait directive
 */

#include <assert.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define NUM_TASKS   50000
#define NUM_REPS    1

int o = 0;
int pp = 0;

void na(float value)
{
    o++;
}

void sscal(float value, float *a)
{
    *a = *a * value;
}

void presscal(float value, float *a)
{
    #pragma omp task
    {
        sscal(value, a);
    }

    #pragma omp task
    {
        na(value);
    }

    #pragma omp taskwait
}

int main(int argc, char *argv[])
{
    int i, r, nthreads;
    double *time, avg_time = 0.0;
    char *str, *endptr;
    float *a;
    double time2 = 0.0;

    #pragma omp parallel
    {
        #pragma omp master
        {
            nthreads = omp_get_num_threads();
        }
    }

    if (argc > 1) {
        str = argv[1];
    }

    int ntasks = argc > 1 ? strtoll(str, &endptr, 10) : NUM_TASKS;
    if (ntasks < nthreads)
        ntasks = nthreads;

    int rep = (argc > 2) ? atoi(argv[2]) : NUM_REPS;

    time = malloc(sizeof(double) * rep);

    a = malloc(sizeof(float) * ntasks);

    for (i = 0; i < ntasks; i++) {
        a[i] = i + 100.0f;
    }

    for (r = 0; r < rep; r++) {
        time[r] = omp_get_wtime();
        #pragma omp parallel
        {
            #pragma omp single
            {
                time2 = omp_get_wtime();
                for (i = 0; i < ntasks; i++) {
                    #pragma omp task firstprivate(i)
                    {
                        presscal(0.9f, &a[i]);
                    }
                }
                time2 = omp_get_wtime() - time2;
            }
        }
        time[r] = omp_get_wtime() - time[r];
        avg_time += time[r];

    }
    for (i = 0; i < ntasks; i++) {
        if (a[i] != (i + 100.0f) * 0.9f) {
            printf("error: a[%d]=%2.f expected %2.f\n", i,
                   a[i], (i + 100.0f) * 0.9f);
        }
    }
    avg_time /= rep;

    printf("nthreads: %d\nntasks: %d\nTime(s):%f\nCreation Time: %f\n",
           nthreads, ntasks, avg_time, time2);
    printf("o=%d and it should be %d\n", o, ntasks);
    printf("pp=%d and it should be %d\n", pp, ntasks);

    return EXIT_SUCCESS;
}

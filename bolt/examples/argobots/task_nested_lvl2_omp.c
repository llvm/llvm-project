/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */

/*
 * See LICENSE.txt in top-level directory.
 */

#include <assert.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define NUM_TASKS       50000
#define NUM_REPS        1
#define LEVELS          2

int o = 0;
void sscal(float value, float *a)
{
    *a = *a * value;
}

void na(float value)
{
    o++;
}

void presscal(float value, float *a, int lvl, int i)
{
    if (lvl > 1) {
        lvl--;
        #pragma omp task
        {
            presscal(value, a, lvl, i);
        }
        #pragma omp task
        {
            presscal(value, a, lvl, i);
        }
    }
    else {
        #pragma omp task
        {
            sscal(value, a);
        }

        #pragma omp task
        {
            na(value);
        }
    }
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

    int lvl = (argc > 2) ? atoi(argv[2]) : LEVELS;

    int rep = (argc > 3) ? atoi(argv[3]) : NUM_REPS;

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
                        presscal(0.9f, &a[i], lvl, i);
                    }
                }
                time2 = omp_get_wtime() - time2;
            }
        }
        time[r] = omp_get_wtime() - time[r];
        avg_time += time[r];

    }

    // TODO: Just works with one repetition
    for (i = 0; i < ntasks; i++) {
        if (a[i] != (i + 100.0f) * 0.9f) {
            printf("error: a[%d]=%2.f expected %2.f\n", i,
                   a[i], (i + 100.0f) * 0.9f);
        }
    }
    avg_time /= rep;
    printf("nthreads: %d\nntasks: %d\nTime(s):%f\nCreation Time: %f\n",
           nthreads, ntasks, avg_time, time2);
    printf("o=%d deberia valer %d\n", o, ntasks);

    return EXIT_SUCCESS;
}

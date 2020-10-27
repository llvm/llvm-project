/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */

/*
 * See LICENSE.txt in top-level directory.
 */

#include <assert.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define NUM_TASKS       5000000
#define NUM_REPS        1

void sscal(float value, float *a)
{
    *a = *a * value;
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
                sleep(2);
                printf("Thread %d\n", omp_get_thread_num());
                time2 = omp_get_wtime();
                for (i = 0; i < ntasks; i++) {
                    #pragma omp task firstprivate(i)
                    {
                        printf("Task %d executed by Thread %d Stolen? %s\n",
                               i, omp_get_thread_num(),
                               (i % nthreads == omp_get_thread_num())
                                ? "NO" : "YES");
                        sscal(0.9f, &a[i]);
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

    return EXIT_SUCCESS;
}

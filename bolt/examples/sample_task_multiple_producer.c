/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */

/*
 * See LICENSE.txt in top-level directory.
 */

#include <omp.h>
#include <stdio.h>
#include <sys/time.h>

int main(int argc, char * argv[]) {

    int i,num=(argc>1)?atoi(argv[1]):100;
    int nthreads;
    struct timeval t_start, t_end;
    double time;
    double *a = (double *)malloc(sizeof(double)*num);

    #pragma omp parallel
    {
        nthreads=omp_get_num_threads();
    }


    for(i=0;i<num;i++){
        a[i]=i;
    }

    gettimeofday(&t_start,NULL);

    #pragma omp parallel
    {
        #pragma omp for
        for(i=0;i<num;i++){
            #pragma omp task
            {
                a[i]*=0.9;
            }
       }
    }

    gettimeofday(&t_end,NULL);

    time=(t_end.tv_sec * 1000000 + t_end.tv_usec) -
         (t_start.tv_sec * 1000000 + t_start.tv_usec);

    printf("%d %f\n",nthreads,time/1000000.0);

    for(i=0;i<num;i++){
        if(a[i]!=i*0.9){
            printf("a[%d]=%f != %f\n",i,a[i],i*0.9);
            return 1;
        }
    }
}

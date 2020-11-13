/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */

/*
 * See LICENSE.txt in top-level directory.
 */

#include <omp.h>
#include <stdio.h>
#include <sys/time.h>



int main(int argc, char * argv[]) {

    int size=(argc>1)?atoi(argv[1]):100;
    int i,j,k=0;
    int nthreads;
    struct timeval t_start, t_end;
    double time;

    double *a = (double *)malloc(sizeof(double)*size*size);

    #pragma omp parallel
    {
        nthreads=omp_get_num_threads();
    }

    for(i=0;i<size*size;i++){
        a[i]=i;
    }

    gettimeofday(&t_start,NULL);

    #pragma omp parallel for
    for(i=0;i<size;i++){
        #pragma omp parallel for
        for(j=0;j<size;j++){
            a[i*size+j]=a[i*size+j]*0.9;
        }
    }

    gettimeofday(&t_end,NULL);

	

    time=(t_end.tv_sec * 1000000 + t_end.tv_usec) -
         (t_start.tv_sec * 1000000 + t_start.tv_usec);


    printf("%d %f\n",nthreads,time/1000000.0);

    for(i=0;i<size*size;i++){
        if(a[i]!=i*0.9){
	    printf("a[%d]=%f\n",i,a[i]);
	    return 1;
        }       
    }

}

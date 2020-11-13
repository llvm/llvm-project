// RUN: %libomp-compile-and-run
// UNSUPPORTED: gcc
// This test is incompatible with gcc because of the explicit call to
// __kmpc_doacross_fini().  gcc relies on an implicit call to this function
// when the last iteration is executed inside the GOMP_loop_*_next() functions.
// Hence, in gcc, having the explicit call leads to __kmpc_doacross_fini()
// being called twice.
#include <stdio.h>

#define N   1000

struct dim {
  long long lo; // lower
  long long up; // upper
  long long st; // stride
};
extern void __kmpc_doacross_init(void*, int, int, struct dim *);
extern void __kmpc_doacross_wait(void*, int, long long*);
extern void __kmpc_doacross_post(void*, int, long long*);
extern void __kmpc_doacross_fini(void*, int);
extern int __kmpc_global_thread_num(void*);

int main()
{
  int i;
  int iter[N];
  struct dim dims;
  for( i = 0; i < N; ++i )
    iter[i] = 1;
  dims.lo = 1;
  dims.up = N-1;
  dims.st = 1;
  #pragma omp parallel num_threads(4)
  {
    int i, gtid;
    long long vec;
    gtid = __kmpc_global_thread_num(NULL);
    __kmpc_doacross_init(NULL,gtid,1,&dims); // thread starts the loop
    #pragma omp for nowait schedule(dynamic)
    for( i = 1; i < N; ++i )
    {
      // runtime call corresponding to #pragma omp ordered depend(sink:i-1)
      vec=i-1;
      __kmpc_doacross_wait(NULL,gtid,&vec);
      // user's code
      iter[i] = iter[i-1] + 1;
      // runtime call corresponding to #pragma omp ordered depend(source)
      vec=i;
      __kmpc_doacross_post(NULL,gtid,&vec);
    }
    // thread finishes the loop (should be before the loop barrier)
    __kmpc_doacross_fini(NULL,gtid);
  }
  if( iter[N-1] == N ) {
    printf("passed\n");
  } else {
    printf("failed %d != %d\n", iter[N-1], N);
    return 1;
  }
  return 0;
}


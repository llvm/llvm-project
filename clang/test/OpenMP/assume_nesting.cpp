// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -ast-print %s | FileCheck %s
// expected-no-diagnostics

extern void bar();

void foo()
{
  #pragma omp assume no_openmp_routines
  // CHECK: omp assume no_openmp_routines
  {
    #pragma omp assume no_parallelism
    // CHECK: omp assume no_parallelism
    {}
  }

  #pragma omp target
  // CHECK: omp target
  {
    #pragma omp assume holds(1==1)
    // CHECK: omp assume holds(1 == 1)
    {}
  }

  #pragma omp assume no_parallelism
  // CHECK: omp assume no_parallelism
  {
    #pragma omp target
    // CHECK: omp target
    {}
  }

  #pragma omp assume absent(parallel)
  // CHECK: omp assume absent(parallel)
  {
    #pragma omp assume contains(target, loop)
    // CHECK: omp assume contains(target, loop)
    {
      #pragma omp assume holds(1==1)
      // CHECK: omp assume holds(1 == 1)
      {
        #pragma omp assume absent(teams)
        // CHECK: omp assume absent(teams)
        {
          #pragma omp assume no_openmp_routines
          // CHECK: omp assume no_openmp_routines
          {
            bar();
          }
        }
      }
    }
  }
}

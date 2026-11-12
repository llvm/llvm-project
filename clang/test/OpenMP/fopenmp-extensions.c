// RUN: %clang_cc1 -verify=ompx -fopenmp %s
// RUN: %clang_cc1 -verify=ompx -fopenmp-simd %s

// RUN: %clang_cc1 -verify=ompx -fopenmp -fopenmp-extensions %s
// RUN: %clang_cc1 -verify=ompx -fopenmp-simd -fopenmp-extensions %s

// RUN: %clang_cc1 -verify=omp -fopenmp -fno-openmp-extensions %s
// RUN: %clang_cc1 -verify=omp -fopenmp-simd -fno-openmp-extensions %s

// RUN: %clang_cc1 -verify=omp -fopenmp \
// RUN:     -fopenmp-extensions -fno-openmp-extensions %s
// RUN: %clang_cc1 -verify=omp -fopenmp-simd \
// RUN:     -fopenmp-extensions -fno-openmp-extensions %s

// RUN: %clang_cc1 -verify=ompx -fopenmp \
// RUN:     -fno-openmp-extensions -fopenmp-extensions %s
// RUN: %clang_cc1 -verify=ompx -fopenmp-simd \
// RUN:     -fno-openmp-extensions -fopenmp-extensions %s

void foo(void) {
  int x;
  // ompx-no-diagnostics
  // omp-error@+1 {{incorrect map type modifier}}
  #pragma omp target map(ompx_hold, alloc: x)
  ;
}

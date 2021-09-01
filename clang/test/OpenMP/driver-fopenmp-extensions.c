// RUN: %clang -c -Xclang -verify=ompx -fopenmp %s
// RUN: %clang -c -Xclang -verify=ompx -fopenmp-simd %s

// RUN: %clang -c -Xclang -verify=ompx -fopenmp -fopenmp-extensions %s
// RUN: %clang -c -Xclang -verify=ompx -fopenmp-simd -fopenmp-extensions %s

// RUN: %clang -c -Xclang -verify=omp -fopenmp -fno-openmp-extensions %s
// RUN: %clang -c -Xclang -verify=omp -fopenmp-simd -fno-openmp-extensions %s

// RUN: %clang -c -Xclang -verify=omp -fopenmp \
// RUN:     -fopenmp-extensions -fno-openmp-extensions %s
// RUN: %clang -c -Xclang -verify=omp -fopenmp-simd \
// RUN:     -fopenmp-extensions -fno-openmp-extensions %s

// RUN: %clang -c -Xclang -verify=ompx -fopenmp \
// RUN:     -fno-openmp-extensions -fopenmp-extensions %s
// RUN: %clang -c -Xclang -verify=ompx -fopenmp-simd \
// RUN:     -fno-openmp-extensions -fopenmp-extensions %s

void foo() {
  int x;
  // ompx-no-diagnostics
  // omp-error@+1 {{incorrect map type modifier}}
  #pragma omp target map(ompx_hold, alloc: x)
  ;
}

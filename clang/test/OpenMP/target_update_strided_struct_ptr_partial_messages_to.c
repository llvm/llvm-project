// RUN: %clang_cc1 -verify -fopenmp -ferror-limit 100 %s -Wuninitialized
// RUN: %clang_cc1 -verify -fopenmp-simd -ferror-limit 100 %s -Wuninitialized
// expected-no-diagnostics

#define N 24
typedef struct {
  double *data;
  int len; 
} T;

int main(int argc, char **argv) {
  T s; 
  s.len = N;
  s.data = (double *)__builtin_alloca(N * sizeof(double));
  
  // Valid partial strided updates with pointer member (to clause)
  #pragma omp target update to(s.data[0:2:10]) // OK - partial coverage (stride > length)
  
  // Valid: complex expressions
  int offset = 1;

  // Runtime-dependent stride expressions
  #pragma omp target update to(s.data[0:4:offset+1]) // OK
  
  return 0;
}

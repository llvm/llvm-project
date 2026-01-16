// RUN: %clang_cc1 -verify -fopenmp -ferror-limit 100 %s -Wuninitialized
// RUN: %clang_cc1 -verify -fopenmp-simd -ferror-limit 100 %s -Wuninitialized
// expected-no-diagnostics

#define N 11
typedef struct { 
  double data[N]; 
  int len; 
} T;

int main(int argc, char **argv) {
  T s; 
  s.len = N;

  // Valid partial strided updates
  #pragma omp target update to(s.data[0:4:3]) // OK
  {}

  // Stride larger than length
  #pragma omp target update to(s.data[0:2:10]) // OK
  {}

  // Valid: complex expressions
  int offset = 1;

  // Potentially invalid stride expressions depending on runtime values
  #pragma omp target update to(s.data[0:4:offset-1]) // OK if offset > 1
  {}

  return 0;
}

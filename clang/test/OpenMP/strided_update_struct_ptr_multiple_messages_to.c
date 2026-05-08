// RUN: %clang_cc1 -verify -fopenmp -ferror-limit 100 %s -Wuninitialized
// RUN: %clang_cc1 -verify -fopenmp-simd -ferror-limit 100 %s -Wuninitialized

#define N 20
typedef struct { 
  double *data; 
  int len; 
} T;

int main(int argc, char **argv) {
  T s1, s2;
  s1.len = N;
  s1.data = (double *)__builtin_alloca(N * sizeof(double));
  s2.len = N;
  s2.data = (double *)__builtin_alloca(N * sizeof(double));
  
  // Multiple valid strided updates (to clause)
  #pragma omp target update to(s1.data[0:10:2], s2.data[0:7:3]) // OK
  
  // Mixed: one with stride, one without
  #pragma omp target update to(s1.data[0:N], s2.data[0:5:2]) // OK
  
  int stride1 = 2;
  int stride2 = 3;
  
  // Multiple with expression strides
  #pragma omp target update to(s1.data[1:5:stride1], s2.data[0:4:stride2]) // OK
  
  // One valid, one invalid
  #pragma omp target update to(s1.data[0:5:2], s2.data[0:4:0]) // expected-error {{section stride is evaluated to a non-positive value 0}}
  
  #pragma omp target update to(s1.data[0:5:-1], s2.data[0:4:2]) // expected-error {{section stride is evaluated to a non-positive value -1}}
  
  #pragma omp target update to(s1.data[0:5:0], s2.data[0:4:1]) // expected-error {{section stride is evaluated to a non-positive value 0}}
  
  // Syntax errors
  #pragma omp target update to(s1.data[0:5:2], s2.data[0:4 3]) // expected-error {{expected ']'}} expected-note {{to match this '['}}
  
  #pragma omp target update to(s1.data[0:5:2:3], s2.data[0:4:2]) // expected-error {{expected ']'}} expected-note {{to match this '['}}
  
  return 0;
}

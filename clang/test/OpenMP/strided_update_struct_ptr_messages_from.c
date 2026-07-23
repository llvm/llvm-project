// RUN: %clang_cc1 -verify -fopenmp -ferror-limit 100 %s -Wuninitialized
// RUN: %clang_cc1 -verify -fopenmp-simd -ferror-limit 100 %s -Wuninitialized

#define N 16
typedef struct { 
  double *data; 
  int len; 
} T;

int main(int argc, char **argv) {
  T s;
  s.len = N;
  s.data = (double *)__builtin_alloca(N * sizeof(double));
  
  // Valid strided array sections with pointer member
  #pragma omp target update from(s.data[0:4:2]) // OK
  
  #pragma omp target update from(s.data[1:3:2]) // OK
  
  // Missing stride (default = 1)
  #pragma omp target update from(s.data[0:4]) // OK
  
  // Invalid stride expressions
  #pragma omp target update from(s.data[0:4:0]) // expected-error {{section stride is evaluated to a non-positive value 0}} expected-error {{expected at least one 'to' clause or 'from' clause specified to '#pragma omp target update'}}
  
  #pragma omp target update from(s.data[0:4:-1]) // expected-error {{section stride is evaluated to a non-positive value -1}} expected-error {{expected at least one 'to' clause or 'from' clause specified to '#pragma omp target update'}}
  
  // Missing colon 
  #pragma omp target update from(s.data[0:4 2]) // expected-error {{expected ']'}} expected-note {{to match this '['}} expected-error {{expected at least one 'to' clause or 'from' clause specified to '#pragma omp target update'}}
  
  // Too many colons
  #pragma omp target update from(s.data[0:4:2:1]) // expected-error {{expected ']'}} expected-note {{to match this '['}} expected-error {{expected at least one 'to' clause or 'from' clause specified to '#pragma omp target update'}}
  
  return 0;
}

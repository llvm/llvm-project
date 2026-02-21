// RUN: %clang_cc1 -verify -fopenmp -ferror-limit 100 %s -Wuninitialized
// RUN: %clang_cc1 -verify -fopenmp-simd -ferror-limit 100 %s -Wuninitialized

int main(int argc, char **argv) {
  int len = 12;
  double *data;
  double *data1;
  double *data2;

  // Valid multiple strided array sections
  #pragma omp target update to(data1[0:6:2], data2[0:4:3]) // OK - different strides
  {}

  #pragma omp target update to(data1[1:2:3], data2[2:3:2]) // OK - with offsets
  {}

  // Mixed strided and regular sections
  #pragma omp target update to(data1[0:len], data2[0:4:2]) // OK - mixed
  {}

  #pragma omp target update to(data1[1:3:2], data2[0:len]) // OK - reversed mix
  {}

  // Using the single data pointer with strides
  #pragma omp target update to(data[0:4:2]) // OK - single pointer
  {}

  // Invalid stride in one of multiple sections
  #pragma omp target update to(data1[0:3:4], data2[0:2:0]) // expected-error {{section stride is evaluated to a non-positive value 0}}

  #pragma omp target update to(data1[0:3:-1], data2[0:2:2]) // expected-error {{section stride is evaluated to a non-positive value -1}}

  #pragma omp target update to(data[0:4:0], data1[0:2:1]) // expected-error {{section stride is evaluated to a non-positive value 0}}

  // Complex expressions in multiple arrays
  int stride1 = 2, stride2 = 3;
  #pragma omp target update to(data1[1:4:stride1+1], data2[0:3:stride2-1]) // OK - expressions
  {}

  // Mix all three pointers
  #pragma omp target update to(data[0:2:3], data1[1:3:2], data2[2:2:4]) // OK - three arrays
  {}

  // Syntax errors in multiple arrays
  #pragma omp target update to(data1[0:4:2], data2[0:3 4]) // expected-error {{expected ']'}} expected-note {{to match this '['}}

  #pragma omp target update to(data1[0:4:2:3], data2[0:3:2]) // expected-error {{expected ']'}} expected-note {{to match this '['}}

  #pragma omp target update to(data[0:4:2], data1[0:3:2:1], data2[0:2:3]) // expected-error {{expected ']'}} expected-note {{to match this '['}}

  return 0;
}
// RUN: %clang_cc1 -verify -fopenmp -ferror-limit 100 %s -Wuninitialized
// RUN: %clang_cc1 -verify -fopenmp-simd -ferror-limit 100 %s -Wuninitialized
// expected-no-diagnostics

int main(int argc, char **argv) {
  int len = 11;
  double data[len];

  // Valid partial strided updates
  #pragma omp target update to(data[0:4:3]) // OK
  {}

  // Stride larger than length
  #pragma omp target update to(data[0:2:10]) // OK
  {}

  // Valid: complex expressions
  int offset = 1;

  // Potentially invalid stride expressions depending on runtime values
  #pragma omp target update to(data[0:4:offset-1]) // OK if offset > 1
  {}

  return 0;
}

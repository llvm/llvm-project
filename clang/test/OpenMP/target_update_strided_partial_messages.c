// RUN: %clang_cc1 -verify -fopenmp -ferror-limit 100 %s -Wuninitialized
// RUN: %clang_cc1 -verify -fopenmp-simd -ferror-limit 100 %s -Wuninitialized

void foo(void) {}

int main(int argc, char **argv) {
  int len = 11;
  double data[len];
  
  // Valid partial strided updates
  #pragma omp target update from(data[0:4:3]) // OK
  {}
  
  // Stride larger than length
  #pragma omp target update from(data[0:2:10]) // OK
  {}
  
  // Valid: complex expressions
  int offset = 1;
  int count = 3;
  int stride = 2;
  #pragma omp target update from(data[offset:count:stride]) // OK
  {}
  
  // Invalid stride expressions
  #pragma omp target update from(data[0:4:offset-1]) // OK if offset > 1
  {}
  
  #pragma omp target update from(data[0:count:0]) // expected-error {{section stride is evaluated to a non-positive value 0}} expected-error {{expected at least one 'to' clause or 'from' clause specified to '#pragma omp target update'}}
  
  return 0;
}
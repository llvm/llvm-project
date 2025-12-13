// RUN: %clang_cc1 -verify -fopenmp -ferror-limit 100 %s -Wuninitialized
// RUN: %clang_cc1 -verify -fopenmp-simd -ferror-limit 100 %s -Wuninitialized

void foo(void) {}

int main(int argc, char **argv) {
  int len = 8;
  double data[len];
  
  // Valid strided array sections
  #pragma omp target update from(data[0:4:2]) // OK
  {}
  
  #pragma omp target update to(data[0:len/2:2]) // OK
  {}
  
  #pragma omp target update from(data[1:3:2]) // OK
  {}
  
  // Missing stride (default = 1)
  #pragma omp target update from(data[0:4]) // OK
  {}
  
  // Invalid stride expressions
  #pragma omp target update from(data[0:4:0]) // expected-error {{section stride is evaluated to a non-positive value 0}} expected-error {{expected at least one 'to' clause or 'from' clause specified to '#pragma omp target update'}}
  
  #pragma omp target update from(data[0:4:-1]) // expected-error {{section stride is evaluated to a non-positive value -1}} expected-error {{expected at least one 'to' clause or 'from' clause specified to '#pragma omp target update'}}
  
  // Missing colon 
  #pragma omp target update from(data[0:4 2]) // expected-error {{expected ']'}} expected-note {{to match this '['}} expected-error {{expected at least one 'to' clause or 'from' clause specified to '#pragma omp target update'}}
  {}
  
  // Too many colons
  #pragma omp target update from(data[0:4:2:1]) // expected-error {{expected ']'}} expected-note {{to match this '['}} expected-error {{expected at least one 'to' clause or 'from' clause specified to '#pragma omp target update'}}
  {}
  
  return 0;
}
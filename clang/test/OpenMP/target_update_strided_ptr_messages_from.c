// RUN: %clang_cc1 -verify -fopenmp -ferror-limit 100 %s -Wuninitialized
// RUN: %clang_cc1 -verify -fopenmp-simd -ferror-limit 100 %s -Wuninitialized

extern void *malloc(__SIZE_TYPE__);
extern void free(void *);

int main(int argc, char **argv) {
  int len = 16;
  double *data = (double *)malloc(len * sizeof(double));
  
  // Valid strided array sections with FROM
  #pragma omp target update from(data[0:8:2]) // OK - even indices
  {}
  
  #pragma omp target update from(data[1:4:3]) // OK - odd start with stride
  {}
  
  #pragma omp target update from(data[2:3:5]) // OK - large stride
  {}
  
  // Missing stride (default = 1)
  #pragma omp target update from(data[0:8]) // OK - default stride
  {}
  
  #pragma omp target update from(data[4:len-4]) // OK - computed length
  {}
  
  // Invalid stride expressions
  #pragma omp target update from(data[0:8:0]) // expected-error {{section stride is evaluated to a non-positive value 0}} expected-error {{expected at least one 'to' clause or 'from' clause specified to '#pragma omp target update'}}
  
  #pragma omp target update from(data[0:4:-1]) // expected-error {{section stride is evaluated to a non-positive value -1}} expected-error {{expected at least one 'to' clause or 'from' clause specified to '#pragma omp target update'}}
  
  #pragma omp target update from(data[1:5:-2]) // expected-error {{section stride is evaluated to a non-positive value -2}} expected-error {{expected at least one 'to' clause or 'from' clause specified to '#pragma omp target update'}}
  
  // Syntax errors
  #pragma omp target update from(data[0:4 2]) // expected-error {{expected ']'}} expected-note {{to match this '['}} expected-error {{expected at least one 'to' clause or 'from' clause specified to '#pragma omp target update'}}
  {}
  
  #pragma omp target update from(data[0:4:2:1]) // expected-error {{expected ']'}} expected-note {{to match this '['}} expected-error {{expected at least one 'to' clause or 'from' clause specified to '#pragma omp target update'}}
  {}
  
  free(data);
  return 0;
}
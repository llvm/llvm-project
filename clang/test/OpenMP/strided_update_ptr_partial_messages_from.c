// RUN: %clang_cc1 -verify -fopenmp -ferror-limit 100 %s -Wuninitialized
// RUN: %clang_cc1 -verify -fopenmp-simd -ferror-limit 100 %s -Wuninitialized

int main(int argc, char **argv) {
  int len = 11;
  double *data;
  
  // Valid partial strided sections with FROM
  #pragma omp target update from(data[0:2:3]) // OK - partial coverage
  {}
  
  #pragma omp target update from(data[1:3:4]) // OK - offset with partial stride
  {}
  
  #pragma omp target update from(data[2:2:5]) // OK - large partial stride
  {}
  
  // Stride larger than remaining elements
  #pragma omp target update from(data[0:2:10]) // OK - stride > array size
  {}
  
  #pragma omp target update from(data[0:3:len]) // OK - stride = len
  {}
  
  // Complex expressions
  int offset = 1;
  int stride = 2;
  
  // Runtime-dependent invalid strides
  #pragma omp target update from(data[0:4:offset-1]) // OK if offset > 1
  {}
  
  // Compile-time invalid strides
  #pragma omp target update from(data[1:2:-3]) // expected-error {{section stride is evaluated to a non-positive value -3}} expected-error {{expected at least one 'to' clause or 'from' clause specified to '#pragma omp target update'}}
  
  return 0;
}
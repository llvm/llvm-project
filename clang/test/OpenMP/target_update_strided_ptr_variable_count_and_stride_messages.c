// RUN: %clang_cc1 -verify -fopenmp -ferror-limit 100 %s -Wuninitialized
// RUN: %clang_cc1 -verify -fopenmp-simd -ferror-limit 100 %s -Wuninitialized

int main(int argc, char **argv) {
  int len = 16;
  int count = 8;
  int stride = 2;
  int stride_large = 5;
  double *data;
  
  // Valid strided array sections with both variable count and variable stride (FROM)
  #pragma omp target update from(data[0:count:stride]) // OK - both variable
  {}
  
  #pragma omp target update from(data[0:len/2:stride]) // OK - count expression, variable stride
  {}
  
  #pragma omp target update from(data[0:count:stride_large]) // OK - variable count, different stride
  {}
  
  #pragma omp target update from(data[1:len-2:stride]) // OK - with offset, count expression
  {}
  
  #pragma omp target update from(data[0:count/2:stride*2]) // OK - both expressions
  {}
  
  #pragma omp target update from(data[0:(len+1)/2:stride+1]) // OK - complex expressions
  {}
  
  #pragma omp target update from(data[2:count-2:len/4]) // OK - all expressions
  {}
  
  // Edge cases
  int stride_one = 1;
  #pragma omp target update from(data[0:count:stride_one]) // OK - variable count, stride=1
  {}
  
  #pragma omp target update from(data[0:len/stride:stride]) // OK - count depends on stride
  {}
  
  // Invalid compile-time constant strides with variable count
  #pragma omp target update from(data[0:count:0]) // expected-error {{section stride is evaluated to a non-positive value 0}} expected-error {{expected at least one 'to' clause or 'from' clause specified to '#pragma omp target update'}}
  
  #pragma omp target update from(data[0:len/2:-1]) // expected-error {{section stride is evaluated to a non-positive value -1}} expected-error {{expected at least one 'to' clause or 'from' clause specified to '#pragma omp target update'}}
  
  #pragma omp target update from(data[1:count:-2]) // expected-error {{section stride is evaluated to a non-positive value -2}} expected-error {{expected at least one 'to' clause or 'from' clause specified to '#pragma omp target update'}}
  
  // Valid strided array sections with variable count and stride (TO)
  #pragma omp target update to(data[0:count:stride]) // OK
  {}
  
  #pragma omp target update to(data[0:len/2:stride]) // OK
  {}
  
  #pragma omp target update to(data[0:count:stride*2]) // OK
  {}
  
  // Invalid stride with TO
  #pragma omp target update to(data[0:count:0]) // expected-error {{section stride is evaluated to a non-positive value 0}} expected-error {{expected at least one 'to' clause or 'from' clause specified to '#pragma omp target update'}}
  
  return 0;
}

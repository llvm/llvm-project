// RUN: %clang_cc1 -verify -fopenmp -ferror-limit 100 %s -Wuninitialized
// RUN: %clang_cc1 -verify -fopenmp-simd -ferror-limit 100 %s -Wuninitialized
// expected-no-diagnostics

int main(int argc, char **argv) {
  int len = 16;
  int count = 8;
  int stride = 2;
  int stride_large = 5;
  double *data;
  
  // Valid strided array sections with both variable count and variable stride (FROM)
  #pragma omp target update from(data[0:count:stride]) // OK - both variable
  
  #pragma omp target update from(data[0:len/2:stride]) // OK - count expression, variable stride
  
  #pragma omp target update from(data[0:count:stride_large]) // OK - variable count, different stride
  
  #pragma omp target update from(data[1:len-2:stride]) // OK - with offset, count expression
  
  #pragma omp target update from(data[0:count/2:stride*2]) // OK - both expressions
  
  #pragma omp target update from(data[0:(len+1)/2:stride+1]) // OK - complex expressions
  
  #pragma omp target update from(data[2:count-2:len/4]) // OK - all expressions
  
  #pragma omp target update from(data[0:len/stride:stride]) // OK - count depends on stride
  
  // Valid strided array sections with variable count and stride (TO)
  #pragma omp target update to(data[0:count:stride]) // OK
  
  #pragma omp target update to(data[0:len/2:stride]) // OK
  
  #pragma omp target update to(data[0:count:stride*2]) // OK
  
  return 0;
}

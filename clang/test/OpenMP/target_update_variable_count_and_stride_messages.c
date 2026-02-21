// RUN: %clang_cc1 -verify -fopenmp -ferror-limit 100 %s -Wuninitialized
// RUN: %clang_cc1 -verify -fopenmp-simd -ferror-limit 100 %s -Wuninitialized
// expected-no-diagnostics

int main(int argc, char **argv) {
  int len = 16;
  int count = 8;
  int stride = 2;
  int divisor = 2;
  double data[100];
  
  // Valid strided array sections with variable count expressions (FROM)
  #pragma omp target update from(data[0:count:2]) // OK - variable count
  
  #pragma omp target update from(data[0:len/2:2]) // OK - count expression
  
  #pragma omp target update from(data[0:len-4:3]) // OK - count with subtraction
  
  #pragma omp target update from(data[1:(len+1)/2:2]) // OK - complex count expression
  
  #pragma omp target update from(data[0:count*2:3]) // OK - count multiplication
  
  #pragma omp target update from(data[2:len%divisor:2]) // OK - count with modulo
  
  // Variable stride with constant/variable count
  #pragma omp target update from(data[0:10:stride]) // OK - constant count, variable stride
  
  #pragma omp target update from(data[0:count:stride]) // OK - both variable
  
  #pragma omp target update from(data[0:len/2:stride]) // OK - count expression, variable stride
  
  #pragma omp target update from(data[0:count:stride*2]) // OK - variable count, stride expression
  
  #pragma omp target update from(data[0:len/divisor:stride+1]) // OK - both expressions
  
  // Variable count with stride = 1 (contiguous)
  #pragma omp target update from(data[0:count]) // OK - variable count, implicit stride
  
  #pragma omp target update from(data[0:len/divisor]) // OK - expression count, implicit stride
  
  #pragma omp target update from(data[0:len/stride:stride]) // OK - count depends on stride
  
  // Valid strided array sections with variable count expressions (TO)
  #pragma omp target update to(data[0:count:2]) // OK
  
  #pragma omp target update to(data[0:len/2:stride]) // OK
  
  #pragma omp target update to(data[0:count:stride]) // OK
  
  #pragma omp target update to(data[0:len/divisor:stride+1]) // OK
  
  return 0;
}

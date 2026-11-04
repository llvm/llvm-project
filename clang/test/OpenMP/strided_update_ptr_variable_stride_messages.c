// RUN: %clang_cc1 -verify -fopenmp -ferror-limit 100 %s -Wuninitialized
// RUN: %clang_cc1 -verify -fopenmp-simd -ferror-limit 100 %s -Wuninitialized
// expected-no-diagnostics

int main(int argc, char **argv) {
  int len = 16;
  int stride = 2;
  int stride_large = 5;
  double *data;
  
  // Valid strided array sections with variable stride (FROM)
  #pragma omp target update from(data[0:8:stride]) // OK - variable stride
  
  #pragma omp target update from(data[0:4:stride_large]) // OK - different variable stride
  
  #pragma omp target update from(data[1:6:stride]) // OK - with offset
  
  #pragma omp target update from(data[0:5:stride+1]) // OK - stride expression
  
  #pragma omp target update from(data[0:4:stride*2]) // OK - stride multiplication
  
  #pragma omp target update from(data[2:3:len/4]) // OK - stride from expression
  
  // Invalid variable stride expressions  
  int zero_stride = 0;
  int neg_stride = -1;
  
  // Note: These are runtime checks, so no compile-time error
  #pragma omp target update from(data[0:8:zero_stride]) // OK at compile-time (runtime will fail)
  
  #pragma omp target update from(data[0:4:neg_stride]) // OK at compile-time (runtime will fail)
  
  // Valid strided array sections with variable stride (TO)
  #pragma omp target update to(data[0:8:stride]) // OK
  
  #pragma omp target update to(data[0:5:stride+1]) // OK
  
  #pragma omp target update to(data[0:4:stride*2]) // OK
  
  return 0;
}

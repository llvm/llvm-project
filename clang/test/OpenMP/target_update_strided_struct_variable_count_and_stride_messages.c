// RUN: %clang_cc1 -verify -fopenmp -ferror-limit 100 %s -Wuninitialized
// RUN: %clang_cc1 -verify -fopenmp-simd -ferror-limit 100 %s -Wuninitialized

#define N 20
typedef struct { 
  double data[N]; 
  int len;
  int stride;
} T;

int main(int argc, char **argv) {
  T s;
  s.len = 16;
  s.stride = 2;
  int count = 8;
  int ext_stride = 3;
  
  // Valid strided struct member array sections with variable count/stride (FROM)
  #pragma omp target update from(s.data[0:s.len/2:2]) // OK - member count expression
  
  #pragma omp target update from(s.data[0:count:s.stride]) // OK - external count, member stride
  
  #pragma omp target update from(s.data[0:s.len:ext_stride]) // OK - member count, external stride
  
  #pragma omp target update from(s.data[0:count:ext_stride]) // OK - both external
  
  #pragma omp target update from(s.data[0:s.len/2:s.stride]) // OK - both from struct
  
  #pragma omp target update from(s.data[1:(s.len-2)/2:s.stride]) // OK - complex count expression
  
  #pragma omp target update from(s.data[0:count*2:s.stride+1]) // OK - expressions for both
  
  #pragma omp target update from(s.data[0:s.len/s.stride:s.stride]) // OK - count depends on stride
  
  // Invalid compile-time constant strides with variable count
  #pragma omp target update from(s.data[0:s.len:0]) // expected-error {{section stride is evaluated to a non-positive value 0}} expected-error {{expected at least one 'to' clause or 'from' clause specified to '#pragma omp target update'}}
  
  #pragma omp target update from(s.data[0:count:-1]) // expected-error {{section stride is evaluated to a non-positive value -1}} expected-error {{expected at least one 'to' clause or 'from' clause specified to '#pragma omp target update'}}
  
  #pragma omp target update from(s.data[1:s.len/2:-2]) // expected-error {{section stride is evaluated to a non-positive value -2}} expected-error {{expected at least one 'to' clause or 'from' clause specified to '#pragma omp target update'}}
  
  // Valid strided struct member array sections with variable count and stride (TO)
  #pragma omp target update to(s.data[0:s.len/2:2]) // OK
  
  #pragma omp target update to(s.data[0:count:s.stride]) // OK
  
  #pragma omp target update to(s.data[0:s.len:ext_stride]) // OK
  
  #pragma omp target update to(s.data[0:count*2:s.stride+1]) // OK
  
  // Invalid stride with TO
  #pragma omp target update to(s.data[0:s.len:0]) // expected-error {{section stride is evaluated to a non-positive value 0}} expected-error {{expected at least one 'to' clause or 'from' clause specified to '#pragma omp target update'}}
  
  return 0;
}

// RUN: %clang_cc1 -verify -fopenmp -ferror-limit 100 %s -Wuninitialized
// RUN: %clang_cc1 -verify -fopenmp-simd -ferror-limit 100 %s -Wuninitialized

void foo(void) {}

typedef struct {
  int len;
  double data[12];
} S;

int main(int argc, char **argv) {
  int len = 12;
  double data1[len], data2[len];
  S s;
  
  // Valid multiple strided array sections
  #pragma omp target update from(data1[0:4:2], data2[0:2:5]) // OK
  {}
  
  #pragma omp target update to(data1[1:2:3], data2[2:3:2]) // OK
  {}
  
  // Mixed strided and regular array sections
  #pragma omp target update from(data1[0:len], data2[0:4:2]) // OK
  {}
  
  // Struct member arrays with strides
  #pragma omp target update from(s.data[0:4:2]) // OK
  {}
  
  #pragma omp target update from(s.data[0:s.len/2:2]) // OK
  {}
  
  // Invalid stride in one of multiple sections
  #pragma omp target update from(data1[0:3:4], data2[0:2:0]) // expected-error {{section stride is evaluated to a non-positive value 0}}
  
  // Complex expressions in multiple arrays
  int stride1 = 2, stride2 = 3;
  #pragma omp target update from(data1[0:len/2:stride1], data2[1:len/3:stride2]) // OK
  {}
  
  // Missing colon 
  #pragma omp target update from(data1[0:4:2], data2[0:3 4]) // expected-error {{expected ']'}} expected-note {{to match this '['}}
  
  return 0;
}
// RUN: %clang_cc1 -verify -fopenmp -ferror-limit 100 %s -Wuninitialized
// RUN: %clang_cc1 -verify -fopenmp-simd -ferror-limit 100 %s -Wuninitialized

int main(int argc, char **argv) {
  int len = 12;
  double data1[len], data2[len];
  
  // Valid multiple strided array sections
  #pragma omp target update from(data1[0:4:2], data2[0:2:5]) // OK
  {}
  
  // Mixed strided and regular array sections
  #pragma omp target update from(data1[0:len], data2[0:4:2]) // OK
  {}
  
  // Invalid stride in one of multiple sections
  #pragma omp target update from(data1[0:3:4], data2[0:2:0]) // expected-error {{section stride is evaluated to a non-positive value 0}}
  
  // Missing colon 
  #pragma omp target update from(data1[0:4:2], data2[0:3 4]) // expected-error {{expected ']'}} expected-note {{to match this '['}}
  
  return 0;
}

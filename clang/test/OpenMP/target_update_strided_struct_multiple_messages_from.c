// RUN: %clang_cc1 -verify -fopenmp -ferror-limit 100 %s -Wuninitialized
// RUN: %clang_cc1 -verify -fopenmp-simd -ferror-limit 100 %s -Wuninitialized

#define N 12
typedef struct {
  double data[N];
  int len;
} T;

int main(int argc, char **argv) {
  T s1, s2;
  s1.len = N;
  s2.len = N;

  // Valid multiple strided array sections
  #pragma omp target update from(s1.data[0:4:2], s2.data[0:2:5]) // OK
  {}

  // Mixed strided and regular array sections
  #pragma omp target update from(s1.data[0:s1.len], s2.data[0:4:2]) // OK
  {}

  // Invalid stride in one of multiple sections
  #pragma omp target update from(s1.data[0:3:4], s2.data[0:2:0]) // expected-error {{section stride is evaluated to a non-positive value 0}}

  // Missing colon
  #pragma omp target update from(s1.data[0:4:2], s2.data[0:3 4]) // expected-error {{expected ']'}} expected-note {{to match this '['}}

  return 0;
}

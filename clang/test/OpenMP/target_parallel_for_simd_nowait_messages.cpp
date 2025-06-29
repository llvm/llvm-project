// RUN: %clang_cc1 -verify -fopenmp %s -Wuninitialized

// RUN: %clang_cc1 -verify -fopenmp-simd %s -Wuninitialized

void foo() {
}

int main(int argc, char **argv) {
  int i;
  #pragma omp target parallel for simd nowait( //  expected-error {{expected expression}} //  expected-error {{expected ')'}} // expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();
  #pragma omp target parallel for simd nowait (argc)) // expected-warning {{extra tokens at the end of '#pragma omp target parallel for simd' are ignored}}
  for (i = 0; i < argc; ++i) foo();
  #pragma omp target parallel for simd nowait device (-10u)
  for (i = 0; i < argc; ++i) foo();
  #pragma omp target parallel for simd nowait (3.14) device (-10u) // expected-error {{arguments of OpenMP clause 'nowait' with bitwise operators cannot be of floating type}}
  for (i = 0; i < argc; ++i) foo();

  return 0;
}

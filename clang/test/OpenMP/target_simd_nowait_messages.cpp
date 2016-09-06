// RUN: %clang_cc1 -verify -fopenmp %s

void foo() {
}

int main(int argc, char **argv) {
  int i;
  #pragma omp target simd nowait( // expected-warning {{extra tokens at the end of '#pragma omp target simd' are ignored}}
  for (i = 0; i < argc; ++i) foo();
  #pragma omp target simd nowait (argc)) // expected-warning {{extra tokens at the end of '#pragma omp target simd' are ignored}}
  for (i = 0; i < argc; ++i) foo();
  #pragma omp target simd nowait device (-10u)
  for (i = 0; i < argc; ++i) foo();
  #pragma omp target simd nowait (3.14) device (-10u) // expected-warning {{extra tokens at the end of '#pragma omp target simd' are ignored}}
  for (i = 0; i < argc; ++i) foo();

  return 0;
}

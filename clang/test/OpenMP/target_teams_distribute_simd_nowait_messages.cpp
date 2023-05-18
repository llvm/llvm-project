// RUN: %clang_cc1 -fsyntax-only -verify -fopenmp %s -Wuninitialized

// RUN: %clang_cc1 -fsyntax-only -verify -fopenmp-simd %s -Wuninitialized

void foo() {
}

int main(int argc, char **argv) {
  int i;
#pragma omp target teams distribute simd nowait( // expected-warning {{extra tokens at the end of '#pragma omp target teams distribute simd' are ignored}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute simd nowait (argc)) // expected-warning {{extra tokens at the end of '#pragma omp target teams distribute simd' are ignored}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute simd nowait device (-10u)
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute simd nowait (3.14) device (-10u) // expected-warning {{extra tokens at the end of '#pragma omp target teams distribute simd' are ignored}}
  for (i = 0; i < argc; ++i) foo();

  return 0;
}

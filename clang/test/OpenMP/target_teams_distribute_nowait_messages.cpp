// RUN: %clang_cc1 -verify -fopenmp -o - %s

// RUN: %clang_cc1 -verify -fopenmp-simd -o - %s

void foo() {
}

int main(int argc, char **argv) {
  int i;
#pragma omp target teams distribute nowait( // expected-warning {{extra tokens at the end of '#pragma omp target teams distribute' are ignored}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute nowait (argc)) // expected-warning {{extra tokens at the end of '#pragma omp target teams distribute' are ignored}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute nowait device (-10u)
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute nowait (3.14) device (-10u) // expected-warning {{extra tokens at the end of '#pragma omp target teams distribute' are ignored}}
  for (i = 0; i < argc; ++i) foo();

  return 0;
}

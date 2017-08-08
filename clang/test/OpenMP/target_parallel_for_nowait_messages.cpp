// RUN: %clang_cc1 -triple x86_64-apple-macos10.7.0 -verify -fopenmp -ferror-limit 100 -o - %s

void foo() {
}

int main(int argc, char **argv) {
  int i;
  #pragma omp target parallel for nowait( // expected-warning {{extra tokens at the end of '#pragma omp target parallel for' are ignored}}
  for (i = 0; i < argc; ++i) foo();
  #pragma omp target parallel for nowait (argc)) // expected-warning {{extra tokens at the end of '#pragma omp target parallel for' are ignored}}
  for (i = 0; i < argc; ++i) foo();
  #pragma omp target parallel for nowait device (-10u)
  for (i = 0; i < argc; ++i) foo();
  #pragma omp target parallel for nowait (3.14) device (-10u) // expected-warning {{extra tokens at the end of '#pragma omp target parallel for' are ignored}}
  for (i = 0; i < argc; ++i) foo();

  return 0;
}

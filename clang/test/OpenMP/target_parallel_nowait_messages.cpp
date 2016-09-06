// RUN: %clang_cc1 -triple x86_64-apple-macos10.7.0 -verify -fopenmp -ferror-limit 100 -o - %s

void foo() {
}

int main(int argc, char **argv) {
  #pragma omp target parallel nowait( // expected-warning {{extra tokens at the end of '#pragma omp target parallel' are ignored}}
  foo();
  #pragma omp target parallel nowait (argc)) // expected-warning {{extra tokens at the end of '#pragma omp target parallel' are ignored}}
  foo();
  #pragma omp target parallel nowait device (-10u)
  foo();
  #pragma omp target parallel nowait (3.14) device (-10u) // expected-warning {{extra tokens at the end of '#pragma omp target parallel' are ignored}}
  foo();

  return 0;
}

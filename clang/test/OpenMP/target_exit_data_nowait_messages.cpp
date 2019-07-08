// RUN: %clang_cc1 -triple x86_64-apple-macos10.7.0 -verify -fopenmp -ferror-limit 100 -o - %s -Wuninitialized

// RUN: %clang_cc1 -triple x86_64-apple-macos10.7.0 -verify -fopenmp-simd -ferror-limit 100 -o - %s -Wuninitialized

int main(int argc, char **argv) {
  int i;

  #pragma omp nowait target exit data map(from: i) // expected-error {{expected an OpenMP directive}}
  #pragma omp target nowait exit data map(from: i) // expected-warning {{extra tokens at the end of '#pragma omp target' are ignored}}
  #pragma omp target exit nowait data map(from: i) // expected-error {{expected an OpenMP directive}}
  #pragma omp target exit data nowait() map(from: i) // expected-warning {{extra tokens at the end of '#pragma omp target exit data' are ignored}} expected-error {{expected at least one 'map' clause for '#pragma omp target exit data'}}
  #pragma omp target exit data map(from: i) nowait( // expected-warning {{extra tokens at the end of '#pragma omp target exit data' are ignored}}
  #pragma omp target exit data map(from: i) nowait (argc)) // expected-warning {{extra tokens at the end of '#pragma omp target exit data' are ignored}}
  #pragma omp target exit data map(from: i) nowait device (-10u)
  #pragma omp target exit data map(from: i) nowait (3.14) device (-10u) // expected-warning {{extra tokens at the end of '#pragma omp target exit data' are ignored}}
  #pragma omp target exit data map(from: i) nowait nowait // expected-error {{directive '#pragma omp target exit data' cannot contain more than one 'nowait' clause}}
  #pragma omp target exit data nowait map(from: i) nowait // expected-error {{directive '#pragma omp target exit data' cannot contain more than one 'nowait' clause}}
  return 0;
}

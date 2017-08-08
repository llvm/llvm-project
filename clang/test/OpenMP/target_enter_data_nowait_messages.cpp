// RUN: %clang_cc1 -triple x86_64-apple-macos10.7.0 -verify -fopenmp -ferror-limit 100 -o - %s

int main(int argc, char **argv) {
  int i;

  #pragma omp nowait target enter data map(to: i) // expected-error {{expected an OpenMP directive}}
  #pragma omp target nowait enter data map(to: i) // expected-warning {{extra tokens at the end of '#pragma omp target' are ignored}}
  #pragma omp target enter nowait data map(to: i) // expected-error {{expected an OpenMP directive}}
  #pragma omp target enter data nowait() map(to: i) // expected-warning {{extra tokens at the end of '#pragma omp target enter data' are ignored}} expected-error {{expected at least one 'map' clause for '#pragma omp target enter data'}}
  #pragma omp target enter data map(to: i) nowait( // expected-warning {{extra tokens at the end of '#pragma omp target enter data' are ignored}}
  #pragma omp target enter data map(to: i) nowait (argc)) // expected-warning {{extra tokens at the end of '#pragma omp target enter data' are ignored}}
  #pragma omp target enter data map(to: i) nowait device (-10u)
  #pragma omp target enter data map(to: i) nowait (3.14) device (-10u) // expected-warning {{extra tokens at the end of '#pragma omp target enter data' are ignored}}
  #pragma omp target enter data map(to: i) nowait nowait // expected-error {{directive '#pragma omp target enter data' cannot contain more than one 'nowait' clause}}
  #pragma omp target enter data nowait map(to: i) nowait // expected-error {{directive '#pragma omp target enter data' cannot contain more than one 'nowait' clause}}
  return 0;
}

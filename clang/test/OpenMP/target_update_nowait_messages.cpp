// RUN: %clang_cc1 -verify -fopenmp -ferror-limit 100 %s

// RUN: %clang_cc1 -verify -fopenmp-simd -ferror-limit 100 %s

int main(int argc, char **argv) {
  int i;

  #pragma omp nowait target update to(i) // expected-error {{expected an OpenMP directive}}
  #pragma omp target nowait update to(i) // expected-error {{unexpected OpenMP clause 'update' in directive '#pragma omp target'}} expected-error {{unexpected OpenMP clause 'to' in directive '#pragma omp target'}}
  {}
  #pragma omp target update nowait() to(i) // expected-warning {{extra tokens at the end of '#pragma omp target update' are ignored}} expected-error {{expected at least one 'to' clause or 'from' clause specified to '#pragma omp target update'}}
  #pragma omp target update to(i) nowait( // expected-warning {{extra tokens at the end of '#pragma omp target update' are ignored}}
  #pragma omp target update to(i) nowait (argc)) // expected-warning {{extra tokens at the end of '#pragma omp target update' are ignored}}
  #pragma omp target update to(i) nowait device (-10u)
  #pragma omp target update to(i) nowait (3.14) device (-10u) // expected-warning {{extra tokens at the end of '#pragma omp target update' are ignored}}
  #pragma omp target update to(i) nowait nowait // expected-error {{directive '#pragma omp target update' cannot contain more than one 'nowait' clause}}
  #pragma omp target update nowait to(i) nowait // expected-error {{directive '#pragma omp target update' cannot contain more than one 'nowait' clause}}
  return 0;
}

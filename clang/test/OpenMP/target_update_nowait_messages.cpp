// RUN: %clang_cc1 -verify -fopenmp -ferror-limit 100 %s -Wuninitialized

// RUN: %clang_cc1 -verify -fopenmp-simd -ferror-limit 100 %s -Wuninitialized

int main(int argc, char **argv) {
  int i, z;

  #pragma omp nowait target update to(i) // expected-error {{expected an OpenMP directive}}
  #pragma omp target nowait update to(i) // expected-error {{unexpected OpenMP clause 'update' in directive '#pragma omp target'}} expected-error {{unexpected OpenMP clause 'to' in directive '#pragma omp target'}}
  {}
  #pragma omp target update nowait() to(i)  //  expected-error {{expected expression}} 
  #pragma omp target update to(i) nowait(  //  expected-error {{expected expression}} //  expected-error {{expected ')'}} // expected-note {{to match this '('}}
  #pragma omp target update to(i) nowait (argc)) // expected-warning {{extra tokens at the end of '#pragma omp target update' are ignored}}
  #pragma omp target update to(i) nowait device (-10u)
  #pragma omp target update to(i) nowait (3.14) device (-10u)
  #pragma omp target update to(i) nowait nowait // expected-error {{directive '#pragma omp target update' cannot contain more than one 'nowait' clause}}
  #pragma omp target update nowait to(i) nowait // expected-error {{directive '#pragma omp target update' cannot contain more than one 'nowait' clause}}
  #pragma omp target update to(i) nowait (argc>> z)
  #pragma omp target update to(i) nowait (argv[1] = 2) // expected-error {{expected ')'}} expected-note {{to match this '('}}
  #pragma omp target update to(i) nowait (argc > 0 ? argv[1] : argv[2])
  #pragma omp target update to(i) nowait (S1) // expected-error {{use of undeclared identifier 'S1'}}
  return 0;
}

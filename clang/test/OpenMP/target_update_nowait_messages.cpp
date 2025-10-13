// RUN: %clang_cc1 -triple x86_64-apple-macos10.7.0 -fopenmp -fopenmp-version=45 -verify=expected,omp-52-and-earlier -ferror-limit 100 -o - %s -Wuninitialized
// RUN: %clang_cc1 -triple x86_64-apple-macos10.7.0 -fopenmp -fopenmp-version=51 -verify=expected,omp-52-and-earlier -ferror-limit 100 -o - %s -Wuninitialized
// RUN: %clang_cc1 -triple x86_64-apple-macos10.7.0 -fopenmp -fopenmp-version=52 -verify=expected,omp-52-and-earlier -ferror-limit 100 -o - %s -Wuninitialized
// RUN: %clang_cc1 -triple x86_64-apple-macos10.7.0 -fopenmp -fopenmp-version=60 -verify=expected,omp-60-and-later   -ferror-limit 100 -o - %s -Wuninitialized

// RUN: %clang_cc1 -triple x86_64-apple-macos10.7.0 -fopenmp-simd -fopenmp-version=45 -verify=expected,omp-52-and-earlier  -ferror-limit 100 -o - %s -Wuninitialized
// RUN: %clang_cc1 -triple x86_64-apple-macos10.7.0 -fopenmp-simd -fopenmp-version=51 -verify=expected,omp-52-and-earlier  -ferror-limit 100 -o - %s -Wuninitialized
// RUN: %clang_cc1 -triple x86_64-apple-macos10.7.0 -fopenmp-simd -fopenmp-version=52 -verify=expected,omp-52-and-earlier  -ferror-limit 100 -o - %s -Wuninitialized
// RUN: %clang_cc1 -triple x86_64-apple-macos10.7.0 -fopenmp-simd -fopenmp-version=60 -verify=expected,omp-60-and-later    -ferror-limit 100 -o - %s -Wuninitialized

int main(int argc, char **argv) {
  int i, z;

  #pragma omp nowait target update to(i) // expected-error {{expected an OpenMP directive}}
  #pragma omp target nowait update to(i) // expected-error {{unexpected OpenMP clause 'update' in directive '#pragma omp target'}} expected-error {{unexpected OpenMP clause 'to' in directive '#pragma omp target'}}
  {}
  #pragma omp target update to(i) nowait()  // omp-52-and-earlier-warning {{extra tokens at the end of '#pragma omp target update' are ignored}} omp-60-and-later-error {{expected expression}}
  #pragma omp target update to(i) nowait(  // omp-52-and-earlier-warning {{extra tokens at the end of '#pragma omp target update' are ignored}} omp-60-and-later-error {{expected expression}} omp-60-and-later-error {{expected ')'}} omp-60-and-later-note {{to match this '('}}
  #pragma omp target update to(i) nowait (argc)) // omp-52-and-earlier-warning {{extra tokens at the end of '#pragma omp target update' are ignored}} omp-60-and-later-warning {{extra tokens at the end of '#pragma omp target_update' are ignored}}
  #pragma omp target update to(i) nowait device (-10u)
  #pragma omp target update to(i) nowait (3.14) device (-10u) // omp-52-and-earlier-warning {{extra tokens at the end of '#pragma omp target update' are ignored}}
  #pragma omp target update to(i) nowait nowait // omp-52-and-earlier-error {{directive '#pragma omp target update' cannot contain more than one 'nowait' clause}} omp-60-and-later-error {{directive '#pragma omp target_update' cannot contain more than one 'nowait' clause}}
  #pragma omp target update nowait to(i) nowait // omp-52-and-earlier-error {{directive '#pragma omp target update' cannot contain more than one 'nowait' clause}} omp-60-and-later-error {{directive '#pragma omp target_update' cannot contain more than one 'nowait' clause}}
  #pragma omp target update to(i) nowait (argc>> z) // omp-52-and-earlier-warning {{extra tokens at the end of '#pragma omp target update' are ignored}}
  #pragma omp target update to(i) nowait (argv[1] = 2) // omp-52-and-earlier-warning {{extra tokens at the end of '#pragma omp target update' are ignored}} omp-60-and-later-error {{expected ')'}} omp-60-and-later-note {{to match this '('}}
  #pragma omp target update to(i) nowait (argc > 0 ? argv[1] : argv[2]) // omp-52-and-earlier-warning {{extra tokens at the end of '#pragma omp target update' are ignored}}
  #pragma omp target update to(i) nowait (S1) // omp-52-and-earlier-warning {{extra tokens at the end of '#pragma omp target update' are ignored}} omp-60-and-later-error {{use of undeclared identifier 'S1'}}
  return 0;
}

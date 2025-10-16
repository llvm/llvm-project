// RUN: %clang_cc1 -triple x86_64-apple-macos10.7.0 -fopenmp -fopenmp-version=45 -verify=expected,omp-52-and-earlier -ferror-limit 100 -o - %s -Wuninitialized
// RUN: %clang_cc1 -triple x86_64-apple-macos10.7.0 -fopenmp -fopenmp-version=51 -verify=expected,omp-52-and-earlier -ferror-limit 100 -o - %s -Wuninitialized
// RUN: %clang_cc1 -triple x86_64-apple-macos10.7.0 -fopenmp -fopenmp-version=52 -verify=expected,omp-52-and-earlier -ferror-limit 100 -o - %s -Wuninitialized
// RUN: %clang_cc1 -triple x86_64-apple-macos10.7.0 -fopenmp -fopenmp-version=60 -verify=expected,omp-60-and-later   -ferror-limit 100 -o - %s -Wuninitialized

// RUN: %clang_cc1 -triple x86_64-apple-macos10.7.0 -fopenmp-simd -fopenmp-version=45 -verify=expected,omp-52-and-earlier  -ferror-limit 100 -o - %s -Wuninitialized
// RUN: %clang_cc1 -triple x86_64-apple-macos10.7.0 -fopenmp-simd -fopenmp-version=51 -verify=expected,omp-52-and-earlier  -ferror-limit 100 -o - %s -Wuninitialized
// RUN: %clang_cc1 -triple x86_64-apple-macos10.7.0 -fopenmp-simd -fopenmp-version=52 -verify=expected,omp-52-and-earlier  -ferror-limit 100 -o - %s -Wuninitialized
// RUN: %clang_cc1 -triple x86_64-apple-macos10.7.0 -fopenmp-simd -fopenmp-version=60 -verify=expected,omp-60-and-later    -ferror-limit 100 -o - %s -Wuninitialized

int main(int argc, char **argv) {
  int i;

  #pragma omp nowait target enter data map(to: i) // expected-error {{expected an OpenMP directive}}
  {}
  #pragma omp target nowait foo data map(to: i) // expected-warning {{extra tokens at the end of '#pragma omp target' are ignored}}
  {}
  #pragma omp target nowait enter data map(to: i) // expected-warning {{extra tokens at the end of '#pragma omp target' are ignored}} expected-error {{unexpected OpenMP clause 'enter' in directive '#pragma omp target'}} expected-error {{expected '(' after 'enter'}}
  {}
  #pragma omp target enter nowait data map(to: i) // expected-error {{expected an OpenMP directive}}
  {}
  #pragma omp target enter data map(to: i) nowait() // omp-52-and-earlier-warning {{extra tokens at the end of '#pragma omp target enter data' are ignored}} omp-60-and-later-error {{expected expression}}
  {}
  #pragma omp target enter data map(to: i) nowait( // omp-52-and-earlier-warning {{extra tokens at the end of '#pragma omp target enter data' are ignored}} omp-60-and-later-error {{expected expression}} omp-60-and-later-error {{expected ')'}} omp-60-and-later-note {{to match this '('}}
  {}
  #pragma omp target enter data map(to: i) nowait (argc)) // omp-52-and-earlier-warning {{extra tokens at the end of '#pragma omp target enter data' are ignored}} omp-60-and-later-warning {{extra tokens at the end of '#pragma omp target_enter_data' are ignored}}
  {}
  #pragma omp target enter data map(to: i) nowait device (-10u)
  {}
  #pragma omp target enter data map(to: i) nowait (3.14) device (-10u) // omp-52-and-earlier-warning {{extra tokens at the end of '#pragma omp target enter data' are ignored}}
  {}
  #pragma omp target enter data map(to: i) nowait (argc>> i) // omp-52-and-earlier-warning {{extra tokens at the end of '#pragma omp target enter data' are ignored}}
  {}
  #pragma omp target enter data map(to: i) nowait (argv[1] = 2) // omp-52-and-earlier-warning {{extra tokens at the end of '#pragma omp target enter data' are ignored}} omp-60-and-later-error {{expected ')'}} omp-60-and-later-note {{to match this '('}}
  {}
  #pragma omp target enter data map(to: i) nowait (argc > 0 ? argv[1] : argv[2]) // omp-52-and-earlier-warning {{extra tokens at the end of '#pragma omp target enter data' are ignored}}
  {}
  #pragma omp target enter data map(to: i) nowait (S1) // omp-52-and-earlier-warning {{extra tokens at the end of '#pragma omp target enter data' are ignored}} omp-60-and-later-error {{use of undeclared identifier 'S1'}}
  {}
  #pragma omp target enter data map(to: i) nowait nowait // omp-52-and-earlier-error {{directive '#pragma omp target enter data' cannot contain more than one 'nowait' clause}} omp-60-and-later-error {{directive '#pragma omp target_enter_data' cannot contain more than one 'nowait' clause}}
  {}
  #pragma omp target enter data nowait map(to: i) nowait // omp-52-and-earlier-error {{directive '#pragma omp target enter data' cannot contain more than one 'nowait' clause}} omp-60-and-later-error {{directive '#pragma omp target_enter_data' cannot contain more than one 'nowait' clause}}
  {}
  return 0;
}

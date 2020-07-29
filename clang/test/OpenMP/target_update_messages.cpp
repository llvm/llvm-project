// RUN: %clang_cc1 -verify=expected,omp45 -fopenmp -fopenmp-version=45 -ferror-limit 100 -o - -std=c++11 %s -Wuninitialized
// RUN: %clang_cc1 -verify=expected,omp50 -fopenmp -fopenmp-version=50 -ferror-limit 100 -o - -std=c++11 %s -Wuninitialized

// RUN: %clang_cc1 -verify=expected,omp45 -fopenmp-simd -fopenmp-version=45 -ferror-limit 100 -o - -std=c++11 %s -Wuninitialized
// RUN: %clang_cc1 -verify=expected,omp50 -fopenmp-simd -fopenmp-version=50 -ferror-limit 100 -o - -std=c++11 %s -Wuninitialized

void xxx(int argc) {
  int x; // expected-note {{initialize the variable 'x' to silence this warning}}
#pragma omp target update to(x)
  argc = x; // expected-warning {{variable 'x' is uninitialized when used here}}
}

void foo() {
}

bool foobool(int argc) {
  return argc;
}

struct S1; // Aexpected-note {{declared here}}

template <class T, class S> // Aexpected-note {{declared here}}
int tmain(T argc, S **argv) {
  int n;
  return 0;
}

int main(int argc, char **argv) {
  int m;
  #pragma omp target update // expected-error {{expected at least one 'to' clause or 'from' clause specified to '#pragma omp target update'}}
  #pragma omp target update to(m) { // expected-warning {{extra tokens at the end of '#pragma omp target update' are ignored}}
  #pragma omp target update to(m) ( // expected-warning {{extra tokens at the end of '#pragma omp target update' are ignored}}
  #pragma omp target update to(m) [ // expected-warning {{extra tokens at the end of '#pragma omp target update' are ignored}}
  #pragma omp target update to(m) ] // expected-warning {{extra tokens at the end of '#pragma omp target update' are ignored}}
  #pragma omp target update to(m) ) // expected-warning {{extra tokens at the end of '#pragma omp target update' are ignored}}

  #pragma omp target update from(m) allocate(m) // expected-error {{unexpected OpenMP clause 'allocate' in directive '#pragma omp target update'}}
  {
    foo();
  }

  int iarr[5][5];
#pragma omp target update to(iarr[0:][1:2:-1]) // omp50-error {{section stride is evaluated to a non-positive value -1}} omp45-error {{expected ']'}} omp45-note {{to match this '['}} expected-error {{expected at least one 'to' clause or 'from' clause specified to '#pragma omp target update'}}
  {}
#pragma omp target update from(iarr[0:][1:2:-1]) // omp50-error {{section stride is evaluated to a non-positive value -1}} omp45-error {{expected ']'}} omp45-note {{to match this '['}} expected-error {{expected at least one 'to' clause or 'from' clause specified to '#pragma omp target update'}}

  return tmain(argc, argv);
}

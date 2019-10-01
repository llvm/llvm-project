// RUN: %clang_cc1 -verify -fopenmp -std=c++11 -o - %s

// RUN: %clang_cc1 -verify -fopenmp-simd -std=c++11 -o - %s

void foo() {
}

#pragma omp teams // expected-error {{unexpected OpenMP directive '#pragma omp teams'}}

int main(int argc, char **argv) {
  #pragma omp target
  #pragma omp teams
  f; // expected-error {{use of undeclared identifier 'f'}}
  #pragma omp target
  #pragma omp teams { // expected-warning {{extra tokens at the end of '#pragma omp teams' are ignored}}
  foo();
  #pragma omp target
  #pragma omp teams ( // expected-warning {{extra tokens at the end of '#pragma omp teams' are ignored}}
  foo();
  #pragma omp target
  #pragma omp teams [ // expected-warning {{extra tokens at the end of '#pragma omp teams' are ignored}}
  foo();
  #pragma omp target
  #pragma omp teams ] // expected-warning {{extra tokens at the end of '#pragma omp teams' are ignored}}
  foo();
  #pragma omp target
  #pragma omp teams ) // expected-warning {{extra tokens at the end of '#pragma omp teams' are ignored}}
  foo();
  #pragma omp target
  #pragma omp teams } // expected-warning {{extra tokens at the end of '#pragma omp teams' are ignored}}
  foo();
  #pragma omp target
  #pragma omp teams
  foo();
  // expected-warning@+2 {{extra tokens at the end of '#pragma omp teams' are ignored}}
  #pragma omp target
  #pragma omp teams unknown()
  foo();
  L1:
    foo();
  #pragma omp target
  #pragma omp teams
  ;
  #pragma omp target
  #pragma omp teams
  {
    goto L1; // expected-error {{use of undeclared label 'L1'}}
    argc++;
  }

  for (int i = 0; i < 10; ++i) {
    switch(argc) {
     case (0):
      #pragma omp target
      #pragma omp teams
      {
        foo();
        break; // expected-error {{'break' statement not in loop or switch statement}}
        continue; // expected-error {{'continue' statement not in loop statement}}
      }
      default:
       break;
    }
  }
  #pragma omp target
  #pragma omp teams default(none) // expected-note {{explicit data sharing attribute requested here}}
  ++argc; // expected-error {{variable 'argc' must have explicitly specified data sharing attributes}}

  goto L2; // expected-error {{use of undeclared label 'L2'}}
  #pragma omp target
  #pragma omp teams
  L2:
  foo();
  #pragma omp target
  #pragma omp teams
  {
    return 1; // expected-error {{cannot return from OpenMP region}}
  }

  [[]] // expected-error {{an attribute list cannot appear here}}
  #pragma omp target
  #pragma omp teams
  for (int n = 0; n < 100; ++n) {}

  return 0;
}


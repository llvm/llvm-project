// RUN: %clang_cc1 -verify -fopenmp -ferror-limit 100 -std=c++11 -o - %s

// RUN: %clang_cc1 -verify -fopenmp-simd -ferror-limit 100 -std=c++11 -o - %s

void foo() {
}

#pragma omp parallel // expected-error {{unexpected OpenMP directive '#pragma omp parallel'}}

struct S;
S& bar();
int main(int argc, char **argv) {
  S &s = bar();
  #pragma omp parallel
  (void)&s;
  #pragma omp parallel { // expected-warning {{extra tokens at the end of '#pragma omp parallel' are ignored}}
  foo();
  #pragma omp parallel ( // expected-warning {{extra tokens at the end of '#pragma omp parallel' are ignored}}
  foo();
  #pragma omp parallel [ // expected-warning {{extra tokens at the end of '#pragma omp parallel' are ignored}}
  foo();
  #pragma omp parallel ] // expected-warning {{extra tokens at the end of '#pragma omp parallel' are ignored}}
  foo();
  #pragma omp parallel ) // expected-warning {{extra tokens at the end of '#pragma omp parallel' are ignored}}
  foo();
  #pragma omp parallel } // expected-warning {{extra tokens at the end of '#pragma omp parallel' are ignored}}
  foo();
  #pragma omp parallel
  // expected-warning@+1 {{extra tokens at the end of '#pragma omp parallel' are ignored}}
  #pragma omp parallel unknown()
  foo();
  L1:
    foo();
  #pragma omp parallel
  ;
  #pragma omp parallel
  {
    goto L1; // expected-error {{use of undeclared label 'L1'}}
    argc++;
  }

  for (int i = 0; i < 10; ++i) {
    switch(argc) {
     case (0):
      #pragma omp parallel
      {
        foo();
        break; // expected-error {{'break' statement not in loop or switch statement}}
        continue; // expected-error {{'continue' statement not in loop statement}}
      }
      default:
       break;
    }
  }
  #pragma omp parallel default(none)
  ++argc; // expected-error {{variable 'argc' must have explicitly specified data sharing attributes}}

  goto L2; // expected-error {{use of undeclared label 'L2'}}
  #pragma omp parallel
  L2:
  foo();
  #pragma omp parallel
  {
    return 1; // expected-error {{cannot return from OpenMP region}}
  }

  [[]] // expected-error {{an attribute list cannot appear here}}
  #pragma omp parallel
  for (int n = 0; n < 100; ++n) {}

  return 0;
}


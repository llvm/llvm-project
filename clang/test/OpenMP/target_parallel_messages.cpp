// RUN: %clang_cc1 -verify -fopenmp -std=c++11 -o - %s -Wuninitialized
// RUN: not %clang_cc1 -fopenmp -std=c++11 -fopenmp-targets=aaa-bbb-ccc-ddd -o - %s 2>&1 | FileCheck %s

// RUN: %clang_cc1 -verify -fopenmp-simd -std=c++11 -o - %s -Wuninitialized
// CHECK: error: OpenMP target is invalid: 'aaa-bbb-ccc-ddd'

void xxx(int argc) {
  int x; // expected-note {{initialize the variable 'x' to silence this warning}}
#pragma omp target parallel
  argc = x; // expected-warning {{variable 'x' is uninitialized when used here}}
}

void foo() {
}

static int pvt;
#pragma omp threadprivate(pvt)

#pragma omp target parallel // expected-error {{unexpected OpenMP directive '#pragma omp target parallel'}}

int main(int argc, char **argv) {
  #pragma omp target parallel { // expected-warning {{extra tokens at the end of '#pragma omp target parallel' are ignored}}
  foo();
  #pragma omp target parallel ( // expected-warning {{extra tokens at the end of '#pragma omp target parallel' are ignored}}
  foo();
  #pragma omp target parallel [ // expected-warning {{extra tokens at the end of '#pragma omp target parallel' are ignored}}
  foo();
  #pragma omp target parallel ] // expected-warning {{extra tokens at the end of '#pragma omp target parallel' are ignored}}
  foo();
  #pragma omp target parallel ) // expected-warning {{extra tokens at the end of '#pragma omp target parallel' are ignored}}
  foo();
  #pragma omp target parallel } // expected-warning {{extra tokens at the end of '#pragma omp target parallel' are ignored}}
  foo();
  #pragma omp target parallel
  foo();
  // expected-warning@+1 {{extra tokens at the end of '#pragma omp target parallel' are ignored}}
  #pragma omp target parallel unknown()
  foo();
  L1:
    foo();
  #pragma omp target parallel
  ;
  #pragma omp target parallel
  {
    goto L1; // expected-error {{use of undeclared label 'L1'}}
    argc++;
  }

  for (int i = 0; i < 10; ++i) {
    switch(argc) {
     case (0):
      #pragma omp target parallel
      {
        foo();
        break; // expected-error {{'break' statement not in loop or switch statement}}
        continue; // expected-error {{'continue' statement not in loop statement}}
      }
      default:
       break;
    }
  }

  goto L2; // expected-error {{use of undeclared label 'L2'}}
  #pragma omp target parallel
  L2:
  foo();
  #pragma omp target parallel
  {
    return 1; // expected-error {{cannot return from OpenMP region}}
  }

  [[]] // expected-error {{an attribute list cannot appear here}}
  #pragma omp target parallel
  for (int n = 0; n < 100; ++n) {}

  #pragma omp target parallel copyin(pvt) // expected-error {{unexpected OpenMP clause 'copyin' in directive '#pragma omp target parallel'}}
  foo();

  #pragma omp target parallel
  {
#pragma omp cancel // expected-error {{one of 'for', 'parallel', 'sections' or 'taskgroup' is expected}}
#pragma omp cancellation point // expected-error {{one of 'for', 'parallel', 'sections' or 'taskgroup' is expected}}
#pragma omp cancel for // expected-error {{region cannot be closely nested inside 'target parallel' region}}
#pragma omp cancellation point for // expected-error {{region cannot be closely nested inside 'target parallel' region}}
#pragma omp cancel sections // expected-error {{region cannot be closely nested inside 'target parallel' region}}
#pragma omp cancellation point sections // expected-error {{region cannot be closely nested inside 'target parallel' region}}
#pragma omp cancel taskgroup // expected-error {{region cannot be closely nested inside 'target parallel' region}}
#pragma omp cancellation point taskgroup // expected-error {{region cannot be closely nested inside 'target parallel' region}}
#pragma omp cancel parallel
#pragma omp cancellation point parallel
  }

  return 0;
}


// RUN: %clang_cc1 -verify -fopenmp -std=c++11 -o - %s
// RUN: not %clang_cc1 -fopenmp -std=c++11 -fopenmp-targets=aaa-bbb-ccc-ddd -o - %s 2>&1 | FileCheck %s
// CHECK: error: OpenMP target is invalid: 'aaa-bbb-ccc-ddd'
// RUN: not %clang_cc1 -fopenmp -std=c++11 -triple nvptx64-nvidia-cuda -o - %s 2>&1 | FileCheck --check-prefix CHECK-UNSUPPORTED-HOST-TARGET %s
// RUN: not %clang_cc1 -fopenmp -std=c++11 -triple nvptx-nvidia-cuda -o - %s 2>&1 | FileCheck --check-prefix CHECK-UNSUPPORTED-HOST-TARGET %s
// CHECK-UNSUPPORTED-HOST-TARGET: error: The target '{{nvptx64-nvidia-cuda|nvptx-nvidia-cuda}}' is not a supported OpenMP host target.

void foo() {
}

#pragma omp target // expected-error {{unexpected OpenMP directive '#pragma omp target'}}

int main(int argc, char **argv) {
  #pragma omp target { // expected-warning {{extra tokens at the end of '#pragma omp target' are ignored}}
  foo();
  #pragma omp target ( // expected-warning {{extra tokens at the end of '#pragma omp target' are ignored}}
  foo();
  #pragma omp target [ // expected-warning {{extra tokens at the end of '#pragma omp target' are ignored}}
  foo();
  #pragma omp target ] // expected-warning {{extra tokens at the end of '#pragma omp target' are ignored}}
  foo();
  #pragma omp target ) // expected-warning {{extra tokens at the end of '#pragma omp target' are ignored}}
  foo();
  #pragma omp target } // expected-warning {{extra tokens at the end of '#pragma omp target' are ignored}}
  foo();
  #pragma omp target
  foo();
  // expected-warning@+1 {{extra tokens at the end of '#pragma omp target' are ignored}}
  #pragma omp target unknown()
  foo();
  L1:
    foo();
  #pragma omp target
  ;
  #pragma omp target
  {
    goto L1; // expected-error {{use of undeclared label 'L1'}}
    argc++;
  }

  for (int i = 0; i < 10; ++i) {
    switch(argc) {
     case (0):
      #pragma omp target
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
  #pragma omp target
  L2:
  foo();
  #pragma omp target
  {
    return 1; // expected-error {{cannot return from OpenMP region}}
  }

  [[]] // expected-error {{an attribute list cannot appear here}}
  #pragma omp target
  for (int n = 0; n < 100; ++n) {}

  return 0;
}


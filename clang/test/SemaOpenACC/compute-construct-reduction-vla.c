// RUN: %clang_cc1 %s -fopenacc -verify

// Regression test for llvm/llvm-project#199162:
// Variable-length arrays cannot be used in OpenACC 'reduction' clauses.

void vla_reduction(int n) {
  int arr[n];
  // expected-error@+1{{variable length array cannot be used in OpenACC 'reduction' clause}}
#pragma acc parallel reduction(+ : arr)
  while (1)
    ;
}

void vla_reduction_serial(int n) {
  int arr[n];
  // expected-error@+1{{variable length array cannot be used in OpenACC 'reduction' clause}}
#pragma acc serial reduction(& : arr)
  while (1)
    ;
}

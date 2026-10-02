// RUN: %clang_cc1 %s -fopenacc -verify

// Regression test for llvm/llvm-project#199162:
// Variable-length arrays cannot be used in OpenACC 'reduction' clauses.

void vla_reduction(int n) {
  int arr[n];
  // expected-warning@+4{{variable of array type 'int[n]' referenced in OpenACC 'reduction' clause does not have constant bounds}}
  // expected-error@+3{{invalid type 'int[n]' used in OpenACC 'reduction' variable reference; type is not an array with constant length}}
  // expected-note@+2{{used as element type of array type 'int[n]'}}
  // expected-note@+1{{OpenACC 'reduction' variable reference must be a scalar variable or a composite of scalars, or an array, sub-array, or element of scalar types}}
#pragma acc parallel reduction(+ : arr)
  while (1)
    ;
}

void vla_reduction_serial(int n) {
  int arr[n];
  // expected-warning@+4{{variable of array type 'int[n]' referenced in OpenACC 'reduction' clause does not have constant bounds}}
  // expected-error@+3{{invalid type 'int[n]' used in OpenACC 'reduction' variable reference; type is not an array with constant length}}
  // expected-note@+2{{used as element type of array type 'int[n]'}}
  // expected-note@+1{{OpenACC 'reduction' variable reference must be a scalar variable or a composite of scalars, or an array, sub-array, or element of scalar types}}
#pragma acc serial reduction(& : arr)
  while (1)
    ;
}

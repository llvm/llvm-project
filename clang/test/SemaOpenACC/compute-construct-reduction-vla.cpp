// RUN: %clang_cc1 %s -fopenacc -verify -Wno-vla-cxx-extension

// C++ companion to compute-construct-reduction-vla.c.
// Regression test for llvm/llvm-project#199162.

void vla_reduction_cxx(int n) {
  int arr[n];
  // expected-warning@+4{{variable of array type 'int[n]' referenced in OpenACC 'reduction' clause does not have constant bounds}}
  // expected-error@+3{{invalid type 'int[n]' used in OpenACC 'reduction' variable reference; type is not an array with constant length}}
  // expected-note@+2{{used as element type of array type 'int[n]'}}
  // expected-note@+1{{OpenACC 'reduction' variable reference must be a scalar variable or a composite of scalars, or an array, sub-array, or element of scalar types}}
#pragma acc parallel reduction(+ : arr)
  while (1)
    ;
}

template <int Pad>
void vla_reduction_template(int n) {
  int arr[n + Pad];
  // expected-error@+3{{invalid type 'int[n + Pad]' used in OpenACC 'reduction' variable reference; type is not an array with constant length}}
  // expected-note@+2{{used as element type of array type 'int[n + Pad]'}}
  // expected-note@+1{{OpenACC 'reduction' variable reference must be a scalar variable or a composite of scalars, or an array, sub-array, or element of scalar types}}
#pragma acc parallel reduction(+ : arr)
  while (1)
    ;
}

void instantiate_template(int n) {
  vla_reduction_template<1>(n);
}

void vla_reduction_combined(int n) {
  int arr[n];
  // expected-warning@+4{{variable of array type 'int[n]' referenced in OpenACC 'reduction' clause does not have constant bounds}}
  // expected-error@+3{{invalid type 'int[n]' used in OpenACC 'reduction' variable reference; type is not an array with constant length}}
  // expected-note@+2{{used as element type of array type 'int[n]'}}
  // expected-note@+1{{OpenACC 'reduction' variable reference must be a scalar variable or a composite of scalars, or an array, sub-array, or element of scalar types}}
#pragma acc parallel loop reduction(+ : arr)
  for (int k = 0; k < 10; ++k)
    ;
}

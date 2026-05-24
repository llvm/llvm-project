// RUN: %clang_cc1 %s -fopenacc -verify -Wno-vla-cxx-extension

// C++ companion to compute-construct-reduction-vla.c.
// Regression test for llvm/llvm-project#199162.

void vla_reduction_cxx(int n) {
  int arr[n];
  // expected-error@+1{{variable length array cannot be used in OpenACC 'reduction' clause}}
#pragma acc parallel reduction(+ : arr)
  while (1)
    ;
}

void vla_reduction_reference(int n) {
  int storage[n];
  int(&arr)[n] = storage;
  // expected-error@+1{{variable length array cannot be used in OpenACC 'reduction' clause}}
#pragma acc parallel reduction(+ : arr)
  while (1)
    ;
}

template <int Pad>
void vla_reduction_template(int n) {
  int arr[n + Pad];
  // expected-error@+1{{variable length array cannot be used in OpenACC 'reduction' clause}}
#pragma acc parallel reduction(+ : arr)
  while (1)
    ;
}

void instantiate_template(int n) {
  vla_reduction_template<1>(n); // expected-note{{in instantiation of function template specialization}}
}

void vla_reduction_combined(int n) {
  int arr[n];
  // expected-error@+1{{variable length array cannot be used in OpenACC 'reduction' clause}}
#pragma acc parallel loop reduction(+ : arr)
  for (int k = 0; k < 10; ++k)
    ;
}

// RUN: %clang_cc1 -fsyntax-only -verify=host,com -x hip %s
// RUN: %clang_cc1 -fsyntax-only -fcuda-is-device -verify=dev,com -x hip %s

#include "Inputs/cuda.h"

template<typename T>
__device__ int fun1(T x) {
  // Check type-dependent constant is allowed in initializer.
  static __device__ int a = sizeof(x);
  static __device__ int b = x;
  // com-error@-1 {{dynamic initialization is not supported for __device__, __constant__, __shared__, and __managed__ variables}}
  return  a + b;
}

__device__ int fun1_caller() {
  return fun1(1);
  // com-note@-1 {{in instantiation of function template specialization 'fun1<int>' requested here}}
}

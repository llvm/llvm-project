// Verifies correct inheritance of target attributes during template
// instantiation and specialization.

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -fsyntax-only -fcuda-is-device -verify %s

#include "Inputs/cuda.h"

// Function must inherit target attributes during instantiation, but not during
// specialization.
template <typename T> __host__ __device__ T function_template(const T &a);

// Specialized functions have their own attributes.
// expected-note@+1 {{candidate function not viable: call to __host__ function from __device__ function}}
template <> __host__ float function_template<float>(const float &from);

// expected-note@+1 {{candidate function not viable: call to __device__ function from __host__ function}}
template <> __device__ double function_template<double>(const double &from);

__host__ void hf() {
  function_template<float>(1.0f); // OK. Specialization is __host__.
  function_template<double>(2.0); // expected-error {{no matching function for call to 'function_template'}}
  function_template(1);           // OK. Instantiated function template is HD.
}
__device__ void df() {
  function_template<float>(3.0f); // expected-error {{no matching function for call to 'function_template'}}
  function_template<double>(4.0); // OK. Specialization is __device__.
  function_template(1);           // OK. Instantiated function template is HD.
}

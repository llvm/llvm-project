// RUN: %clang_cc1 -fsyntax-only -Wundefined-internal -verify %s
// RUN: %clang_cc1 -fsyntax-only -Wundefined-internal -fcuda-is-device -verify %s

// RUN: %clang_cc1 -fsyntax-only -Wundefined-internal -fgpu-rdc -verify=rdc %s
// RUN: %clang_cc1 -fsyntax-only -Wundefined-internal -fcuda-is-device -fgpu-rdc -verify=rdc %s

// Most of these declarations are fine in separate compilation mode.

#include "Inputs/cuda.h"

__device__ void foo() {
  extern __shared__ int x; // expected-error {{__shared__ variable 'x' cannot be 'extern'}}
  extern __shared__ int arr[];  // ok
  extern __shared__ int arr0[0]; // expected-error {{__shared__ variable 'arr0' cannot be 'extern'}}
  extern __shared__ int arr1[1]; // expected-error {{__shared__ variable 'arr1' cannot be 'extern'}}
  extern __shared__ int* ptr ; // expected-error {{__shared__ variable 'ptr' cannot be 'extern'}}
}

__host__ __device__ void bar() {
  extern __shared__ int arr[];  // ok
  extern __shared__ int arr0[0]; // expected-error {{__shared__ variable 'arr0' cannot be 'extern'}}
  extern __shared__ int arr1[1]; // expected-error {{__shared__ variable 'arr1' cannot be 'extern'}}
  extern __shared__ int* ptr ; // expected-error {{__shared__ variable 'ptr' cannot be 'extern'}}
}

extern __shared__ int global; // expected-error {{__shared__ variable 'global' cannot be 'extern'}}
extern __shared__ int global_arr[]; // ok
extern __shared__ int global_arr1[1]; // expected-error {{__shared__ variable 'global_arr1' cannot be 'extern'}}

// Check that, iff we're not in rdc mode, extern __shared__ can appear in an
// anonymous namespace / in a static function without generating a warning
// about a variable with internal linkage but no definition
// (-Wundefined-internal).
namespace {
extern __shared__ int global_arr[]; // rdc-warning {{has internal linkage but is not defined}}
__global__ void in_anon_ns() {
  extern __shared__ int local_arr[]; // rdc-warning {{has internal linkage but is not defined}}

  // Touch arrays to generate the warning.
  local_arr[0] = 0;  // rdc-note {{used here}}
  global_arr[0] = 0; // rdc-note {{used here}}
}
} // namespace

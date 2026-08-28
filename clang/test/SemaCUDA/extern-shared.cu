// RUN: %clang_cc1 -fsyntax-only -Wundefined-internal -verify %s
// RUN: %clang_cc1 -fsyntax-only -Wundefined-internal -fcuda-is-device -verify %s

// RUN: %clang_cc1 -fsyntax-only -Wundefined-internal -fgpu-rdc -verify=rdc %s
// RUN: %clang_cc1 -fsyntax-only -Wundefined-internal -fcuda-is-device -fgpu-rdc -verify=rdc %s

// RUN: %clang_cc1 -fsyntax-only -Wcuda-extern-shared -verify=warn %s
// RUN: %clang_cc1 -fsyntax-only -Wcuda-extern-shared -fcuda-is-device -verify=warn %s

// The warning is off by default, so the first two RUN lines expect no diagnostics.
// expected-no-diagnostics

// Most of these declarations are fine in separate compilation mode.

#include "Inputs/cuda.h"

__device__ void foo() {
  // Scalar types.
  extern __shared__ int x; // warn-warning {{'extern __shared__' variable 'x' is not an incomplete array type}}
  extern __shared__ float f; // warn-warning {{'extern __shared__' variable 'f' is not an incomplete array type}}
  extern __shared__ double d; // warn-warning {{'extern __shared__' variable 'd' is not an incomplete array type}}
  extern __shared__ char c; // warn-warning {{'extern __shared__' variable 'c' is not an incomplete array type}}

  // Incomplete array (the only standard-conforming form).
  extern __shared__ int arr[];  // ok
  extern __shared__ float farr[]; // ok
  extern __shared__ double darr[]; // ok

  // Fixed-size arrays and pointers are not incomplete arrays.
  extern __shared__ int arr0[0]; // warn-warning {{'extern __shared__' variable 'arr0' is not an incomplete array type}}
  extern __shared__ int arr1[1]; // warn-warning {{'extern __shared__' variable 'arr1' is not an incomplete array type}}
  extern __shared__ float farr1[4]; // warn-warning {{'extern __shared__' variable 'farr1' is not an incomplete array type}}
  extern __shared__ int* ptr; // warn-warning {{'extern __shared__' variable 'ptr' is not an incomplete array type}}
}

__host__ __device__ void bar() {
  extern __shared__ int arr[];  // ok
  extern __shared__ int arr0[0]; // warn-warning {{'extern __shared__' variable 'arr0' is not an incomplete array type}}
  extern __shared__ int arr1[1]; // warn-warning {{'extern __shared__' variable 'arr1' is not an incomplete array type}}
  extern __shared__ int* ptr; // warn-warning {{'extern __shared__' variable 'ptr' is not an incomplete array type}}
}

extern __shared__ int global; // warn-warning {{'extern __shared__' variable 'global' is not an incomplete array type}}
extern __shared__ int global_arr[]; // ok
extern __shared__ int global_arr1[1]; // warn-warning {{'extern __shared__' variable 'global_arr1' is not an incomplete array type}}

// Struct types (common NCCL pattern: overlay dynamic shared memory with a struct).
struct ShmemData { int x; float y; };
extern __shared__ ShmemData shmem; // warn-warning {{'extern __shared__' variable 'shmem' is not an incomplete array type}}
__device__ void use_shmem() {
  shmem.x = 1;
  shmem.y = 2.0f;
}

// Struct incomplete array (standard-conforming alternative).
extern __shared__ ShmemData shmem_arr[]; // ok
__device__ void use_shmem_arr() {
  shmem_arr[0].x = 1;
}

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

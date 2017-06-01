// expected-no-diagnostics

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -fsyntax-only -fcuda-is-device -verify %s

#include "Inputs/cuda.h"

struct S {
  __host__ static void operator delete(void*, size_t) {}
  __device__ static void operator delete(void*, size_t) {}
};

__host__ __device__ void test(S* s) {
  // This shouldn't be ambiguous -- we call the host overload in host mode and
  // the device overload in device mode.
  delete s;
}

// Code should work with no explicit declarations/definitions of
// allocator functions.
__host__ __device__ void test_default_global_delete_hd(int *ptr) {
  // Again, there should be no ambiguity between which operator delete we call.
  ::delete ptr;
}

__device__ void test_default_global_delete(int *ptr) {
  // Again, there should be no ambiguity between which operator delete we call.
  ::delete ptr;
}
__host__ void test_default_global_delete(int *ptr) {
  // Again, there should be no ambiguity between which operator delete we call.
  ::delete ptr;
}

// It should work with only some of allocators (re-)declared.
__device__ void operator delete(void *ptr);

__host__ __device__ void test_partial_global_delete_hd(int *ptr) {
  // Again, there should be no ambiguity between which operator delete we call.
  ::delete ptr;
}

__device__ void test_partial_global_delete(int *ptr) {
  // Again, there should be no ambiguity between which operator delete we call.
  ::delete ptr;
}
__host__ void test_partial_global_delete(int *ptr) {
  // Again, there should be no ambiguity between which operator delete we call.
  ::delete ptr;
}


// We should be able to define both host and device variants.
__host__ void operator delete(void *ptr) {}
__device__ void operator delete(void *ptr) {}

__host__ __device__ void test_overloaded_global_delete_hd(int *ptr) {
  // Again, there should be no ambiguity between which operator delete we call.
  ::delete ptr;
}

__device__ void test_overloaded_global_delete(int *ptr) {
  // Again, there should be no ambiguity between which operator delete we call.
  ::delete ptr;
}
__host__ void test_overloaded_global_delete(int *ptr) {
  // Again, there should be no ambiguity between which operator delete we call.
  ::delete ptr;
}

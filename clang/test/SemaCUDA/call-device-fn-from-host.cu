// RUN: %clang_cc1 %s --std=c++11 -triple x86_64-unknown-linux -emit-llvm -o - \
// RUN:   -verify -verify-ignore-unexpected=note

// Note: This test won't work with -fsyntax-only, because some of these errors
// are emitted during codegen.

#include "Inputs/cuda.h"

__device__ void device_fn() {}
// expected-note@-1 5 {{'device_fn' declared here}}

struct S {
  __device__ S() {}
  // expected-note@-1 2 {{'S' declared here}}
  __device__ ~S() { device_fn(); }
  // expected-note@-1 {{'~S' declared here}}
  int x;
};

struct T {
  __host__ __device__ void hd() { device_fn(); }
  // expected-error@-1 {{reference to __device__ function 'device_fn' in __host__ __device__ function}}

  // No error; this is (implicitly) inline and is never called, so isn't
  // codegen'ed.
  __host__ __device__ void hd2() { device_fn(); }

  __host__ __device__ void hd3();

  __device__ void d() {}
  // expected-note@-1 {{'d' declared here}}
};

__host__ __device__ void T::hd3() {
  device_fn();
  // expected-error@-1 {{reference to __device__ function 'device_fn' in __host__ __device__ function}}
}

template <typename T> __host__ __device__ void hd2() { device_fn(); }
// expected-error@-1 {{reference to __device__ function 'device_fn' in __host__ __device__ function}}
void host_fn() { hd2<int>(); }

__host__ __device__ void hd() { device_fn(); }
// expected-error@-1 {{reference to __device__ function 'device_fn' in __host__ __device__ function}}

// No error because this is never instantiated.
template <typename T> __host__ __device__ void hd3() { device_fn(); }

__host__ __device__ void local_var() {
  S s;
  // expected-error@-1 {{reference to __device__ function 'S' in __host__ __device__ function}}
}

__host__ __device__ void placement_new(char *ptr) {
  ::new(ptr) S();
  // expected-error@-1 {{reference to __device__ function 'S' in __host__ __device__ function}}
}

__host__ __device__ void explicit_destructor(S *s) {
  s->~S();
  // expected-error@-1 {{reference to __device__ function '~S' in __host__ __device__ function}}
}

__host__ __device__ void hd_member_fn() {
  T t;
  // Necessary to trigger an error on T::hd.  It's (implicitly) inline, so
  // isn't codegen'ed until we call it.
  t.hd();
}

__host__ __device__ void h_member_fn() {
  T t;
  t.d();
  // expected-error@-1 {{reference to __device__ function 'd' in __host__ __device__ function}}
}

__host__ __device__ void fn_ptr() {
  auto* ptr = &device_fn;
  // expected-error@-1 {{reference to __device__ function 'device_fn' in __host__ __device__ function}}
}

template <typename T>
__host__ __device__ void fn_ptr_template() {
  auto* ptr = &device_fn;  // Not an error because the template isn't instantiated.
}

// Launching a kernel from a host function does not result in code generation
// for it, so calling HD function which calls a D function should not trigger
// errors.
static __host__ __device__ void hd_func() { device_fn(); }
__global__ void kernel() { hd_func(); }
void host_func(void) { kernel<<<1, 1>>>(); }

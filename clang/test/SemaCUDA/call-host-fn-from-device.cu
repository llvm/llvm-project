// RUN: %clang_cc1 %s --std=c++11 -triple nvptx-unknown-unknown -fcuda-is-device -emit-llvm -o - -verify

// Note: This test won't work with -fsyntax-only, because some of these errors
// are emitted during codegen.

#include "Inputs/cuda.h"

extern "C" void host_fn() {}
// expected-note@-1 {{'host_fn' declared here}}
// expected-note@-2 {{'host_fn' declared here}}
// expected-note@-3 {{'host_fn' declared here}}
// expected-note@-4 {{'host_fn' declared here}}
// expected-note@-5 {{'host_fn' declared here}}
// expected-note@-6 {{'host_fn' declared here}}

struct S {
  S() {}
  // expected-note@-1 {{'S' declared here}}
  // expected-note@-2 {{'S' declared here}}
  ~S() { host_fn(); }
  // expected-note@-1 {{'~S' declared here}}
  int x;
};

struct T {
  __host__ __device__ void hd() { host_fn(); }
  // expected-error@-1 {{reference to __host__ function 'host_fn' in __host__ __device__ function}}

  // No error; this is (implicitly) inline and is never called, so isn't
  // codegen'ed.
  __host__ __device__ void hd2() { host_fn(); }

  __host__ __device__ void hd3();

  void h() {}
  // expected-note@-1 {{'h' declared here}}
};

__host__ __device__ void T::hd3() {
  host_fn();
  // expected-error@-1 {{reference to __host__ function 'host_fn' in __host__ __device__ function}}
}

template <typename T> __host__ __device__ void hd2() { host_fn(); }
// expected-error@-1 {{reference to __host__ function 'host_fn' in __host__ __device__ function}}
__global__ void kernel() { hd2<int>(); }

__host__ __device__ void hd() { host_fn(); }
// expected-error@-1 {{reference to __host__ function 'host_fn' in __host__ __device__ function}}

template <typename T> __host__ __device__ void hd3() { host_fn(); }
// expected-error@-1 {{reference to __host__ function 'host_fn' in __host__ __device__ function}}
__device__ void device_fn() { hd3<int>(); }

// No error because this is never instantiated.
template <typename T> __host__ __device__ void hd4() { host_fn(); }

__host__ __device__ void local_var() {
  S s;
  // expected-error@-1 {{reference to __host__ function 'S' in __host__ __device__ function}}
}

__host__ __device__ void placement_new(char *ptr) {
  ::new(ptr) S();
  // expected-error@-1 {{reference to __host__ function 'S' in __host__ __device__ function}}
}

__host__ __device__ void explicit_destructor(S *s) {
  s->~S();
  // expected-error@-1 {{reference to __host__ function '~S' in __host__ __device__ function}}
}

__host__ __device__ void hd_member_fn() {
  T t;
  // Necessary to trigger an error on T::hd.  It's (implicitly) inline, so
  // isn't codegen'ed until we call it.
  t.hd();
}

__host__ __device__ void h_member_fn() {
  T t;
  t.h();
  // expected-error@-1 {{reference to __host__ function 'h' in __host__ __device__ function}}
}

__host__ __device__ void fn_ptr() {
  auto* ptr = &host_fn;
  // expected-error@-1 {{reference to __host__ function 'host_fn' in __host__ __device__ function}}
}

template <typename T>
__host__ __device__ void fn_ptr_template() {
  auto* ptr = &host_fn;  // Not an error because the template isn't instantiated.
}

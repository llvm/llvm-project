// RUN: %clang_cc1 %s --std=c++11 -triple nvptx-unknown-unknown -fcuda-is-device \
// RUN:   -emit-llvm -o /dev/null -verify -verify-ignore-unexpected=note

// Note: This test won't work with -fsyntax-only, because some of these errors
// are emitted during codegen.

#include "Inputs/cuda.h"

extern "C" void host_fn() {}
// expected-note@-1 7 {{'host_fn' declared here}}

struct Dummy {};

struct S {
  S() {}
  // expected-note@-1 2 {{'S' declared here}}
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

  void operator+();
  // expected-note@-1 {{'operator+' declared here}}

  void operator-(const T&) {}
  // expected-note@-1 {{'operator-' declared here}}

  operator Dummy() { return Dummy(); }
  // expected-note@-1 {{'operator Dummy' declared here}}

  __host__ void operator delete(void *) { host_fn(); };
  __device__ void operator delete(void*, __SIZE_TYPE__);
};

struct U {
  __device__ void operator delete(void*, __SIZE_TYPE__) = delete;
  __host__ __device__ void operator delete(void*);
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

__host__ __device__ void class_specific_delete(T *t, U *u) {
  delete t; // ok, call sized device delete even though host has preferable non-sized version
  delete u; // ok, call non-sized HD delete rather than sized D delete
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

__host__ __device__ void unaryOp() {
  T t;
  (void) +t; // expected-error {{reference to __host__ function 'operator+' in __host__ __device__ function}}
}

__host__ __device__ void binaryOp() {
  T t;
  (void) (t - t); // expected-error {{reference to __host__ function 'operator-' in __host__ __device__ function}}
}

__host__ __device__ void implicitConversion() {
  T t;
  Dummy d = t; // expected-error {{reference to __host__ function 'operator Dummy' in __host__ __device__ function}}
}

template <typename T>
struct TmplStruct {
  template <typename U> __host__ __device__ void fn() {}
};

template <>
template <>
__host__ __device__ void TmplStruct<int>::fn<int>() { host_fn(); }
// expected-error@-1 {{reference to __host__ function 'host_fn' in __host__ __device__ function}}

__device__ void double_specialization() { TmplStruct<int>().fn<int>(); }

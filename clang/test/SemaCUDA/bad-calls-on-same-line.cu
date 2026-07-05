// RUN: %clang_cc1 -fsyntax-only -verify %s

// The hd function template is instantiated three times.
//
// Two of those instantiations call a device function, which is an error when
// compiling for host.  Clang should report both errors.

#include "Inputs/cuda.h"

template <typename T>
struct Selector {};

template <>
struct Selector<int> {
  __host__ void f() {}
};

template <>
struct Selector<float> {
  __device__ void f() {} // expected-note {{declared here}}
};

template <>
struct Selector<double> {
  __device__ void f() {} // expected-note {{declared here}}
};

template <typename T>
inline __host__ __device__ void hd() {
  Selector<T>().f();
  // expected-error@-1 2 {{reference to __device__ function}}
}

void host_fn() {
  hd<int>();
  hd<double>();
  // expected-note@-1 {{called by 'host_fn'}}
  hd<float>();
  // expected-note@-1 {{called by 'host_fn'}}
}

// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -fcuda-is-device -fsyntax-only -verify -Wno-vla %s

#include "Inputs/cuda.h"

// We should emit an error for hd_fn's use of a VLA.  This would have been
// legal if hd_fn were never codegen'ed on the device, so we should also print
// out a callstack showing how we determine that hd_fn is known-emitted.
//
// Compare to no-call-stack-for-deferred-err.cu.

inline __host__ __device__ void hd_fn(int n);
inline __device__ void device_fn2() { hd_fn(42); } // expected-note {{called by 'device_fn2'}}

__global__ void kernel() { device_fn2(); } // expected-note {{called by 'kernel'}}

inline __host__ __device__ void hd_fn(int n) {
  int vla[n]; // expected-error {{variable-length array}}
}

// A constructor calls the constructors of its bases and members from its
// initializer list.
struct Member {
  __host__ __device__ Member() {
    int n = 42;
    int vla[n]; // expected-error {{variable-length array}}
  }
};
struct HasMember { // expected-note {{called by 'HasMember'}}
  Member m;
};

struct Base {
  __host__ __device__ Base() {
    int n = 42;
    int vla[n]; // expected-error {{variable-length array}}
  }
};
struct Derived : Base {}; // expected-note {{called by 'Derived'}}

__global__ void ctor_kernel() {
  HasMember h; // expected-note {{which is called by 'ctor_kernel'}}
  Derived d;   // expected-note {{which is called by 'ctor_kernel'}}
}

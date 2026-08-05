// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fcuda-is-device -x hip %s \
// RUN:   -fsyntax-only -verify
// RUN: %clang_cc1 -triple x86_64 -x hip %s \
// RUN:   -fsyntax-only -verify=host

// host-no-diagnostics

#include "Inputs/cuda.h"

// Test const var initialized with address of a const var.
// Both are promoted to device side.

namespace Test1 {
const int a = 1;

struct B {
    static const int *const p;
    __device__ static const int *const p2;
};
const int *const B::p = &a;
// Const variable 'a' is treated as __constant__ on device side,
// therefore its address can be used as initializer for another
// device variable.
__device__ const int *const B::p2 = &a;

__device__ void f() {
  int y = a;
  const int *x = B::p;
  const int *z = B::p2;
}
}

// Test const var initialized with address of a non-cost var.
// Neither is promoted to device side.

namespace Test2 {
int a = 1;
// expected-note@-1{{host variable declared here}}

struct B {
    static int *const p;
};
int *const B::p = &a;
// expected-note@-1{{const variable cannot be emitted on device side due to dynamic initialization}}

__device__ void f() {
  int y = a;
  // expected-error@-1{{reference to __host__ variable 'a' in __device__ function}}
  const int *x = B::p;
  // expected-error@-1{{reference to __host__ variable 'p' in __device__ function}}
}
}

// Test device var initialized with address of a non-const host var, __shared var,
// __managed__ var, __device__ var, __constant__ var, texture var, surface var.

namespace Test3 {
struct textureReference {
  int desc;
};

enum ReadMode {
  ElementType = 0,
  NormalizedFloat = 1
};

template <typename T, int dim = 1, enum ReadMode mode = ElementType>
struct __attribute__((device_builtin_texture_type)) texture : public textureReference {
};

struct surfaceReference {
  int desc;
};

template <typename T, int dim = 1>
struct __attribute__((device_builtin_surface_type)) surface : public surfaceReference {
};

// Partial specialization over `void`.
template<int dim>
struct __attribute__((device_builtin_surface_type)) surface<void, dim> : public surfaceReference {
};

texture<float, 2, ElementType> tex;
surface<void, 2> surf;

int a = 1;
__shared__ int b;
__managed__ int c = 1;
__device__ int d = 1;
__constant__ int e = 1;
struct B {
    __device__ static int *const p1;
    __device__ static int *const p2;
    __device__ static int *const p3;
    __device__ static int *const p4;
    __device__ static int *const p5;
    __device__ static texture<float, 2, ElementType> *const p6;
    __device__ static surface<void, 2> *const p7;
};
__device__ int *const B::p1 = &a;
// expected-error@-1{{dynamic initialization is not supported for __device__, __constant__, __shared__, and __managed__ variables}}
__device__ int *const B::p2 = &b;
// expected-error@-1{{dynamic initialization is not supported for __device__, __constant__, __shared__, and __managed__ variables}}
__device__ int *const B::p3 = &c;
// expected-error@-1{{dynamic initialization is not supported for __device__, __constant__, __shared__, and __managed__ variables}}
__device__ int *const B::p4 = &d;
__device__ int *const B::p5 = &e;
__device__ texture<float, 2, ElementType> *const B::p6 = &tex;
__device__ surface<void, 2> *const B::p7 = &surf;
}

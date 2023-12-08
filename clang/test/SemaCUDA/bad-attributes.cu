// Tests handling of CUDA attributes that are bad either because they're
// applied to the wrong sort of thing, or because they're given in illegal
// combinations.
//
// You should be able to run this file through nvcc for compatibility testing.
//
// RUN: %clang_cc1 -fsyntax-only -Wcuda-compat -verify -DEXPECT_INLINE_WARNING %s
// RUN: %clang_cc1 -fcuda-is-device -fsyntax-only -Wcuda-compat -verify %s

#include "Inputs/cuda.h"

// Try applying attributes to functions and variables.  Some should generate
// warnings; others not.
__device__ int a1;
__device__ void a2();
__host__ int b1; // expected-warning {{attribute only applies to functions}}
__host__ void b2();
__constant__ int c1;
__constant__ void c2(); // expected-warning {{attribute only applies to variables}}
__shared__ int d1;
__shared__ void d2(); // expected-warning {{attribute only applies to variables}}
__global__ int e1; // expected-warning {{attribute only applies to functions}}
__global__ void e2();

// Try all pairs of attributes which can be present on a function or a
// variable.  Check both orderings of the attributes, as that can matter in
// clang.
__device__ __host__ void z1();
__device__ __constant__ int z2;
__device__ __shared__ int z3;
__device__ __global__ void z4(); // expected-error {{attributes are not compatible}}
// expected-note@-1 {{conflicting attribute is here}}

__host__ __device__ void z5();
__host__ __global__ void z6();  // expected-error {{attributes are not compatible}}
// expected-note@-1 {{conflicting attribute is here}}

__constant__ __device__ int z7;
__constant__ __shared__ int z8;  // expected-error {{attributes are not compatible}}
// expected-note@-1 {{conflicting attribute is here}}

__shared__ __device__ int z9;
__shared__ __constant__ int z10;  // expected-error {{attributes are not compatible}}
// expected-note@-1 {{conflicting attribute is here}}
__constant__ __shared__ int z10a;  // expected-error {{attributes are not compatible}}
// expected-note@-1 {{conflicting attribute is here}}

__global__ __device__ void z11();  // expected-error {{attributes are not compatible}}
// expected-note@-1 {{conflicting attribute is here}}
__global__ __host__ void z12();  // expected-error {{attributes are not compatible}}
// expected-note@-1 {{conflicting attribute is here}}

struct S {
  __global__ void foo() {};  // expected-error {{must be a free function or static member function}}
  __global__ static void bar(); // expected-warning {{kernel function 'bar' is a member function}}
  // Although this is implicitly inline, we shouldn't warn.
  __global__ static void baz() {}; // expected-warning {{kernel function 'baz' is a member function}}
};

__global__ static inline void foobar() {};
#ifdef EXPECT_INLINE_WARNING
// expected-warning@-2 {{ignored 'inline' attribute on kernel function 'foobar'}}
#endif

__constant__ int global_constant;
void host_fn() {
  __constant__ int c; // expected-error {{__constant__, __device__, and __managed__ are not allowed on non-static local variables}}
  __shared__ int s; // expected-error {{__shared__ local variables not allowed in __host__ functions}}
}
__device__ void device_fn() {
  __constant__ int c; // expected-error {{__constant__, __device__, and __managed__ are not allowed on non-static local variables}}
}

typedef __attribute__((device_builtin_surface_type)) unsigned long long s0_ty; // expected-warning {{'device_builtin_surface_type' attribute only applies to classes}}
typedef __attribute__((device_builtin_texture_type)) unsigned long long t0_ty; // expected-warning {{'device_builtin_texture_type' attribute only applies to classes}}

struct __attribute__((device_builtin_surface_type)) s1_ref {}; // expected-error {{illegal device builtin surface reference type 's1_ref' declared here}}
// expected-note@-1 {{'s1_ref' needs to be instantiated from a class template with proper template arguments}}
struct __attribute__((device_builtin_texture_type)) t1_ref {}; // expected-error {{illegal device builtin texture reference type 't1_ref' declared here}}
// expected-note@-1 {{'t1_ref' needs to be instantiated from a class template with proper template arguments}}

template <typename T>
struct __attribute__((device_builtin_surface_type)) s2_cls_template {}; // expected-error {{illegal device builtin surface reference class template 's2_cls_template' declared here}}
// expected-note@-1 {{'s2_cls_template' needs to have exactly 2 template parameters}}
template <typename T>
struct __attribute__((device_builtin_texture_type)) t2_cls_template {}; // expected-error {{illegal device builtin texture reference class template 't2_cls_template' declared here}}
// expected-note@-1 {{'t2_cls_template' needs to have exactly 3 template parameters}}

template <int val, void *ptr>
struct __attribute__((device_builtin_surface_type)) s3_cls_template {}; // expected-error {{illegal device builtin surface reference class template 's3_cls_template' declared here}}
// expected-note@-1 {{the 1st template parameter of 's3_cls_template' needs to be a type}}
// expected-note@-2 {{the 2nd template parameter of 's3_cls_template' needs to be an integer or enum value}}
template <int val, int type, typename T>
struct __attribute__((device_builtin_texture_type)) t3_cls_template {}; // expected-error {{illegal device builtin texture reference class template 't3_cls_template' declared here}}
// expected-note@-1 {{the 1st template parameter of 't3_cls_template' needs to be a type}}
// expected-note@-2 {{the 3rd template parameter of 't3_cls_template' needs to be an integer or enum value}}

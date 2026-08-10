// RUN: %clang_cc1 -fcuda-is-device -fsyntax-only -verify=dev,com %s \
// RUN:   -std=c++11 -fgpu-defer-diag
// RUN: %clang_cc1 -fsyntax-only -verify=host,com %s \
// RUN:   -std=c++11 -fgpu-defer-diag
// RUN: %clang_cc1 -fopenmp -fsyntax-only -verify=host,com %s \
// RUN:   -std=c++11 -fgpu-defer-diag

// With -fgpu-defer-diag, clang defers overloading resolution induced
// diagnostics when the full candidates set include host device
// functions or wrong-sided candidates. This roughly matches nvcc's
// behavior.

#include "Inputs/cuda.h"

// When callee is called by a host function with integer arguments, there is an error for ambiguity.
// It should be deferred since it involves wrong-sided candidates.
__device__ void callee(int);
__host__ void callee(float); // host-note {{candidate function}}
__host__ void callee(double); // host-note {{candidate function}}

// When callee2 is called by a device function without arguments, there is an error for 'no matching function'.
// It should be deferred since it involves wrong-sided candidates.
__host__ void callee2(); // dev-note{{candidate function not viable: call to __host__ function from __device__ function}}

// When callee3 is called by a device function without arguments, there is an error for 'no matching function'.
// It should be deferred since it involves wrong-sided candidates.
__host__ void callee3(); // dev-note{{candidate function not viable: call to __host__ function from __device__ function}}
__device__ void callee3(int); // dev-note{{candidate function not viable: requires 1 argument, but 0 were provided}}

// When callee4 is called by a host or device function without arguments, there is an error for 'no matching function'.
// It should be immediate since it involves no wrong-sided candidates (it is not a viable candiate due to signature).
__host__ void callee4(int); // com-note 2{{candidate function not viable: requires 1 argument, but 0 were provided}}

// When callee5 is called by a host function with integer arguments, there is an error for ambiguity.
// It should be immediate since it involves no wrong-sided candidates.
__host__ void callee5(float); // com-note {{candidate function}}
__host__ void callee5(double); // com-note {{candidate function}}

// When '<<` operator is called by a device function, there is error for 'invalid operands'.
// It should be deferred since it involves wrong-sided candidates.
struct S {
  __host__ S &operator <<(int i); // dev-note {{candidate function not viable}}
};

__host__ void hf() {
 callee(1); // host-error {{call to 'callee' is ambiguous}}
 callee2();
 callee3();
 callee4(); // com-error {{no matching function for call to 'callee4'}}
 callee5(1); // com-error {{call to 'callee5' is ambiguous}}
 S s;
 s << 1;
 undeclared_func(); // com-error {{use of undeclared identifier 'undeclared_func'}}
}

__device__ void df() {
 callee(1);
 callee2(); // dev-error {{no matching function for call to 'callee2'}}
 callee3(); // dev-error {{no matching function for call to 'callee3'}}
 callee4(); // com-error {{no matching function for call to 'callee4'}}
 S s;
 s << 1;    // dev-error {{invalid operands to binary expression}}
}

struct A { int x; typedef int isA; };
struct B { int x; };

// This function is invalid for A and B by SFINAE.
// This fails to substitue for A but no diagnostic
// should be emitted.
template<typename T, typename T::foo* = nullptr>
__host__ __device__ void sfinae(T t) { // host-note {{candidate template ignored: substitution failure [with T = B]}}
  t.x = 1;
}

// This function is defined for A only by SFINAE.
// Calling it with A should succeed, with B should fail.
// The error should not be deferred since it happens in
// file scope.

template<typename T, typename T::isA* = nullptr>
__host__ __device__ void sfinae(T t) { // host-note {{candidate template ignored: substitution failure [with T = B]}}
  t.x = 1;
}

void test_sfinae() {
  sfinae(A());
  sfinae(B()); // host-error{{no matching function for call to 'sfinae'}}
}

// Make sure throw is diagnosed in OpenMP parallel region in host function.
void test_openmp() {
  #pragma omp parallel for
  for (int i = 0; i < 10; i++) {
    throw 1;
  }
}

// If a syntax error causes a function not declared, it cannot
// be deferred.

inline __host__ __device__ void bad_func() { // com-note {{to match this '{'}}
// com-error {{expected '}'}}

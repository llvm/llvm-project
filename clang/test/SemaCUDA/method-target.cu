// RUN: %clang_cc1 -fsyntax-only -verify %s

#include "Inputs/cuda.h"

//------------------------------------------------------------------------------
// Test 1: host method called from device function

struct S1 {
  void method() {} // expected-note {{'method' declared here}}
};

__device__ void foo1(S1& s) {
  s.method(); // expected-error {{reference to __host__ function 'method' in __device__ function}}
}

//------------------------------------------------------------------------------
// Test 2: host method called from device function, for overloaded method

struct S2 {
  void method(int) {} // expected-note {{candidate function not viable: call to __host__ function from __device__ function}}
  void method(float) {} // expected-note {{candidate function not viable: call to __host__ function from __device__ function}}
};

__device__ void foo2(S2& s, int i, float f) {
  s.method(f); // expected-error {{no matching member function}}
}

//------------------------------------------------------------------------------
// Test 3: device method called from host function

struct S3 {
  __device__ void method() {} // expected-note {{'method' declared here}};
};

void foo3(S3& s) {
  s.method(); // expected-error {{reference to __device__ function 'method' in __host__ function}}
}

//------------------------------------------------------------------------------
// Test 4: device method called from host&device function

struct S4 {
  __device__ void method() {}
};

__host__ __device__ void foo4(S4& s) {
  s.method();
}

//------------------------------------------------------------------------------
// Test 5: overloaded operators

struct S5 {
  S5() {}
  S5& operator=(const S5&) {return *this;} // expected-note {{candidate function not viable}}
};

__device__ void foo5(S5& s, S5& t) {
  s = t; // expected-error {{no viable overloaded '='}}
}

//------------------------------------------------------------------------------
// Test 6: call method through pointer

struct S6 {
  void method() {} // expected-note {{'method' declared here}};
};

__device__ void foo6(S6* s) {
  s->method(); // expected-error {{reference to __host__ function 'method' in __device__ function}}
}

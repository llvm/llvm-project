// RUN: %clang_cc1 %s -triple x86_64-linux-unknown -fsyntax-only -o - -verify=expected,host
// RUN: %clang_cc1 %s -fcuda-is-device -triple nvptx -fsyntax-only -o - -verify=expected,device

#include "Inputs/cuda.h"

// Check that we get an error if we try to call a __device__ function from a
// module initializer.

struct S {
  // expected-note@-1 {{candidate constructor (the implicit copy constructor) not viable: requires 1 argument, but 0 were provided}}
  // expected-note@-2 {{candidate constructor (the implicit move constructor) not viable: requires 1 argument, but 0 were provided}}
  __device__ S() {}
  // expected-note@-1 {{candidate constructor not viable: call to __device__ function from __host__ function}}
};

S s;
// expected-error@-1 {{no matching constructor for initialization of 'S'}}

struct T {
  __host__ __device__ T() {}
};
T t;  // No error, this is OK.

struct U {
  // expected-note@-1 {{candidate constructor (the implicit copy constructor) not viable: no known conversion from 'int' to 'const U' for 1st argument}}
  // expected-note@-2 {{candidate constructor (the implicit move constructor) not viable: no known conversion from 'int' to 'U' for 1st argument}}
  __host__ U() {}
  // expected-note@-1 {{candidate constructor not viable: requires 0 arguments, but 1 was provided}}
  __device__ U(int) {}
  // expected-note@-1 {{candidate constructor not viable: call to __device__ function from __host__ function}}
};
U u(42);
// expected-error@-1 {{no matching constructor for initialization of 'U'}}

__device__ int device_fn() { return 42; }
// expected-note@-1 {{candidate function not viable: call to __device__ function from __host__ function}}
int n = device_fn();
// expected-error@-1 {{no matching function for call to 'device_fn'}}

// Check host/device-based overloding resolution in global variable initializer.
double pow(double, double);

__device__ double pow(double, int);

double X = pow(1.0, 1);
__device__ double Y = pow(2.0, 2); // expected-error{{dynamic initialization is not supported for __device__, __constant__, __shared__, and __managed__ variables}}

constexpr double cpow(double, double) { return 1.0; }

constexpr __device__ double cpow(double, int) { return 2.0; }

const double CX = cpow(1.0, 1);
const __device__ double CY = cpow(2.0, 2);

struct A {
  double pow(double, double);

  __device__ double pow(double, int);

  constexpr double cpow(double, double) const { return 1.0; }

  constexpr __device__ double cpow(double, int) const { return 1.0; }

};

A a;
double AX = a.pow(1.0, 1);
__device__ double AY = a.pow(2.0, 2); // expected-error{{dynamic initialization is not supported for __device__, __constant__, __shared__, and __managed__ variables}}

const A ca;
const double CAX = ca.cpow(1.0, 1);
const __device__ double CAY = ca.cpow(2.0, 2);

namespace ns1 {
  // host-note@+3 {{'value_func' declared here}}
  // expected-note@+2 9{{'value_func' declared here}}
  // expected-note@+1 {{candidate function not viable: call to __device__ function from __host__ function}}
__device__ constexpr inline int value_func() {
  return 32;
}
__device__ constexpr inline int another_value_func() {
  return 32;
}
}

namespace ns2 {
  using namespace ns1;
  // diagnosed via overloading.
  constexpr static unsigned var0 = value_func();
  // expected-error@-1 {{no matching function for call to 'value_func'}}

  // diagnosed via SemaCUDA::checkAllowedInitializer
  constexpr static unsigned var1 = ns1::value_func();
  // host-error@-1 {{reference to __device__ function 'value_func' in global initializer}}

  // FIXME: Inconsistency with var1 - non constexpr cases are diagnosed for both host and device.
  static int var2 = 1 + ns1::value_func();
  // expected-error@-1 {{reference to __device__ function 'value_func' in global initializer}}
  static int var3 {ns1::value_func()};
  // expected-error@-1 {{reference to __device__ function 'value_func' in global initializer}}

  struct A {
    unsigned b;
  };
  A b{ns1::value_func()};
  // expected-error@-1 {{reference to __device__ function 'value_func' in global initializer}}

  int foo(int);
  int nested = foo(ns1::value_func());
  // expected-error@-1 {{reference to __device__ function 'value_func' in global initializer}}

  void foobar() {
    // diagnosed via SemaCUDA::checkCall
    static int var2 = 1 + ns1::value_func();
  // host-error@-1 {{reference to __device__ function 'value_func' in __host__ function}}
  // device-error@-2 {{reference to __device__ function 'value_func' in global initializer}}
  }

  struct DefInit {
    unsigned a = 1 + value_func();
  };
  DefInit testDefInit;
  // expected-error@-1 {{reference to __device__ function 'value_func' in global initializer}}

  struct DefArg {
    int data;
    DefArg(int a = 1 + value_func()) : data(a) {}
  };
  DefArg testDefArg;
  // expected-error@-1 {{reference to __device__ function 'value_func' in global initializer}}

  unsigned twotimes = ns1::value_func() + ns1::another_value_func();
  // expected-error@-1 {{reference to __device__ function 'value_func' in global initializer}}

  unsigned twotimes1 = ns1::value_func() + ns1::value_func();
  // expected-error@-1 {{reference to __device__ function 'value_func' in global initializer}}

}

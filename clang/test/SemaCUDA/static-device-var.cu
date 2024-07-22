// REQUIRES: x86-registered-target
// REQUIRES: amdgpu-registered-target

// RUN: %clang_cc1 -triple nvptx -fcuda-is-device -std=c++11 \
// RUN:    -o - %s -fsyntax-only -verify=dev,com

// RUN: %clang_cc1 -triple x86_64-gnu-linux -std=c++11 \
// RUN:    -o - %s -fsyntax-only -verify=host,com

// Checks allowed usage of file-scope and function-scope static variables.

#include "Inputs/cuda.h"

// Checks static variables are allowed in device functions.

__device__ void f1() {
  const static int b = 123;
  static int a;
}

// Checks static variables are allowd in global functions.

__global__ void k1() {
  const static int b = 123;
  static int a;
}

// Checks static device and constant variables are allowed in device and
// host functions, and static host variables are not allowed in device
// functions.

static __device__ int x;
static __constant__ int y;
static int z; // dev-note {{host variable declared here}}

__global__ void kernel(int *a) {
  a[0] = x;
  a[1] = y;
  a[2] = z;
  // dev-error@-1 {{reference to __host__ variable 'z' in __global__ function}}
}

// Check dynamic initialization of static device variable is not allowed.

namespace TestStaticVarInLambda {
class A {
public:
  A(char *);
};
class B {
public:
  __device__ B(char *);
};
void fun() {
  (void) [](char *c) {
    static A var1(c);
    static __device__ B var2(c);
    // com-error@-1 {{dynamic initialization is not supported for __device__, __constant__, __shared__, and __managed__ variables}}
    (void) var1;
    (void) var2;
  };
}
}

int* getDeviceSymbol(int *x);

void foo() {
  getDeviceSymbol(&x);
  getDeviceSymbol(&y);
}

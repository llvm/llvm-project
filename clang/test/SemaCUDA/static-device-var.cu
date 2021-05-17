// REQUIRES: x86-registered-target
// REQUIRES: amdgpu-registered-target

// RUN: %clang_cc1 -triple nvptx -fcuda-is-device \
// RUN:    -emit-llvm -o - %s -fsyntax-only -verify=dev

// RUN: %clang_cc1 -triple x86_64-gnu-linux \
// RUN:    -emit-llvm -o - %s -fsyntax-only -verify=host

// Checks allowed usage of file-scope and function-scope static variables.

// host-no-diagnostics

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
static int z;

__global__ void kernel(int *a) {
  a[0] = x;
  a[1] = y;
  a[2] = z;
  // dev-error@-1 {{reference to __host__ variable 'z' in __global__ function}}
}

int* getDeviceSymbol(int *x);

void foo() {
  getDeviceSymbol(&x);
  getDeviceSymbol(&y);
}

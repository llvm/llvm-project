// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fcuda-is-device -emit-llvm -x hip %s -o - | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -x hip %s -o - | FileCheck -check-prefix=HOST %s

#include "Inputs/cuda.h"

// Coerced struct from `struct S` without all generic pointers lowered into
// global ones.
// CHECK: %struct.S.coerce = type { i32 addrspace(1)*, float addrspace(1)* }
// CHECK: %struct.T.coerce = type { [2 x float addrspace(1)*] }

// On the host-side compilation, generic pointer won't be coerced.
// HOST-NOT: %struct.S.coerce
// HOST-NOT: %struct.T.coerce

// CHECK: define amdgpu_kernel void  @_Z7kernel1Pi(i32 addrspace(1)* %x.coerce)
// HOST: define void @_Z22__device_stub__kernel1Pi(i32* %x)
__global__ void kernel1(int *x) {
  x[0]++;
}

// CHECK: define amdgpu_kernel void  @_Z7kernel2Ri(i32 addrspace(1)* nonnull align 4 dereferenceable(4) %x.coerce)
// HOST: define void @_Z22__device_stub__kernel2Ri(i32* nonnull align 4 dereferenceable(4) %x)
__global__ void kernel2(int &x) {
  x++;
}

// CHECK: define amdgpu_kernel void  @_Z7kernel3PU3AS2iPU3AS1i(i32 addrspace(2)* %x, i32 addrspace(1)* %y)
// HOST: define void @_Z22__device_stub__kernel3PU3AS2iPU3AS1i(i32 addrspace(2)* %x, i32 addrspace(1)* %y)
__global__ void kernel3(__attribute__((address_space(2))) int *x,
                        __attribute__((address_space(1))) int *y) {
  y[0] = x[0];
}

// CHECK: define void @_Z4funcPi(i32* %x)
__device__ void func(int *x) {
  x[0]++;
}

struct S {
  int *x;
  float *y;
};
// `by-val` struct will be coerced into a similar struct with all generic
// pointers lowerd into global ones.
// CHECK: define amdgpu_kernel void @_Z7kernel41S(%struct.S.coerce %s.coerce)
// HOST: define void @_Z22__device_stub__kernel41S(i32* %s.coerce0, float* %s.coerce1)
__global__ void kernel4(struct S s) {
  s.x[0]++;
  s.y[0] += 1.f;
}

// If a pointer to struct is passed, only the pointer itself is coerced into the global one.
// CHECK: define amdgpu_kernel void @_Z7kernel5P1S(%struct.S addrspace(1)* %s.coerce)
// HOST: define void @_Z22__device_stub__kernel5P1S(%struct.S* %s)
__global__ void kernel5(struct S *s) {
  s->x[0]++;
  s->y[0] += 1.f;
}

struct T {
  float *x[2];
};
// `by-val` array is also coerced.
// CHECK: define amdgpu_kernel void @_Z7kernel61T(%struct.T.coerce %t.coerce)
// HOST: define void @_Z22__device_stub__kernel61T(float* %t.coerce0, float* %t.coerce1)
__global__ void kernel6(struct T t) {
  t.x[0][0] += 1.f;
  t.x[1][0] += 2.f;
}

// Check that coerced pointers retain the noalias attribute when qualified with __restrict.
// CHECK: define amdgpu_kernel void @_Z7kernel7Pi(i32 addrspace(1)* noalias %x.coerce)
// HOST: define void @_Z22__device_stub__kernel7Pi(i32* noalias %x)
__global__ void kernel7(int *__restrict x) {
  x[0]++;
}

// REQUIRES: x86-registered-target
// REQUIRES: amdgpu-registered-target

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fcuda-is-device -emit-llvm -x hip %s -o - | FileCheck --check-prefixes=COMMON,CHECK %s
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fcuda-is-device -emit-llvm -x hip %s -disable-O0-optnone -o - | opt -S -O2 | FileCheck %s --check-prefixes=COMMON,OPT
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -x hip %s -o - | FileCheck -check-prefix=HOST %s

#include "Inputs/cuda.h"

// Coerced struct from `struct S` without all generic pointers lowered into
// global ones.
// COMMON: %struct.S.coerce = type { i32 addrspace(1)*, float addrspace(1)* }
// COMMON: %struct.T.coerce = type { [2 x float addrspace(1)*] }

// On the host-side compilation, generic pointer won't be coerced.
// HOST-NOT: %struct.S.coerce
// HOST-NOT: %struct.T.coerce

// HOST: define void @_Z22__device_stub__kernel1Pi(i32* %x)
// COMMON-LABEL: define amdgpu_kernel void @_Z7kernel1Pi(i32 addrspace(1)*{{.*}} %x.coerce)
// CHECK:     = addrspacecast [[TYPE:.*]] addrspace(1)* %{{.*}} to [[TYPE]]*
// CHECK-NOT: = addrspacecast [[TYPE:.*]] addrspace(1)* %{{.*}} to [[TYPE]]*
// OPT: [[VAL:%.*]] = load i32, i32 addrspace(1)* %x.coerce, align 4
// OPT: [[INC:%.*]] = add nsw i32 [[VAL]], 1
// OPT: store i32 [[INC]], i32 addrspace(1)* %x.coerce, align 4
// OPT: ret void
__global__ void kernel1(int *x) {
  x[0]++;
}

// HOST: define void @_Z22__device_stub__kernel2Ri(i32* nonnull align 4 dereferenceable(4) %x)
// COMMON-LABEL: define amdgpu_kernel void @_Z7kernel2Ri(i32 addrspace(1)*{{.*}} nonnull align 4 dereferenceable(4) %x.coerce)
// CHECK:     = addrspacecast [[TYPE:.*]] addrspace(1)* %{{.*}} to [[TYPE]]*
// CHECK-NOT: = addrspacecast [[TYPE:.*]] addrspace(1)* %{{.*}} to [[TYPE]]*
// OPT: [[VAL:%.*]] = load i32, i32 addrspace(1)* %x.coerce, align 4
// OPT: [[INC:%.*]] = add nsw i32 [[VAL]], 1
// OPT: store i32 [[INC]], i32 addrspace(1)* %x.coerce, align 4
// OPT: ret void
__global__ void kernel2(int &x) {
  x++;
}

// HOST: define void @_Z22__device_stub__kernel3PU3AS2iPU3AS1i(i32 addrspace(2)* %x, i32 addrspace(1)* %y)
// CHECK-LABEL: define amdgpu_kernel void  @_Z7kernel3PU3AS2iPU3AS1i(i32 addrspace(2)*{{.*}} %x, i32 addrspace(1)*{{.*}} %y)
// CHECK-NOT: = addrspacecast [[TYPE:.*]] addrspace(1)* %{{.*}} to [[TYPE]]*
__global__ void kernel3(__attribute__((address_space(2))) int *x,
                        __attribute__((address_space(1))) int *y) {
  y[0] = x[0];
}

// COMMON-LABEL: define void @_Z4funcPi(i32*{{.*}} %x)
// CHECK-NOT: = addrspacecast [[TYPE:.*]] addrspace(1)* %{{.*}} to [[TYPE]]*
__device__ void func(int *x) {
  x[0]++;
}

struct S {
  int *x;
  float *y;
};
// `by-val` struct will be coerced into a similar struct with all generic
// pointers lowerd into global ones.
// HOST: define void @_Z22__device_stub__kernel41S(i32* %s.coerce0, float* %s.coerce1)
// COMMON-LABEL: define amdgpu_kernel void @_Z7kernel41S(%struct.S.coerce %s.coerce)
// OPT: [[P0:%.*]] = extractvalue %struct.S.coerce %s.coerce, 0
// OPT: [[P1:%.*]] = extractvalue %struct.S.coerce %s.coerce, 1
// OPT: [[V0:%.*]] = load i32, i32 addrspace(1)* [[P0]], align 4
// OPT: [[INC:%.*]] = add nsw i32 [[V0]], 1
// OPT: store i32 [[INC]], i32 addrspace(1)* [[P0]], align 4
// OPT: [[V1:%.*]] = load float, float addrspace(1)* [[P1]], align 4
// OPT: [[ADD:%.*]] = fadd contract float [[V1]], 1.000000e+00
// OPT: store float [[ADD]], float addrspace(1)* [[P1]], align 4
// OPT: ret void
__global__ void kernel4(struct S s) {
  s.x[0]++;
  s.y[0] += 1.f;
}

// If a pointer to struct is passed, only the pointer itself is coerced into the global one.
// HOST: define void @_Z22__device_stub__kernel5P1S(%struct.S* %s)
// COMMON-LABEL: define amdgpu_kernel void @_Z7kernel5P1S(%struct.S addrspace(1)*{{.*}} %s.coerce)
__global__ void kernel5(struct S *s) {
  s->x[0]++;
  s->y[0] += 1.f;
}

struct T {
  float *x[2];
};
// `by-val` array is also coerced.
// HOST: define void @_Z22__device_stub__kernel61T(float* %t.coerce0, float* %t.coerce1)
// COMMON-LABEL: define amdgpu_kernel void @_Z7kernel61T(%struct.T.coerce %t.coerce)
// OPT: [[ARR:%.*]] = extractvalue %struct.T.coerce %t.coerce, 0
// OPT: [[P0:%.*]] = extractvalue [2 x float addrspace(1)*] [[ARR]], 0
// OPT: [[P1:%.*]] = extractvalue [2 x float addrspace(1)*] [[ARR]], 1
// OPT: [[V0:%.*]] = load float, float addrspace(1)* [[P0]], align 4
// OPT: [[ADD0:%.*]] = fadd contract float [[V0]], 1.000000e+00
// OPT: store float [[ADD0]], float addrspace(1)* [[P0]], align 4
// OPT: [[V1:%.*]] = load float, float addrspace(1)* [[P1]], align 4
// OPT: [[ADD1:%.*]] = fadd contract float [[V1]], 2.000000e+00
// OPT: store float [[ADD1]], float addrspace(1)* [[P1]], align 4
// OPT: ret void
__global__ void kernel6(struct T t) {
  t.x[0][0] += 1.f;
  t.x[1][0] += 2.f;
}

// Check that coerced pointers retain the noalias attribute when qualified with __restrict.
// HOST: define void @_Z22__device_stub__kernel7Pi(i32* noalias %x)
// COMMON-LABEL: define amdgpu_kernel void @_Z7kernel7Pi(i32 addrspace(1)* noalias{{.*}} %x.coerce)
__global__ void kernel7(int *__restrict x) {
  x[0]++;
}

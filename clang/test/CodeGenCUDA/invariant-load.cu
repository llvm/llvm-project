// RUN: %clang_cc1 -triple amdgpu7.00-amd-amdhsa -fcuda-is-device -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple amdgpu7.00-amd-amdhsa -fcuda-is-device -fsanitize=array-bounds,pointer-overflow -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple amdgpu7.00-amd-amdhsa -fcuda-is-device -fwrapv-pointer -emit-llvm -o - %s | FileCheck %s

#include "Inputs/cuda.h"

__constant__ int constant_value;
__constant__ int constant_array[4];
__device__ int device_value;

struct S {
  int member;
};

__constant__ S constant_struct;
__device__ S device_struct;

__device__ int constant_load() {
  return constant_value;
}

// CHECK-LABEL: define{{.*}}@_Z13constant_loadv(
// CHECK: load i32, ptr addrspacecast (ptr addrspace(4) @constant_value to ptr), align 4, !invariant.load [[INVARIANT:![0-9]+]]

__device__ int constant_array_load(int index) {
  return constant_array[index];
}

// CHECK-LABEL: define{{.*}}@_Z19constant_array_loadi(
// CHECK: load i32, ptr %{{.*}}, align 4, !invariant.load [[INVARIANT]]

__device__ int constant_member_load() {
  return constant_struct.member;
}

// CHECK-LABEL: define{{.*}}@_Z20constant_member_loadv(
// CHECK: load i32, ptr addrspacecast (ptr addrspace(4) @constant_struct to ptr), align 4, !invariant.load [[INVARIANT]]

__device__ int device_load() {
  return device_value;
}

// CHECK-LABEL: define{{.*}}@_Z11device_loadv(
// CHECK: load i32, ptr addrspacecast (ptr addrspace(1) @device_value to ptr), align 4{{$}}

struct Inner {
  int x;
  int a[4];
};

struct Outer {
  Inner in[2];
  int *p;
  int &r;
};

__constant__ Inner constant_inner_array[4];
__constant__ Outer constant_outer = {{}, &device_value, device_value};

__device__ int nested_array_member_load(int i, int j) {
  return constant_inner_array[i].a[j];
}

// CHECK-LABEL: define{{.*}}@_Z24nested_array_member_loadii(
// CHECK: load i32, ptr %arrayidx{{[0-9]*}}, align 4, !invariant.load [[INVARIANT]]

__device__ int addressed_member_load(int i, int j) {
  return (&constant_outer)->in[i].a[j];
}

// CHECK-LABEL: define{{.*}}@_Z21addressed_member_loadii(
// CHECK: load i32, ptr %arrayidx{{[0-9]*}}, align 4, !invariant.load [[INVARIANT]]

extern __constant__ int &constant_reference;

__device__ int reference_load() {
  return constant_reference;
}

// CHECK-LABEL: define{{.*}}@_Z14reference_loadv(
// CHECK: [[REF:%.*]] = load ptr, ptr addrspacecast (ptr addrspace(4) @constant_reference to ptr)
// CHECK: load i32, ptr [[REF]], align 4{{$}}

__device__ int pointer_member_index(int i) {
  return constant_outer.p[i];
}

// CHECK-LABEL: define{{.*}}@_Z20pointer_member_indexi(
// CHECK: load ptr, ptr getelementptr {{.*}}@constant_outer{{.*}}, align 8, !invariant.load [[INVARIANT]]
// CHECK: load i32, ptr %arrayidx, align 4{{$}}

__device__ int reference_member() {
  return constant_outer.r;
}

// CHECK-LABEL: define{{.*}}@_Z16reference_memberv(
// CHECK: [[R:%.*]] = load ptr, ptr getelementptr {{.*}}@constant_outer
// CHECK: load i32, ptr [[R]], align 4{{$}}

__device__ int conditional_member_load(bool cond) {
  return (cond ? constant_struct : device_struct).member;
}

// CHECK-LABEL: define{{.*}}@_Z23conditional_member_loadb(
// CHECK: phi ptr
// CHECK: load i32, ptr %{{.*}}, align 4{{$}}

// CHECK: [[INVARIANT]] = !{}

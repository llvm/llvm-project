// RUN: %clang_cc1 -triple amdgpu7.00-amd-amdhsa -fcuda-is-device -emit-llvm -o - %s | FileCheck %s

#include "Inputs/cuda.h"

__constant__ int constant_value;
__constant__ int constant_array[4];
__device__ int device_value;

struct S {
  int member;
};

__constant__ S constant_struct;

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

// CHECK: [[INVARIANT]] = !{}

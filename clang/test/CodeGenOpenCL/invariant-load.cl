// RUN: %clang_cc1 -triple amdgpu7.00-amd-amdhsa -cl-std=CL2.0 -O0 -emit-llvm -o - %s | FileCheck %s

kernel void constant_load(global int *out, constant int *in) {
  out[0] = in[0];
}

// CHECK-LABEL: define{{.*}}@constant_load(
// CHECK: load i32, ptr addrspace(4) %{{.*}}, align 4, !invariant.load [[INVARIANT:![0-9]+]]

kernel void global_const_load(global int *out, global const int *in) {
  out[0] = in[0];
}

// CHECK-LABEL: define{{.*}}@global_const_load(
// CHECK: load i32, ptr addrspace(1) %{{.*}}, align 4{{$}}

// CHECK: [[INVARIANT]] = !{}

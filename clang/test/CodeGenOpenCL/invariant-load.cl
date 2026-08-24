// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -cl-std=CL2.0 -O0 -emit-llvm -o - %s | FileCheck %s --check-prefix=AMDGCN
// RUN: %clang_cc1 -triple spir64-unknown-unknown -cl-std=CL2.0 -O0 -emit-llvm -o - %s | FileCheck %s --check-prefix=SPIR

kernel void constant_load(global int *out, constant int *in) {
  out[0] = in[0];
}

// AMDGCN-LABEL: define{{.*}}@constant_load(
// AMDGCN: load i32, ptr addrspace(4) %{{.*}}, align 4, !invariant.load [[INVARIANT:![0-9]+]]
// SPIR-LABEL: define{{.*}}@constant_load(
// SPIR: load i32, ptr addrspace(2) %{{.*}}, align 4, !invariant.load [[INVARIANT:![0-9]+]]

kernel void global_const_load(global int *out, global const int *in) {
  out[0] = in[0];
}

// AMDGCN-LABEL: define{{.*}}@global_const_load(
// AMDGCN: load i32, ptr addrspace(1) %{{.*}}, align 4{{$}}
// SPIR-LABEL: define{{.*}}@global_const_load(
// SPIR: load i32, ptr addrspace(1) %{{.*}}, align 4{{$}}

// AMDGCN: [[INVARIANT]] = !{}
// SPIR: [[INVARIANT]] = !{}

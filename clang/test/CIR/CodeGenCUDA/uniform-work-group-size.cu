// Based on the 'uniform-work-group-size' portion of
// clang/test/CodeGenCUDA/amdgpu-kernel-attrs.cu and
// clang/test/CodeGenHIP/default-attributes.hip

// 'uniform-work-group-size' comes from a language option rather than a decl
// attribute, so it lands on every function and every call site. It defaults to
// on for CUDA/HIP.

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -fcuda-is-device \
// RUN:   -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -fcuda-is-device \
// RUN:   -foffload-uniform-block -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -fcuda-is-device \
// RUN:   -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -fcuda-is-device \
// RUN:   -foffload-uniform-block -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefix=LLVM

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -fcuda-is-device \
// RUN:   -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=OGCG
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -fcuda-is-device \
// RUN:   -foffload-uniform-block -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=OGCG

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -fcuda-is-device \
// RUN:   -fno-offload-uniform-block -fclangir -emit-cir %s -o %t-noub.cir
// RUN: FileCheck --input-file=%t-noub.cir %s --check-prefix=CIR-NOUB
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -fcuda-is-device \
// RUN:   -fno-offload-uniform-block -fclangir -emit-llvm %s -o %t-noub-cir.ll
// RUN: FileCheck --input-file=%t-noub-cir.ll %s --check-prefix=NOUB
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -fcuda-is-device \
// RUN:   -fno-offload-uniform-block -emit-llvm %s -o %t-noub.ll
// RUN: FileCheck --input-file=%t-noub.ll %s --check-prefix=NOUB

#include "Inputs/cuda.h"

__device__ void extern_func();

__device__ void func() {
  extern_func();
}
// CIR: cir.func{{.*}}@_Z4funcv()
// CIR-SAME: uniform_work_group_size
// CIR: cir.call @_Z11extern_funcv()
// CIR-SAME: uniform_work_group_size

// CIR: cir.func private @_Z11extern_funcv()
// CIR-SAME: uniform_work_group_size

// LLVM: define{{.*}} void @_Z4funcv() [[FUNC:#[0-9]+]]
// LLVM: call void @_Z11extern_funcv() [[CALL:#[0-9]+]]
// OGCG: define{{.*}} void @_Z4funcv() [[FUNC:#[0-9]+]]
// OGCG: call void @_Z11extern_funcv() [[CALL:#[0-9]+]]

__global__ void kernel() {
  extern_func();
}
// CIR: cir.func{{.*}}@_Z6kernelv() cc(amdgpu_kernel)
// CIR-SAME: uniform_work_group_size
// CIR: cir.call @_Z11extern_funcv()
// CIR-SAME: uniform_work_group_size

// LLVM: define{{.*}} amdgpu_kernel void @_Z6kernelv() [[KERNEL:#[0-9]+]]
// LLVM: call void @_Z11extern_funcv() [[CALL]]
// OGCG: define{{.*}} amdgpu_kernel void @_Z6kernelv() [[KERNEL:#[0-9]+]]
// OGCG: call void @_Z11extern_funcv() [[CALL]]

// The attribute is present on both definitions and call sites.

// LLVM-DAG: attributes [[FUNC]] = {{.*}}"uniform-work-group-size"
// LLVM-DAG: attributes [[KERNEL]] = {{.*}}"uniform-work-group-size"
// LLVM-DAG: attributes [[CALL]] = {{.*}}"uniform-work-group-size"

// OGCG-DAG: attributes [[FUNC]] = {{.*}}"uniform-work-group-size"
// OGCG-DAG: attributes [[KERNEL]] = {{.*}}"uniform-work-group-size"
// OGCG-DAG: attributes [[CALL]] = {{.*}}"uniform-work-group-size"

// CIR-NOUB-NOT: uniform_work_group_size
// NOUB-NOT: "uniform-work-group-size"

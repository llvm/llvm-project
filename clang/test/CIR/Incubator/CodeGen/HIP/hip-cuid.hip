#include "../Inputs/cuda.h"

// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple=amdgcn-amd-amdhsa -x hip -fclangir \
// RUN:            -fcuda-is-device -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR %s --input-file=%t.cir

// RUN: %clang_cc1 -triple=amdgcn-amd-amdhsa -x hip -fclangir \
// RUN:            -fcuda-is-device -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM %s --input-file=%t.ll

// RUN: %clang_cc1 -triple=amdgcn-amd-amdhsa -x hip \
// RUN:            -fcuda-is-device -emit-llvm %s -o %t.ogcg.ll
// RUN: FileCheck --check-prefix=OGCG %s --input-file=%t.ogcg.ll

// Test that HIP compiler unit ID global is emitted

// CIR: cir.global external lang_address_space(offload_global) @__hip_cuid_{{.*}} = #cir.int<0> : !u8i

// LLVM: @__hip_cuid_{{.*}} = addrspace(1) global i8 0
// LLVM: @llvm.compiler.used = {{.*}}@__hip_cuid_

// OGCG: @__hip_cuid_{{.*}} = addrspace(1) global i8 0
// OGCG: @llvm.compiler.used = {{.*}}@__hip_cuid_

__global__ void kernel() {}

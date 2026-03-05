#include "../Inputs/cuda.h"

// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple=amdgcn-amd-amdhsa -x hip -fclangir \
// RUN:            -fcuda-is-device -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR %s --input-file=%t.cir

// RUN: %clang_cc1 -triple=amdgcn-amd-amdhsa -x hip -fclangir \
// RUN:            -fcuda-is-device -emit-llvm %s -o %t.cir.ll
// RUN: FileCheck --check-prefix=LLVM %s --input-file=%t.cir.ll

// RUN: %clang_cc1 -triple=amdgcn-amd-amdhsa -x hip \
// RUN:            -fcuda-is-device -emit-llvm %s -o %t.ogcg.ll
// RUN: FileCheck --check-prefix=OGCG %s --input-file=%t.ogcg.ll

// Test that AMDGPU module flags are emitted correctly.

// CIR: module {{.*}} attributes {
// CIR-SAME: cir.amdgpu_printf_kind = "hostcall"
// CIR-SAME: cir.amdhsa_code_object_version = 600

// LLVM: !llvm.module.flags = !{
// LLVM-DAG: !{i32 1, !"amdhsa_code_object_version", i32 600}
// LLVM-DAG: !{i32 1, !"amdgpu_printf_kind", !"hostcall"}

// OGCG: !llvm.module.flags = !{
// OGCG-DAG: !{i32 1, !"amdhsa_code_object_version", i32 600}
// OGCG-DAG: !{i32 1, !"amdgpu_printf_kind", !"hostcall"}

__global__ void kernel() {}

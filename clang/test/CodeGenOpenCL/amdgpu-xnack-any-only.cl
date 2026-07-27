// Test that xnack module flags are emitted for all targets, regardless of support.
// Targets without FEATURE_XNACK_ON_OFF_MODES (like gfx12-5-generic, gfx1250, gfx1251)
// will ignore the module flag during codegen, but it is still emitted by clang.
// TODO: In the future, clang should not emit the flag for targets that don't support
// xnack control.

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -target-cpu gfx12-5-generic \
// RUN:   -mxnack -emit-llvm -o - %s | FileCheck %s --check-prefix=XNACK-ON

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -target-cpu gfx12-5-generic \
// RUN:   -mno-xnack -emit-llvm -o - %s | FileCheck %s --check-prefix=XNACK-OFF

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -target-cpu gfx12-5-generic \
// RUN:   -emit-llvm -o - %s | FileCheck %s --check-prefix=NO-FLAGS

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -target-cpu gfx1250 \
// RUN:   -mxnack -emit-llvm -o - %s | FileCheck %s --check-prefix=XNACK-ON

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -target-cpu gfx1250 \
// RUN:   -mno-xnack -emit-llvm -o - %s | FileCheck %s --check-prefix=XNACK-OFF

// Module flags are emitted regardless of target support
// XNACK-ON-DAG: !{i32 1, !"amdgpu.xnack", i32 1}
// XNACK-OFF-DAG: !{i32 1, !"amdgpu.xnack", i32 0}
// NO-FLAGS-NOT: !"amdgpu.xnack"

kernel void test() {}

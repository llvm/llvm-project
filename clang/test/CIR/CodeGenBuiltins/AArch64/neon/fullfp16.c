// REQUIRES: aarch64-registered-target

// RUN: %clang_cc1 -triple aarch64 -target-feature +fullfp16 -disable-O0-optnone -Werror -Wall -fclangir -emit-cir  -o - %s                                       | FileCheck %s --check-prefixes=ALL,CIR
// RUN: %clang_cc1 -triple aarch64 -target-feature +fullfp16 -disable-O0-optnone -Werror -Wall -fclangir -emit-llvm -o - %s |  opt -S -passes=mem2reg,simplifycfg | FileCheck %s --check-prefixes=ALL,LLVM
// RUN: %clang_cc1 -triple aarch64 -target-feature +fullfp16 -disable-O0-optnone -Werror -Wall           -emit-llvm -o - %s |  opt -S -passes=mem2reg,simplifycfg | FileCheck %s --check-prefixes=ALL,LLVM


//=============================================================================
// NOTES
//
// Tests for unconstrained intrinsics that require the fullfp16 extension.
//
// As these intrinsics expand to code with multiple compound and declaration
// stmts, the LLVM output has been simplified with opt. `simplifycfg` was added
// specifically for the CIR lowering path.
//
// Minor differences between RUNs (e.g. presence of `noundef` attached to
// argumens, align` attribute attached to pointers), are matched using
// catch-alls like {{.*}}.
//
// TODO: Merge this file with clang/test/CodeGen/AArch64/v8.2a-fp16-intrinsics.c
// (the source of these tests).
//=============================================================================

#include <arm_fp16.h>

// ALL-LABEL: @test_vabsh_f16
float16_t test_vabsh_f16(float16_t a) {
// CIR: {{%.*}}  = cir.fabs {{%.*}} : !cir.f16

// LLVM-SAME: (half{{.*}} [[A:%.*]])
// LLVM:  [[ABS:%.*]] =  call half @llvm.fabs.f16(half [[A]])
// LLVM:  ret half [[ABS]]
  return vabsh_f16(a);
}

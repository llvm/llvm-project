// REQUIRES: aarch64-registered-target

// RUN:                   %clang_cc1 -triple arm64-none-linux-gnu -target-feature +fullfp16 -disable-O0-optnone           -emit-llvm -o - %s | opt -S -passes=mem2reg             | FileCheck %s --check-prefixes=ALL,LLVM
// RUN: %if cir-enabled %{%clang_cc1 -triple arm64-none-linux-gnu -target-feature +fullfp16 -disable-O0-optnone -fclangir -emit-llvm -o - %s | opt -S -passes=mem2reg,simplifycfg | FileCheck %s --check-prefixes=ALL,LLVM %}
// RUN: %if cir-enabled %{%clang_cc1 -triple arm64-none-linux-gnu -target-feature +fullfp16 -disable-O0-optnone -fclangir -emit-cir  -o - %s |                                      FileCheck %s --check-prefixes=ALL,CIR %}

//=============================================================================
// NOTES
//
// Tests for unconstrained intrinsics that require the fullfp16 extension.
//
// These intrinsics expand to code containing multiple compound and declaration
// statements rather than just plain function calls, which leads to:
//  * "scopes" at the CIR level, and then
//  * redundant branches at the LLVM IR level.  
// The default lowering path never generates those redundant LLVM IR branches,
// hence for CIR we use `opt -passes=simplifycfg` to reduce the control flow
// and to make LLVM IR match for all paths.
//
// Minor differences between RUN lines (e.g., the presence of `noundef` on
// arguments or the `align` attribute on pointers) are matched using
// catch-alls such as `{{.*}}`.
//
// TODO: Remove `-simplifycfg` once CIR lowering includes the relevant
//       optimizations to reduce the CFG.
//
// TODO: Merge this file with
//        * clang/test/CodeGen/AArch64/v8.2a-fp16-intrinsics.c
//       (the source of these tests).
//=============================================================================

#include <arm_fp16.h>

// ALL-LABEL: @test_vabsh_f16
float16_t test_vabsh_f16(float16_t a) {
// CIR: {{%.*}} = cir.fabs {{%.*}} : !cir.f16

// LLVM-SAME: (half{{.*}} [[A:%.*]])
// LLVM:  [[ABS:%.*]] = call half @llvm.fabs.f16(half [[A]])
// LLVM:  ret half [[ABS]]
  return vabsh_f16(a);
}

// ALL-LABEL: @test_vnegh_f16
float16_t test_vnegh_f16(float16_t a) {
// CIR: cir.unary(minus, {{.*}}) : !cir.f16

// LLVM-SAME: half{{.*}} [[A:%.*]])
// LLVM: [[NEG:%.*]] = fneg half [[A:%.*]]
// LLVM: ret half [[NEG]]
  return vnegh_f16(a);
}

// ALL-LABEL: test_vfmah_f16
float16_t test_vfmah_f16(float16_t a, float16_t b, float16_t c) {
// CIR: cir.call_llvm_intrinsic "fma" {{.*}} : (!cir.f16, !cir.f16, !cir.f16) -> !cir.f16

// LLVM-SAME: half{{.*}} [[A:%.*]], half{{.*}} [[B:%.*]], half{{.*}} [[C:%.*]])
// LLVM:  [[FMA:%.*]] = call half @llvm.fma.f16(half [[B]], half [[C]], half [[A]])
// LLVM:  ret half [[FMA]]
  return vfmah_f16(a, b, c);
}

// ALL-LABEL: test_vfmsh_f16
float16_t test_vfmsh_f16(float16_t a, float16_t b, float16_t c) {
// CIR: [[SUB:%.*]] = cir.unary(minus, %{{.*}}) : !cir.f16, !cir.f16
// CIR: cir.call_llvm_intrinsic "fma" [[SUB]], {{.*}} : (!cir.f16, !cir.f16, !cir.f16) -> !cir.f16

// LLVM-SAME: half{{.*}} [[A:%.*]], half{{.*}} [[B:%.*]], half{{.*}} [[C:%.*]])
// LLVM:  [[SUB:%.*]] = fneg half [[B]]
// LLVM:  [[ADD:%.*]] = call half @llvm.fma.f16(half [[SUB]], half [[C]], half [[A]])
// LLVM:  ret half [[ADD]]
  return vfmsh_f16(a, b, c);
}

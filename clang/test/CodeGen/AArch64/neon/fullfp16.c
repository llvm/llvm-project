// REQUIRES: aarch64-registered-target

// RUN:                   %clang_cc1 -triple arm64-none-linux-gnu -target-feature +fullfp16 -disable-O0-optnone           -emit-llvm -o - %s | opt -S -passes=mem2reg             | FileCheck %s --check-prefixes=ALL,LLVM
// RUN: %if cir-enabled %{%clang_cc1 -triple arm64-none-linux-gnu -target-feature +fullfp16 -disable-O0-optnone -fclangir -emit-llvm -o - %s | opt -S -passes=mem2reg,simplifycfg | FileCheck %s --check-prefixes=ALL,LLVM %}
// RUN: %if cir-enabled %{%clang_cc1 -triple arm64-none-linux-gnu -target-feature +fullfp16 -disable-O0-optnone -fclangir -emit-cir  -o - %s |                                      FileCheck %s --check-prefixes=ALL,CIR %}

//=============================================================================
// NOTES
//
// Minor differences between RUNs (e.g. presence of `noundef` attached to
// argumens, `align` attribute attached to pointers), are matched using
// catch-alls like {{.*}}.
//
// Different labels for CIR stem from an additional function call that is
// present at the AST and CIR levels, but is inlined at the LLVM IR level.
//=============================================================================

#include <arm_fp16.h>

// ALL-LABEL: @test_vnegh_f16
float16_t test_vnegh_f16(float16_t a) {
// CIR: cir.unary(minus, {{.*}}) : !cir.f16

// LLVM-SAME: half{{.*}} [[A:%.*]])
// LLVM: [[NEG:%.*]] = fneg half [[A:%.*]]
// LLVM: ret half [[NEG]]
  return vnegh_f16(a);
}

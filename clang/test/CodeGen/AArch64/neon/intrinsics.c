// REQUIRES: aarch64-registered-target || arm-registered-target

// RUN:           %clang_cc1 -triple arm64-none-linux-gnu -target-feature +neon -disable-O0-optnone -flax-vector-conversions=none           -emit-llvm -o - %s | opt -S -passes=mem2reg,sroa             | FileCheck %s --check-prefixes=LLVM
// RUN: %if cir %{%clang_cc1 -triple arm64-none-linux-gnu -target-feature +neon -disable-O0-optnone -flax-vector-conversions=none -fclangir -emit-llvm -o - %s | opt -S -passes=mem2reg,sroa,instcombine | FileCheck %s --check-prefixes=LLVM %}
// RUN: %if cir %{%clang_cc1 -triple arm64-none-linux-gnu -target-feature +neon -disable-O0-optnone -flax-vector-conversions=none -fclangir -emit-cir  -o - %s |                                           FileCheck %s --check-prefixes=CIR %}

//=============================================================================
// NOTES
//
// Minor differences between RUNs (e.g. presence of `noundef` attached to
// argumens, `align` attribute attached to pointers), are matched using
// catch-alls like {{.*}}.
//
// Different labels for CIR stem from an additional function call that is
// present at the AST and CIR levels, but is inlined at the LLVM IR level.
//
// For `-fclangir`, `instcombine` is used to e.g. fold 1-element vectors to
// scalars.
//=============================================================================

#include <arm_neon.h>

// LLVM-LABEL: @test_vceqzd_s64
// CIR-LABEL: @vceqzd_s64
uint64_t test_vceqzd_s64(int64_t a) {
// CIR:   [[C_0:%.*]] = cir.const #cir.int<0>
// CIR:   [[LHS:%.*]] = cir.vec.splat {{.*}} : !s64i, !cir.vector<1 x !s64i>
// CIR:   [[RHS:%.*]] = cir.vec.splat [[C_0]] : !s64i, !cir.vector<1 x !s64i>
// CIR:   [[CMP:%.*]] = cir.vec.cmp(eq, [[LHS]], [[RHS]]) : !cir.vector<1 x !s64i>, !cir.vector<1 x !s64i>
// CIR:   [[C_0_1:%.*]] = cir.const #cir.int<0> : !u64i
// CIR:   [[RES:%.*]] = cir.vec.extract [[CMP]][[[C_0_1]] : !u64i] : !cir.vector<1 x !s64i>
// CIR:   cir.cast bitcast [[RES]] : !s64i -> !u64i

// LLVM-SAME: i64{{.*}} [[A:%.*]])
// LLVM:          [[TMP0:%.*]] = icmp eq i64 [[A]], 0
// LLVM-NEXT:    [[VCEQZ_I:%.*]] = sext i1 [[TMP0]] to i64
// LLVM-NEXT:    ret i64 [[VCEQZ_I]]
  return (uint64_t)vceqzd_s64(a);
}

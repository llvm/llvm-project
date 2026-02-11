// REQUIRES: aarch64-registered-target || arm-registered-target

// RUN:                   %clang_cc1 -triple arm64-none-linux-gnu -target-feature +neon -disable-O0-optnone -flax-vector-conversions=none           -emit-llvm -o - %s | opt -S -passes=mem2reg,sroa | FileCheck %s --check-prefixes=LLVM
// RUN: %if cir-enabled %{%clang_cc1 -triple arm64-none-linux-gnu -target-feature +neon -disable-O0-optnone -flax-vector-conversions=none -fclangir -emit-llvm -o - %s | opt -S -passes=mem2reg,sroa | FileCheck %s --check-prefixes=LLVM %}
// RUN: %if cir-enabled %{%clang_cc1 -triple arm64-none-linux-gnu -target-feature +neon -disable-O0-optnone -flax-vector-conversions=none -fclangir -emit-cir  -o - %s |                               FileCheck %s --check-prefixes=CIR %}

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

#include <arm_neon.h>

// LLVM-LABEL: @test_vceqzd_s64
// CIR-LABEL: @vceqzd_s64
uint64_t test_vceqzd_s64(int64_t a) {
// CIR:   [[C_0:%.*]] = cir.const #cir.int<0>
// CIR:   [[CMP:%.*]] = cir.cmp(eq, %{{.*}}, [[C_0]]) : !s64i, !cir.bool
// CIR:   [[RES:%.*]] = cir.cast bool_to_int [[CMP]] : !cir.bool -> !cir.int<s, 1>
// CIR:   cir.cast integral [[RES]] : !cir.int<s, 1> -> !u64i

// LLVM-SAME: i64{{.*}} [[A:%.*]])
// LLVM:          [[TMP0:%.*]] = icmp eq i64 [[A]], 0
// LLVM-NEXT:    [[VCEQZ_I:%.*]] = sext i1 [[TMP0]] to i64
// LLVM-NEXT:    ret i64 [[VCEQZ_I]]
  return (uint64_t)vceqzd_s64(a);
}

// LLVM-LABEL: @test_vnegd_s64
// CIR-LABEL: @vnegd_s64
int64_t test_vnegd_s64(int64_t a) {
// CIR: cir.unary(minus, {{.*}}) : !s64

// LLVM-SAME: i64{{.*}} [[A:%.*]])
// LLVM:          [[VNEGD_I:%.*]] = sub i64 0, [[A]]
// LLVM-NEXT:     ret i64 [[VNEGD_I]]
  return (int64_t)vnegd_s64(a);
}

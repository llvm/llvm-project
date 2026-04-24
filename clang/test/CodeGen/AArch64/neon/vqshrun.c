// REQUIRES: aarch64-registered-target || arm-registered-target

// RUN:                   %clang_cc1 -triple arm64-none-linux-gnu -target-feature +neon -disable-O0-optnone -flax-vector-conversions=none           -emit-llvm -o - %s | opt -S -passes=mem2reg,sroa,simplifycfg | FileCheck %s --check-prefixes=LLVM
// RUN: %if cir-enabled %{%clang_cc1 -triple arm64-none-linux-gnu -target-feature +neon -disable-O0-optnone -flax-vector-conversions=none -fclangir -emit-llvm -o - %s | opt -S -passes=mem2reg,sroa,simplifycfg | FileCheck %s --check-prefixes=LLVM %}
// RUN: %if cir-enabled %{%clang_cc1 -triple arm64-none-linux-gnu -target-feature +neon -disable-O0-optnone -flax-vector-conversions=none -fclangir -emit-cir  -o - %s |                               FileCheck %s --check-prefixes=CIR %}

#include <arm_neon.h>

// LLVM-LABEL: @test_vqshrun_n_s16(
// CIR-LABEL: @test_vqshrun_n_s16(
uint8x8_t test_vqshrun_n_s16(int16x8_t a) {
// CIR:      cir.call_llvm_intrinsic "aarch64.neon.sqshrun" {{.*}} : (!cir.vector<8 x !s16i>, !s32i) -> !cir.vector<8 x !u8i>

// LLVM-SAME: <8 x i16> {{.*}} [[A:%.*]]) {{.*}} {
// LLVM:      [[R:%.*]] = call <8 x i8> @llvm.aarch64.neon.sqshrun.v8i8(<8 x i16> {{.*}}, i32 3)
// LLVM-NEXT: ret <8 x i8> [[R]]
  return vqshrun_n_s16(a, 3);
}

// LLVM-LABEL: @test_vqshrun_n_s32(
// CIR-LABEL: @test_vqshrun_n_s32(
uint16x4_t test_vqshrun_n_s32(int32x4_t a) {
// CIR:      cir.call_llvm_intrinsic "aarch64.neon.sqshrun" {{.*}} : (!cir.vector<4 x !s32i>, !s32i) -> !cir.vector<4 x !u16i>

// LLVM-SAME: <4 x i32> {{.*}} [[A:%.*]]) {{.*}} {
// LLVM:      [[R:%.*]] = call <4 x i16> @llvm.aarch64.neon.sqshrun.v4i16(<4 x i32> {{.*}}, i32 9)
// LLVM-NEXT: ret <4 x i16> [[R]]
  return vqshrun_n_s32(a, 9);
}

// LLVM-LABEL: @test_vqshrun_n_s64(
// CIR-LABEL: @test_vqshrun_n_s64(
uint32x2_t test_vqshrun_n_s64(int64x2_t a) {
// CIR:      cir.call_llvm_intrinsic "aarch64.neon.sqshrun" {{.*}} : (!cir.vector<2 x !s64i>, !s32i) -> !cir.vector<2 x !u32i>

// LLVM-SAME: <2 x i64> {{.*}} [[A:%.*]]) {{.*}} {
// LLVM:      [[R:%.*]] = call <2 x i32> @llvm.aarch64.neon.sqshrun.v2i32(<2 x i64> {{.*}}, i32 19)
// LLVM-NEXT: ret <2 x i32> [[R]]
  return vqshrun_n_s64(a, 19);
}

// RUN: %clang_cc1 -triple arm64-none-linux-gnu -target-feature +neon -emit-llvm -o - %s | FileCheck %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple arm64-none-linux-gnu -target-feature +neon -fclangir -emit-cir -o - %s | FileCheck %s --check-prefix=CIR

#include <arm_neon.h>

// LLVM-LABEL: @test_vqshrun_n_s16(
// CIR-LABEL: @test_vqshrun_n_s16(
uint8x8_t test_vqshrun_n_s16(int16x8_t a) {
  // CIR: cir.call_llvm_intrinsic "aarch64.neon.sqshrun"
  // LLVM: call <8 x i8> @llvm.aarch64.neon.sqshrun.v8i8
  return vqshrun_n_s16(a, 3);
}

// LLVM-LABEL: @test_vqshrun_n_s32(
// CIR-LABEL: @test_vqshrun_n_s32(
uint16x4_t test_vqshrun_n_s32(int32x4_t a) {
  // CIR: cir.call_llvm_intrinsic "aarch64.neon.sqshrun"
  // LLVM: call <4 x i16> @llvm.aarch64.neon.sqshrun.v4i16
  return vqshrun_n_s32(a, 9);
}

// LLVM-LABEL: @test_vqshrun_n_s64(
// CIR-LABEL: @test_vqshrun_n_s64(
uint32x2_t test_vqshrun_n_s64(int64x2_t a) {
  // CIR: cir.call_llvm_intrinsic "aarch64.neon.sqshrun"
  // LLVM: call <2 x i32> @llvm.aarch64.neon.sqshrun.v2i32
  return vqshrun_n_s64(a, 19);
}
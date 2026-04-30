// REQUIRES: aarch64-registered-target || arm-registered-target
// RUN: %clang_cc1 -triple arm64-none-linux-gnu -target-feature +neon \
// RUN:   -disable-O0-optnone -flax-vector-conversions=none \
// RUN:   -emit-llvm -o - %s | FileCheck %s --check-prefix=LLVM
// RUN: %if cir-enabled %{ %clang_cc1 -triple arm64-none-linux-gnu \
// RUN:   -target-feature +neon -disable-O0-optnone \
// RUN:   -flax-vector-conversions=none -fclangir -emit-cir \
// RUN:   -o - %s | FileCheck %s --check-prefix=CIR %}

#include <arm_neon.h>

// LLVM-LABEL: @test_vqshrn_n_s16(
// CIR-LABEL: @vqshrn_n_s16(
int8x8_t test_vqshrn_n_s16(int16x8_t a) {
  // CIR: {{.*}}cir.call_llvm_intrinsic "aarch64.neon.sqshrn"
  // CIR-SAME: (!cir.vector<8 x !s16i>) -> !cir.vector<8 x !s8i>
  // LLVM: [[RES:%.*]] = call <8 x i8> @llvm.aarch64.neon.sqshrn.v8i8
  // LLVM-NEXT: ret <8 x i8> [[RES]]
  return vqshrn_n_s16(a, 3);
}

// LLVM-LABEL: @test_vqshrn_n_s32(
// CIR-LABEL: @vqshrn_n_s32(
int16x4_t test_vqshrn_n_s32(int32x4_t a) {
  // CIR: {{.*}}cir.call_llvm_intrinsic "aarch64.neon.sqshrn"
  // CIR-SAME: (!cir.vector<4 x !s32i>) -> !cir.vector<4 x !s16i>
  // LLVM: [[RES:%.*]] = call <4 x i16> @llvm.aarch64.neon.sqshrn.v4i16
  // LLVM-NEXT: ret <4 x i16> [[RES]]
  return vqshrn_n_s32(a, 9);
}

// LLVM-LABEL: @test_vqshrn_n_s64(
// CIR-LABEL: @vqshrn_n_s64(
int32x2_t test_vqshrn_n_s64(int64x2_t a) {
  // CIR: {{.*}}cir.call_llvm_intrinsic "aarch64.neon.sqshrn"
  // CIR-SAME: (!cir.vector<2 x !s64i>) -> !cir.vector<2 x !s32i>
  // LLVM: [[RES:%.*]] = call <2 x i32> @llvm.aarch64.neon.sqshrn.v2i32
  // LLVM-NEXT: ret <2 x i32> [[RES]]
  return vqshrn_n_s64(a, 19);
}

// LLVM-LABEL: @test_vqshrn_n_u16(
// CIR-LABEL: @vqshrn_n_u16(
uint8x8_t test_vqshrn_n_u16(uint16x8_t a) {
  // CIR: {{.*}}cir.call_llvm_intrinsic "aarch64.neon.uqshrn"
  // CIR-SAME: (!cir.vector<8 x !u16i>) -> !cir.vector<8 x !u8i>
  // LLVM: [[RES:%.*]] = call <8 x i8> @llvm.aarch64.neon.uqshrn.v8i8
  // LLVM-NEXT: ret <8 x i8> [[RES]]
  return vqshrn_n_u16(a, 3);
}

// LLVM-LABEL: @test_vqshrn_n_u32(
// CIR-LABEL: @vqshrn_n_u32(
uint16x4_t test_vqshrn_n_u32(uint32x4_t a) {
  // CIR: {{.*}}cir.call_llvm_intrinsic "aarch64.neon.uqshrn"
  // CIR-SAME: (!cir.vector<4 x !u32i>) -> !cir.vector<4 x !u16i>
  // LLVM: [[RES:%.*]] = call <4 x i16> @llvm.aarch64.neon.uqshrn.v4i16
  // LLVM-NEXT: ret <4 x i16> [[RES]]
  return vqshrn_n_u32(a, 9);
}

// LLVM-LABEL: @test_vqshrn_n_u64(
// CIR-LABEL: @vqshrn_n_u64(
uint32x2_t test_vqshrn_n_u64(uint64x2_t a) {
  // CIR: {{.*}}cir.call_llvm_intrinsic "aarch64.neon.uqshrn"
  // CIR-SAME: (!cir.vector<2 x !u64i>) -> !cir.vector<2 x !u32i>
  // LLVM: [[RES:%.*]] = call <2 x i32> @llvm.aarch64.neon.uqshrn.v2i32
  // LLVM-NEXT: ret <2 x i32> [[RES]]
  return vqshrn_n_u64(a, 19);
}

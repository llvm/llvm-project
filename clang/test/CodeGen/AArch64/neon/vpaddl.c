// REQUIRES: aarch64-registered-target || arm-registered-target

// RUN:                   %clang_cc1 -triple arm64-none-linux-gnu -target-feature +neon -disable-O0-optnone -flax-vector-conversions=none           -emit-llvm -o - %s | opt -S -passes=mem2reg,sroa | FileCheck %s --check-prefixes=LLVM
// RUN: %if cir-enabled %{%clang_cc1 -triple arm64-none-linux-gnu -target-feature +neon -disable-O0-optnone -flax-vector-conversions=none -fclangir -emit-llvm -o - %s | opt -S -passes=mem2reg,sroa | FileCheck %s --check-prefixes=LLVM %}
// RUN: %if cir-enabled %{%clang_cc1 -triple arm64-none-linux-gnu -target-feature +neon -disable-O0-optnone -flax-vector-conversions=none -fclangir -emit-cir  -o - %s |                               FileCheck %s --check-prefixes=CIR %}

#include <arm_neon.h>


//===------------------------------------------------------===//
// 2.1.1.12.2 Pairwise addition and widen
//===------------------------------------------------------===//
// LLVM-LABEL: @test_vpaddl_s8(
// CIR-LABEL: @vpaddl_s8(
int16x4_t test_vpaddl_s8(int8x8_t a) {
// CIR:      cir.call_llvm_intrinsic "aarch64.neon.saddlp" %{{.*}} : (!cir.vector<8 x !s8i>) -> !cir.vector<4 x !s16i>

// LLVM-SAME: <8 x i8> {{.*}} [[A:%.*]]) {{.*}} {
// LLVM:         [[VPADDL_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.saddlp.v4i16.v8i8(<8 x i8> [[A]])
// LLVM-NEXT:    ret <4 x i16> [[VPADDL_I]]
  return vpaddl_s8(a);
}

// LLVM-LABEL: @test_vpaddlq_s8(
// CIR-LABEL: @vpaddlq_s8(
int16x8_t test_vpaddlq_s8(int8x16_t a) {
// CIR:      cir.call_llvm_intrinsic "aarch64.neon.saddlp" %{{.*}} : (!cir.vector<16 x !s8i>) -> !cir.vector<8 x !s16i>

// LLVM-SAME: <16 x i8> {{.*}} [[A:%.*]]) {{.*}} {
// LLVM:         [[VPADDL_I:%.*]] = call <8 x i16> @llvm.aarch64.neon.saddlp.v8i16.v16i8(<16 x i8> [[A]])
// LLVM-NEXT:    ret <8 x i16> [[VPADDL_I]]
  return vpaddlq_s8(a);
}

// LLVM-LABEL: @test_vpaddl_s16(
// CIR-LABEL: @vpaddl_s16(
int32x2_t test_vpaddl_s16(int16x4_t a) {
// CIR:      cir.call_llvm_intrinsic "aarch64.neon.saddlp" %{{.*}} : (!cir.vector<4 x !s16i>) -> !cir.vector<2 x !s32i>

// LLVM-SAME: <4 x i16> {{.*}} [[A:%.*]]) {{.*}} {
// LLVM:         [[TMP0:%.*]] = bitcast <4 x i16> [[A]] to <8 x i8>
// LLVM-NEXT:    [[VPADDL_I:%.*]] = bitcast <8 x i8> [[TMP0]] to <4 x i16>
// LLVM-NEXT:    [[VPADDL1_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.saddlp.v2i32.v4i16(<4 x i16> [[VPADDL_I]])
// LLVM-NEXT:    ret <2 x i32> [[VPADDL1_I]]
  return vpaddl_s16(a);
}

// LLVM-LABEL: @test_vpaddlq_s16(
// CIR-LABEL: @vpaddlq_s16(
int32x4_t test_vpaddlq_s16(int16x8_t a) {
// CIR:      cir.call_llvm_intrinsic "aarch64.neon.saddlp" %{{.*}} : (!cir.vector<8 x !s16i>) -> !cir.vector<4 x !s32i>

// LLVM-SAME: <8 x i16> {{.*}} [[A:%.*]]) {{.*}} {
// LLVM:         [[TMP0:%.*]] = bitcast <8 x i16> [[A]] to <16 x i8>
// LLVM-NEXT:    [[VPADDL_I:%.*]] = bitcast <16 x i8> [[TMP0]] to <8 x i16>
// LLVM-NEXT:    [[VPADDL1_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.saddlp.v4i32.v8i16(<8 x i16> [[VPADDL_I]])
// LLVM-NEXT:    ret <4 x i32> [[VPADDL1_I]]
  return vpaddlq_s16(a);
}

// LLVM-LABEL: @test_vpaddl_s32(
// CIR-LABEL: @vpaddl_s32(
int64x1_t test_vpaddl_s32(int32x2_t a) {
// CIR:      cir.call_llvm_intrinsic "aarch64.neon.saddlp" %{{.*}} : (!cir.vector<2 x !s32i>) -> !cir.vector<1 x !s64i>

// LLVM-SAME: <2 x i32> {{.*}} [[A:%.*]]) {{.*}} {
// LLVM:         [[TMP0:%.*]] = bitcast <2 x i32> [[A]] to <8 x i8>
// LLVM-NEXT:    [[VPADDL_I:%.*]] = bitcast <8 x i8> [[TMP0]] to <2 x i32>
// LLVM-NEXT:    [[VPADDL1_I:%.*]] = call <1 x i64> @llvm.aarch64.neon.saddlp.v1i64.v2i32(<2 x i32> [[VPADDL_I]])
// LLVM-NEXT:    ret <1 x i64> [[VPADDL1_I]]
  return vpaddl_s32(a);
}

// LLVM-LABEL: @test_vpaddlq_s32(
// CIR-LABEL: @vpaddlq_s32(
int64x2_t test_vpaddlq_s32(int32x4_t a) {
// CIR:      cir.call_llvm_intrinsic "aarch64.neon.saddlp" %{{.*}} : (!cir.vector<4 x !s32i>) -> !cir.vector<2 x !s64i>

// LLVM-SAME: <4 x i32> {{.*}} [[A:%.*]]) {{.*}} {
// LLVM:         [[TMP0:%.*]] = bitcast <4 x i32> [[A]] to <16 x i8>
// LLVM-NEXT:    [[VPADDL_I:%.*]] = bitcast <16 x i8> [[TMP0]] to <4 x i32>
// LLVM-NEXT:    [[VPADDL1_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.saddlp.v2i64.v4i32(<4 x i32> [[VPADDL_I]])
// LLVM-NEXT:    ret <2 x i64> [[VPADDL1_I]]
  return vpaddlq_s32(a);
}

// LLVM-LABEL: @test_vpaddl_u8(
// CIR-LABEL: @vpaddl_u8(
uint16x4_t test_vpaddl_u8(uint8x8_t a) {
// CIR:      cir.call_llvm_intrinsic "aarch64.neon.uaddlp" %{{.*}} : (!cir.vector<8 x !u8i>) -> !cir.vector<4 x !u16i>

// LLVM-SAME: <8 x i8> {{.*}} [[A:%.*]]) {{.*}} {
// LLVM:         [[VPADDL_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.uaddlp.v4i16.v8i8(<8 x i8> [[A]])
// LLVM-NEXT:    ret <4 x i16> [[VPADDL_I]]
  return vpaddl_u8(a);
}

// LLVM-LABEL: @test_vpaddlq_u8(
// CIR-LABEL: @vpaddlq_u8(
uint16x8_t test_vpaddlq_u8(uint8x16_t a) {
// CIR:      cir.call_llvm_intrinsic "aarch64.neon.uaddlp" %{{.*}} : (!cir.vector<16 x !u8i>) -> !cir.vector<8 x !u16i>

// LLVM-SAME: <16 x i8> {{.*}} [[A:%.*]]) {{.*}} {
// LLVM:         [[VPADDL_I:%.*]] = call <8 x i16> @llvm.aarch64.neon.uaddlp.v8i16.v16i8(<16 x i8> [[A]])
// LLVM-NEXT:    ret <8 x i16> [[VPADDL_I]]
  return vpaddlq_u8(a);
}

// LLVM-LABEL: @test_vpaddl_u16(
// CIR-LABEL: @vpaddl_u16(
uint32x2_t test_vpaddl_u16(uint16x4_t a) {
// CIR:      cir.call_llvm_intrinsic "aarch64.neon.uaddlp" %{{.*}} : (!cir.vector<4 x !u16i>) -> !cir.vector<2 x !u32i>

// LLVM-SAME: <4 x i16> {{.*}} [[A:%.*]]) {{.*}} {
// LLVM:         [[TMP0:%.*]] = bitcast <4 x i16> [[A]] to <8 x i8>
// LLVM-NEXT:    [[VPADDL_I:%.*]] = bitcast <8 x i8> [[TMP0]] to <4 x i16>
// LLVM-NEXT:    [[VPADDL1_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.uaddlp.v2i32.v4i16(<4 x i16> [[VPADDL_I]])
// LLVM-NEXT:    ret <2 x i32> [[VPADDL1_I]]
  return vpaddl_u16(a);
}

// LLVM-LABEL: @test_vpaddlq_u16(
// CIR-LABEL: @vpaddlq_u16(
uint32x4_t test_vpaddlq_u16(uint16x8_t a) {
// CIR:      cir.call_llvm_intrinsic "aarch64.neon.uaddlp" %{{.*}} : (!cir.vector<8 x !u16i>) -> !cir.vector<4 x !u32i>

// LLVM-SAME: <8 x i16> {{.*}} [[A:%.*]]) {{.*}} {
// LLVM:         [[TMP0:%.*]] = bitcast <8 x i16> [[A]] to <16 x i8>
// LLVM-NEXT:    [[VPADDL_I:%.*]] = bitcast <16 x i8> [[TMP0]] to <8 x i16>
// LLVM-NEXT:    [[VPADDL1_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.uaddlp.v4i32.v8i16(<8 x i16> [[VPADDL_I]])
// LLVM-NEXT:    ret <4 x i32> [[VPADDL1_I]]
  return vpaddlq_u16(a);
}

// LLVM-LABEL: @test_vpaddl_u32(
// CIR-LABEL: @vpaddl_u32(
uint64x1_t test_vpaddl_u32(uint32x2_t a) {
// CIR:      cir.call_llvm_intrinsic "aarch64.neon.uaddlp" %{{.*}} : (!cir.vector<2 x !u32i>) -> !cir.vector<1 x !u64i>

// LLVM-SAME: <2 x i32> {{.*}} [[A:%.*]]) {{.*}} {
// LLVM:         [[TMP0:%.*]] = bitcast <2 x i32> [[A]] to <8 x i8>
// LLVM-NEXT:    [[VPADDL_I:%.*]] = bitcast <8 x i8> [[TMP0]] to <2 x i32>
// LLVM-NEXT:    [[VPADDL1_I:%.*]] = call <1 x i64> @llvm.aarch64.neon.uaddlp.v1i64.v2i32(<2 x i32> [[VPADDL_I]])
// LLVM-NEXT:    ret <1 x i64> [[VPADDL1_I]]
  return vpaddl_u32(a);
}

// LLVM-LABEL: @test_vpaddlq_u32(
// CIR-LABEL: @vpaddlq_u32(
uint64x2_t test_vpaddlq_u32(uint32x4_t a) {
// CIR:      cir.call_llvm_intrinsic "aarch64.neon.uaddlp" %{{.*}} : (!cir.vector<4 x !u32i>) -> !cir.vector<2 x !u64i>

// LLVM-SAME: <4 x i32> {{.*}} [[A:%.*]]) {{.*}} {
// LLVM:         [[TMP0:%.*]] = bitcast <4 x i32> [[A]] to <16 x i8>
// LLVM-NEXT:    [[VPADDL_I:%.*]] = bitcast <16 x i8> [[TMP0]] to <4 x i32>
// LLVM-NEXT:    [[VPADDL1_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.uaddlp.v2i64.v4i32(<4 x i32> [[VPADDL_I]])
// LLVM-NEXT:    ret <2 x i64> [[VPADDL1_I]]
  return vpaddlq_u32(a);
}

// LLVM-LABEL: @test_vpadal_s8(
// CIR-LABEL: @vpadal_s8(
int16x4_t test_vpadal_s8(int16x4_t a, int8x8_t b) {
// CIR:      [[VPADAL_I:%.*]] = cir.call_llvm_intrinsic "aarch64.neon.saddlp" %{{.*}} : (!cir.vector<8 x !s8i>) -> !cir.vector<4 x !s16i>
// CIR:      [[TMP:%.*]] = cir.add [[VPADAL_I]], %{{.*}} : !cir.vector<4 x !s16i>

// LLVM-SAME: <4 x i16> {{.*}} [[A:%.*]], <8 x i8> {{.*}} [[B:%.*]]) {{.*}} {
// LLVM:    [[TMP0:%.*]] = bitcast <4 x i16> [[A]] to <8 x i8>
// LLVM-NEXT:    [[VPADAL_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.saddlp.v4i16.v8i8(<8 x i8> [[B]])
// LLVM-NEXT:    [[TMP1:%.*]] = bitcast <8 x i8> [[TMP0]] to <4 x i16>
// LLVM-NEXT:    [[TMP2:%.*]] = add <4 x i16> [[VPADAL_I]], [[TMP1]]
// LLVM-NEXT:    ret <4 x i16> [[TMP2]]
  return vpadal_s8(a, b);
}

// LLVM-LABEL: @test_vpadalq_s8(
// CIR-LABEL: @vpadalq_s8(
int16x8_t test_vpadalq_s8(int16x8_t a, int8x16_t b) {
// CIR:      [[VPADAL_I:%.*]] = cir.call_llvm_intrinsic "aarch64.neon.saddlp" %{{.*}} : (!cir.vector<16 x !s8i>) -> !cir.vector<8 x !s16i>
// CIR:      [[TMP10:%.*]] = cir.add [[VPADAL_I]], %{{.*}} : !cir.vector<8 x !s16i>

// LLVM-SAME: <8 x i16> {{.*}} [[A:%.*]], <16 x i8> {{.*}} [[B:%.*]]) {{.*}} {
// LLVM:    [[TMP0:%.*]] = bitcast <8 x i16> [[A]] to <16 x i8>
// LLVM-NEXT:    [[VPADAL_I:%.*]] = call <8 x i16> @llvm.aarch64.neon.saddlp.v8i16.v16i8(<16 x i8> [[B]])
// LLVM-NEXT:    [[TMP1:%.*]] = bitcast <16 x i8> [[TMP0]] to <8 x i16>
// LLVM-NEXT:    [[TMP2:%.*]] = add <8 x i16> [[VPADAL_I]], [[TMP1]]
// LLVM-NEXT:    ret <8 x i16> [[TMP2]]
  return vpadalq_s8(a, b);
}

// LLVM-LABEL: @test_vpadal_s16(
// CIR-LABEL: @vpadal_s16(
int32x2_t test_vpadal_s16(int32x2_t a, int16x4_t b) {
// CIR:      [[VPADAL_I:%.*]] = cir.call_llvm_intrinsic "aarch64.neon.saddlp" %{{.*}} : (!cir.vector<4 x !s16i>) -> !cir.vector<2 x !s32i>
// CIR:      cir.add [[VPADAL_I]], %{{.*}} : !cir.vector<2 x !s32i>

// LLVM-SAME: <2 x i32> {{.*}} [[A:%.*]], <4 x i16> {{.*}} [[B:%.*]]) {{.*}} {
// LLVM:    [[TMP0:%.*]] = bitcast <2 x i32> [[A]] to <8 x i8>
// LLVM-NEXT:    [[TMP1:%.*]] = bitcast <4 x i16> [[B]] to <8 x i8>
// LLVM-NEXT:    [[VPADAL_I:%.*]] = bitcast <8 x i8> [[TMP1]] to <4 x i16>
// LLVM-NEXT:    [[VPADAL1_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.saddlp.v2i32.v4i16(<4 x i16> [[VPADAL_I]])
// LLVM-NEXT:    [[TMP2:%.*]] = bitcast <8 x i8> [[TMP0]] to <2 x i32>
// LLVM-NEXT:    [[TMP3:%.*]] = add <2 x i32> [[VPADAL1_I]], [[TMP2]]
// LLVM-NEXT:    ret <2 x i32> [[TMP3]]
  return vpadal_s16(a, b);
}

// LLVM-LABEL: @test_vpadalq_s16(
// CIR-LABEL: @vpadalq_s16(
int32x4_t test_vpadalq_s16(int32x4_t a, int16x8_t b) {
// CIR:      [[VPADAL_I:%.*]] = cir.call_llvm_intrinsic "aarch64.neon.saddlp" %{{.*}} : (!cir.vector<8 x !s16i>) -> !cir.vector<4 x !s32i>
// CIR:      cir.add [[VPADAL_I]], %{{.*}} : !cir.vector<4 x !s32i>

// LLVM-SAME: <4 x i32> {{.*}} [[A:%.*]], <8 x i16> {{.*}} [[B:%.*]]) {{.*}} {
// LLVM:    [[TMP0:%.*]] = bitcast <4 x i32> [[A]] to <16 x i8>
// LLVM-NEXT:    [[TMP1:%.*]] = bitcast <8 x i16> [[B]] to <16 x i8>
// LLVM-NEXT:    [[VPADAL_I:%.*]] = bitcast <16 x i8> [[TMP1]] to <8 x i16>
// LLVM-NEXT:    [[VPADAL1_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.saddlp.v4i32.v8i16(<8 x i16> [[VPADAL_I]])
// LLVM-NEXT:    [[TMP2:%.*]] = bitcast <16 x i8> [[TMP0]] to <4 x i32>
// LLVM-NEXT:    [[TMP3:%.*]] = add <4 x i32> [[VPADAL1_I]], [[TMP2]]
// LLVM-NEXT:    ret <4 x i32> [[TMP3]]
  return vpadalq_s16(a, b);
}

// LLVM-LABEL: @test_vpadal_s32(
// CIR-LABEL: @vpadal_s32(
int64x1_t test_vpadal_s32(int64x1_t a, int32x2_t b) {
// CIR: [[VPADAL_I:%.*]] = cir.call_llvm_intrinsic "aarch64.neon.saddlp" %{{.*}} : (!cir.vector<2 x !s32i>) -> !cir.vector<1 x !s64i>
// CIR: cir.add [[VPADAL_I]], %{{.*}} : !cir.vector<1 x !s64i>

// LLVM-SAME: <1 x i64> {{.*}} [[A:%.*]], <2 x i32> {{.*}} [[B:%.*]]) {{.*}} {
// LLVM:    [[TMP0:%.*]] = bitcast <1 x i64> [[A]] to <8 x i8>
// LLVM-NEXT:    [[TMP1:%.*]] = bitcast <2 x i32> [[B]] to <8 x i8>
// LLVM-NEXT:    [[VPADAL_I:%.*]] = bitcast <8 x i8> [[TMP1]] to <2 x i32>
// LLVM-NEXT:    [[VPADAL1_I:%.*]] = call <1 x i64> @llvm.aarch64.neon.saddlp.v1i64.v2i32(<2 x i32> [[VPADAL_I]])
// LLVM-NEXT:    [[TMP2:%.*]] = bitcast <8 x i8> [[TMP0]] to <1 x i64>
// LLVM-NEXT:    [[TMP3:%.*]] = add <1 x i64> [[VPADAL1_I]], [[TMP2]]
// LLVM-NEXT:    ret <1 x i64> [[TMP3]]
  return vpadal_s32(a, b);
}

// LLVM-LABEL: @test_vpadalq_s32(
// CIR-LABEL: @vpadalq_s32(
int64x2_t test_vpadalq_s32(int64x2_t a, int32x4_t b) {
// CIR: [[VPADAL_I:%.*]] = cir.call_llvm_intrinsic "aarch64.neon.saddlp" %{{.*}} : (!cir.vector<4 x !s32i>) -> !cir.vector<2 x !s64i>
// CIR: cir.add [[VPADAL_I]], %{{.*}} : !cir.vector<2 x !s64i>

// LLVM-SAME: <2 x i64> {{.*}} [[A:%.*]], <4 x i32> {{.*}} [[B:%.*]]) {{.*}} {
// LLVM:    [[TMP0:%.*]] = bitcast <2 x i64> [[A]] to <16 x i8>
// LLVM-NEXT:    [[TMP1:%.*]] = bitcast <4 x i32> [[B]] to <16 x i8>
// LLVM-NEXT:    [[VPADAL_I:%.*]] = bitcast <16 x i8> [[TMP1]] to <4 x i32>
// LLVM-NEXT:    [[VPADAL1_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.saddlp.v2i64.v4i32(<4 x i32> [[VPADAL_I]])
// LLVM-NEXT:    [[TMP2:%.*]] = bitcast <16 x i8> [[TMP0]] to <2 x i64>
// LLVM-NEXT:    [[TMP3:%.*]] = add <2 x i64> [[VPADAL1_I]], [[TMP2]]
// LLVM-NEXT:    ret <2 x i64> [[TMP3]]
  return vpadalq_s32(a, b);
}

// LLVM-LABEL: @test_vpadal_u8(
// CIR-LABEL: @vpadal_u8(
uint16x4_t test_vpadal_u8(uint16x4_t a, uint8x8_t b) {
// CIR: [[VPADAL_I:%.*]] = cir.call_llvm_intrinsic "aarch64.neon.uaddlp" %{{.*}} : (!cir.vector<8 x !u8i>) -> !cir.vector<4 x !u16i>
// CIR: cir.add [[VPADAL_I]], %{{.*}} : !cir.vector<4 x !u16i>

// LLVM-SAME: <4 x i16> {{.*}} [[A:%.*]], <8 x i8> {{.*}} [[B:%.*]]) {{.*}} {
// LLVM:    [[TMP0:%.*]] = bitcast <4 x i16> [[A]] to <8 x i8>
// LLVM-NEXT:    [[VPADAL_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.uaddlp.v4i16.v8i8(<8 x i8> [[B]])
// LLVM-NEXT:    [[TMP1:%.*]] = bitcast <8 x i8> [[TMP0]] to <4 x i16>
// LLVM-NEXT:    [[TMP2:%.*]] = add <4 x i16> [[VPADAL_I]], [[TMP1]]
// LLVM-NEXT:    ret <4 x i16> [[TMP2]]
  return vpadal_u8(a, b);
}

// LLVM-LABEL: @test_vpadalq_u8(
// CIR-LABEL: @vpadalq_u8(
uint16x8_t test_vpadalq_u8(uint16x8_t a, uint8x16_t b) {
// CIR: [[VPADAL_I:%.*]] = cir.call_llvm_intrinsic "aarch64.neon.uaddlp" %{{.*}} : (!cir.vector<16 x !u8i>) -> !cir.vector<8 x !u16i>
// CIR: cir.add [[VPADAL_I]], %{{.*}} : !cir.vector<8 x !u16i>

// LLVM-SAME: <8 x i16> {{.*}} [[A:%.*]], <16 x i8> {{.*}} [[B:%.*]]) {{.*}} {
// LLVM:    [[TMP0:%.*]] = bitcast <8 x i16> [[A]] to <16 x i8>
// LLVM-NEXT:    [[VPADAL_I:%.*]] = call <8 x i16> @llvm.aarch64.neon.uaddlp.v8i16.v16i8(<16 x i8> [[B]])
// LLVM-NEXT:    [[TMP1:%.*]] = bitcast <16 x i8> [[TMP0]] to <8 x i16>
// LLVM-NEXT:    [[TMP2:%.*]] = add <8 x i16> [[VPADAL_I]], [[TMP1]]
// LLVM-NEXT:    ret <8 x i16> [[TMP2]]
  return vpadalq_u8(a, b);
}

// LLVM-LABEL: @test_vpadal_u16(
// CIR-LABEL: @vpadal_u16(
uint32x2_t test_vpadal_u16(uint32x2_t a, uint16x4_t b) {
// CIR: [[VPADAL_I:%.*]] = cir.call_llvm_intrinsic "aarch64.neon.uaddlp" %{{.*}} : (!cir.vector<4 x !u16i>) -> !cir.vector<2 x !u32i>
// CIR: cir.add [[VPADAL_I]], %{{.*}} : !cir.vector<2 x !u32i>

// LLVM-SAME: <2 x i32> {{.*}} [[A:%.*]], <4 x i16> {{.*}} [[B:%.*]]) {{.*}} {
// LLVM:    [[TMP0:%.*]] = bitcast <2 x i32> [[A]] to <8 x i8>
// LLVM-NEXT:    [[TMP1:%.*]] = bitcast <4 x i16> [[B]] to <8 x i8>
// LLVM-NEXT:    [[VPADAL_I:%.*]] = bitcast <8 x i8> [[TMP1]] to <4 x i16>
// LLVM-NEXT:    [[VPADAL1_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.uaddlp.v2i32.v4i16(<4 x i16> [[VPADAL_I]])
// LLVM-NEXT:    [[TMP2:%.*]] = bitcast <8 x i8> [[TMP0]] to <2 x i32>
// LLVM-NEXT:    [[TMP3:%.*]] = add <2 x i32> [[VPADAL1_I]], [[TMP2]]
// LLVM-NEXT:    ret <2 x i32> [[TMP3]]
  return vpadal_u16(a, b);
}

// LLVM-LABEL: @test_vpadalq_u16(
// CIR-LABEL: @vpadalq_u16(
uint32x4_t test_vpadalq_u16(uint32x4_t a, uint16x8_t b) {
// CIR: [[VPADAL_I:%.*]] = cir.call_llvm_intrinsic "aarch64.neon.uaddlp" %{{.*}} : (!cir.vector<8 x !u16i>) -> !cir.vector<4 x !u32i>
// CIR: cir.add [[VPADAL_I]], %{{.*}} : !cir.vector<4 x !u32i>

// LLVM-SAME: <4 x i32> {{.*}} [[A:%.*]], <8 x i16> {{.*}} [[B:%.*]]) {{.*}} {
// LLVM:    [[TMP0:%.*]] = bitcast <4 x i32> [[A]] to <16 x i8>
// LLVM-NEXT:    [[TMP1:%.*]] = bitcast <8 x i16> [[B]] to <16 x i8>
// LLVM-NEXT:    [[VPADAL_I:%.*]] = bitcast <16 x i8> [[TMP1]] to <8 x i16>
// LLVM-NEXT:    [[VPADAL1_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.uaddlp.v4i32.v8i16(<8 x i16> [[VPADAL_I]])
// LLVM-NEXT:    [[TMP2:%.*]] = bitcast <16 x i8> [[TMP0]] to <4 x i32>
// LLVM-NEXT:    [[TMP3:%.*]] = add <4 x i32> [[VPADAL1_I]], [[TMP2]]
// LLVM-NEXT:    ret <4 x i32> [[TMP3]]
  return vpadalq_u16(a, b);
}

// LLVM-LABEL: @test_vpadal_u32(
// CIR-LABEL: @vpadal_u32(
uint64x1_t test_vpadal_u32(uint64x1_t a, uint32x2_t b) {
// CIR: [[VPADAL_I:%.*]] = cir.call_llvm_intrinsic "aarch64.neon.uaddlp" %{{.*}} : (!cir.vector<2 x !u32i>) -> !cir.vector<1 x !u64i>
// CIR: cir.add [[VPADAL_I]], %{{.*}} : !cir.vector<1 x !u64i>

// LLVM-SAME: <1 x i64> {{.*}} [[A:%.*]], <2 x i32> {{.*}} [[B:%.*]]) {{.*}} {
// LLVM:    [[TMP0:%.*]] = bitcast <1 x i64> [[A]] to <8 x i8>
// LLVM-NEXT:    [[TMP1:%.*]] = bitcast <2 x i32> [[B]] to <8 x i8>
// LLVM-NEXT:    [[VPADAL_I:%.*]] = bitcast <8 x i8> [[TMP1]] to <2 x i32>
// LLVM-NEXT:    [[VPADAL1_I:%.*]] = call <1 x i64> @llvm.aarch64.neon.uaddlp.v1i64.v2i32(<2 x i32> [[VPADAL_I]])
// LLVM-NEXT:    [[TMP2:%.*]] = bitcast <8 x i8> [[TMP0]] to <1 x i64>
// LLVM-NEXT:    [[TMP3:%.*]] = add <1 x i64> [[VPADAL1_I]], [[TMP2]]
// LLVM-NEXT:    ret <1 x i64> [[TMP3]]
  return vpadal_u32(a, b);
}

// LLVM-LABEL: @test_vpadalq_u32(
// CIR-LABEL: @vpadalq_u32(
uint64x2_t test_vpadalq_u32(uint64x2_t a, uint32x4_t b) {
// CIR: [[VPADAL_I:%.*]] = cir.call_llvm_intrinsic "aarch64.neon.uaddlp" %{{.*}} : (!cir.vector<4 x !u32i>) -> !cir.vector<2 x !u64i>
// CIR: cir.add [[VPADAL_I]], %{{.*}} : !cir.vector<2 x !u64i>

// LLVM-SAME: <2 x i64> {{.*}} [[A:%.*]], <4 x i32> {{.*}} [[B:%.*]]) {{.*}} {
// LLVM:    [[TMP0:%.*]] = bitcast <2 x i64> [[A]] to <16 x i8>
// LLVM-NEXT:    [[TMP1:%.*]] = bitcast <4 x i32> [[B]] to <16 x i8>
// LLVM-NEXT:    [[VPADAL_I:%.*]] = bitcast <16 x i8> [[TMP1]] to <4 x i32>
// LLVM-NEXT:    [[VPADAL1_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.uaddlp.v2i64.v4i32(<4 x i32> [[VPADAL_I]])
// LLVM-NEXT:    [[TMP2:%.*]] = bitcast <16 x i8> [[TMP0]] to <2 x i64>
// LLVM-NEXT:    [[TMP3:%.*]] = add <2 x i64> [[VPADAL1_I]], [[TMP2]]
// LLVM-NEXT:    ret <2 x i64> [[TMP3]]
  return vpadalq_u32(a, b);
}

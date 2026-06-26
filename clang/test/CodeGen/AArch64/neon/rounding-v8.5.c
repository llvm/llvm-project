// REQUIRES: aarch64-registered-target

// RUN:                   %clang_cc1_cg_arm64_neon -target-feature +v8.5a           -emit-llvm %s -disable-O0-optnone | opt -S -passes=mem2reg,sroa | FileCheck %s --check-prefixes=LLVM
// RUN: %if cir-enabled %{%clang_cc1_cg_arm64_neon -target-feature +v8.5a -fclangir -emit-llvm %s -disable-O0-optnone | opt -S -passes=mem2reg,sroa | FileCheck %s --check-prefixes=LLVM %}
// RUN: %if cir-enabled %{%clang_cc1_cg_arm64_neon -target-feature +v8.5a -fclangir -emit-cir  %s -disable-O0-optnone |                               FileCheck %s --check-prefixes=CIR %}

#include <arm_neon.h>

//===------------------------------------------------------===//
// 2.12.1.1. Rounding
//https://arm-software.github.io/acle/neon_intrinsics/advsimd.html#rounding-3
//===------------------------------------------------------===//

// LLVM-LABEL: @test_vrnd32x_f32(
// CIR-LABEL: @vrnd32x_f32(
float32x2_t test_vrnd32x_f32(float32x2_t a) {
// CIR: [[LOAD:%.*]] = cir.load {{.*}} : !cir.ptr<!cir.vector<8 x !s8i>>, !cir.vector<8 x !s8i>
// CIR: [[CAST:%.*]] = cir.cast bitcast [[LOAD]] : !cir.vector<8 x !s8i> -> !cir.vector<2 x !cir.float>
// CIR: cir.call_llvm_intrinsic "aarch64.neon.frint32x" [[CAST]] : (!cir.vector<2 x !cir.float>) -> !cir.vector<2 x !cir.float>

// LLVM-SAME: <2 x float> noundef [[A:%.*]])
// LLVM: [[TMP0:%.*]] = bitcast <2 x float> [[A]] to <2 x i32>
// LLVM: [[TMP1:%.*]] = bitcast <2 x i32> [[TMP0]] to <8 x i8>
// LLVM: [[VRND32X_I:%.*]] = bitcast <8 x i8> [[TMP1]] to <2 x float>
// LLVM: [[VRND32X1_I:%.*]] = call <2 x float> @llvm.aarch64.neon.frint32x.v2f32(<2 x float> [[VRND32X_I]])
  return vrnd32x_f32(a);
}

// LLVM-LABEL: @test_vrnd32xq_f32(
// CIR-LABEL: @vrnd32xq_f32(
float32x4_t test_vrnd32xq_f32(float32x4_t a) {
// CIR: [[LOAD:%.*]] = cir.load {{.*}} : !cir.ptr<!cir.vector<16 x !s8i>>, !cir.vector<16 x !s8i>
// CIR: [[CAST:%.*]] = cir.cast bitcast [[LOAD]] : !cir.vector<16 x !s8i> -> !cir.vector<4 x !cir.float>
// CIR: cir.call_llvm_intrinsic "aarch64.neon.frint32x" [[CAST]] : (!cir.vector<4 x !cir.float>) -> !cir.vector<4 x !cir.float>

// LLVM-SAME: <4 x float> noundef [[A:%.*]])
// LLVM: [[TMP0:%.*]] = bitcast <4 x float> [[A]] to <4 x i32>
// LLVM: [[TMP1:%.*]] = bitcast <4 x i32> [[TMP0]] to <16 x i8>
// LLVM: [[VRND32XQ_I:%.*]] = bitcast <16 x i8> [[TMP1]] to <4 x float>
// LLVM: [[VRND32XQ1_I:%.*]] = call <4 x float> @llvm.aarch64.neon.frint32x.v4f32(<4 x float> [[VRND32XQ_I]])
  return vrnd32xq_f32(a);
}

// LLVM-LABEL: @test_vrnd32z_f32(
// CIR-LABEL: @vrnd32z_f32(
float32x2_t test_vrnd32z_f32(float32x2_t a) {
// CIR: [[LOAD:%.*]] = cir.load {{.*}} : !cir.ptr<!cir.vector<8 x !s8i>>, !cir.vector<8 x !s8i>
// CIR: [[CAST:%.*]] = cir.cast bitcast [[LOAD]] : !cir.vector<8 x !s8i> -> !cir.vector<2 x !cir.float>
// CIR: cir.call_llvm_intrinsic "aarch64.neon.frint32z" [[CAST]] : (!cir.vector<2 x !cir.float>) -> !cir.vector<2 x !cir.float>

// LLVM-SAME: <2 x float> noundef [[A:%.*]])
// LLVM: [[TMP0:%.*]] = bitcast <2 x float> [[A]] to <2 x i32>
// LLVM: [[TMP1:%.*]] = bitcast <2 x i32> [[TMP0]] to <8 x i8>
// LLVM: [[VRND32Z_I:%.*]] = bitcast <8 x i8> [[TMP1]] to <2 x float>
// LLVM: [[VRND32Z1_I:%.*]] = call <2 x float> @llvm.aarch64.neon.frint32z.v2f32(<2 x float> [[VRND32Z_I]])
  return vrnd32z_f32(a);
}

// LLVM-LABEL: @test_vrnd32zq_f32(
// CIR-LABEL: @vrnd32zq_f32(
float32x4_t test_vrnd32zq_f32(float32x4_t a) {
// CIR: [[LOAD:%.*]] = cir.load {{.*}} : !cir.ptr<!cir.vector<16 x !s8i>>, !cir.vector<16 x !s8i>
// CIR: [[CAST:%.*]] = cir.cast bitcast [[LOAD]] : !cir.vector<16 x !s8i> -> !cir.vector<4 x !cir.float>
// CIR: cir.call_llvm_intrinsic "aarch64.neon.frint32z" [[CAST]] : (!cir.vector<4 x !cir.float>) -> !cir.vector<4 x !cir.float>

// LLVM-SAME: <4 x float> noundef [[A:%.*]])
// LLVM: [[TMP0:%.*]] = bitcast <4 x float> [[A]] to <4 x i32>
// LLVM: [[TMP1:%.*]] = bitcast <4 x i32> [[TMP0]] to <16 x i8>
// LLVM: [[VRND32ZQ_I:%.*]] = bitcast <16 x i8> [[TMP1]] to <4 x float>
// LLVM: [[VRND32ZQ1_I:%.*]] = call <4 x float> @llvm.aarch64.neon.frint32z.v4f32(<4 x float> [[VRND32ZQ_I]])
  return vrnd32zq_f32(a);
}

// LLVM-LABEL: @test_vrnd64x_f32(
// CIR-LABEL: @vrnd64x_f32(
float32x2_t test_vrnd64x_f32(float32x2_t a) {
// CIR: [[LOAD:%.*]] = cir.load {{.*}} : !cir.ptr<!cir.vector<8 x !s8i>>, !cir.vector<8 x !s8i>
// CIR: [[CAST:%.*]] = cir.cast bitcast [[LOAD]] : !cir.vector<8 x !s8i> -> !cir.vector<2 x !cir.float>
// CIR: cir.call_llvm_intrinsic "aarch64.neon.frint64x" [[CAST]] : (!cir.vector<2 x !cir.float>) -> !cir.vector<2 x !cir.float>

// LLVM-SAME: <2 x float> noundef [[A:%.*]])
// LLVM: [[TMP0:%.*]] = bitcast <2 x float> [[A]] to <2 x i32>
// LLVM: [[TMP1:%.*]] = bitcast <2 x i32> [[TMP0]] to <8 x i8>
// LLVM: [[VRND64X_I:%.*]] = bitcast <8 x i8> [[TMP1]] to <2 x float>
// LLVM: [[VRND64X1_I:%.*]] = call <2 x float> @llvm.aarch64.neon.frint64x.v2f32(<2 x float> [[VRND64X_I]])
  return vrnd64x_f32(a);
}

// LLVM-LABEL: @test_vrnd64xq_f32(
// CIR-LABEL: @vrnd64xq_f32(
float32x4_t test_vrnd64xq_f32(float32x4_t a) {
// CIR: [[LOAD:%.*]] = cir.load {{.*}} : !cir.ptr<!cir.vector<16 x !s8i>>, !cir.vector<16 x !s8i>
// CIR: [[CAST:%.*]] = cir.cast bitcast [[LOAD]] : !cir.vector<16 x !s8i> -> !cir.vector<4 x !cir.float>
// CIR: cir.call_llvm_intrinsic "aarch64.neon.frint64x" [[CAST]] : (!cir.vector<4 x !cir.float>) -> !cir.vector<4 x !cir.float>

// LLVM-SAME: <4 x float> noundef [[A:%.*]])
// LLVM: [[TMP0:%.*]] = bitcast <4 x float> [[A]] to <4 x i32>
// LLVM: [[TMP1:%.*]] = bitcast <4 x i32> [[TMP0]] to <16 x i8>
// LLVM: [[VRND64XQ_I:%.*]] = bitcast <16 x i8> [[TMP1]] to <4 x float>
// LLVM: [[VRND64XQ1_I:%.*]] = call <4 x float> @llvm.aarch64.neon.frint64x.v4f32(<4 x float> [[VRND64XQ_I]])
  return vrnd64xq_f32(a);
}

// LLVM-LABEL: @test_vrnd64z_f32(
// CIR-LABEL: @vrnd64z_f32(
float32x2_t test_vrnd64z_f32(float32x2_t a) {
// CIR: [[LOAD:%.*]] = cir.load {{.*}} : !cir.ptr<!cir.vector<8 x !s8i>>, !cir.vector<8 x !s8i>
// CIR: [[CAST:%.*]] = cir.cast bitcast [[LOAD]] : !cir.vector<8 x !s8i> -> !cir.vector<2 x !cir.float>
// CIR: cir.call_llvm_intrinsic "aarch64.neon.frint64z" [[CAST]] : (!cir.vector<2 x !cir.float>) -> !cir.vector<2 x !cir.float>

// LLVM-SAME: <2 x float> noundef [[A:%.*]])
// LLVM: [[TMP0:%.*]] = bitcast <2 x float> [[A]] to <2 x i32>
// LLVM: [[TMP1:%.*]] = bitcast <2 x i32> [[TMP0]] to <8 x i8>
// LLVM: [[VRND64Z_I:%.*]] = bitcast <8 x i8> [[TMP1]] to <2 x float>
// LLVM: [[VRND64Z1_I:%.*]] = call <2 x float> @llvm.aarch64.neon.frint64z.v2f32(<2 x float> [[VRND64Z_I]])
  return vrnd64z_f32(a);
}

// LLVM-LABEL: @test_vrnd64zq_f32(
// CIR-LABEL: @vrnd64zq_f32(
float32x4_t test_vrnd64zq_f32(float32x4_t a) {
// CIR: [[LOAD:%.*]] = cir.load {{.*}} : !cir.ptr<!cir.vector<16 x !s8i>>, !cir.vector<16 x !s8i>
// CIR: [[CAST:%.*]] = cir.cast bitcast [[LOAD]] : !cir.vector<16 x !s8i> -> !cir.vector<4 x !cir.float>
// CIR: cir.call_llvm_intrinsic "aarch64.neon.frint64z" [[CAST]] : (!cir.vector<4 x !cir.float>) -> !cir.vector<4 x !cir.float>

// LLVM-SAME: <4 x float> noundef [[A:%.*]])
// LLVM: [[TMP0:%.*]] = bitcast <4 x float> [[A]] to <4 x i32>
// LLVM: [[TMP1:%.*]] = bitcast <4 x i32> [[TMP0]] to <16 x i8>
// LLVM: [[VRND64ZQ_I:%.*]] = bitcast <16 x i8> [[TMP1]] to <4 x float>
// LLVM: [[VRND64ZQ1_I:%.*]] = call <4 x float> @llvm.aarch64.neon.frint64z.v4f32(<4 x float> [[VRND64ZQ_I]])
  return vrnd64zq_f32(a);
}

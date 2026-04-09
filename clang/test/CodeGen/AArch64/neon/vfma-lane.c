// REQUIRES: aarch64-registered-target

// RUN: %clang_cc1 -triple arm64-none-linux-gnu -target-feature +neon -target-feature +fullfp16 -disable-O0-optnone -flax-vector-conversions=none -emit-llvm -o - %s | opt -S -passes=mem2reg,sroa | FileCheck %s --check-prefixes=LLVM
// RUN: %if cir-enabled %{%clang_cc1 -triple arm64-none-linux-gnu -target-feature +neon -target-feature +fullfp16 -disable-O0-optnone -flax-vector-conversions=none -fclangir -emit-llvm -o - %s | opt -S -passes=mem2reg,sroa | FileCheck %s --check-prefixes=CIRLLVM %}
// RUN: %if cir-enabled %{%clang_cc1 -triple arm64-none-linux-gnu -target-feature +neon -target-feature +fullfp16 -disable-O0-optnone -flax-vector-conversions=none -fclangir -emit-cir -o - %s | FileCheck %s --check-prefixes=CIR %}

#include <arm_neon.h>

// LLVM-LABEL: define {{[^@]+}}@test_vfma_lane_f16
// CIR-LABEL: @test_vfma_lane_f16(
float16x4_t test_vfma_lane_f16(float16x4_t a, float16x4_t b, float16x4_t c) {
// CIR: cir.vec.shuffle
// CIR: cir.call_llvm_intrinsic "fma"

// CIRLLVM:      shufflevector <4 x half> {{.*}} <i32 3, i32 3, i32 3, i32 3>
// CIRLLVM-NEXT: {{.*}}call <4 x half> @llvm.fma.v4f16({{.*}}

// LLVM-SAME: (<4 x half> {{.*}} [[A:%.*]], <4 x half> {{.*}} [[B:%.*]], <4 x half> {{.*}} [[C:%.*]]) {{.*}} {
// LLVM-NEXT:  entry:
// LLVM-NEXT:    [[TMP0:%.*]] = bitcast <4 x half> [[A]] to <4 x i16>
// LLVM-NEXT:    [[TMP1:%.*]] = bitcast <4 x half> [[B]] to <4 x i16>
// LLVM-NEXT:    [[TMP2:%.*]] = bitcast <4 x half> [[C]] to <4 x i16>
// LLVM-NEXT:    [[TMP3:%.*]] = bitcast <4 x i16> [[TMP0]] to <8 x i8>
// LLVM-NEXT:    [[TMP4:%.*]] = bitcast <4 x i16> [[TMP1]] to <8 x i8>
// LLVM-NEXT:    [[TMP5:%.*]] = bitcast <4 x i16> [[TMP2]] to <8 x i8>
// LLVM-NEXT:    [[TMP6:%.*]] = bitcast <8 x i8> [[TMP5]] to <4 x half>
// LLVM-NEXT:    [[LANE:%.*]] = shufflevector <4 x half> [[TMP6]], <4 x half> [[TMP6]], <4 x i32> <i32 3, i32 3, i32 3, i32 3>
// LLVM-NEXT:    [[FMLA:%.*]] = bitcast <8 x i8> [[TMP4]] to <4 x half>
// LLVM-NEXT:    [[FMLA1:%.*]] = bitcast <8 x i8> [[TMP3]] to <4 x half>
// LLVM-NEXT:    [[FMLA2:%.*]] = call <4 x half> @llvm.fma.v4f16(<4 x half> [[FMLA]], <4 x half> [[LANE]], <4 x half> [[FMLA1]])
// LLVM-NEXT:    ret <4 x half> [[FMLA2]]
  return vfma_lane_f16(a, b, c, 3);
}

// LLVM-LABEL: define {{[^@]+}}@test_vfmaq_lane_f16
// CIR-LABEL: @test_vfmaq_lane_f16(
float16x8_t test_vfmaq_lane_f16(float16x8_t a, float16x8_t b, float16x4_t c) {
// CIR: cir.vec.shuffle
// CIR: cir.call_llvm_intrinsic "fma"

// CIRLLVM:      shufflevector <4 x half> {{.*}} <i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3>
// CIRLLVM-NEXT: {{.*}}call <8 x half> @llvm.fma.v8f16({{.*}}

// LLVM-SAME: (<8 x half> {{.*}} [[A:%.*]], <8 x half> {{.*}} [[B:%.*]], <4 x half> {{.*}} [[C:%.*]]) {{.*}} {
// LLVM-NEXT:  entry:
// LLVM-NEXT:    [[TMP0:%.*]] = bitcast <8 x half> [[A]] to <8 x i16>
// LLVM-NEXT:    [[TMP1:%.*]] = bitcast <8 x half> [[B]] to <8 x i16>
// LLVM-NEXT:    [[TMP2:%.*]] = bitcast <4 x half> [[C]] to <4 x i16>
// LLVM-NEXT:    [[TMP3:%.*]] = bitcast <8 x i16> [[TMP0]] to <16 x i8>
// LLVM-NEXT:    [[TMP4:%.*]] = bitcast <8 x i16> [[TMP1]] to <16 x i8>
// LLVM-NEXT:    [[TMP5:%.*]] = bitcast <4 x i16> [[TMP2]] to <8 x i8>
// LLVM-NEXT:    [[TMP6:%.*]] = bitcast <8 x i8> [[TMP5]] to <4 x half>
// LLVM-NEXT:    [[LANE:%.*]] = shufflevector <4 x half> [[TMP6]], <4 x half> [[TMP6]], <8 x i32> <i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3>
// LLVM-NEXT:    [[FMLA:%.*]] = bitcast <16 x i8> [[TMP4]] to <8 x half>
// LLVM-NEXT:    [[FMLA1:%.*]] = bitcast <16 x i8> [[TMP3]] to <8 x half>
// LLVM-NEXT:    [[FMLA2:%.*]] = call <8 x half> @llvm.fma.v8f16(<8 x half> [[FMLA]], <8 x half> [[LANE]], <8 x half> [[FMLA1]])
// LLVM-NEXT:    ret <8 x half> [[FMLA2]]
  return vfmaq_lane_f16(a, b, c, 3);
}

// LLVM-LABEL: define {{[^@]+}}@test_vfma_laneq_f16
// CIR-LABEL: @test_vfma_laneq_f16(
float16x4_t test_vfma_laneq_f16(float16x4_t a, float16x4_t b, float16x8_t c) {
// CIR: cir.vec.shuffle
// CIR: cir.call_llvm_intrinsic "fma"

// CIRLLVM:      shufflevector <8 x half> {{.*}} <i32 7, i32 7, i32 7, i32 7>
// CIRLLVM-NEXT: {{.*}}call <4 x half> @llvm.fma.v4f16({{.*}}

// LLVM-SAME: (<4 x half> {{.*}} [[A:%.*]], <4 x half> {{.*}} [[B:%.*]], <8 x half> {{.*}} [[C:%.*]]) {{.*}} {
// LLVM-NEXT:  entry:
// LLVM-NEXT:    [[TMP0:%.*]] = bitcast <4 x half> [[A]] to <4 x i16>
// LLVM-NEXT:    [[TMP1:%.*]] = bitcast <4 x half> [[B]] to <4 x i16>
// LLVM-NEXT:    [[TMP2:%.*]] = bitcast <8 x half> [[C]] to <8 x i16>
// LLVM-NEXT:    [[TMP3:%.*]] = bitcast <4 x i16> [[TMP0]] to <8 x i8>
// LLVM-NEXT:    [[TMP4:%.*]] = bitcast <4 x i16> [[TMP1]] to <8 x i8>
// LLVM-NEXT:    [[TMP5:%.*]] = bitcast <8 x i16> [[TMP2]] to <16 x i8>
// LLVM-NEXT:    [[TMP6:%.*]] = bitcast <8 x i8> [[TMP3]] to <4 x half>
// LLVM-NEXT:    [[TMP7:%.*]] = bitcast <8 x i8> [[TMP4]] to <4 x half>
// LLVM-NEXT:    [[TMP8:%.*]] = bitcast <16 x i8> [[TMP5]] to <8 x half>
// LLVM-NEXT:    [[LANE:%.*]] = shufflevector <8 x half> [[TMP8]], <8 x half> [[TMP8]], <4 x i32> <i32 7, i32 7, i32 7, i32 7>
// LLVM-NEXT:    [[TMP9:%.*]] = call <4 x half> @llvm.fma.v4f16(<4 x half> [[LANE]], <4 x half> [[TMP7]], <4 x half> [[TMP6]])
// LLVM-NEXT:    ret <4 x half> [[TMP9]]
  return vfma_laneq_f16(a, b, c, 7);
}

// LLVM-LABEL: define {{[^@]+}}@test_vfmaq_laneq_f16
// CIR-LABEL: @test_vfmaq_laneq_f16(
float16x8_t test_vfmaq_laneq_f16(float16x8_t a, float16x8_t b, float16x8_t c) {
// CIR: cir.vec.shuffle
// CIR: cir.call_llvm_intrinsic "fma"

// CIRLLVM:      shufflevector <8 x half> {{.*}} <i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7>
// CIRLLVM-NEXT: {{.*}}call <8 x half> @llvm.fma.v8f16({{.*}}

// LLVM-SAME: (<8 x half> {{.*}} [[A:%.*]], <8 x half> {{.*}} [[B:%.*]], <8 x half> {{.*}} [[C:%.*]]) {{.*}} {
// LLVM-NEXT:  entry:
// LLVM-NEXT:    [[TMP0:%.*]] = bitcast <8 x half> [[A]] to <8 x i16>
// LLVM-NEXT:    [[TMP1:%.*]] = bitcast <8 x half> [[B]] to <8 x i16>
// LLVM-NEXT:    [[TMP2:%.*]] = bitcast <8 x half> [[C]] to <8 x i16>
// LLVM-NEXT:    [[TMP3:%.*]] = bitcast <8 x i16> [[TMP0]] to <16 x i8>
// LLVM-NEXT:    [[TMP4:%.*]] = bitcast <8 x i16> [[TMP1]] to <16 x i8>
// LLVM-NEXT:    [[TMP5:%.*]] = bitcast <8 x i16> [[TMP2]] to <16 x i8>
// LLVM-NEXT:    [[TMP6:%.*]] = bitcast <16 x i8> [[TMP3]] to <8 x half>
// LLVM-NEXT:    [[TMP7:%.*]] = bitcast <16 x i8> [[TMP4]] to <8 x half>
// LLVM-NEXT:    [[TMP8:%.*]] = bitcast <16 x i8> [[TMP5]] to <8 x half>
// LLVM-NEXT:    [[LANE:%.*]] = shufflevector <8 x half> [[TMP8]], <8 x half> [[TMP8]], <8 x i32> <i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7>
// LLVM-NEXT:    [[TMP9:%.*]] = call <8 x half> @llvm.fma.v8f16(<8 x half> [[LANE]], <8 x half> [[TMP7]], <8 x half> [[TMP6]])
// LLVM-NEXT:    ret <8 x half> [[TMP9]]
  return vfmaq_laneq_f16(a, b, c, 7);
}

// LLVM-LABEL: @test_vfma_lane_f32(
// CIRLLVM-LABEL: @test_vfma_lane_f32(
// CIR-LABEL: @test_vfma_lane_f32(
float32x2_t test_vfma_lane_f32(float32x2_t a, float32x2_t b, float32x2_t v) {
// CIR: cir.vec.shuffle
// CIR: cir.call_llvm_intrinsic "fma"

// CIRLLVM:      shufflevector <2 x float> {{.*}} <i32 1, i32 1>
// CIRLLVM-NEXT: {{.*}}call <2 x float> @llvm.fma.v2f32({{.*}}

// LLVM-SAME: <2 x float> {{.*}} [[A:%.*]], <2 x float> {{.*}} [[B:%.*]], <2 x float> {{.*}} [[V:%.*]]) {{.*}} {
// LLVM-NEXT:  entry:
// LLVM-NEXT:    [[TMP0:%.*]] = bitcast <2 x float> [[A]] to <2 x i32>
// LLVM-NEXT:    [[TMP1:%.*]] = bitcast <2 x float> [[B]] to <2 x i32>
// LLVM-NEXT:    [[TMP2:%.*]] = bitcast <2 x float> [[V]] to <2 x i32>
// LLVM-NEXT:    [[TMP3:%.*]] = bitcast <2 x i32> [[TMP0]] to <8 x i8>
// LLVM-NEXT:    [[TMP4:%.*]] = bitcast <2 x i32> [[TMP1]] to <8 x i8>
// LLVM-NEXT:    [[TMP5:%.*]] = bitcast <2 x i32> [[TMP2]] to <8 x i8>
// LLVM-NEXT:    [[TMP6:%.*]] = bitcast <8 x i8> [[TMP5]] to <2 x float>
// LLVM-NEXT:    [[LANE:%.*]] = shufflevector <2 x float> [[TMP6]], <2 x float> [[TMP6]], <2 x i32> <i32 1, i32 1>
// LLVM-NEXT:    [[FMLA:%.*]] = bitcast <8 x i8> [[TMP4]] to <2 x float>
// LLVM-NEXT:    [[FMLA1:%.*]] = bitcast <8 x i8> [[TMP3]] to <2 x float>
// LLVM-NEXT:    [[FMLA2:%.*]] = call <2 x float> @llvm.fma.v2f32(<2 x float> [[FMLA]], <2 x float> [[LANE]], <2 x float> [[FMLA1]])
// LLVM-NEXT:    ret <2 x float> [[FMLA2]]
  return vfma_lane_f32(a, b, v, 1);
}

// LLVM-LABEL: @test_vfmaq_lane_f32(
// CIRLLVM-LABEL: @test_vfmaq_lane_f32(
// CIR-LABEL: @test_vfmaq_lane_f32(
float32x4_t test_vfmaq_lane_f32(float32x4_t a, float32x4_t b, float32x2_t v) {
// CIR: cir.vec.shuffle
// CIR: cir.call_llvm_intrinsic "fma"

// CIRLLVM:      shufflevector <2 x float> {{.*}} <i32 1, i32 1, i32 1, i32 1>
// CIRLLVM-NEXT: {{.*}}call <4 x float> @llvm.fma.v4f32({{.*}}

// LLVM-SAME: <4 x float> {{.*}} [[A:%.*]], <4 x float> {{.*}} [[B:%.*]], <2 x float> {{.*}} [[V:%.*]]) {{.*}} {
// LLVM-NEXT:  entry:
// LLVM-NEXT:    [[TMP0:%.*]] = bitcast <4 x float> [[A]] to <4 x i32>
// LLVM-NEXT:    [[TMP1:%.*]] = bitcast <4 x float> [[B]] to <4 x i32>
// LLVM-NEXT:    [[TMP2:%.*]] = bitcast <2 x float> [[V]] to <2 x i32>
// LLVM-NEXT:    [[TMP3:%.*]] = bitcast <4 x i32> [[TMP0]] to <16 x i8>
// LLVM-NEXT:    [[TMP4:%.*]] = bitcast <4 x i32> [[TMP1]] to <16 x i8>
// LLVM-NEXT:    [[TMP5:%.*]] = bitcast <2 x i32> [[TMP2]] to <8 x i8>
// LLVM-NEXT:    [[TMP6:%.*]] = bitcast <8 x i8> [[TMP5]] to <2 x float>
// LLVM-NEXT:    [[LANE:%.*]] = shufflevector <2 x float> [[TMP6]], <2 x float> [[TMP6]], <4 x i32> <i32 1, i32 1, i32 1, i32 1>
// LLVM-NEXT:    [[FMLA:%.*]] = bitcast <16 x i8> [[TMP4]] to <4 x float>
// LLVM-NEXT:    [[FMLA1:%.*]] = bitcast <16 x i8> [[TMP3]] to <4 x float>
// LLVM-NEXT:    [[FMLA2:%.*]] = call <4 x float> @llvm.fma.v4f32(<4 x float> [[FMLA]], <4 x float> [[LANE]], <4 x float> [[FMLA1]])
// LLVM-NEXT:    ret <4 x float> [[FMLA2]]
  return vfmaq_lane_f32(a, b, v, 1);
}

// LLVM-LABEL: @test_vfma_laneq_f32(
// CIRLLVM-LABEL: @test_vfma_laneq_f32(
// CIR-LABEL: @test_vfma_laneq_f32(
float32x2_t test_vfma_laneq_f32(float32x2_t a, float32x2_t b, float32x4_t v) {
// CIR: cir.vec.shuffle
// CIR: cir.call_llvm_intrinsic "fma"

// CIRLLVM:      shufflevector <4 x float> {{.*}} <i32 3, i32 3>
// CIRLLVM-NEXT: {{.*}}call <2 x float> @llvm.fma.v2f32({{.*}}

// LLVM-SAME: <2 x float> {{.*}} [[A:%.*]], <2 x float> {{.*}} [[B:%.*]], <4 x float> {{.*}} [[V:%.*]]) {{.*}} {
// LLVM-NEXT:  entry:
// LLVM-NEXT:    [[TMP0:%.*]] = bitcast <2 x float> [[A]] to <2 x i32>
// LLVM-NEXT:    [[TMP1:%.*]] = bitcast <2 x float> [[B]] to <2 x i32>
// LLVM-NEXT:    [[TMP2:%.*]] = bitcast <4 x float> [[V]] to <4 x i32>
// LLVM-NEXT:    [[TMP3:%.*]] = bitcast <2 x i32> [[TMP0]] to <8 x i8>
// LLVM-NEXT:    [[TMP4:%.*]] = bitcast <2 x i32> [[TMP1]] to <8 x i8>
// LLVM-NEXT:    [[TMP5:%.*]] = bitcast <4 x i32> [[TMP2]] to <16 x i8>
// LLVM-NEXT:    [[TMP6:%.*]] = bitcast <8 x i8> [[TMP3]] to <2 x float>
// LLVM-NEXT:    [[TMP7:%.*]] = bitcast <8 x i8> [[TMP4]] to <2 x float>
// LLVM-NEXT:    [[TMP8:%.*]] = bitcast <16 x i8> [[TMP5]] to <4 x float>
// LLVM-NEXT:    [[LANE:%.*]] = shufflevector <4 x float> [[TMP8]], <4 x float> [[TMP8]], <2 x i32> <i32 3, i32 3>
// LLVM-NEXT:    [[TMP9:%.*]] = call <2 x float> @llvm.fma.v2f32(<2 x float> [[LANE]], <2 x float> [[TMP7]], <2 x float> [[TMP6]])
// LLVM-NEXT:    ret <2 x float> [[TMP9]]
  return vfma_laneq_f32(a, b, v, 3);
}

// LLVM-LABEL: @test_vfmaq_laneq_f32(
// CIRLLVM-LABEL: @test_vfmaq_laneq_f32(
// CIR-LABEL: @test_vfmaq_laneq_f32(
float32x4_t test_vfmaq_laneq_f32(float32x4_t a, float32x4_t b, float32x4_t v) {
// CIR: cir.vec.shuffle
// CIR: cir.call_llvm_intrinsic "fma"

// CIRLLVM:      shufflevector <4 x float> {{.*}} <i32 3, i32 3, i32 3, i32 3>
// CIRLLVM-NEXT: {{.*}}call <4 x float> @llvm.fma.v4f32({{.*}}

// LLVM-SAME: <4 x float> {{.*}} [[A:%.*]], <4 x float> {{.*}} [[B:%.*]], <4 x float> {{.*}} [[V:%.*]]) {{.*}} {
// LLVM-NEXT:  entry:
// LLVM-NEXT:    [[TMP0:%.*]] = bitcast <4 x float> [[A]] to <4 x i32>
// LLVM-NEXT:    [[TMP1:%.*]] = bitcast <4 x float> [[B]] to <4 x i32>
// LLVM-NEXT:    [[TMP2:%.*]] = bitcast <4 x float> [[V]] to <4 x i32>
// LLVM-NEXT:    [[TMP3:%.*]] = bitcast <4 x i32> [[TMP0]] to <16 x i8>
// LLVM-NEXT:    [[TMP4:%.*]] = bitcast <4 x i32> [[TMP1]] to <16 x i8>
// LLVM-NEXT:    [[TMP5:%.*]] = bitcast <4 x i32> [[TMP2]] to <16 x i8>
// LLVM-NEXT:    [[TMP6:%.*]] = bitcast <16 x i8> [[TMP3]] to <4 x float>
// LLVM-NEXT:    [[TMP7:%.*]] = bitcast <16 x i8> [[TMP4]] to <4 x float>
// LLVM-NEXT:    [[TMP8:%.*]] = bitcast <16 x i8> [[TMP5]] to <4 x float>
// LLVM-NEXT:    [[LANE:%.*]] = shufflevector <4 x float> [[TMP8]], <4 x float> [[TMP8]], <4 x i32> <i32 3, i32 3, i32 3, i32 3>
// LLVM-NEXT:    [[TMP9:%.*]] = call <4 x float> @llvm.fma.v4f32(<4 x float> [[LANE]], <4 x float> [[TMP7]], <4 x float> [[TMP6]])
// LLVM-NEXT:    ret <4 x float> [[TMP9]]
  return vfmaq_laneq_f32(a, b, v, 3);
}

// LLVM-LABEL: define dso_local <1 x double> @test_vfma_lane_f64(
// CIR-LABEL: @test_vfma_lane_f64(
float64x1_t test_vfma_lane_f64(float64x1_t a, float64x1_t b, float64x1_t v) {
// CIR: cir.vec.shuffle
// CIR: cir.call_llvm_intrinsic "fma"

// CIRLLVM:      shufflevector <1 x double> {{.*}} zeroinitializer
// CIRLLVM-NEXT: {{.*}}call <1 x double> @llvm.fma.v1f64({{.*}}

// LLVM-SAME: <1 x double> {{.*}} [[A:%.*]], <1 x double> {{.*}} [[B:%.*]], <1 x double> {{.*}} [[V:%.*]]) {{.*}} {
// LLVM-NEXT:  [[ENTRY:.*:]]
// LLVM-NEXT:    [[TMP0:%.*]] = bitcast <1 x double> [[A]] to i64
// LLVM-NEXT:    [[__S0_SROA_0_0_VEC_INSERT:%.*]] = insertelement <1 x i64> undef, i64 [[TMP0]], i32 0
// LLVM-NEXT:    [[TMP1:%.*]] = bitcast <1 x double> [[B]] to i64
// LLVM-NEXT:    [[__S1_SROA_0_0_VEC_INSERT:%.*]] = insertelement <1 x i64> undef, i64 [[TMP1]], i32 0
// LLVM-NEXT:    [[TMP2:%.*]] = bitcast <1 x double> [[V]] to i64
// LLVM-NEXT:    [[__S2_SROA_0_0_VEC_INSERT:%.*]] = insertelement <1 x i64> undef, i64 [[TMP2]], i32 0
// LLVM-NEXT:    [[TMP3:%.*]] = bitcast <1 x i64> [[__S0_SROA_0_0_VEC_INSERT]] to <8 x i8>
// LLVM-NEXT:    [[TMP4:%.*]] = bitcast <1 x i64> [[__S1_SROA_0_0_VEC_INSERT]] to <8 x i8>
// LLVM-NEXT:    [[TMP5:%.*]] = bitcast <1 x i64> [[__S2_SROA_0_0_VEC_INSERT]] to <8 x i8>
// LLVM-NEXT:    [[TMP6:%.*]] = bitcast <8 x i8> [[TMP5]] to <1 x double>
// LLVM-NEXT:    [[LANE:%.*]] = shufflevector <1 x double> [[TMP6]], <1 x double> [[TMP6]], <1 x i32> zeroinitializer
// LLVM-NEXT:    [[FMLA:%.*]] = bitcast <8 x i8> [[TMP4]] to <1 x double>
// LLVM-NEXT:    [[FMLA1:%.*]] = bitcast <8 x i8> [[TMP3]] to <1 x double>
// LLVM-NEXT:    [[FMLA2:%.*]] = call <1 x double> @llvm.fma.v1f64(<1 x double> [[FMLA]], <1 x double> [[LANE]], <1 x double> [[FMLA1]])
// LLVM-NEXT:    ret <1 x double> [[FMLA2]]
  return vfma_lane_f64(a, b, v, 0);
}

// LLVM-LABEL: @test_vfmaq_lane_f64(
// CIRLLVM-LABEL: @test_vfmaq_lane_f64(
// CIR-LABEL: @test_vfmaq_lane_f64(
float64x2_t test_vfmaq_lane_f64(float64x2_t a, float64x2_t b, float64x1_t v) {
// CIR: cir.vec.shuffle
// CIR: cir.call_llvm_intrinsic "fma"

// CIRLLVM:      shufflevector <1 x double> {{.*}} <2 x i32> zeroinitializer
// CIRLLVM-NEXT: {{.*}}call <2 x double> @llvm.fma.v2f64({{.*}}

// LLVM-SAME: <2 x double> {{.*}} [[A:%.*]], <2 x double> {{.*}} [[B:%.*]], <1 x double> {{.*}} [[V:%.*]]) {{.*}} {
// LLVM-NEXT:  entry:
// LLVM-NEXT:    [[TMP0:%.*]] = bitcast <2 x double> [[A]] to <2 x i64>
// LLVM-NEXT:    [[TMP1:%.*]] = bitcast <2 x double> [[B]] to <2 x i64>
// LLVM-NEXT:    [[TMP2:%.*]] = bitcast <1 x double> [[V]] to i64
// LLVM-NEXT:    [[__S2_SROA_0_0_VEC_INSERT:%.*]] = insertelement <1 x i64> undef, i64 [[TMP2]], i32 0
// LLVM-NEXT:    [[TMP3:%.*]] = bitcast <2 x i64> [[TMP0]] to <16 x i8>
// LLVM-NEXT:    [[TMP4:%.*]] = bitcast <2 x i64> [[TMP1]] to <16 x i8>
// LLVM-NEXT:    [[TMP5:%.*]] = bitcast <1 x i64> [[__S2_SROA_0_0_VEC_INSERT]] to <8 x i8>
// LLVM-NEXT:    [[TMP6:%.*]] = bitcast <8 x i8> [[TMP5]] to <1 x double>
// LLVM-NEXT:    [[LANE:%.*]] = shufflevector <1 x double> [[TMP6]], <1 x double> [[TMP6]], <2 x i32> zeroinitializer
// LLVM-NEXT:    [[FMLA:%.*]] = bitcast <16 x i8> [[TMP4]] to <2 x double>
// LLVM-NEXT:    [[FMLA1:%.*]] = bitcast <16 x i8> [[TMP3]] to <2 x double>
// LLVM-NEXT:    [[FMLA2:%.*]] = call <2 x double> @llvm.fma.v2f64(<2 x double> [[FMLA]], <2 x double> [[LANE]], <2 x double> [[FMLA1]])
// LLVM-NEXT:    ret <2 x double> [[FMLA2]]
  return vfmaq_lane_f64(a, b, v, 0);
}

// LLVM-LABEL: define dso_local <1 x double> @test_vfma_laneq_f64(
// CIR-LABEL: @test_vfma_laneq_f64(
float64x1_t test_vfma_laneq_f64(float64x1_t a, float64x1_t b, float64x2_t v) {
// CIR: cir.vec.extract
// CIR: cir.call_llvm_intrinsic "fma"

// CIRLLVM:      extractelement <2 x double> {{.*}}, i32 1
// CIRLLVM-NEXT: {{.*}}call double @llvm.fma.f64({{.*}}

// LLVM-SAME: <1 x double> {{.*}} [[A:%.*]], <1 x double> {{.*}} [[B:%.*]], <2 x double> {{.*}} [[V:%.*]]) {{.*}} {
// LLVM-NEXT:  [[ENTRY:.*:]]
// LLVM-NEXT:    [[TMP0:%.*]] = bitcast <1 x double> [[A]] to i64
// LLVM-NEXT:    [[__S0_SROA_0_0_VEC_INSERT:%.*]] = insertelement <1 x i64> undef, i64 [[TMP0]], i32 0
// LLVM-NEXT:    [[TMP1:%.*]] = bitcast <1 x double> [[B]] to i64
// LLVM-NEXT:    [[__S1_SROA_0_0_VEC_INSERT:%.*]] = insertelement <1 x i64> undef, i64 [[TMP1]], i32 0
// LLVM-NEXT:    [[TMP2:%.*]] = bitcast <2 x double> [[V]] to <2 x i64>
// LLVM-NEXT:    [[TMP3:%.*]] = bitcast <1 x i64> [[__S0_SROA_0_0_VEC_INSERT]] to <8 x i8>
// LLVM-NEXT:    [[TMP4:%.*]] = bitcast <1 x i64> [[__S1_SROA_0_0_VEC_INSERT]] to <8 x i8>
// LLVM-NEXT:    [[TMP5:%.*]] = bitcast <2 x i64> [[TMP2]] to <16 x i8>
// LLVM-NEXT:    [[TMP6:%.*]] = bitcast <8 x i8> [[TMP3]] to double
// LLVM-NEXT:    [[TMP7:%.*]] = bitcast <8 x i8> [[TMP4]] to double
// LLVM-NEXT:    [[TMP8:%.*]] = bitcast <16 x i8> [[TMP5]] to <2 x double>
// LLVM-NEXT:    [[EXTRACT:%.*]] = extractelement <2 x double> [[TMP8]], i32 1
// LLVM-NEXT:    [[TMP9:%.*]] = call double @llvm.fma.f64(double [[TMP7]], double [[EXTRACT]], double [[TMP6]])
// LLVM-NEXT:    [[TMP10:%.*]] = bitcast double [[TMP9]] to <1 x double>
// LLVM-NEXT:    ret <1 x double> [[TMP10]]
  return vfma_laneq_f64(a, b, v, 1);
}

// LLVM-LABEL: @test_vfmaq_laneq_f64(
// CIRLLVM-LABEL: @test_vfmaq_laneq_f64(
// CIR-LABEL: @test_vfmaq_laneq_f64(
float64x2_t test_vfmaq_laneq_f64(float64x2_t a, float64x2_t b, float64x2_t v) {
// CIR: cir.vec.shuffle
// CIR: cir.call_llvm_intrinsic "fma"

// CIRLLVM:      shufflevector <2 x double> {{.*}} <i32 1, i32 1>
// CIRLLVM-NEXT: {{.*}}call <2 x double> @llvm.fma.v2f64({{.*}}

// LLVM-SAME: <2 x double> {{.*}} [[A:%.*]], <2 x double> {{.*}} [[B:%.*]], <2 x double> {{.*}} [[V:%.*]]) {{.*}} {
// LLVM-NEXT:  entry:
// LLVM-NEXT:    [[TMP0:%.*]] = bitcast <2 x double> [[A]] to <2 x i64>
// LLVM-NEXT:    [[TMP1:%.*]] = bitcast <2 x double> [[B]] to <2 x i64>
// LLVM-NEXT:    [[TMP2:%.*]] = bitcast <2 x double> [[V]] to <2 x i64>
// LLVM-NEXT:    [[TMP3:%.*]] = bitcast <2 x i64> [[TMP0]] to <16 x i8>
// LLVM-NEXT:    [[TMP4:%.*]] = bitcast <2 x i64> [[TMP1]] to <16 x i8>
// LLVM-NEXT:    [[TMP5:%.*]] = bitcast <2 x i64> [[TMP2]] to <16 x i8>
// LLVM-NEXT:    [[TMP6:%.*]] = bitcast <16 x i8> [[TMP3]] to <2 x double>
// LLVM-NEXT:    [[TMP7:%.*]] = bitcast <16 x i8> [[TMP4]] to <2 x double>
// LLVM-NEXT:    [[TMP8:%.*]] = bitcast <16 x i8> [[TMP5]] to <2 x double>
// LLVM-NEXT:    [[LANE:%.*]] = shufflevector <2 x double> [[TMP8]], <2 x double> [[TMP8]], <2 x i32> <i32 1, i32 1>
// LLVM-NEXT:    [[TMP9:%.*]] = call <2 x double> @llvm.fma.v2f64(<2 x double> [[LANE]], <2 x double> [[TMP7]], <2 x double> [[TMP6]])
// LLVM-NEXT:    ret <2 x double> [[TMP9]]
  return vfmaq_laneq_f64(a, b, v, 1);
}

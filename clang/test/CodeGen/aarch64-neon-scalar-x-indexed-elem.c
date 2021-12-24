// RUN: %clang_cc1 -triple arm64-none-linux-gnu -target-feature +neon -target-cpu cyclone \
// RUN: -disable-O0-optnone -emit-llvm -o - %s | opt -S -mem2reg | FileCheck %s

// REQUIRES: aarch64-registered-target || arm-registered-target

#include <arm_neon.h>


// CHECK-LABEL: define{{.*}} float @test_vmuls_lane_f32(float %a, <2 x float> %b) #0 {
// CHECK:   [[VGET_LANE:%.*]] = extractelement <2 x float> %b, i32 1
// CHECK:   [[MUL:%.*]] = fmul float %a, [[VGET_LANE]]
// CHECK:   ret float [[MUL]]
float32_t test_vmuls_lane_f32(float32_t a, float32x2_t b) {
  return vmuls_lane_f32(a, b, 1);
}

// CHECK-LABEL: define{{.*}} double @test_vmuld_lane_f64(double %a, <1 x double> %b) #0 {
// CHECK:   [[VGET_LANE:%.*]] = extractelement <1 x double> %b, i32 0
// CHECK:   [[MUL:%.*]] = fmul double %a, [[VGET_LANE]]
// CHECK:   ret double [[MUL]]
float64_t test_vmuld_lane_f64(float64_t a, float64x1_t b) {
  return vmuld_lane_f64(a, b, 0);
}

// CHECK-LABEL: define{{.*}} float @test_vmuls_laneq_f32(float %a, <4 x float> %b) #1 {
// CHECK:   [[VGETQ_LANE:%.*]] = extractelement <4 x float> %b, i32 3
// CHECK:   [[MUL:%.*]] = fmul float %a, [[VGETQ_LANE]]
// CHECK:   ret float [[MUL]]
float32_t test_vmuls_laneq_f32(float32_t a, float32x4_t b) {
  return vmuls_laneq_f32(a, b, 3);
}

// CHECK-LABEL: define{{.*}} double @test_vmuld_laneq_f64(double %a, <2 x double> %b) #1 {
// CHECK:   [[VGETQ_LANE:%.*]] = extractelement <2 x double> %b, i32 1
// CHECK:   [[MUL:%.*]] = fmul double %a, [[VGETQ_LANE]]
// CHECK:   ret double [[MUL]]
float64_t test_vmuld_laneq_f64(float64_t a, float64x2_t b) {
  return vmuld_laneq_f64(a, b, 1);
}

// CHECK-LABEL: define{{.*}} <1 x double> @test_vmul_n_f64(<1 x double> %a, double %b) #0 {
// CHECK:   [[TMP2:%.*]] = bitcast <1 x double> %a to double
// CHECK:   [[TMP3:%.*]] = fmul double [[TMP2]], %b
// CHECK:   [[TMP4:%.*]] = bitcast double [[TMP3]] to <1 x double>
// CHECK:   ret <1 x double> [[TMP4]]
float64x1_t test_vmul_n_f64(float64x1_t a, float64_t b) {
  return vmul_n_f64(a, b);
}

// CHECK-LABEL: define{{.*}} float @test_vmulxs_lane_f32(float %a, <2 x float> %b) #0 {
// CHECK:   [[VGET_LANE:%.*]] = extractelement <2 x float> %b, i32 1
// CHECK:   [[VMULXS_F32_I:%.*]] = call float @llvm.aarch64.neon.fmulx.f32(float %a, float [[VGET_LANE]])
// CHECK:   ret float [[VMULXS_F32_I]]
float32_t test_vmulxs_lane_f32(float32_t a, float32x2_t b) {
  return vmulxs_lane_f32(a, b, 1);
}

// CHECK-LABEL: define{{.*}} float @test_vmulxs_laneq_f32(float %a, <4 x float> %b) #1 {
// CHECK:   [[VGETQ_LANE:%.*]] = extractelement <4 x float> %b, i32 3
// CHECK:   [[VMULXS_F32_I:%.*]] = call float @llvm.aarch64.neon.fmulx.f32(float %a, float [[VGETQ_LANE]])
// CHECK:   ret float [[VMULXS_F32_I]]
float32_t test_vmulxs_laneq_f32(float32_t a, float32x4_t b) {
  return vmulxs_laneq_f32(a, b, 3);
}

// CHECK-LABEL: define{{.*}} double @test_vmulxd_lane_f64(double %a, <1 x double> %b) #0 {
// CHECK:   [[VGET_LANE:%.*]] = extractelement <1 x double> %b, i32 0
// CHECK:   [[VMULXD_F64_I:%.*]] = call double @llvm.aarch64.neon.fmulx.f64(double %a, double [[VGET_LANE]])
// CHECK:   ret double [[VMULXD_F64_I]]
float64_t test_vmulxd_lane_f64(float64_t a, float64x1_t b) {
  return vmulxd_lane_f64(a, b, 0);
}

// CHECK-LABEL: define{{.*}} double @test_vmulxd_laneq_f64(double %a, <2 x double> %b) #1 {
// CHECK:   [[VGETQ_LANE:%.*]] = extractelement <2 x double> %b, i32 1
// CHECK:   [[VMULXD_F64_I:%.*]] = call double @llvm.aarch64.neon.fmulx.f64(double %a, double [[VGETQ_LANE]])
// CHECK:   ret double [[VMULXD_F64_I]]
float64_t test_vmulxd_laneq_f64(float64_t a, float64x2_t b) {
  return vmulxd_laneq_f64(a, b, 1);
}

// CHECK-LABEL: define{{.*}} <1 x double> @test_vmulx_lane_f64(<1 x double> %a, <1 x double> %b) #0 {
// CHECK:   [[VGET_LANE:%.*]] = extractelement <1 x double> %a, i32 0
// CHECK:   [[VGET_LANE6:%.*]] = extractelement <1 x double> %b, i32 0
// CHECK:   [[VMULXD_F64_I:%.*]] = call double @llvm.aarch64.neon.fmulx.f64(double [[VGET_LANE]], double [[VGET_LANE6]])
// CHECK:   [[VSET_LANE:%.*]] = insertelement <1 x double> %a, double [[VMULXD_F64_I]], i32 0
// CHECK:   ret <1 x double> [[VSET_LANE]]
float64x1_t test_vmulx_lane_f64(float64x1_t a, float64x1_t b) {
  return vmulx_lane_f64(a, b, 0);
}


// CHECK-LABEL: define{{.*}} <1 x double> @test_vmulx_laneq_f64_0(<1 x double> %a, <2 x double> %b) #1 {
// CHECK:   [[VGET_LANE:%.*]] = extractelement <1 x double> %a, i32 0
// CHECK:   [[VGETQ_LANE:%.*]] = extractelement <2 x double> %b, i32 0
// CHECK:   [[VMULXD_F64_I:%.*]] = call double @llvm.aarch64.neon.fmulx.f64(double [[VGET_LANE]], double [[VGETQ_LANE]])
// CHECK:   [[VSET_LANE:%.*]] = insertelement <1 x double> %a, double [[VMULXD_F64_I]], i32 0
// CHECK:   ret <1 x double> [[VSET_LANE]]
float64x1_t test_vmulx_laneq_f64_0(float64x1_t a, float64x2_t b) {
  return vmulx_laneq_f64(a, b, 0);
}

// CHECK-LABEL: define{{.*}} <1 x double> @test_vmulx_laneq_f64_1(<1 x double> %a, <2 x double> %b) #1 {
// CHECK:   [[VGET_LANE:%.*]] = extractelement <1 x double> %a, i32 0
// CHECK:   [[VGETQ_LANE:%.*]] = extractelement <2 x double> %b, i32 1
// CHECK:   [[VMULXD_F64_I:%.*]] = call double @llvm.aarch64.neon.fmulx.f64(double [[VGET_LANE]], double [[VGETQ_LANE]])
// CHECK:   [[VSET_LANE:%.*]] = insertelement <1 x double> %a, double [[VMULXD_F64_I]], i32 0
// CHECK:   ret <1 x double> [[VSET_LANE]]
float64x1_t test_vmulx_laneq_f64_1(float64x1_t a, float64x2_t b) {
  return vmulx_laneq_f64(a, b, 1);
}


// CHECK-LABEL: define{{.*}} float @test_vfmas_lane_f32(float %a, float %b, <2 x float> %c) #0 {
// CHECK:   [[EXTRACT:%.*]] = extractelement <2 x float> %c, i32 1
// CHECK:   [[TMP2:%.*]] = call float @llvm.fma.f32(float %b, float [[EXTRACT]], float %a)
// CHECK:   ret float [[TMP2]]
float32_t test_vfmas_lane_f32(float32_t a, float32_t b, float32x2_t c) {
  return vfmas_lane_f32(a, b, c, 1);
}

// CHECK-LABEL: define{{.*}} double @test_vfmad_lane_f64(double %a, double %b, <1 x double> %c) #0 {
// CHECK:   [[EXTRACT:%.*]] = extractelement <1 x double> %c, i32 0
// CHECK:   [[TMP2:%.*]] = call double @llvm.fma.f64(double %b, double [[EXTRACT]], double %a)
// CHECK:   ret double [[TMP2]]
float64_t test_vfmad_lane_f64(float64_t a, float64_t b, float64x1_t c) {
  return vfmad_lane_f64(a, b, c, 0);
}

// CHECK-LABEL: define{{.*}} double @test_vfmad_laneq_f64(double %a, double %b, <2 x double> %c) #1 {
// CHECK:   [[EXTRACT:%.*]] = extractelement <2 x double> %c, i32 1
// CHECK:   [[TMP2:%.*]] = call double @llvm.fma.f64(double %b, double [[EXTRACT]], double %a)
// CHECK:   ret double [[TMP2]]
float64_t test_vfmad_laneq_f64(float64_t a, float64_t b, float64x2_t c) {
  return vfmad_laneq_f64(a, b, c, 1);
}

// CHECK-LABEL: define{{.*}} float @test_vfmss_lane_f32(float %a, float %b, <2 x float> %c) #0 {
// CHECK:   [[SUB:%.*]] = fneg float %b
// CHECK:   [[EXTRACT:%.*]] = extractelement <2 x float> %c, i32 1
// CHECK:   [[TMP2:%.*]] = call float @llvm.fma.f32(float [[SUB]], float [[EXTRACT]], float %a)
// CHECK:   ret float [[TMP2]]
float32_t test_vfmss_lane_f32(float32_t a, float32_t b, float32x2_t c) {
  return vfmss_lane_f32(a, b, c, 1);
}

// CHECK-LABEL: define{{.*}} <1 x double> @test_vfma_lane_f64(<1 x double> %a, <1 x double> %b, <1 x double> %v) #0 {
// CHECK:   [[TMP0:%.*]] = bitcast <1 x double> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <1 x double> %b to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <1 x double> %v to <8 x i8>
// CHECK:   [[TMP3:%.*]] = bitcast <8 x i8> [[TMP2]] to <1 x double>
// CHECK:   [[LANE:%.*]] = shufflevector <1 x double> [[TMP3]], <1 x double> [[TMP3]], <1 x i32> zeroinitializer
// CHECK:   [[FMLA:%.*]] = bitcast <8 x i8> [[TMP1]] to <1 x double>
// CHECK:   [[FMLA1:%.*]] = bitcast <8 x i8> [[TMP0]] to <1 x double>
// CHECK:   [[FMLA2:%.*]] = call <1 x double> @llvm.fma.v1f64(<1 x double> [[FMLA]], <1 x double> [[LANE]], <1 x double> [[FMLA1]])
// CHECK:   ret <1 x double> [[FMLA2]]
float64x1_t test_vfma_lane_f64(float64x1_t a, float64x1_t b, float64x1_t v) {
  return vfma_lane_f64(a, b, v, 0);
}

// CHECK-LABEL: define{{.*}} <1 x double> @test_vfms_lane_f64(<1 x double> %a, <1 x double> %b, <1 x double> %v) #0 {
// CHECK:   [[SUB:%.*]] = fneg <1 x double> %b
// CHECK:   [[TMP0:%.*]] = bitcast <1 x double> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <1 x double> [[SUB]] to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <1 x double> %v to <8 x i8>
// CHECK:   [[TMP3:%.*]] = bitcast <8 x i8> [[TMP2]] to <1 x double>
// CHECK:   [[LANE:%.*]] = shufflevector <1 x double> [[TMP3]], <1 x double> [[TMP3]], <1 x i32> zeroinitializer
// CHECK:   [[FMLA:%.*]] = bitcast <8 x i8> [[TMP1]] to <1 x double>
// CHECK:   [[FMLA1:%.*]] = bitcast <8 x i8> [[TMP0]] to <1 x double>
// CHECK:   [[FMLA2:%.*]] = call <1 x double> @llvm.fma.v1f64(<1 x double> [[FMLA]], <1 x double> [[LANE]], <1 x double> [[FMLA1]])
// CHECK:   ret <1 x double> [[FMLA2]]
float64x1_t test_vfms_lane_f64(float64x1_t a, float64x1_t b, float64x1_t v) {
  return vfms_lane_f64(a, b, v, 0);
}

// CHECK-LABEL: define{{.*}} <1 x double> @test_vfma_laneq_f64(<1 x double> %a, <1 x double> %b, <2 x double> %v) #1 {
// CHECK:   [[TMP0:%.*]] = bitcast <1 x double> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <1 x double> %b to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <2 x double> %v to <16 x i8>
// CHECK:   [[TMP3:%.*]] = bitcast <8 x i8> [[TMP0]] to double
// CHECK:   [[TMP4:%.*]] = bitcast <8 x i8> [[TMP1]] to double
// CHECK:   [[TMP5:%.*]] = bitcast <16 x i8> [[TMP2]] to <2 x double>
// CHECK:   [[EXTRACT:%.*]] = extractelement <2 x double> [[TMP5]], i32 0
// CHECK:   [[TMP6:%.*]] = call double @llvm.fma.f64(double [[TMP4]], double [[EXTRACT]], double [[TMP3]])
// CHECK:   [[TMP7:%.*]] = bitcast double [[TMP6]] to <1 x double>
// CHECK:   ret <1 x double> [[TMP7]]
float64x1_t test_vfma_laneq_f64(float64x1_t a, float64x1_t b, float64x2_t v) {
  return vfma_laneq_f64(a, b, v, 0);
}

// CHECK-LABEL: define{{.*}} <1 x double> @test_vfms_laneq_f64(<1 x double> %a, <1 x double> %b, <2 x double> %v) #1 {
// CHECK:   [[SUB:%.*]] = fneg <1 x double> %b
// CHECK:   [[TMP0:%.*]] = bitcast <1 x double> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <1 x double> [[SUB]] to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <2 x double> %v to <16 x i8>
// CHECK:   [[TMP3:%.*]] = bitcast <8 x i8> [[TMP0]] to double
// CHECK:   [[TMP4:%.*]] = bitcast <8 x i8> [[TMP1]] to double
// CHECK:   [[TMP5:%.*]] = bitcast <16 x i8> [[TMP2]] to <2 x double>
// CHECK:   [[EXTRACT:%.*]] = extractelement <2 x double> [[TMP5]], i32 0
// CHECK:   [[TMP6:%.*]] = call double @llvm.fma.f64(double [[TMP4]], double [[EXTRACT]], double [[TMP3]])
// CHECK:   [[TMP7:%.*]] = bitcast double [[TMP6]] to <1 x double>
// CHECK:   ret <1 x double> [[TMP7]]
float64x1_t test_vfms_laneq_f64(float64x1_t a, float64x1_t b, float64x2_t v) {
  return vfms_laneq_f64(a, b, v, 0);
}

// CHECK-LABEL: define{{.*}} i32 @test_vqdmullh_lane_s16(i16 %a, <4 x i16> %b) #0 {
// CHECK:   [[VGET_LANE:%.*]] = extractelement <4 x i16> %b, i32 3
// CHECK:   [[TMP2:%.*]] = insertelement <4 x i16> undef, i16 %a, i64 0
// CHECK:   [[TMP3:%.*]] = insertelement <4 x i16> undef, i16 [[VGET_LANE]], i64 0
// CHECK:   [[VQDMULLH_S16_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.sqdmull.v4i32(<4 x i16> [[TMP2]], <4 x i16> [[TMP3]])
// CHECK:   [[TMP4:%.*]] = extractelement <4 x i32> [[VQDMULLH_S16_I]], i64 0
// CHECK:   ret i32 [[TMP4]]
int32_t test_vqdmullh_lane_s16(int16_t a, int16x4_t b) {
  return vqdmullh_lane_s16(a, b, 3);
}

// CHECK-LABEL: define{{.*}} i64 @test_vqdmulls_lane_s32(i32 %a, <2 x i32> %b) #0 {
// CHECK:   [[VGET_LANE:%.*]] = extractelement <2 x i32> %b, i32 1
// CHECK:   [[VQDMULLS_S32_I:%.*]] = call i64 @llvm.aarch64.neon.sqdmulls.scalar(i32 %a, i32 [[VGET_LANE]])
// CHECK:   ret i64 [[VQDMULLS_S32_I]]
int64_t test_vqdmulls_lane_s32(int32_t a, int32x2_t b) {
  return vqdmulls_lane_s32(a, b, 1);
}

// CHECK-LABEL: define{{.*}} i32 @test_vqdmullh_laneq_s16(i16 %a, <8 x i16> %b) #1 {
// CHECK:   [[VGETQ_LANE:%.*]] = extractelement <8 x i16> %b, i32 7
// CHECK:   [[TMP2:%.*]] = insertelement <4 x i16> undef, i16 %a, i64 0
// CHECK:   [[TMP3:%.*]] = insertelement <4 x i16> undef, i16 [[VGETQ_LANE]], i64 0
// CHECK:   [[VQDMULLH_S16_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.sqdmull.v4i32(<4 x i16> [[TMP2]], <4 x i16> [[TMP3]])
// CHECK:   [[TMP4:%.*]] = extractelement <4 x i32> [[VQDMULLH_S16_I]], i64 0
// CHECK:   ret i32 [[TMP4]]
int32_t test_vqdmullh_laneq_s16(int16_t a, int16x8_t b) {
  return vqdmullh_laneq_s16(a, b, 7);
}

// CHECK-LABEL: define{{.*}} i64 @test_vqdmulls_laneq_s32(i32 %a, <4 x i32> %b) #1 {
// CHECK:   [[VGETQ_LANE:%.*]] = extractelement <4 x i32> %b, i32 3
// CHECK:   [[VQDMULLS_S32_I:%.*]] = call i64 @llvm.aarch64.neon.sqdmulls.scalar(i32 %a, i32 [[VGETQ_LANE]])
// CHECK:   ret i64 [[VQDMULLS_S32_I]]
int64_t test_vqdmulls_laneq_s32(int32_t a, int32x4_t b) {
  return vqdmulls_laneq_s32(a, b, 3);
}

// CHECK-LABEL: define{{.*}} i16 @test_vqdmulhh_lane_s16(i16 %a, <4 x i16> %b) #0 {
// CHECK:   [[VGET_LANE:%.*]] = extractelement <4 x i16> %b, i32 3
// CHECK:   [[TMP2:%.*]] = insertelement <4 x i16> undef, i16 %a, i64 0
// CHECK:   [[TMP3:%.*]] = insertelement <4 x i16> undef, i16 [[VGET_LANE]], i64 0
// CHECK:   [[VQDMULHH_S16_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.sqdmulh.v4i16(<4 x i16> [[TMP2]], <4 x i16> [[TMP3]])
// CHECK:   [[TMP4:%.*]] = extractelement <4 x i16> [[VQDMULHH_S16_I]], i64 0
// CHECK:   ret i16 [[TMP4]]
int16_t test_vqdmulhh_lane_s16(int16_t a, int16x4_t b) {
  return vqdmulhh_lane_s16(a, b, 3);
}

// CHECK-LABEL: define{{.*}} i32 @test_vqdmulhs_lane_s32(i32 %a, <2 x i32> %b) #0 {
// CHECK:   [[VGET_LANE:%.*]] = extractelement <2 x i32> %b, i32 1
// CHECK:   [[VQDMULHS_S32_I:%.*]] = call i32 @llvm.aarch64.neon.sqdmulh.i32(i32 %a, i32 [[VGET_LANE]])
// CHECK:   ret i32 [[VQDMULHS_S32_I]]
int32_t test_vqdmulhs_lane_s32(int32_t a, int32x2_t b) {
  return vqdmulhs_lane_s32(a, b, 1);
}


// CHECK-LABEL: define{{.*}} i16 @test_vqdmulhh_laneq_s16(i16 %a, <8 x i16> %b) #1 {
// CHECK:   [[VGETQ_LANE:%.*]] = extractelement <8 x i16> %b, i32 7
// CHECK:   [[TMP2:%.*]] = insertelement <4 x i16> undef, i16 %a, i64 0
// CHECK:   [[TMP3:%.*]] = insertelement <4 x i16> undef, i16 [[VGETQ_LANE]], i64 0
// CHECK:   [[VQDMULHH_S16_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.sqdmulh.v4i16(<4 x i16> [[TMP2]], <4 x i16> [[TMP3]])
// CHECK:   [[TMP4:%.*]] = extractelement <4 x i16> [[VQDMULHH_S16_I]], i64 0
// CHECK:   ret i16 [[TMP4]]
int16_t test_vqdmulhh_laneq_s16(int16_t a, int16x8_t b) {
  return vqdmulhh_laneq_s16(a, b, 7);
}


// CHECK-LABEL: define{{.*}} i32 @test_vqdmulhs_laneq_s32(i32 %a, <4 x i32> %b) #1 {
// CHECK:   [[VGETQ_LANE:%.*]] = extractelement <4 x i32> %b, i32 3
// CHECK:   [[VQDMULHS_S32_I:%.*]] = call i32 @llvm.aarch64.neon.sqdmulh.i32(i32 %a, i32 [[VGETQ_LANE]])
// CHECK:   ret i32 [[VQDMULHS_S32_I]]
int32_t test_vqdmulhs_laneq_s32(int32_t a, int32x4_t b) {
  return vqdmulhs_laneq_s32(a, b, 3);
}

// CHECK-LABEL: define{{.*}} i16 @test_vqrdmulhh_lane_s16(i16 %a, <4 x i16> %b) #0 {
// CHECK:   [[VGET_LANE:%.*]] = extractelement <4 x i16> %b, i32 3
// CHECK:   [[TMP2:%.*]] = insertelement <4 x i16> undef, i16 %a, i64 0
// CHECK:   [[TMP3:%.*]] = insertelement <4 x i16> undef, i16 [[VGET_LANE]], i64 0
// CHECK:   [[VQRDMULHH_S16_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.sqrdmulh.v4i16(<4 x i16> [[TMP2]], <4 x i16> [[TMP3]])
// CHECK:   [[TMP4:%.*]] = extractelement <4 x i16> [[VQRDMULHH_S16_I]], i64 0
// CHECK:   ret i16 [[TMP4]]
int16_t test_vqrdmulhh_lane_s16(int16_t a, int16x4_t b) {
  return vqrdmulhh_lane_s16(a, b, 3);
}

// CHECK-LABEL: define{{.*}} i32 @test_vqrdmulhs_lane_s32(i32 %a, <2 x i32> %b) #0 {
// CHECK:   [[VGET_LANE:%.*]] = extractelement <2 x i32> %b, i32 1
// CHECK:   [[VQRDMULHS_S32_I:%.*]] = call i32 @llvm.aarch64.neon.sqrdmulh.i32(i32 %a, i32 [[VGET_LANE]])
// CHECK:   ret i32 [[VQRDMULHS_S32_I]]
int32_t test_vqrdmulhs_lane_s32(int32_t a, int32x2_t b) {
  return vqrdmulhs_lane_s32(a, b, 1);
}


// CHECK-LABEL: define{{.*}} i16 @test_vqrdmulhh_laneq_s16(i16 %a, <8 x i16> %b) #1 {
// CHECK:   [[VGETQ_LANE:%.*]] = extractelement <8 x i16> %b, i32 7
// CHECK:   [[TMP2:%.*]] = insertelement <4 x i16> undef, i16 %a, i64 0
// CHECK:   [[TMP3:%.*]] = insertelement <4 x i16> undef, i16 [[VGETQ_LANE]], i64 0
// CHECK:   [[VQRDMULHH_S16_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.sqrdmulh.v4i16(<4 x i16> [[TMP2]], <4 x i16> [[TMP3]])
// CHECK:   [[TMP4:%.*]] = extractelement <4 x i16> [[VQRDMULHH_S16_I]], i64 0
// CHECK:   ret i16 [[TMP4]]
int16_t test_vqrdmulhh_laneq_s16(int16_t a, int16x8_t b) {
  return vqrdmulhh_laneq_s16(a, b, 7);
}


// CHECK-LABEL: define{{.*}} i32 @test_vqrdmulhs_laneq_s32(i32 %a, <4 x i32> %b) #1 {
// CHECK:   [[VGETQ_LANE:%.*]] = extractelement <4 x i32> %b, i32 3
// CHECK:   [[VQRDMULHS_S32_I:%.*]] = call i32 @llvm.aarch64.neon.sqrdmulh.i32(i32 %a, i32 [[VGETQ_LANE]])
// CHECK:   ret i32 [[VQRDMULHS_S32_I]]
int32_t test_vqrdmulhs_laneq_s32(int32_t a, int32x4_t b) {
  return vqrdmulhs_laneq_s32(a, b, 3);
}

// CHECK-LABEL: define{{.*}} i32 @test_vqdmlalh_lane_s16(i32 %a, i16 %b, <4 x i16> %c) #0 {
// CHECK:   [[LANE:%.*]] = extractelement <4 x i16> %c, i32 3
// CHECK:   [[TMP2:%.*]] = insertelement <4 x i16> undef, i16 %b, i64 0
// CHECK:   [[TMP3:%.*]] = insertelement <4 x i16> undef, i16 [[LANE]], i64 0
// CHECK:   [[VQDMLXL:%.*]] = call <4 x i32> @llvm.aarch64.neon.sqdmull.v4i32(<4 x i16> [[TMP2]], <4 x i16> [[TMP3]])
// CHECK:   [[LANE0:%.*]] = extractelement <4 x i32> [[VQDMLXL]], i64 0
// CHECK:   [[VQDMLXL1:%.*]] = call i32 @llvm.aarch64.neon.sqadd.i32(i32 %a, i32 [[LANE0]])
// CHECK:   ret i32 [[VQDMLXL1]]
int32_t test_vqdmlalh_lane_s16(int32_t a, int16_t b, int16x4_t c) {
  return vqdmlalh_lane_s16(a, b, c, 3);
}

// CHECK-LABEL: define{{.*}} i64 @test_vqdmlals_lane_s32(i64 %a, i32 %b, <2 x i32> %c) #0 {
// CHECK:   [[LANE:%.*]] = extractelement <2 x i32> %c, i32 1
// CHECK:   [[VQDMLXL:%.*]] = call i64 @llvm.aarch64.neon.sqdmulls.scalar(i32 %b, i32 [[LANE]])
// CHECK:   [[VQDMLXL1:%.*]] = call i64 @llvm.aarch64.neon.sqadd.i64(i64 %a, i64 [[VQDMLXL]])
// CHECK:   ret i64 [[VQDMLXL1]]
int64_t test_vqdmlals_lane_s32(int64_t a, int32_t b, int32x2_t c) {
  return vqdmlals_lane_s32(a, b, c, 1);
}

// CHECK-LABEL: define{{.*}} i32 @test_vqdmlalh_laneq_s16(i32 %a, i16 %b, <8 x i16> %c) #1 {
// CHECK:   [[LANE:%.*]] = extractelement <8 x i16> %c, i32 7
// CHECK:   [[TMP2:%.*]] = insertelement <4 x i16> undef, i16 %b, i64 0
// CHECK:   [[TMP3:%.*]] = insertelement <4 x i16> undef, i16 [[LANE]], i64 0
// CHECK:   [[VQDMLXL:%.*]] = call <4 x i32> @llvm.aarch64.neon.sqdmull.v4i32(<4 x i16> [[TMP2]], <4 x i16> [[TMP3]])
// CHECK:   [[LANE0:%.*]] = extractelement <4 x i32> [[VQDMLXL]], i64 0
// CHECK:   [[VQDMLXL1:%.*]] = call i32 @llvm.aarch64.neon.sqadd.i32(i32 %a, i32 [[LANE0]])
// CHECK:   ret i32 [[VQDMLXL1]]
int32_t test_vqdmlalh_laneq_s16(int32_t a, int16_t b, int16x8_t c) {
  return vqdmlalh_laneq_s16(a, b, c, 7);
}

// CHECK-LABEL: define{{.*}} i64 @test_vqdmlals_laneq_s32(i64 %a, i32 %b, <4 x i32> %c) #1 {
// CHECK:   [[LANE:%.*]] = extractelement <4 x i32> %c, i32 3
// CHECK:   [[VQDMLXL:%.*]] = call i64 @llvm.aarch64.neon.sqdmulls.scalar(i32 %b, i32 [[LANE]])
// CHECK:   [[VQDMLXL1:%.*]] = call i64 @llvm.aarch64.neon.sqadd.i64(i64 %a, i64 [[VQDMLXL]])
// CHECK:   ret i64 [[VQDMLXL1]]
int64_t test_vqdmlals_laneq_s32(int64_t a, int32_t b, int32x4_t c) {
  return vqdmlals_laneq_s32(a, b, c, 3);
}

// CHECK-LABEL: define{{.*}} i32 @test_vqdmlslh_lane_s16(i32 %a, i16 %b, <4 x i16> %c) #0 {
// CHECK:   [[LANE:%.*]] = extractelement <4 x i16> %c, i32 3
// CHECK:   [[TMP2:%.*]] = insertelement <4 x i16> undef, i16 %b, i64 0
// CHECK:   [[TMP3:%.*]] = insertelement <4 x i16> undef, i16 [[LANE]], i64 0
// CHECK:   [[VQDMLXL:%.*]] = call <4 x i32> @llvm.aarch64.neon.sqdmull.v4i32(<4 x i16> [[TMP2]], <4 x i16> [[TMP3]])
// CHECK:   [[LANE0:%.*]] = extractelement <4 x i32> [[VQDMLXL]], i64 0
// CHECK:   [[VQDMLXL1:%.*]] = call i32 @llvm.aarch64.neon.sqsub.i32(i32 %a, i32 [[LANE0]])
// CHECK:   ret i32 [[VQDMLXL1]]
int32_t test_vqdmlslh_lane_s16(int32_t a, int16_t b, int16x4_t c) {
  return vqdmlslh_lane_s16(a, b, c, 3);
}

// CHECK-LABEL: define{{.*}} i64 @test_vqdmlsls_lane_s32(i64 %a, i32 %b, <2 x i32> %c) #0 {
// CHECK:   [[LANE:%.*]] = extractelement <2 x i32> %c, i32 1
// CHECK:   [[VQDMLXL:%.*]] = call i64 @llvm.aarch64.neon.sqdmulls.scalar(i32 %b, i32 [[LANE]])
// CHECK:   [[VQDMLXL1:%.*]] = call i64 @llvm.aarch64.neon.sqsub.i64(i64 %a, i64 [[VQDMLXL]])
// CHECK:   ret i64 [[VQDMLXL1]]
int64_t test_vqdmlsls_lane_s32(int64_t a, int32_t b, int32x2_t c) {
  return vqdmlsls_lane_s32(a, b, c, 1);
}

// CHECK-LABEL: define{{.*}} i32 @test_vqdmlslh_laneq_s16(i32 %a, i16 %b, <8 x i16> %c) #1 {
// CHECK:   [[LANE:%.*]] = extractelement <8 x i16> %c, i32 7
// CHECK:   [[TMP2:%.*]] = insertelement <4 x i16> undef, i16 %b, i64 0
// CHECK:   [[TMP3:%.*]] = insertelement <4 x i16> undef, i16 [[LANE]], i64 0
// CHECK:   [[VQDMLXL:%.*]] = call <4 x i32> @llvm.aarch64.neon.sqdmull.v4i32(<4 x i16> [[TMP2]], <4 x i16> [[TMP3]])
// CHECK:   [[LANE0:%.*]] = extractelement <4 x i32> [[VQDMLXL]], i64 0
// CHECK:   [[VQDMLXL1:%.*]] = call i32 @llvm.aarch64.neon.sqsub.i32(i32 %a, i32 [[LANE0]])
// CHECK:   ret i32 [[VQDMLXL1]]
int32_t test_vqdmlslh_laneq_s16(int32_t a, int16_t b, int16x8_t c) {
  return vqdmlslh_laneq_s16(a, b, c, 7);
}

// CHECK-LABEL: define{{.*}} i64 @test_vqdmlsls_laneq_s32(i64 %a, i32 %b, <4 x i32> %c) #1 {
// CHECK:   [[LANE:%.*]] = extractelement <4 x i32> %c, i32 3
// CHECK:   [[VQDMLXL:%.*]] = call i64 @llvm.aarch64.neon.sqdmulls.scalar(i32 %b, i32 [[LANE]])
// CHECK:   [[VQDMLXL1:%.*]] = call i64 @llvm.aarch64.neon.sqsub.i64(i64 %a, i64 [[VQDMLXL]])
// CHECK:   ret i64 [[VQDMLXL1]]
int64_t test_vqdmlsls_laneq_s32(int64_t a, int32_t b, int32x4_t c) {
  return vqdmlsls_laneq_s32(a, b, c, 3);
}

// CHECK-LABEL: define{{.*}} <1 x double> @test_vmulx_lane_f64_0() #0 {
// CHECK:   [[TMP0:%.*]] = bitcast i64 4599917171378402754 to <1 x double>
// CHECK:   [[TMP1:%.*]] = bitcast i64 4606655882138939123 to <1 x double>
// CHECK:   [[VGET_LANE:%.*]] = extractelement <1 x double> [[TMP0]], i32 0
// CHECK:   [[VGET_LANE7:%.*]] = extractelement <1 x double> [[TMP1]], i32 0
// CHECK:   [[VMULXD_F64_I:%.*]] = call double @llvm.aarch64.neon.fmulx.f64(double [[VGET_LANE]], double [[VGET_LANE7]])
// CHECK:   [[VSET_LANE:%.*]] = insertelement <1 x double> [[TMP0]], double [[VMULXD_F64_I]], i32 0
// CHECK:   ret <1 x double> [[VSET_LANE]]
float64x1_t test_vmulx_lane_f64_0() {
      float64x1_t arg1;
      float64x1_t arg2;
      float64x1_t result;
      float64_t sarg1, sarg2, sres;
      arg1 = vcreate_f64(UINT64_C(0x3fd6304bc43ab5c2));
      arg2 = vcreate_f64(UINT64_C(0x3fee211e215aeef3));
      result = vmulx_lane_f64(arg1, arg2, 0);
      return result;
}

// CHECK-LABEL: define{{.*}} <1 x double> @test_vmulx_laneq_f64_2() #1 {
// CHECK:   [[TMP0:%.*]] = bitcast i64 4599917171378402754 to <1 x double>
// CHECK:   [[TMP1:%.*]] = bitcast i64 4606655882138939123 to <1 x double>
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <1 x double> [[TMP0]], <1 x double> [[TMP1]], <2 x i32> <i32 0, i32 1>
// CHECK:   [[VGET_LANE:%.*]] = extractelement <1 x double> [[TMP0]], i32 0
// CHECK:   [[VGETQ_LANE:%.*]] = extractelement <2 x double> [[SHUFFLE_I]], i32 1
// CHECK:   [[VMULXD_F64_I:%.*]] = call double @llvm.aarch64.neon.fmulx.f64(double [[VGET_LANE]], double [[VGETQ_LANE]])
// CHECK:   [[VSET_LANE:%.*]] = insertelement <1 x double> [[TMP0]], double [[VMULXD_F64_I]], i32 0
// CHECK:   ret <1 x double> [[VSET_LANE]]
float64x1_t test_vmulx_laneq_f64_2() {
      float64x1_t arg1;
      float64x1_t arg2;
      float64x2_t arg3;
      float64x1_t result;
      float64_t sarg1, sarg2, sres;
      arg1 = vcreate_f64(UINT64_C(0x3fd6304bc43ab5c2));
      arg2 = vcreate_f64(UINT64_C(0x3fee211e215aeef3));
      arg3 = vcombine_f64(arg1, arg2);
      result = vmulx_laneq_f64(arg1, arg3, 1);
      return result;
}

// CHECK: attributes #0 ={{.*}}"min-legal-vector-width"="64"
// CHECK: attributes #1 ={{.*}}"min-legal-vector-width"="128"

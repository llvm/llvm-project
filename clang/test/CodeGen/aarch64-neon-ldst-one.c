// RUN: %clang_cc1 -no-opaque-pointers -triple arm64-none-linux-gnu -target-feature +neon \
// RUN:  -disable-O0-optnone -fallow-half-arguments-and-returns -emit-llvm -o - %s \
// RUN: | opt -S -mem2reg | FileCheck %s

// REQUIRES: aarch64-registered-target || arm-registered-target

#include <arm_neon.h>

// CHECK-LABEL: define{{.*}} <16 x i8> @test_vld1q_dup_u8(i8* noundef %a) #0 {
// CHECK:   [[TMP0:%.*]] = load i8, i8* %a
// CHECK:   [[TMP1:%.*]] = insertelement <16 x i8> undef, i8 [[TMP0]], i32 0
// CHECK:   [[LANE:%.*]] = shufflevector <16 x i8> [[TMP1]], <16 x i8> [[TMP1]], <16 x i32> zeroinitializer
// CHECK:   ret <16 x i8> [[LANE]]
uint8x16_t test_vld1q_dup_u8(uint8_t  *a) {
  return vld1q_dup_u8(a);
}

// CHECK-LABEL: define{{.*}} <8 x i16> @test_vld1q_dup_u16(i16* noundef %a) #0 {
// CHECK:   [[TMP0:%.*]] = bitcast i16* %a to i8*
// CHECK:   [[TMP1:%.*]] = bitcast i8* [[TMP0]] to i16*
// CHECK:   [[TMP2:%.*]] = load i16, i16* [[TMP1]]
// CHECK:   [[TMP3:%.*]] = insertelement <8 x i16> undef, i16 [[TMP2]], i32 0
// CHECK:   [[LANE:%.*]] = shufflevector <8 x i16> [[TMP3]], <8 x i16> [[TMP3]], <8 x i32> zeroinitializer
// CHECK:   ret <8 x i16> [[LANE]]
uint16x8_t test_vld1q_dup_u16(uint16_t  *a) {
  return vld1q_dup_u16(a);
}

// CHECK-LABEL: define{{.*}} <4 x i32> @test_vld1q_dup_u32(i32* noundef %a) #0 {
// CHECK:   [[TMP0:%.*]] = bitcast i32* %a to i8*
// CHECK:   [[TMP1:%.*]] = bitcast i8* [[TMP0]] to i32*
// CHECK:   [[TMP2:%.*]] = load i32, i32* [[TMP1]]
// CHECK:   [[TMP3:%.*]] = insertelement <4 x i32> undef, i32 [[TMP2]], i32 0
// CHECK:   [[LANE:%.*]] = shufflevector <4 x i32> [[TMP3]], <4 x i32> [[TMP3]], <4 x i32> zeroinitializer
// CHECK:   ret <4 x i32> [[LANE]]
uint32x4_t test_vld1q_dup_u32(uint32_t  *a) {
  return vld1q_dup_u32(a);
}

// CHECK-LABEL: define{{.*}} <2 x i64> @test_vld1q_dup_u64(i64* noundef %a) #0 {
// CHECK:   [[TMP0:%.*]] = bitcast i64* %a to i8*
// CHECK:   [[TMP1:%.*]] = bitcast i8* [[TMP0]] to i64*
// CHECK:   [[TMP2:%.*]] = load i64, i64* [[TMP1]]
// CHECK:   [[TMP3:%.*]] = insertelement <2 x i64> undef, i64 [[TMP2]], i32 0
// CHECK:   [[LANE:%.*]] = shufflevector <2 x i64> [[TMP3]], <2 x i64> [[TMP3]], <2 x i32> zeroinitializer
// CHECK:   ret <2 x i64> [[LANE]]
uint64x2_t test_vld1q_dup_u64(uint64_t  *a) {
  return vld1q_dup_u64(a);
}

// CHECK-LABEL: define{{.*}} <16 x i8> @test_vld1q_dup_s8(i8* noundef %a) #0 {
// CHECK:   [[TMP0:%.*]] = load i8, i8* %a
// CHECK:   [[TMP1:%.*]] = insertelement <16 x i8> undef, i8 [[TMP0]], i32 0
// CHECK:   [[LANE:%.*]] = shufflevector <16 x i8> [[TMP1]], <16 x i8> [[TMP1]], <16 x i32> zeroinitializer
// CHECK:   ret <16 x i8> [[LANE]]
int8x16_t test_vld1q_dup_s8(int8_t  *a) {
  return vld1q_dup_s8(a);
}

// CHECK-LABEL: define{{.*}} <8 x i16> @test_vld1q_dup_s16(i16* noundef %a) #0 {
// CHECK:   [[TMP0:%.*]] = bitcast i16* %a to i8*
// CHECK:   [[TMP1:%.*]] = bitcast i8* [[TMP0]] to i16*
// CHECK:   [[TMP2:%.*]] = load i16, i16* [[TMP1]]
// CHECK:   [[TMP3:%.*]] = insertelement <8 x i16> undef, i16 [[TMP2]], i32 0
// CHECK:   [[LANE:%.*]] = shufflevector <8 x i16> [[TMP3]], <8 x i16> [[TMP3]], <8 x i32> zeroinitializer
// CHECK:   ret <8 x i16> [[LANE]]
int16x8_t test_vld1q_dup_s16(int16_t  *a) {
  return vld1q_dup_s16(a);
}

// CHECK-LABEL: define{{.*}} <4 x i32> @test_vld1q_dup_s32(i32* noundef %a) #0 {
// CHECK:   [[TMP0:%.*]] = bitcast i32* %a to i8*
// CHECK:   [[TMP1:%.*]] = bitcast i8* [[TMP0]] to i32*
// CHECK:   [[TMP2:%.*]] = load i32, i32* [[TMP1]]
// CHECK:   [[TMP3:%.*]] = insertelement <4 x i32> undef, i32 [[TMP2]], i32 0
// CHECK:   [[LANE:%.*]] = shufflevector <4 x i32> [[TMP3]], <4 x i32> [[TMP3]], <4 x i32> zeroinitializer
// CHECK:   ret <4 x i32> [[LANE]]
int32x4_t test_vld1q_dup_s32(int32_t  *a) {
  return vld1q_dup_s32(a);
}

// CHECK-LABEL: define{{.*}} <2 x i64> @test_vld1q_dup_s64(i64* noundef %a) #0 {
// CHECK:   [[TMP0:%.*]] = bitcast i64* %a to i8*
// CHECK:   [[TMP1:%.*]] = bitcast i8* [[TMP0]] to i64*
// CHECK:   [[TMP2:%.*]] = load i64, i64* [[TMP1]]
// CHECK:   [[TMP3:%.*]] = insertelement <2 x i64> undef, i64 [[TMP2]], i32 0
// CHECK:   [[LANE:%.*]] = shufflevector <2 x i64> [[TMP3]], <2 x i64> [[TMP3]], <2 x i32> zeroinitializer
// CHECK:   ret <2 x i64> [[LANE]]
int64x2_t test_vld1q_dup_s64(int64_t  *a) {
  return vld1q_dup_s64(a);
}

// CHECK-LABEL: define{{.*}} <8 x half> @test_vld1q_dup_f16(half* noundef %a) #0 {
// CHECK:   [[TMP0:%.*]] = bitcast half* %a to i8*
// CHECK:   [[TMP1:%.*]] = bitcast i8* [[TMP0]] to half*
// CHECK:   [[TMP2:%.*]] = load half, half* [[TMP1]]
// CHECK:   [[TMP3:%.*]] = insertelement <8 x half> undef, half [[TMP2]], i32 0
// CHECK:   [[LANE:%.*]] = shufflevector <8 x half> [[TMP3]], <8 x half> [[TMP3]], <8 x i32> zeroinitializer
// CHECK:   ret <8 x half> [[LANE]]
float16x8_t test_vld1q_dup_f16(float16_t  *a) {
  return vld1q_dup_f16(a);
}

// CHECK-LABEL: define{{.*}} <4 x float> @test_vld1q_dup_f32(float* noundef %a) #0 {
// CHECK:   [[TMP0:%.*]] = bitcast float* %a to i8*
// CHECK:   [[TMP1:%.*]] = bitcast i8* [[TMP0]] to float*
// CHECK:   [[TMP2:%.*]] = load float, float* [[TMP1]]
// CHECK:   [[TMP3:%.*]] = insertelement <4 x float> undef, float [[TMP2]], i32 0
// CHECK:   [[LANE:%.*]] = shufflevector <4 x float> [[TMP3]], <4 x float> [[TMP3]], <4 x i32> zeroinitializer
// CHECK:   ret <4 x float> [[LANE]]
float32x4_t test_vld1q_dup_f32(float32_t  *a) {
  return vld1q_dup_f32(a);
}

// CHECK-LABEL: define{{.*}} <2 x double> @test_vld1q_dup_f64(double* noundef %a) #0 {
// CHECK:   [[TMP0:%.*]] = bitcast double* %a to i8*
// CHECK:   [[TMP1:%.*]] = bitcast i8* [[TMP0]] to double*
// CHECK:   [[TMP2:%.*]] = load double, double* [[TMP1]]
// CHECK:   [[TMP3:%.*]] = insertelement <2 x double> undef, double [[TMP2]], i32 0
// CHECK:   [[LANE:%.*]] = shufflevector <2 x double> [[TMP3]], <2 x double> [[TMP3]], <2 x i32> zeroinitializer
// CHECK:   ret <2 x double> [[LANE]]
float64x2_t test_vld1q_dup_f64(float64_t  *a) {
  return vld1q_dup_f64(a);
}

// CHECK-LABEL: define{{.*}} <16 x i8> @test_vld1q_dup_p8(i8* noundef %a) #0 {
// CHECK:   [[TMP0:%.*]] = load i8, i8* %a
// CHECK:   [[TMP1:%.*]] = insertelement <16 x i8> undef, i8 [[TMP0]], i32 0
// CHECK:   [[LANE:%.*]] = shufflevector <16 x i8> [[TMP1]], <16 x i8> [[TMP1]], <16 x i32> zeroinitializer
// CHECK:   ret <16 x i8> [[LANE]]
poly8x16_t test_vld1q_dup_p8(poly8_t  *a) {
  return vld1q_dup_p8(a);
}

// CHECK-LABEL: define{{.*}} <8 x i16> @test_vld1q_dup_p16(i16* noundef %a) #0 {
// CHECK:   [[TMP0:%.*]] = bitcast i16* %a to i8*
// CHECK:   [[TMP1:%.*]] = bitcast i8* [[TMP0]] to i16*
// CHECK:   [[TMP2:%.*]] = load i16, i16* [[TMP1]]
// CHECK:   [[TMP3:%.*]] = insertelement <8 x i16> undef, i16 [[TMP2]], i32 0
// CHECK:   [[LANE:%.*]] = shufflevector <8 x i16> [[TMP3]], <8 x i16> [[TMP3]], <8 x i32> zeroinitializer
// CHECK:   ret <8 x i16> [[LANE]]
poly16x8_t test_vld1q_dup_p16(poly16_t  *a) {
  return vld1q_dup_p16(a);
}

// CHECK-LABEL: define{{.*}} <2 x i64> @test_vld1q_dup_p64(i64* noundef %a) #0 {
// CHECK:   [[TMP0:%.*]] = bitcast i64* %a to i8*
// CHECK:   [[TMP1:%.*]] = bitcast i8* [[TMP0]] to i64*
// CHECK:   [[TMP2:%.*]] = load i64, i64* [[TMP1]]
// CHECK:   [[TMP3:%.*]] = insertelement <2 x i64> undef, i64 [[TMP2]], i32 0
// CHECK:   [[LANE:%.*]] = shufflevector <2 x i64> [[TMP3]], <2 x i64> [[TMP3]], <2 x i32> zeroinitializer
// CHECK:   ret <2 x i64> [[LANE]]
poly64x2_t test_vld1q_dup_p64(poly64_t  *a) {
  return vld1q_dup_p64(a);
}

// CHECK-LABEL: define{{.*}} <8 x i8> @test_vld1_dup_u8(i8* noundef %a) #1 {
// CHECK:   [[TMP0:%.*]] = load i8, i8* %a
// CHECK:   [[TMP1:%.*]] = insertelement <8 x i8> undef, i8 [[TMP0]], i32 0
// CHECK:   [[LANE:%.*]] = shufflevector <8 x i8> [[TMP1]], <8 x i8> [[TMP1]], <8 x i32> zeroinitializer
// CHECK:   ret <8 x i8> [[LANE]]
uint8x8_t test_vld1_dup_u8(uint8_t  *a) {
  return vld1_dup_u8(a);
}

// CHECK-LABEL: define{{.*}} <4 x i16> @test_vld1_dup_u16(i16* noundef %a) #1 {
// CHECK:   [[TMP0:%.*]] = bitcast i16* %a to i8*
// CHECK:   [[TMP1:%.*]] = bitcast i8* [[TMP0]] to i16*
// CHECK:   [[TMP2:%.*]] = load i16, i16* [[TMP1]]
// CHECK:   [[TMP3:%.*]] = insertelement <4 x i16> undef, i16 [[TMP2]], i32 0
// CHECK:   [[LANE:%.*]] = shufflevector <4 x i16> [[TMP3]], <4 x i16> [[TMP3]], <4 x i32> zeroinitializer
// CHECK:   ret <4 x i16> [[LANE]]
uint16x4_t test_vld1_dup_u16(uint16_t  *a) {
  return vld1_dup_u16(a);
}

// CHECK-LABEL: define{{.*}} <2 x i32> @test_vld1_dup_u32(i32* noundef %a) #1 {
// CHECK:   [[TMP0:%.*]] = bitcast i32* %a to i8*
// CHECK:   [[TMP1:%.*]] = bitcast i8* [[TMP0]] to i32*
// CHECK:   [[TMP2:%.*]] = load i32, i32* [[TMP1]]
// CHECK:   [[TMP3:%.*]] = insertelement <2 x i32> undef, i32 [[TMP2]], i32 0
// CHECK:   [[LANE:%.*]] = shufflevector <2 x i32> [[TMP3]], <2 x i32> [[TMP3]], <2 x i32> zeroinitializer
// CHECK:   ret <2 x i32> [[LANE]]
uint32x2_t test_vld1_dup_u32(uint32_t  *a) {
  return vld1_dup_u32(a);
}

// CHECK-LABEL: define{{.*}} <1 x i64> @test_vld1_dup_u64(i64* noundef %a) #1 {
// CHECK:   [[TMP0:%.*]] = bitcast i64* %a to i8*
// CHECK:   [[TMP1:%.*]] = bitcast i8* [[TMP0]] to i64*
// CHECK:   [[TMP2:%.*]] = load i64, i64* [[TMP1]]
// CHECK:   [[TMP3:%.*]] = insertelement <1 x i64> undef, i64 [[TMP2]], i32 0
// CHECK:   [[LANE:%.*]] = shufflevector <1 x i64> [[TMP3]], <1 x i64> [[TMP3]], <1 x i32> zeroinitializer
// CHECK:   ret <1 x i64> [[LANE]]
uint64x1_t test_vld1_dup_u64(uint64_t  *a) {
  return vld1_dup_u64(a);
}

// CHECK-LABEL: define{{.*}} <8 x i8> @test_vld1_dup_s8(i8* noundef %a) #1 {
// CHECK:   [[TMP0:%.*]] = load i8, i8* %a
// CHECK:   [[TMP1:%.*]] = insertelement <8 x i8> undef, i8 [[TMP0]], i32 0
// CHECK:   [[LANE:%.*]] = shufflevector <8 x i8> [[TMP1]], <8 x i8> [[TMP1]], <8 x i32> zeroinitializer
// CHECK:   ret <8 x i8> [[LANE]]
int8x8_t test_vld1_dup_s8(int8_t  *a) {
  return vld1_dup_s8(a);
}

// CHECK-LABEL: define{{.*}} <4 x i16> @test_vld1_dup_s16(i16* noundef %a) #1 {
// CHECK:   [[TMP0:%.*]] = bitcast i16* %a to i8*
// CHECK:   [[TMP1:%.*]] = bitcast i8* [[TMP0]] to i16*
// CHECK:   [[TMP2:%.*]] = load i16, i16* [[TMP1]]
// CHECK:   [[TMP3:%.*]] = insertelement <4 x i16> undef, i16 [[TMP2]], i32 0
// CHECK:   [[LANE:%.*]] = shufflevector <4 x i16> [[TMP3]], <4 x i16> [[TMP3]], <4 x i32> zeroinitializer
// CHECK:   ret <4 x i16> [[LANE]]
int16x4_t test_vld1_dup_s16(int16_t  *a) {
  return vld1_dup_s16(a);
}

// CHECK-LABEL: define{{.*}} <2 x i32> @test_vld1_dup_s32(i32* noundef %a) #1 {
// CHECK:   [[TMP0:%.*]] = bitcast i32* %a to i8*
// CHECK:   [[TMP1:%.*]] = bitcast i8* [[TMP0]] to i32*
// CHECK:   [[TMP2:%.*]] = load i32, i32* [[TMP1]]
// CHECK:   [[TMP3:%.*]] = insertelement <2 x i32> undef, i32 [[TMP2]], i32 0
// CHECK:   [[LANE:%.*]] = shufflevector <2 x i32> [[TMP3]], <2 x i32> [[TMP3]], <2 x i32> zeroinitializer
// CHECK:   ret <2 x i32> [[LANE]]
int32x2_t test_vld1_dup_s32(int32_t  *a) {
  return vld1_dup_s32(a);
}

// CHECK-LABEL: define{{.*}} <1 x i64> @test_vld1_dup_s64(i64* noundef %a) #1 {
// CHECK:   [[TMP0:%.*]] = bitcast i64* %a to i8*
// CHECK:   [[TMP1:%.*]] = bitcast i8* [[TMP0]] to i64*
// CHECK:   [[TMP2:%.*]] = load i64, i64* [[TMP1]]
// CHECK:   [[TMP3:%.*]] = insertelement <1 x i64> undef, i64 [[TMP2]], i32 0
// CHECK:   [[LANE:%.*]] = shufflevector <1 x i64> [[TMP3]], <1 x i64> [[TMP3]], <1 x i32> zeroinitializer
// CHECK:   ret <1 x i64> [[LANE]]
int64x1_t test_vld1_dup_s64(int64_t  *a) {
  return vld1_dup_s64(a);
}

// CHECK-LABEL: define{{.*}} <4 x half> @test_vld1_dup_f16(half* noundef %a) #1 {
// CHECK:   [[TMP0:%.*]] = bitcast half* %a to i8*
// CHECK:   [[TMP1:%.*]] = bitcast i8* [[TMP0]] to half*
// CHECK:   [[TMP2:%.*]] = load half, half* [[TMP1]]
// CHECK:   [[TMP3:%.*]] = insertelement <4 x half> undef, half [[TMP2]], i32 0
// CHECK:   [[LANE:%.*]] = shufflevector <4 x half> [[TMP3]], <4 x half> [[TMP3]], <4 x i32> zeroinitializer
// CHECK:   ret <4 x half> [[LANE]]
float16x4_t test_vld1_dup_f16(float16_t  *a) {
  return vld1_dup_f16(a);
}

// CHECK-LABEL: define{{.*}} <2 x float> @test_vld1_dup_f32(float* noundef %a) #1 {
// CHECK:   [[TMP0:%.*]] = bitcast float* %a to i8*
// CHECK:   [[TMP1:%.*]] = bitcast i8* [[TMP0]] to float*
// CHECK:   [[TMP2:%.*]] = load float, float* [[TMP1]]
// CHECK:   [[TMP3:%.*]] = insertelement <2 x float> undef, float [[TMP2]], i32 0
// CHECK:   [[LANE:%.*]] = shufflevector <2 x float> [[TMP3]], <2 x float> [[TMP3]], <2 x i32> zeroinitializer
// CHECK:   ret <2 x float> [[LANE]]
float32x2_t test_vld1_dup_f32(float32_t  *a) {
  return vld1_dup_f32(a);
}

// CHECK-LABEL: define{{.*}} <1 x double> @test_vld1_dup_f64(double* noundef %a) #1 {
// CHECK:   [[TMP0:%.*]] = bitcast double* %a to i8*
// CHECK:   [[TMP1:%.*]] = bitcast i8* [[TMP0]] to double*
// CHECK:   [[TMP2:%.*]] = load double, double* [[TMP1]]
// CHECK:   [[TMP3:%.*]] = insertelement <1 x double> undef, double [[TMP2]], i32 0
// CHECK:   [[LANE:%.*]] = shufflevector <1 x double> [[TMP3]], <1 x double> [[TMP3]], <1 x i32> zeroinitializer
// CHECK:   ret <1 x double> [[LANE]]
float64x1_t test_vld1_dup_f64(float64_t  *a) {
  return vld1_dup_f64(a);
}

// CHECK-LABEL: define{{.*}} <8 x i8> @test_vld1_dup_p8(i8* noundef %a) #1 {
// CHECK:   [[TMP0:%.*]] = load i8, i8* %a
// CHECK:   [[TMP1:%.*]] = insertelement <8 x i8> undef, i8 [[TMP0]], i32 0
// CHECK:   [[LANE:%.*]] = shufflevector <8 x i8> [[TMP1]], <8 x i8> [[TMP1]], <8 x i32> zeroinitializer
// CHECK:   ret <8 x i8> [[LANE]]
poly8x8_t test_vld1_dup_p8(poly8_t  *a) {
  return vld1_dup_p8(a);
}

// CHECK-LABEL: define{{.*}} <4 x i16> @test_vld1_dup_p16(i16* noundef %a) #1 {
// CHECK:   [[TMP0:%.*]] = bitcast i16* %a to i8*
// CHECK:   [[TMP1:%.*]] = bitcast i8* [[TMP0]] to i16*
// CHECK:   [[TMP2:%.*]] = load i16, i16* [[TMP1]]
// CHECK:   [[TMP3:%.*]] = insertelement <4 x i16> undef, i16 [[TMP2]], i32 0
// CHECK:   [[LANE:%.*]] = shufflevector <4 x i16> [[TMP3]], <4 x i16> [[TMP3]], <4 x i32> zeroinitializer
// CHECK:   ret <4 x i16> [[LANE]]
poly16x4_t test_vld1_dup_p16(poly16_t  *a) {
  return vld1_dup_p16(a);
}

// CHECK-LABEL: define{{.*}} <1 x i64> @test_vld1_dup_p64(i64* noundef %a) #1 {
// CHECK:   [[TMP0:%.*]] = bitcast i64* %a to i8*
// CHECK:   [[TMP1:%.*]] = bitcast i8* [[TMP0]] to i64*
// CHECK:   [[TMP2:%.*]] = load i64, i64* [[TMP1]]
// CHECK:   [[TMP3:%.*]] = insertelement <1 x i64> undef, i64 [[TMP2]], i32 0
// CHECK:   [[LANE:%.*]] = shufflevector <1 x i64> [[TMP3]], <1 x i64> [[TMP3]], <1 x i32> zeroinitializer
// CHECK:   ret <1 x i64> [[LANE]]
poly64x1_t test_vld1_dup_p64(poly64_t  *a) {
  return vld1_dup_p64(a);
}

// CHECK-LABEL: define{{.*}} %struct.uint64x2x2_t @test_vld2q_dup_u64(i64* noundef %a) #2 {
// CHECK:   [[RETVAL:%.*]] = alloca %struct.uint64x2x2_t, align 16
// CHECK:   [[__RET:%.*]] = alloca %struct.uint64x2x2_t, align 16
// CHECK:   [[TMP0:%.*]] = bitcast %struct.uint64x2x2_t* [[__RET]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast i64* %a to i8*
// CHECK:   [[TMP2:%.*]] = bitcast i8* [[TMP1]] to i64*
// CHECK:   [[VLD2:%.*]] = call { <2 x i64>, <2 x i64> } @llvm.aarch64.neon.ld2r.v2i64.p0i64(i64* [[TMP2]])
// CHECK:   [[TMP3:%.*]] = bitcast i8* [[TMP0]] to { <2 x i64>, <2 x i64> }*
// CHECK:   store { <2 x i64>, <2 x i64> } [[VLD2]], { <2 x i64>, <2 x i64> }* [[TMP3]]
// CHECK:   [[TMP4:%.*]] = bitcast %struct.uint64x2x2_t* [[RETVAL]] to i8*
// CHECK:   [[TMP5:%.*]] = bitcast %struct.uint64x2x2_t* [[__RET]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP4]], i8* align 16 [[TMP5]], i64 32, i1 false)
// CHECK:   [[TMP6:%.*]] = load %struct.uint64x2x2_t, %struct.uint64x2x2_t* [[RETVAL]], align 16
// CHECK:   ret %struct.uint64x2x2_t [[TMP6]]
uint64x2x2_t test_vld2q_dup_u64(uint64_t  *a) {
  return vld2q_dup_u64(a);
}

// CHECK-LABEL: define{{.*}} %struct.int64x2x2_t @test_vld2q_dup_s64(i64* noundef %a) #2 {
// CHECK:   [[RETVAL:%.*]] = alloca %struct.int64x2x2_t, align 16
// CHECK:   [[__RET:%.*]] = alloca %struct.int64x2x2_t, align 16
// CHECK:   [[TMP0:%.*]] = bitcast %struct.int64x2x2_t* [[__RET]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast i64* %a to i8*
// CHECK:   [[TMP2:%.*]] = bitcast i8* [[TMP1]] to i64*
// CHECK:   [[VLD2:%.*]] = call { <2 x i64>, <2 x i64> } @llvm.aarch64.neon.ld2r.v2i64.p0i64(i64* [[TMP2]])
// CHECK:   [[TMP3:%.*]] = bitcast i8* [[TMP0]] to { <2 x i64>, <2 x i64> }*
// CHECK:   store { <2 x i64>, <2 x i64> } [[VLD2]], { <2 x i64>, <2 x i64> }* [[TMP3]]
// CHECK:   [[TMP4:%.*]] = bitcast %struct.int64x2x2_t* [[RETVAL]] to i8*
// CHECK:   [[TMP5:%.*]] = bitcast %struct.int64x2x2_t* [[__RET]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP4]], i8* align 16 [[TMP5]], i64 32, i1 false)
// CHECK:   [[TMP6:%.*]] = load %struct.int64x2x2_t, %struct.int64x2x2_t* [[RETVAL]], align 16
// CHECK:   ret %struct.int64x2x2_t [[TMP6]]
int64x2x2_t test_vld2q_dup_s64(int64_t  *a) {
  return vld2q_dup_s64(a);
}

// CHECK-LABEL: define{{.*}} %struct.float64x2x2_t @test_vld2q_dup_f64(double* noundef %a) #2 {
// CHECK:   [[RETVAL:%.*]] = alloca %struct.float64x2x2_t, align 16
// CHECK:   [[__RET:%.*]] = alloca %struct.float64x2x2_t, align 16
// CHECK:   [[TMP0:%.*]] = bitcast %struct.float64x2x2_t* [[__RET]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast double* %a to i8*
// CHECK:   [[TMP2:%.*]] = bitcast i8* [[TMP1]] to double*
// CHECK:   [[VLD2:%.*]] = call { <2 x double>, <2 x double> } @llvm.aarch64.neon.ld2r.v2f64.p0f64(double* [[TMP2]])
// CHECK:   [[TMP3:%.*]] = bitcast i8* [[TMP0]] to { <2 x double>, <2 x double> }*
// CHECK:   store { <2 x double>, <2 x double> } [[VLD2]], { <2 x double>, <2 x double> }* [[TMP3]]
// CHECK:   [[TMP4:%.*]] = bitcast %struct.float64x2x2_t* [[RETVAL]] to i8*
// CHECK:   [[TMP5:%.*]] = bitcast %struct.float64x2x2_t* [[__RET]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP4]], i8* align 16 [[TMP5]], i64 32, i1 false)
// CHECK:   [[TMP6:%.*]] = load %struct.float64x2x2_t, %struct.float64x2x2_t* [[RETVAL]], align 16
// CHECK:   ret %struct.float64x2x2_t [[TMP6]]
float64x2x2_t test_vld2q_dup_f64(float64_t  *a) {
  return vld2q_dup_f64(a);
}

// CHECK-LABEL: define{{.*}} %struct.poly64x2x2_t @test_vld2q_dup_p64(i64* noundef %a) #2 {
// CHECK:   [[RETVAL:%.*]] = alloca %struct.poly64x2x2_t, align 16
// CHECK:   [[__RET:%.*]] = alloca %struct.poly64x2x2_t, align 16
// CHECK:   [[TMP0:%.*]] = bitcast %struct.poly64x2x2_t* [[__RET]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast i64* %a to i8*
// CHECK:   [[TMP2:%.*]] = bitcast i8* [[TMP1]] to i64*
// CHECK:   [[VLD2:%.*]] = call { <2 x i64>, <2 x i64> } @llvm.aarch64.neon.ld2r.v2i64.p0i64(i64* [[TMP2]])
// CHECK:   [[TMP3:%.*]] = bitcast i8* [[TMP0]] to { <2 x i64>, <2 x i64> }*
// CHECK:   store { <2 x i64>, <2 x i64> } [[VLD2]], { <2 x i64>, <2 x i64> }* [[TMP3]]
// CHECK:   [[TMP4:%.*]] = bitcast %struct.poly64x2x2_t* [[RETVAL]] to i8*
// CHECK:   [[TMP5:%.*]] = bitcast %struct.poly64x2x2_t* [[__RET]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP4]], i8* align 16 [[TMP5]], i64 32, i1 false)
// CHECK:   [[TMP6:%.*]] = load %struct.poly64x2x2_t, %struct.poly64x2x2_t* [[RETVAL]], align 16
// CHECK:   ret %struct.poly64x2x2_t [[TMP6]]
poly64x2x2_t test_vld2q_dup_p64(poly64_t  *a) {
  return vld2q_dup_p64(a);
}

// CHECK-LABEL: define{{.*}} %struct.float64x1x2_t @test_vld2_dup_f64(double* noundef %a) #2 {
// CHECK:   [[RETVAL:%.*]] = alloca %struct.float64x1x2_t, align 8
// CHECK:   [[__RET:%.*]] = alloca %struct.float64x1x2_t, align 8
// CHECK:   [[TMP0:%.*]] = bitcast %struct.float64x1x2_t* [[__RET]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast double* %a to i8*
// CHECK:   [[TMP2:%.*]] = bitcast i8* [[TMP1]] to double*
// CHECK:   [[VLD2:%.*]] = call { <1 x double>, <1 x double> } @llvm.aarch64.neon.ld2r.v1f64.p0f64(double* [[TMP2]])
// CHECK:   [[TMP3:%.*]] = bitcast i8* [[TMP0]] to { <1 x double>, <1 x double> }*
// CHECK:   store { <1 x double>, <1 x double> } [[VLD2]], { <1 x double>, <1 x double> }* [[TMP3]]
// CHECK:   [[TMP4:%.*]] = bitcast %struct.float64x1x2_t* [[RETVAL]] to i8*
// CHECK:   [[TMP5:%.*]] = bitcast %struct.float64x1x2_t* [[__RET]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[TMP4]], i8* align 8 [[TMP5]], i64 16, i1 false)
// CHECK:   [[TMP6:%.*]] = load %struct.float64x1x2_t, %struct.float64x1x2_t* [[RETVAL]], align 8
// CHECK:   ret %struct.float64x1x2_t [[TMP6]]
float64x1x2_t test_vld2_dup_f64(float64_t  *a) {
  return vld2_dup_f64(a);
}

// CHECK-LABEL: define{{.*}} %struct.poly64x1x2_t @test_vld2_dup_p64(i64* noundef %a) #2 {
// CHECK:   [[RETVAL:%.*]] = alloca %struct.poly64x1x2_t, align 8
// CHECK:   [[__RET:%.*]] = alloca %struct.poly64x1x2_t, align 8
// CHECK:   [[TMP0:%.*]] = bitcast %struct.poly64x1x2_t* [[__RET]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast i64* %a to i8*
// CHECK:   [[TMP2:%.*]] = bitcast i8* [[TMP1]] to i64*
// CHECK:   [[VLD2:%.*]] = call { <1 x i64>, <1 x i64> } @llvm.aarch64.neon.ld2r.v1i64.p0i64(i64* [[TMP2]])
// CHECK:   [[TMP3:%.*]] = bitcast i8* [[TMP0]] to { <1 x i64>, <1 x i64> }*
// CHECK:   store { <1 x i64>, <1 x i64> } [[VLD2]], { <1 x i64>, <1 x i64> }* [[TMP3]]
// CHECK:   [[TMP4:%.*]] = bitcast %struct.poly64x1x2_t* [[RETVAL]] to i8*
// CHECK:   [[TMP5:%.*]] = bitcast %struct.poly64x1x2_t* [[__RET]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[TMP4]], i8* align 8 [[TMP5]], i64 16, i1 false)
// CHECK:   [[TMP6:%.*]] = load %struct.poly64x1x2_t, %struct.poly64x1x2_t* [[RETVAL]], align 8
// CHECK:   ret %struct.poly64x1x2_t [[TMP6]]
poly64x1x2_t test_vld2_dup_p64(poly64_t  *a) {
  return vld2_dup_p64(a);
}

// CHECK-LABEL: define{{.*}} %struct.uint64x2x3_t @test_vld3q_dup_u64(i64* noundef %a) #2 {
// CHECK:   [[RETVAL:%.*]] = alloca %struct.uint64x2x3_t, align 16
// CHECK:   [[__RET:%.*]] = alloca %struct.uint64x2x3_t, align 16
// CHECK:   [[TMP0:%.*]] = bitcast %struct.uint64x2x3_t* [[__RET]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast i64* %a to i8*
// CHECK:   [[TMP2:%.*]] = bitcast i8* [[TMP1]] to i64*
// CHECK:   [[VLD3:%.*]] = call { <2 x i64>, <2 x i64>, <2 x i64> } @llvm.aarch64.neon.ld3r.v2i64.p0i64(i64* [[TMP2]])
// CHECK:   [[TMP3:%.*]] = bitcast i8* [[TMP0]] to { <2 x i64>, <2 x i64>, <2 x i64> }*
// CHECK:   store { <2 x i64>, <2 x i64>, <2 x i64> } [[VLD3]], { <2 x i64>, <2 x i64>, <2 x i64> }* [[TMP3]]
// CHECK:   [[TMP4:%.*]] = bitcast %struct.uint64x2x3_t* [[RETVAL]] to i8*
// CHECK:   [[TMP5:%.*]] = bitcast %struct.uint64x2x3_t* [[__RET]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP4]], i8* align 16 [[TMP5]], i64 48, i1 false)
// CHECK:   [[TMP6:%.*]] = load %struct.uint64x2x3_t, %struct.uint64x2x3_t* [[RETVAL]], align 16
// CHECK:   ret %struct.uint64x2x3_t [[TMP6]]
uint64x2x3_t test_vld3q_dup_u64(uint64_t  *a) {
  return vld3q_dup_u64(a);
  // [{{x[0-9]+|sp}}]
}

// CHECK-LABEL: define{{.*}} %struct.int64x2x3_t @test_vld3q_dup_s64(i64* noundef %a) #2 {
// CHECK:   [[RETVAL:%.*]] = alloca %struct.int64x2x3_t, align 16
// CHECK:   [[__RET:%.*]] = alloca %struct.int64x2x3_t, align 16
// CHECK:   [[TMP0:%.*]] = bitcast %struct.int64x2x3_t* [[__RET]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast i64* %a to i8*
// CHECK:   [[TMP2:%.*]] = bitcast i8* [[TMP1]] to i64*
// CHECK:   [[VLD3:%.*]] = call { <2 x i64>, <2 x i64>, <2 x i64> } @llvm.aarch64.neon.ld3r.v2i64.p0i64(i64* [[TMP2]])
// CHECK:   [[TMP3:%.*]] = bitcast i8* [[TMP0]] to { <2 x i64>, <2 x i64>, <2 x i64> }*
// CHECK:   store { <2 x i64>, <2 x i64>, <2 x i64> } [[VLD3]], { <2 x i64>, <2 x i64>, <2 x i64> }* [[TMP3]]
// CHECK:   [[TMP4:%.*]] = bitcast %struct.int64x2x3_t* [[RETVAL]] to i8*
// CHECK:   [[TMP5:%.*]] = bitcast %struct.int64x2x3_t* [[__RET]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP4]], i8* align 16 [[TMP5]], i64 48, i1 false)
// CHECK:   [[TMP6:%.*]] = load %struct.int64x2x3_t, %struct.int64x2x3_t* [[RETVAL]], align 16
// CHECK:   ret %struct.int64x2x3_t [[TMP6]]
int64x2x3_t test_vld3q_dup_s64(int64_t  *a) {
  return vld3q_dup_s64(a);
  // [{{x[0-9]+|sp}}]
}

// CHECK-LABEL: define{{.*}} %struct.float64x2x3_t @test_vld3q_dup_f64(double* noundef %a) #2 {
// CHECK:   [[RETVAL:%.*]] = alloca %struct.float64x2x3_t, align 16
// CHECK:   [[__RET:%.*]] = alloca %struct.float64x2x3_t, align 16
// CHECK:   [[TMP0:%.*]] = bitcast %struct.float64x2x3_t* [[__RET]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast double* %a to i8*
// CHECK:   [[TMP2:%.*]] = bitcast i8* [[TMP1]] to double*
// CHECK:   [[VLD3:%.*]] = call { <2 x double>, <2 x double>, <2 x double> } @llvm.aarch64.neon.ld3r.v2f64.p0f64(double* [[TMP2]])
// CHECK:   [[TMP3:%.*]] = bitcast i8* [[TMP0]] to { <2 x double>, <2 x double>, <2 x double> }*
// CHECK:   store { <2 x double>, <2 x double>, <2 x double> } [[VLD3]], { <2 x double>, <2 x double>, <2 x double> }* [[TMP3]]
// CHECK:   [[TMP4:%.*]] = bitcast %struct.float64x2x3_t* [[RETVAL]] to i8*
// CHECK:   [[TMP5:%.*]] = bitcast %struct.float64x2x3_t* [[__RET]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP4]], i8* align 16 [[TMP5]], i64 48, i1 false)
// CHECK:   [[TMP6:%.*]] = load %struct.float64x2x3_t, %struct.float64x2x3_t* [[RETVAL]], align 16
// CHECK:   ret %struct.float64x2x3_t [[TMP6]]
float64x2x3_t test_vld3q_dup_f64(float64_t  *a) {
  return vld3q_dup_f64(a);
  // [{{x[0-9]+|sp}}]
}

// CHECK-LABEL: define{{.*}} %struct.poly64x2x3_t @test_vld3q_dup_p64(i64* noundef %a) #2 {
// CHECK:   [[RETVAL:%.*]] = alloca %struct.poly64x2x3_t, align 16
// CHECK:   [[__RET:%.*]] = alloca %struct.poly64x2x3_t, align 16
// CHECK:   [[TMP0:%.*]] = bitcast %struct.poly64x2x3_t* [[__RET]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast i64* %a to i8*
// CHECK:   [[TMP2:%.*]] = bitcast i8* [[TMP1]] to i64*
// CHECK:   [[VLD3:%.*]] = call { <2 x i64>, <2 x i64>, <2 x i64> } @llvm.aarch64.neon.ld3r.v2i64.p0i64(i64* [[TMP2]])
// CHECK:   [[TMP3:%.*]] = bitcast i8* [[TMP0]] to { <2 x i64>, <2 x i64>, <2 x i64> }*
// CHECK:   store { <2 x i64>, <2 x i64>, <2 x i64> } [[VLD3]], { <2 x i64>, <2 x i64>, <2 x i64> }* [[TMP3]]
// CHECK:   [[TMP4:%.*]] = bitcast %struct.poly64x2x3_t* [[RETVAL]] to i8*
// CHECK:   [[TMP5:%.*]] = bitcast %struct.poly64x2x3_t* [[__RET]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP4]], i8* align 16 [[TMP5]], i64 48, i1 false)
// CHECK:   [[TMP6:%.*]] = load %struct.poly64x2x3_t, %struct.poly64x2x3_t* [[RETVAL]], align 16
// CHECK:   ret %struct.poly64x2x3_t [[TMP6]]
poly64x2x3_t test_vld3q_dup_p64(poly64_t  *a) {
  return vld3q_dup_p64(a);
  // [{{x[0-9]+|sp}}]
}

// CHECK-LABEL: define{{.*}} %struct.float64x1x3_t @test_vld3_dup_f64(double* noundef %a) #2 {
// CHECK:   [[RETVAL:%.*]] = alloca %struct.float64x1x3_t, align 8
// CHECK:   [[__RET:%.*]] = alloca %struct.float64x1x3_t, align 8
// CHECK:   [[TMP0:%.*]] = bitcast %struct.float64x1x3_t* [[__RET]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast double* %a to i8*
// CHECK:   [[TMP2:%.*]] = bitcast i8* [[TMP1]] to double*
// CHECK:   [[VLD3:%.*]] = call { <1 x double>, <1 x double>, <1 x double> } @llvm.aarch64.neon.ld3r.v1f64.p0f64(double* [[TMP2]])
// CHECK:   [[TMP3:%.*]] = bitcast i8* [[TMP0]] to { <1 x double>, <1 x double>, <1 x double> }*
// CHECK:   store { <1 x double>, <1 x double>, <1 x double> } [[VLD3]], { <1 x double>, <1 x double>, <1 x double> }* [[TMP3]]
// CHECK:   [[TMP4:%.*]] = bitcast %struct.float64x1x3_t* [[RETVAL]] to i8*
// CHECK:   [[TMP5:%.*]] = bitcast %struct.float64x1x3_t* [[__RET]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[TMP4]], i8* align 8 [[TMP5]], i64 24, i1 false)
// CHECK:   [[TMP6:%.*]] = load %struct.float64x1x3_t, %struct.float64x1x3_t* [[RETVAL]], align 8
// CHECK:   ret %struct.float64x1x3_t [[TMP6]]
float64x1x3_t test_vld3_dup_f64(float64_t  *a) {
  return vld3_dup_f64(a);
  // [{{x[0-9]+|sp}}]
}

// CHECK-LABEL: define{{.*}} %struct.poly64x1x3_t @test_vld3_dup_p64(i64* noundef %a) #2 {
// CHECK:   [[RETVAL:%.*]] = alloca %struct.poly64x1x3_t, align 8
// CHECK:   [[__RET:%.*]] = alloca %struct.poly64x1x3_t, align 8
// CHECK:   [[TMP0:%.*]] = bitcast %struct.poly64x1x3_t* [[__RET]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast i64* %a to i8*
// CHECK:   [[TMP2:%.*]] = bitcast i8* [[TMP1]] to i64*
// CHECK:   [[VLD3:%.*]] = call { <1 x i64>, <1 x i64>, <1 x i64> } @llvm.aarch64.neon.ld3r.v1i64.p0i64(i64* [[TMP2]])
// CHECK:   [[TMP3:%.*]] = bitcast i8* [[TMP0]] to { <1 x i64>, <1 x i64>, <1 x i64> }*
// CHECK:   store { <1 x i64>, <1 x i64>, <1 x i64> } [[VLD3]], { <1 x i64>, <1 x i64>, <1 x i64> }* [[TMP3]]
// CHECK:   [[TMP4:%.*]] = bitcast %struct.poly64x1x3_t* [[RETVAL]] to i8*
// CHECK:   [[TMP5:%.*]] = bitcast %struct.poly64x1x3_t* [[__RET]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[TMP4]], i8* align 8 [[TMP5]], i64 24, i1 false)
// CHECK:   [[TMP6:%.*]] = load %struct.poly64x1x3_t, %struct.poly64x1x3_t* [[RETVAL]], align 8
// CHECK:   ret %struct.poly64x1x3_t [[TMP6]]
poly64x1x3_t test_vld3_dup_p64(poly64_t  *a) {
  return vld3_dup_p64(a);
  // [{{x[0-9]+|sp}}]
}

// CHECK-LABEL: define{{.*}} %struct.uint64x2x4_t @test_vld4q_dup_u64(i64* noundef %a) #2 {
// CHECK:   [[RETVAL:%.*]] = alloca %struct.uint64x2x4_t, align 16
// CHECK:   [[__RET:%.*]] = alloca %struct.uint64x2x4_t, align 16
// CHECK:   [[TMP0:%.*]] = bitcast %struct.uint64x2x4_t* [[__RET]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast i64* %a to i8*
// CHECK:   [[TMP2:%.*]] = bitcast i8* [[TMP1]] to i64*
// CHECK:   [[VLD4:%.*]] = call { <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } @llvm.aarch64.neon.ld4r.v2i64.p0i64(i64* [[TMP2]])
// CHECK:   [[TMP3:%.*]] = bitcast i8* [[TMP0]] to { <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> }*
// CHECK:   store { <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } [[VLD4]], { <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> }* [[TMP3]]
// CHECK:   [[TMP4:%.*]] = bitcast %struct.uint64x2x4_t* [[RETVAL]] to i8*
// CHECK:   [[TMP5:%.*]] = bitcast %struct.uint64x2x4_t* [[__RET]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP4]], i8* align 16 [[TMP5]], i64 64, i1 false)
// CHECK:   [[TMP6:%.*]] = load %struct.uint64x2x4_t, %struct.uint64x2x4_t* [[RETVAL]], align 16
// CHECK:   ret %struct.uint64x2x4_t [[TMP6]]
uint64x2x4_t test_vld4q_dup_u64(uint64_t  *a) {
  return vld4q_dup_u64(a);
}

// CHECK-LABEL: define{{.*}} %struct.int64x2x4_t @test_vld4q_dup_s64(i64* noundef %a) #2 {
// CHECK:   [[RETVAL:%.*]] = alloca %struct.int64x2x4_t, align 16
// CHECK:   [[__RET:%.*]] = alloca %struct.int64x2x4_t, align 16
// CHECK:   [[TMP0:%.*]] = bitcast %struct.int64x2x4_t* [[__RET]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast i64* %a to i8*
// CHECK:   [[TMP2:%.*]] = bitcast i8* [[TMP1]] to i64*
// CHECK:   [[VLD4:%.*]] = call { <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } @llvm.aarch64.neon.ld4r.v2i64.p0i64(i64* [[TMP2]])
// CHECK:   [[TMP3:%.*]] = bitcast i8* [[TMP0]] to { <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> }*
// CHECK:   store { <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } [[VLD4]], { <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> }* [[TMP3]]
// CHECK:   [[TMP4:%.*]] = bitcast %struct.int64x2x4_t* [[RETVAL]] to i8*
// CHECK:   [[TMP5:%.*]] = bitcast %struct.int64x2x4_t* [[__RET]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP4]], i8* align 16 [[TMP5]], i64 64, i1 false)
// CHECK:   [[TMP6:%.*]] = load %struct.int64x2x4_t, %struct.int64x2x4_t* [[RETVAL]], align 16
// CHECK:   ret %struct.int64x2x4_t [[TMP6]]
int64x2x4_t test_vld4q_dup_s64(int64_t  *a) {
  return vld4q_dup_s64(a);
}

// CHECK-LABEL: define{{.*}} %struct.float64x2x4_t @test_vld4q_dup_f64(double* noundef %a) #2 {
// CHECK:   [[RETVAL:%.*]] = alloca %struct.float64x2x4_t, align 16
// CHECK:   [[__RET:%.*]] = alloca %struct.float64x2x4_t, align 16
// CHECK:   [[TMP0:%.*]] = bitcast %struct.float64x2x4_t* [[__RET]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast double* %a to i8*
// CHECK:   [[TMP2:%.*]] = bitcast i8* [[TMP1]] to double*
// CHECK:   [[VLD4:%.*]] = call { <2 x double>, <2 x double>, <2 x double>, <2 x double> } @llvm.aarch64.neon.ld4r.v2f64.p0f64(double* [[TMP2]])
// CHECK:   [[TMP3:%.*]] = bitcast i8* [[TMP0]] to { <2 x double>, <2 x double>, <2 x double>, <2 x double> }*
// CHECK:   store { <2 x double>, <2 x double>, <2 x double>, <2 x double> } [[VLD4]], { <2 x double>, <2 x double>, <2 x double>, <2 x double> }* [[TMP3]]
// CHECK:   [[TMP4:%.*]] = bitcast %struct.float64x2x4_t* [[RETVAL]] to i8*
// CHECK:   [[TMP5:%.*]] = bitcast %struct.float64x2x4_t* [[__RET]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP4]], i8* align 16 [[TMP5]], i64 64, i1 false)
// CHECK:   [[TMP6:%.*]] = load %struct.float64x2x4_t, %struct.float64x2x4_t* [[RETVAL]], align 16
// CHECK:   ret %struct.float64x2x4_t [[TMP6]]
float64x2x4_t test_vld4q_dup_f64(float64_t  *a) {
  return vld4q_dup_f64(a);
}

// CHECK-LABEL: define{{.*}} %struct.poly64x2x4_t @test_vld4q_dup_p64(i64* noundef %a) #2 {
// CHECK:   [[RETVAL:%.*]] = alloca %struct.poly64x2x4_t, align 16
// CHECK:   [[__RET:%.*]] = alloca %struct.poly64x2x4_t, align 16
// CHECK:   [[TMP0:%.*]] = bitcast %struct.poly64x2x4_t* [[__RET]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast i64* %a to i8*
// CHECK:   [[TMP2:%.*]] = bitcast i8* [[TMP1]] to i64*
// CHECK:   [[VLD4:%.*]] = call { <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } @llvm.aarch64.neon.ld4r.v2i64.p0i64(i64* [[TMP2]])
// CHECK:   [[TMP3:%.*]] = bitcast i8* [[TMP0]] to { <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> }*
// CHECK:   store { <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } [[VLD4]], { <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> }* [[TMP3]]
// CHECK:   [[TMP4:%.*]] = bitcast %struct.poly64x2x4_t* [[RETVAL]] to i8*
// CHECK:   [[TMP5:%.*]] = bitcast %struct.poly64x2x4_t* [[__RET]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP4]], i8* align 16 [[TMP5]], i64 64, i1 false)
// CHECK:   [[TMP6:%.*]] = load %struct.poly64x2x4_t, %struct.poly64x2x4_t* [[RETVAL]], align 16
// CHECK:   ret %struct.poly64x2x4_t [[TMP6]]
poly64x2x4_t test_vld4q_dup_p64(poly64_t  *a) {
  return vld4q_dup_p64(a);
}

// CHECK-LABEL: define{{.*}} %struct.float64x1x4_t @test_vld4_dup_f64(double* noundef %a) #2 {
// CHECK:   [[RETVAL:%.*]] = alloca %struct.float64x1x4_t, align 8
// CHECK:   [[__RET:%.*]] = alloca %struct.float64x1x4_t, align 8
// CHECK:   [[TMP0:%.*]] = bitcast %struct.float64x1x4_t* [[__RET]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast double* %a to i8*
// CHECK:   [[TMP2:%.*]] = bitcast i8* [[TMP1]] to double*
// CHECK:   [[VLD4:%.*]] = call { <1 x double>, <1 x double>, <1 x double>, <1 x double> } @llvm.aarch64.neon.ld4r.v1f64.p0f64(double* [[TMP2]])
// CHECK:   [[TMP3:%.*]] = bitcast i8* [[TMP0]] to { <1 x double>, <1 x double>, <1 x double>, <1 x double> }*
// CHECK:   store { <1 x double>, <1 x double>, <1 x double>, <1 x double> } [[VLD4]], { <1 x double>, <1 x double>, <1 x double>, <1 x double> }* [[TMP3]]
// CHECK:   [[TMP4:%.*]] = bitcast %struct.float64x1x4_t* [[RETVAL]] to i8*
// CHECK:   [[TMP5:%.*]] = bitcast %struct.float64x1x4_t* [[__RET]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[TMP4]], i8* align 8 [[TMP5]], i64 32, i1 false)
// CHECK:   [[TMP6:%.*]] = load %struct.float64x1x4_t, %struct.float64x1x4_t* [[RETVAL]], align 8
// CHECK:   ret %struct.float64x1x4_t [[TMP6]]
float64x1x4_t test_vld4_dup_f64(float64_t  *a) {
  return vld4_dup_f64(a);
}

// CHECK-LABEL: define{{.*}} %struct.poly64x1x4_t @test_vld4_dup_p64(i64* noundef %a) #2 {
// CHECK:   [[RETVAL:%.*]] = alloca %struct.poly64x1x4_t, align 8
// CHECK:   [[__RET:%.*]] = alloca %struct.poly64x1x4_t, align 8
// CHECK:   [[TMP0:%.*]] = bitcast %struct.poly64x1x4_t* [[__RET]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast i64* %a to i8*
// CHECK:   [[TMP2:%.*]] = bitcast i8* [[TMP1]] to i64*
// CHECK:   [[VLD4:%.*]] = call { <1 x i64>, <1 x i64>, <1 x i64>, <1 x i64> } @llvm.aarch64.neon.ld4r.v1i64.p0i64(i64* [[TMP2]])
// CHECK:   [[TMP3:%.*]] = bitcast i8* [[TMP0]] to { <1 x i64>, <1 x i64>, <1 x i64>, <1 x i64> }*
// CHECK:   store { <1 x i64>, <1 x i64>, <1 x i64>, <1 x i64> } [[VLD4]], { <1 x i64>, <1 x i64>, <1 x i64>, <1 x i64> }* [[TMP3]]
// CHECK:   [[TMP4:%.*]] = bitcast %struct.poly64x1x4_t* [[RETVAL]] to i8*
// CHECK:   [[TMP5:%.*]] = bitcast %struct.poly64x1x4_t* [[__RET]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[TMP4]], i8* align 8 [[TMP5]], i64 32, i1 false)
// CHECK:   [[TMP6:%.*]] = load %struct.poly64x1x4_t, %struct.poly64x1x4_t* [[RETVAL]], align 8
// CHECK:   ret %struct.poly64x1x4_t [[TMP6]]
poly64x1x4_t test_vld4_dup_p64(poly64_t  *a) {
  return vld4_dup_p64(a);
}

// CHECK-LABEL: define{{.*}} <16 x i8> @test_vld1q_lane_u8(i8* noundef %a, <16 x i8> noundef %b) #0 {
// CHECK:   [[TMP0:%.*]] = load i8, i8* %a
// CHECK:   [[VLD1_LANE:%.*]] = insertelement <16 x i8> %b, i8 [[TMP0]], i32 15
// CHECK:   ret <16 x i8> [[VLD1_LANE]]
uint8x16_t test_vld1q_lane_u8(uint8_t  *a, uint8x16_t b) {
  return vld1q_lane_u8(a, b, 15);
}

// CHECK-LABEL: define{{.*}} <8 x i16> @test_vld1q_lane_u16(i16* noundef %a, <8 x i16> noundef %b) #0 {
// CHECK:   [[TMP0:%.*]] = bitcast i16* %a to i8*
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <16 x i8> [[TMP1]] to <8 x i16>
// CHECK:   [[TMP3:%.*]] = bitcast i8* [[TMP0]] to i16*
// CHECK:   [[TMP4:%.*]] = load i16, i16* [[TMP3]]
// CHECK:   [[VLD1_LANE:%.*]] = insertelement <8 x i16> [[TMP2]], i16 [[TMP4]], i32 7
// CHECK:   ret <8 x i16> [[VLD1_LANE]]
uint16x8_t test_vld1q_lane_u16(uint16_t  *a, uint16x8_t b) {
  return vld1q_lane_u16(a, b, 7);
}

// CHECK-LABEL: define{{.*}} <4 x i32> @test_vld1q_lane_u32(i32* noundef %a, <4 x i32> noundef %b) #0 {
// CHECK:   [[TMP0:%.*]] = bitcast i32* %a to i8*
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i32> %b to <16 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <16 x i8> [[TMP1]] to <4 x i32>
// CHECK:   [[TMP3:%.*]] = bitcast i8* [[TMP0]] to i32*
// CHECK:   [[TMP4:%.*]] = load i32, i32* [[TMP3]]
// CHECK:   [[VLD1_LANE:%.*]] = insertelement <4 x i32> [[TMP2]], i32 [[TMP4]], i32 3
// CHECK:   ret <4 x i32> [[VLD1_LANE]]
uint32x4_t test_vld1q_lane_u32(uint32_t  *a, uint32x4_t b) {
  return vld1q_lane_u32(a, b, 3);
}

// CHECK-LABEL: define{{.*}} <2 x i64> @test_vld1q_lane_u64(i64* noundef %a, <2 x i64> noundef %b) #0 {
// CHECK:   [[TMP0:%.*]] = bitcast i64* %a to i8*
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i64> %b to <16 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <16 x i8> [[TMP1]] to <2 x i64>
// CHECK:   [[TMP3:%.*]] = bitcast i8* [[TMP0]] to i64*
// CHECK:   [[TMP4:%.*]] = load i64, i64* [[TMP3]]
// CHECK:   [[VLD1_LANE:%.*]] = insertelement <2 x i64> [[TMP2]], i64 [[TMP4]], i32 1
// CHECK:   ret <2 x i64> [[VLD1_LANE]]
uint64x2_t test_vld1q_lane_u64(uint64_t  *a, uint64x2_t b) {
  return vld1q_lane_u64(a, b, 1);
}

// CHECK-LABEL: define{{.*}} <16 x i8> @test_vld1q_lane_s8(i8* noundef %a, <16 x i8> noundef %b) #0 {
// CHECK:   [[TMP0:%.*]] = load i8, i8* %a
// CHECK:   [[VLD1_LANE:%.*]] = insertelement <16 x i8> %b, i8 [[TMP0]], i32 15
// CHECK:   ret <16 x i8> [[VLD1_LANE]]
int8x16_t test_vld1q_lane_s8(int8_t  *a, int8x16_t b) {
  return vld1q_lane_s8(a, b, 15);
}

// CHECK-LABEL: define{{.*}} <8 x i16> @test_vld1q_lane_s16(i16* noundef %a, <8 x i16> noundef %b) #0 {
// CHECK:   [[TMP0:%.*]] = bitcast i16* %a to i8*
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <16 x i8> [[TMP1]] to <8 x i16>
// CHECK:   [[TMP3:%.*]] = bitcast i8* [[TMP0]] to i16*
// CHECK:   [[TMP4:%.*]] = load i16, i16* [[TMP3]]
// CHECK:   [[VLD1_LANE:%.*]] = insertelement <8 x i16> [[TMP2]], i16 [[TMP4]], i32 7
// CHECK:   ret <8 x i16> [[VLD1_LANE]]
int16x8_t test_vld1q_lane_s16(int16_t  *a, int16x8_t b) {
  return vld1q_lane_s16(a, b, 7);
}

// CHECK-LABEL: define{{.*}} <4 x i32> @test_vld1q_lane_s32(i32* noundef %a, <4 x i32> noundef %b) #0 {
// CHECK:   [[TMP0:%.*]] = bitcast i32* %a to i8*
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i32> %b to <16 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <16 x i8> [[TMP1]] to <4 x i32>
// CHECK:   [[TMP3:%.*]] = bitcast i8* [[TMP0]] to i32*
// CHECK:   [[TMP4:%.*]] = load i32, i32* [[TMP3]]
// CHECK:   [[VLD1_LANE:%.*]] = insertelement <4 x i32> [[TMP2]], i32 [[TMP4]], i32 3
// CHECK:   ret <4 x i32> [[VLD1_LANE]]
int32x4_t test_vld1q_lane_s32(int32_t  *a, int32x4_t b) {
  return vld1q_lane_s32(a, b, 3);
}

// CHECK-LABEL: define{{.*}} <2 x i64> @test_vld1q_lane_s64(i64* noundef %a, <2 x i64> noundef %b) #0 {
// CHECK:   [[TMP0:%.*]] = bitcast i64* %a to i8*
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i64> %b to <16 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <16 x i8> [[TMP1]] to <2 x i64>
// CHECK:   [[TMP3:%.*]] = bitcast i8* [[TMP0]] to i64*
// CHECK:   [[TMP4:%.*]] = load i64, i64* [[TMP3]]
// CHECK:   [[VLD1_LANE:%.*]] = insertelement <2 x i64> [[TMP2]], i64 [[TMP4]], i32 1
// CHECK:   ret <2 x i64> [[VLD1_LANE]]
int64x2_t test_vld1q_lane_s64(int64_t  *a, int64x2_t b) {
  return vld1q_lane_s64(a, b, 1);
}

// CHECK-LABEL: define{{.*}} <8 x half> @test_vld1q_lane_f16(half* noundef %a, <8 x half> noundef %b) #0 {
// CHECK:   [[TMP0:%.*]] = bitcast half* %a to i8*
// CHECK:   [[TMP1:%.*]] = bitcast <8 x half> %b to <16 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <16 x i8> [[TMP1]] to <8 x half>
// CHECK:   [[TMP3:%.*]] = bitcast i8* [[TMP0]] to half*
// CHECK:   [[TMP4:%.*]] = load half, half* [[TMP3]]
// CHECK:   [[VLD1_LANE:%.*]] = insertelement <8 x half> [[TMP2]], half [[TMP4]], i32 7
// CHECK:   ret <8 x half> [[VLD1_LANE]]
float16x8_t test_vld1q_lane_f16(float16_t  *a, float16x8_t b) {
  return vld1q_lane_f16(a, b, 7);
}

// CHECK-LABEL: define{{.*}} <4 x float> @test_vld1q_lane_f32(float* noundef %a, <4 x float> noundef %b) #0 {
// CHECK:   [[TMP0:%.*]] = bitcast float* %a to i8*
// CHECK:   [[TMP1:%.*]] = bitcast <4 x float> %b to <16 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <16 x i8> [[TMP1]] to <4 x float>
// CHECK:   [[TMP3:%.*]] = bitcast i8* [[TMP0]] to float*
// CHECK:   [[TMP4:%.*]] = load float, float* [[TMP3]]
// CHECK:   [[VLD1_LANE:%.*]] = insertelement <4 x float> [[TMP2]], float [[TMP4]], i32 3
// CHECK:   ret <4 x float> [[VLD1_LANE]]
float32x4_t test_vld1q_lane_f32(float32_t  *a, float32x4_t b) {
  return vld1q_lane_f32(a, b, 3);
}

// CHECK-LABEL: define{{.*}} <2 x double> @test_vld1q_lane_f64(double* noundef %a, <2 x double> noundef %b) #0 {
// CHECK:   [[TMP0:%.*]] = bitcast double* %a to i8*
// CHECK:   [[TMP1:%.*]] = bitcast <2 x double> %b to <16 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <16 x i8> [[TMP1]] to <2 x double>
// CHECK:   [[TMP3:%.*]] = bitcast i8* [[TMP0]] to double*
// CHECK:   [[TMP4:%.*]] = load double, double* [[TMP3]]
// CHECK:   [[VLD1_LANE:%.*]] = insertelement <2 x double> [[TMP2]], double [[TMP4]], i32 1
// CHECK:   ret <2 x double> [[VLD1_LANE]]
float64x2_t test_vld1q_lane_f64(float64_t  *a, float64x2_t b) {
  return vld1q_lane_f64(a, b, 1);
}

// CHECK-LABEL: define{{.*}} <16 x i8> @test_vld1q_lane_p8(i8* noundef %a, <16 x i8> noundef %b) #0 {
// CHECK:   [[TMP0:%.*]] = load i8, i8* %a
// CHECK:   [[VLD1_LANE:%.*]] = insertelement <16 x i8> %b, i8 [[TMP0]], i32 15
// CHECK:   ret <16 x i8> [[VLD1_LANE]]
poly8x16_t test_vld1q_lane_p8(poly8_t  *a, poly8x16_t b) {
  return vld1q_lane_p8(a, b, 15);
}

// CHECK-LABEL: define{{.*}} <8 x i16> @test_vld1q_lane_p16(i16* noundef %a, <8 x i16> noundef %b) #0 {
// CHECK:   [[TMP0:%.*]] = bitcast i16* %a to i8*
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <16 x i8> [[TMP1]] to <8 x i16>
// CHECK:   [[TMP3:%.*]] = bitcast i8* [[TMP0]] to i16*
// CHECK:   [[TMP4:%.*]] = load i16, i16* [[TMP3]]
// CHECK:   [[VLD1_LANE:%.*]] = insertelement <8 x i16> [[TMP2]], i16 [[TMP4]], i32 7
// CHECK:   ret <8 x i16> [[VLD1_LANE]]
poly16x8_t test_vld1q_lane_p16(poly16_t  *a, poly16x8_t b) {
  return vld1q_lane_p16(a, b, 7);
}

// CHECK-LABEL: define{{.*}} <2 x i64> @test_vld1q_lane_p64(i64* noundef %a, <2 x i64> noundef %b) #0 {
// CHECK:   [[TMP0:%.*]] = bitcast i64* %a to i8*
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i64> %b to <16 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <16 x i8> [[TMP1]] to <2 x i64>
// CHECK:   [[TMP3:%.*]] = bitcast i8* [[TMP0]] to i64*
// CHECK:   [[TMP4:%.*]] = load i64, i64* [[TMP3]]
// CHECK:   [[VLD1_LANE:%.*]] = insertelement <2 x i64> [[TMP2]], i64 [[TMP4]], i32 1
// CHECK:   ret <2 x i64> [[VLD1_LANE]]
poly64x2_t test_vld1q_lane_p64(poly64_t  *a, poly64x2_t b) {
  return vld1q_lane_p64(a, b, 1);
}

// CHECK-LABEL: define{{.*}} <8 x i8> @test_vld1_lane_u8(i8* noundef %a, <8 x i8> noundef %b) #1 {
// CHECK:   [[TMP0:%.*]] = load i8, i8* %a
// CHECK:   [[VLD1_LANE:%.*]] = insertelement <8 x i8> %b, i8 [[TMP0]], i32 7
// CHECK:   ret <8 x i8> [[VLD1_LANE]]
uint8x8_t test_vld1_lane_u8(uint8_t  *a, uint8x8_t b) {
  return vld1_lane_u8(a, b, 7);
}

// CHECK-LABEL: define{{.*}} <4 x i16> @test_vld1_lane_u16(i16* noundef %a, <4 x i16> noundef %b) #1 {
// CHECK:   [[TMP0:%.*]] = bitcast i16* %a to i8*
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <8 x i8> [[TMP1]] to <4 x i16>
// CHECK:   [[TMP3:%.*]] = bitcast i8* [[TMP0]] to i16*
// CHECK:   [[TMP4:%.*]] = load i16, i16* [[TMP3]]
// CHECK:   [[VLD1_LANE:%.*]] = insertelement <4 x i16> [[TMP2]], i16 [[TMP4]], i32 3
// CHECK:   ret <4 x i16> [[VLD1_LANE]]
uint16x4_t test_vld1_lane_u16(uint16_t  *a, uint16x4_t b) {
  return vld1_lane_u16(a, b, 3);
}

// CHECK-LABEL: define{{.*}} <2 x i32> @test_vld1_lane_u32(i32* noundef %a, <2 x i32> noundef %b) #1 {
// CHECK:   [[TMP0:%.*]] = bitcast i32* %a to i8*
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <8 x i8> [[TMP1]] to <2 x i32>
// CHECK:   [[TMP3:%.*]] = bitcast i8* [[TMP0]] to i32*
// CHECK:   [[TMP4:%.*]] = load i32, i32* [[TMP3]]
// CHECK:   [[VLD1_LANE:%.*]] = insertelement <2 x i32> [[TMP2]], i32 [[TMP4]], i32 1
// CHECK:   ret <2 x i32> [[VLD1_LANE]]
uint32x2_t test_vld1_lane_u32(uint32_t  *a, uint32x2_t b) {
  return vld1_lane_u32(a, b, 1);
}

// CHECK-LABEL: define{{.*}} <1 x i64> @test_vld1_lane_u64(i64* noundef %a, <1 x i64> noundef %b) #1 {
// CHECK:   [[TMP0:%.*]] = bitcast i64* %a to i8*
// CHECK:   [[TMP1:%.*]] = bitcast <1 x i64> %b to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <8 x i8> [[TMP1]] to <1 x i64>
// CHECK:   [[TMP3:%.*]] = bitcast i8* [[TMP0]] to i64*
// CHECK:   [[TMP4:%.*]] = load i64, i64* [[TMP3]]
// CHECK:   [[VLD1_LANE:%.*]] = insertelement <1 x i64> [[TMP2]], i64 [[TMP4]], i32 0
// CHECK:   ret <1 x i64> [[VLD1_LANE]]
uint64x1_t test_vld1_lane_u64(uint64_t  *a, uint64x1_t b) {
  return vld1_lane_u64(a, b, 0);
}

// CHECK-LABEL: define{{.*}} <8 x i8> @test_vld1_lane_s8(i8* noundef %a, <8 x i8> noundef %b) #1 {
// CHECK:   [[TMP0:%.*]] = load i8, i8* %a
// CHECK:   [[VLD1_LANE:%.*]] = insertelement <8 x i8> %b, i8 [[TMP0]], i32 7
// CHECK:   ret <8 x i8> [[VLD1_LANE]]
int8x8_t test_vld1_lane_s8(int8_t  *a, int8x8_t b) {
  return vld1_lane_s8(a, b, 7);
}

// CHECK-LABEL: define{{.*}} <4 x i16> @test_vld1_lane_s16(i16* noundef %a, <4 x i16> noundef %b) #1 {
// CHECK:   [[TMP0:%.*]] = bitcast i16* %a to i8*
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <8 x i8> [[TMP1]] to <4 x i16>
// CHECK:   [[TMP3:%.*]] = bitcast i8* [[TMP0]] to i16*
// CHECK:   [[TMP4:%.*]] = load i16, i16* [[TMP3]]
// CHECK:   [[VLD1_LANE:%.*]] = insertelement <4 x i16> [[TMP2]], i16 [[TMP4]], i32 3
// CHECK:   ret <4 x i16> [[VLD1_LANE]]
int16x4_t test_vld1_lane_s16(int16_t  *a, int16x4_t b) {
  return vld1_lane_s16(a, b, 3);
}

// CHECK-LABEL: define{{.*}} <2 x i32> @test_vld1_lane_s32(i32* noundef %a, <2 x i32> noundef %b) #1 {
// CHECK:   [[TMP0:%.*]] = bitcast i32* %a to i8*
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <8 x i8> [[TMP1]] to <2 x i32>
// CHECK:   [[TMP3:%.*]] = bitcast i8* [[TMP0]] to i32*
// CHECK:   [[TMP4:%.*]] = load i32, i32* [[TMP3]]
// CHECK:   [[VLD1_LANE:%.*]] = insertelement <2 x i32> [[TMP2]], i32 [[TMP4]], i32 1
// CHECK:   ret <2 x i32> [[VLD1_LANE]]
int32x2_t test_vld1_lane_s32(int32_t  *a, int32x2_t b) {
  return vld1_lane_s32(a, b, 1);
}

// CHECK-LABEL: define{{.*}} <1 x i64> @test_vld1_lane_s64(i64* noundef %a, <1 x i64> noundef %b) #1 {
// CHECK:   [[TMP0:%.*]] = bitcast i64* %a to i8*
// CHECK:   [[TMP1:%.*]] = bitcast <1 x i64> %b to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <8 x i8> [[TMP1]] to <1 x i64>
// CHECK:   [[TMP3:%.*]] = bitcast i8* [[TMP0]] to i64*
// CHECK:   [[TMP4:%.*]] = load i64, i64* [[TMP3]]
// CHECK:   [[VLD1_LANE:%.*]] = insertelement <1 x i64> [[TMP2]], i64 [[TMP4]], i32 0
// CHECK:   ret <1 x i64> [[VLD1_LANE]]
int64x1_t test_vld1_lane_s64(int64_t  *a, int64x1_t b) {
  return vld1_lane_s64(a, b, 0);
}

// CHECK-LABEL: define{{.*}} <4 x half> @test_vld1_lane_f16(half* noundef %a, <4 x half> noundef %b) #1 {
// CHECK:   [[TMP0:%.*]] = bitcast half* %a to i8*
// CHECK:   [[TMP1:%.*]] = bitcast <4 x half> %b to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <8 x i8> [[TMP1]] to <4 x half>
// CHECK:   [[TMP3:%.*]] = bitcast i8* [[TMP0]] to half*
// CHECK:   [[TMP4:%.*]] = load half, half* [[TMP3]]
// CHECK:   [[VLD1_LANE:%.*]] = insertelement <4 x half> [[TMP2]], half [[TMP4]], i32 3
// CHECK:   ret <4 x half> [[VLD1_LANE]]
float16x4_t test_vld1_lane_f16(float16_t  *a, float16x4_t b) {
  return vld1_lane_f16(a, b, 3);
}

// CHECK-LABEL: define{{.*}} <2 x float> @test_vld1_lane_f32(float* noundef %a, <2 x float> noundef %b) #1 {
// CHECK:   [[TMP0:%.*]] = bitcast float* %a to i8*
// CHECK:   [[TMP1:%.*]] = bitcast <2 x float> %b to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <8 x i8> [[TMP1]] to <2 x float>
// CHECK:   [[TMP3:%.*]] = bitcast i8* [[TMP0]] to float*
// CHECK:   [[TMP4:%.*]] = load float, float* [[TMP3]]
// CHECK:   [[VLD1_LANE:%.*]] = insertelement <2 x float> [[TMP2]], float [[TMP4]], i32 1
// CHECK:   ret <2 x float> [[VLD1_LANE]]
float32x2_t test_vld1_lane_f32(float32_t  *a, float32x2_t b) {
  return vld1_lane_f32(a, b, 1);
}

// CHECK-LABEL: define{{.*}} <1 x double> @test_vld1_lane_f64(double* noundef %a, <1 x double> noundef %b) #1 {
// CHECK:   [[TMP0:%.*]] = bitcast double* %a to i8*
// CHECK:   [[TMP1:%.*]] = bitcast <1 x double> %b to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <8 x i8> [[TMP1]] to <1 x double>
// CHECK:   [[TMP3:%.*]] = bitcast i8* [[TMP0]] to double*
// CHECK:   [[TMP4:%.*]] = load double, double* [[TMP3]]
// CHECK:   [[VLD1_LANE:%.*]] = insertelement <1 x double> [[TMP2]], double [[TMP4]], i32 0
// CHECK:   ret <1 x double> [[VLD1_LANE]]
float64x1_t test_vld1_lane_f64(float64_t  *a, float64x1_t b) {
  return vld1_lane_f64(a, b, 0);
}

// CHECK-LABEL: define{{.*}} <8 x i8> @test_vld1_lane_p8(i8* noundef %a, <8 x i8> noundef %b) #1 {
// CHECK:   [[TMP0:%.*]] = load i8, i8* %a
// CHECK:   [[VLD1_LANE:%.*]] = insertelement <8 x i8> %b, i8 [[TMP0]], i32 7
// CHECK:   ret <8 x i8> [[VLD1_LANE]]
poly8x8_t test_vld1_lane_p8(poly8_t  *a, poly8x8_t b) {
  return vld1_lane_p8(a, b, 7);
}

// CHECK-LABEL: define{{.*}} <4 x i16> @test_vld1_lane_p16(i16* noundef %a, <4 x i16> noundef %b) #1 {
// CHECK:   [[TMP0:%.*]] = bitcast i16* %a to i8*
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <8 x i8> [[TMP1]] to <4 x i16>
// CHECK:   [[TMP3:%.*]] = bitcast i8* [[TMP0]] to i16*
// CHECK:   [[TMP4:%.*]] = load i16, i16* [[TMP3]]
// CHECK:   [[VLD1_LANE:%.*]] = insertelement <4 x i16> [[TMP2]], i16 [[TMP4]], i32 3
// CHECK:   ret <4 x i16> [[VLD1_LANE]]
poly16x4_t test_vld1_lane_p16(poly16_t  *a, poly16x4_t b) {
  return vld1_lane_p16(a, b, 3);
}

// CHECK-LABEL: define{{.*}} <1 x i64> @test_vld1_lane_p64(i64* noundef %a, <1 x i64> noundef %b) #1 {
// CHECK:   [[TMP0:%.*]] = bitcast i64* %a to i8*
// CHECK:   [[TMP1:%.*]] = bitcast <1 x i64> %b to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <8 x i8> [[TMP1]] to <1 x i64>
// CHECK:   [[TMP3:%.*]] = bitcast i8* [[TMP0]] to i64*
// CHECK:   [[TMP4:%.*]] = load i64, i64* [[TMP3]]
// CHECK:   [[VLD1_LANE:%.*]] = insertelement <1 x i64> [[TMP2]], i64 [[TMP4]], i32 0
// CHECK:   ret <1 x i64> [[VLD1_LANE]]
poly64x1_t test_vld1_lane_p64(poly64_t  *a, poly64x1_t b) {
  return vld1_lane_p64(a, b, 0);
}

// CHECK-LABEL: define{{.*}} %struct.int8x16x2_t @test_vld2q_lane_s8(i8* noundef %ptr, [2 x <16 x i8>] %src.coerce) #2 {
// CHECK:   [[RETVAL:%.*]] = alloca %struct.int8x16x2_t, align 16
// CHECK:   [[SRC:%.*]] = alloca %struct.int8x16x2_t, align 16
// CHECK:   [[__RET:%.*]] = alloca %struct.int8x16x2_t, align 16
// CHECK:   [[__S1:%.*]] = alloca %struct.int8x16x2_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.int8x16x2_t, %struct.int8x16x2_t* [[SRC]], i32 0, i32 0
// CHECK:   store [2 x <16 x i8>] [[SRC]].coerce, [2 x <16 x i8>]* [[COERCE_DIVE]], align 16
// CHECK:   [[TMP0:%.*]] = bitcast %struct.int8x16x2_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.int8x16x2_t* [[SRC]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP0]], i8* align 16 [[TMP1]], i64 32, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast %struct.int8x16x2_t* [[__RET]] to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.int8x16x2_t, %struct.int8x16x2_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <16 x i8>], [2 x <16 x i8>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX]], align 16
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.int8x16x2_t, %struct.int8x16x2_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <16 x i8>], [2 x <16 x i8>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP4:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX2]], align 16
// CHECK:   [[VLD2_LANE:%.*]] = call { <16 x i8>, <16 x i8> } @llvm.aarch64.neon.ld2lane.v16i8.p0i8(<16 x i8> [[TMP3]], <16 x i8> [[TMP4]], i64 15, i8* %ptr)
// CHECK:   [[TMP5:%.*]] = bitcast i8* [[TMP2]] to { <16 x i8>, <16 x i8> }*
// CHECK:   store { <16 x i8>, <16 x i8> } [[VLD2_LANE]], { <16 x i8>, <16 x i8> }* [[TMP5]]
// CHECK:   [[TMP6:%.*]] = bitcast %struct.int8x16x2_t* [[RETVAL]] to i8*
// CHECK:   [[TMP7:%.*]] = bitcast %struct.int8x16x2_t* [[__RET]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP6]], i8* align 16 [[TMP7]], i64 32, i1 false)
// CHECK:   [[TMP8:%.*]] = load %struct.int8x16x2_t, %struct.int8x16x2_t* [[RETVAL]], align 16
// CHECK:   ret %struct.int8x16x2_t [[TMP8]]
int8x16x2_t test_vld2q_lane_s8(int8_t const * ptr, int8x16x2_t src) {
  return vld2q_lane_s8(ptr, src, 15);
}

// CHECK-LABEL: define{{.*}} %struct.uint8x16x2_t @test_vld2q_lane_u8(i8* noundef %ptr, [2 x <16 x i8>] %src.coerce) #2 {
// CHECK:   [[RETVAL:%.*]] = alloca %struct.uint8x16x2_t, align 16
// CHECK:   [[SRC:%.*]] = alloca %struct.uint8x16x2_t, align 16
// CHECK:   [[__RET:%.*]] = alloca %struct.uint8x16x2_t, align 16
// CHECK:   [[__S1:%.*]] = alloca %struct.uint8x16x2_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.uint8x16x2_t, %struct.uint8x16x2_t* [[SRC]], i32 0, i32 0
// CHECK:   store [2 x <16 x i8>] [[SRC]].coerce, [2 x <16 x i8>]* [[COERCE_DIVE]], align 16
// CHECK:   [[TMP0:%.*]] = bitcast %struct.uint8x16x2_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.uint8x16x2_t* [[SRC]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP0]], i8* align 16 [[TMP1]], i64 32, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast %struct.uint8x16x2_t* [[__RET]] to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.uint8x16x2_t, %struct.uint8x16x2_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <16 x i8>], [2 x <16 x i8>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX]], align 16
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.uint8x16x2_t, %struct.uint8x16x2_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <16 x i8>], [2 x <16 x i8>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP4:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX2]], align 16
// CHECK:   [[VLD2_LANE:%.*]] = call { <16 x i8>, <16 x i8> } @llvm.aarch64.neon.ld2lane.v16i8.p0i8(<16 x i8> [[TMP3]], <16 x i8> [[TMP4]], i64 15, i8* %ptr)
// CHECK:   [[TMP5:%.*]] = bitcast i8* [[TMP2]] to { <16 x i8>, <16 x i8> }*
// CHECK:   store { <16 x i8>, <16 x i8> } [[VLD2_LANE]], { <16 x i8>, <16 x i8> }* [[TMP5]]
// CHECK:   [[TMP6:%.*]] = bitcast %struct.uint8x16x2_t* [[RETVAL]] to i8*
// CHECK:   [[TMP7:%.*]] = bitcast %struct.uint8x16x2_t* [[__RET]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP6]], i8* align 16 [[TMP7]], i64 32, i1 false)
// CHECK:   [[TMP8:%.*]] = load %struct.uint8x16x2_t, %struct.uint8x16x2_t* [[RETVAL]], align 16
// CHECK:   ret %struct.uint8x16x2_t [[TMP8]]
uint8x16x2_t test_vld2q_lane_u8(uint8_t const * ptr, uint8x16x2_t src) {
  return vld2q_lane_u8(ptr, src, 15);
}

// CHECK-LABEL: define{{.*}} %struct.poly8x16x2_t @test_vld2q_lane_p8(i8* noundef %ptr, [2 x <16 x i8>] %src.coerce) #2 {
// CHECK:   [[RETVAL:%.*]] = alloca %struct.poly8x16x2_t, align 16
// CHECK:   [[SRC:%.*]] = alloca %struct.poly8x16x2_t, align 16
// CHECK:   [[__RET:%.*]] = alloca %struct.poly8x16x2_t, align 16
// CHECK:   [[__S1:%.*]] = alloca %struct.poly8x16x2_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.poly8x16x2_t, %struct.poly8x16x2_t* [[SRC]], i32 0, i32 0
// CHECK:   store [2 x <16 x i8>] [[SRC]].coerce, [2 x <16 x i8>]* [[COERCE_DIVE]], align 16
// CHECK:   [[TMP0:%.*]] = bitcast %struct.poly8x16x2_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.poly8x16x2_t* [[SRC]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP0]], i8* align 16 [[TMP1]], i64 32, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast %struct.poly8x16x2_t* [[__RET]] to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.poly8x16x2_t, %struct.poly8x16x2_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <16 x i8>], [2 x <16 x i8>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX]], align 16
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.poly8x16x2_t, %struct.poly8x16x2_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <16 x i8>], [2 x <16 x i8>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP4:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX2]], align 16
// CHECK:   [[VLD2_LANE:%.*]] = call { <16 x i8>, <16 x i8> } @llvm.aarch64.neon.ld2lane.v16i8.p0i8(<16 x i8> [[TMP3]], <16 x i8> [[TMP4]], i64 15, i8* %ptr)
// CHECK:   [[TMP5:%.*]] = bitcast i8* [[TMP2]] to { <16 x i8>, <16 x i8> }*
// CHECK:   store { <16 x i8>, <16 x i8> } [[VLD2_LANE]], { <16 x i8>, <16 x i8> }* [[TMP5]]
// CHECK:   [[TMP6:%.*]] = bitcast %struct.poly8x16x2_t* [[RETVAL]] to i8*
// CHECK:   [[TMP7:%.*]] = bitcast %struct.poly8x16x2_t* [[__RET]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP6]], i8* align 16 [[TMP7]], i64 32, i1 false)
// CHECK:   [[TMP8:%.*]] = load %struct.poly8x16x2_t, %struct.poly8x16x2_t* [[RETVAL]], align 16
// CHECK:   ret %struct.poly8x16x2_t [[TMP8]]
poly8x16x2_t test_vld2q_lane_p8(poly8_t const * ptr, poly8x16x2_t src) {
  return vld2q_lane_p8(ptr, src, 15);
}

// CHECK-LABEL: define{{.*}} %struct.int8x16x3_t @test_vld3q_lane_s8(i8* noundef %ptr, [3 x <16 x i8>] %src.coerce) #2 {
// CHECK:   [[RETVAL:%.*]] = alloca %struct.int8x16x3_t, align 16
// CHECK:   [[SRC:%.*]] = alloca %struct.int8x16x3_t, align 16
// CHECK:   [[__RET:%.*]] = alloca %struct.int8x16x3_t, align 16
// CHECK:   [[__S1:%.*]] = alloca %struct.int8x16x3_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.int8x16x3_t, %struct.int8x16x3_t* [[SRC]], i32 0, i32 0
// CHECK:   store [3 x <16 x i8>] [[SRC]].coerce, [3 x <16 x i8>]* [[COERCE_DIVE]], align 16
// CHECK:   [[TMP0:%.*]] = bitcast %struct.int8x16x3_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.int8x16x3_t* [[SRC]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP0]], i8* align 16 [[TMP1]], i64 48, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast %struct.int8x16x3_t* [[__RET]] to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.int8x16x3_t, %struct.int8x16x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <16 x i8>], [3 x <16 x i8>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX]], align 16
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.int8x16x3_t, %struct.int8x16x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <16 x i8>], [3 x <16 x i8>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP4:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX2]], align 16
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds %struct.int8x16x3_t, %struct.int8x16x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <16 x i8>], [3 x <16 x i8>]* [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP5:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX4]], align 16
// CHECK:   [[VLD3_LANE:%.*]] = call { <16 x i8>, <16 x i8>, <16 x i8> } @llvm.aarch64.neon.ld3lane.v16i8.p0i8(<16 x i8> [[TMP3]], <16 x i8> [[TMP4]], <16 x i8> [[TMP5]], i64 15, i8* %ptr)
// CHECK:   [[TMP6:%.*]] = bitcast i8* [[TMP2]] to { <16 x i8>, <16 x i8>, <16 x i8> }*
// CHECK:   store { <16 x i8>, <16 x i8>, <16 x i8> } [[VLD3_LANE]], { <16 x i8>, <16 x i8>, <16 x i8> }* [[TMP6]]
// CHECK:   [[TMP7:%.*]] = bitcast %struct.int8x16x3_t* [[RETVAL]] to i8*
// CHECK:   [[TMP8:%.*]] = bitcast %struct.int8x16x3_t* [[__RET]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP7]], i8* align 16 [[TMP8]], i64 48, i1 false)
// CHECK:   [[TMP9:%.*]] = load %struct.int8x16x3_t, %struct.int8x16x3_t* [[RETVAL]], align 16
// CHECK:   ret %struct.int8x16x3_t [[TMP9]]
int8x16x3_t test_vld3q_lane_s8(int8_t const * ptr, int8x16x3_t src) {
  return vld3q_lane_s8(ptr, src, 15);
}

// CHECK-LABEL: define{{.*}} %struct.uint8x16x3_t @test_vld3q_lane_u8(i8* noundef %ptr, [3 x <16 x i8>] %src.coerce) #2 {
// CHECK:   [[RETVAL:%.*]] = alloca %struct.uint8x16x3_t, align 16
// CHECK:   [[SRC:%.*]] = alloca %struct.uint8x16x3_t, align 16
// CHECK:   [[__RET:%.*]] = alloca %struct.uint8x16x3_t, align 16
// CHECK:   [[__S1:%.*]] = alloca %struct.uint8x16x3_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.uint8x16x3_t, %struct.uint8x16x3_t* [[SRC]], i32 0, i32 0
// CHECK:   store [3 x <16 x i8>] [[SRC]].coerce, [3 x <16 x i8>]* [[COERCE_DIVE]], align 16
// CHECK:   [[TMP0:%.*]] = bitcast %struct.uint8x16x3_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.uint8x16x3_t* [[SRC]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP0]], i8* align 16 [[TMP1]], i64 48, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast %struct.uint8x16x3_t* [[__RET]] to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.uint8x16x3_t, %struct.uint8x16x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <16 x i8>], [3 x <16 x i8>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX]], align 16
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.uint8x16x3_t, %struct.uint8x16x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <16 x i8>], [3 x <16 x i8>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP4:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX2]], align 16
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds %struct.uint8x16x3_t, %struct.uint8x16x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <16 x i8>], [3 x <16 x i8>]* [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP5:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX4]], align 16
// CHECK:   [[VLD3_LANE:%.*]] = call { <16 x i8>, <16 x i8>, <16 x i8> } @llvm.aarch64.neon.ld3lane.v16i8.p0i8(<16 x i8> [[TMP3]], <16 x i8> [[TMP4]], <16 x i8> [[TMP5]], i64 15, i8* %ptr)
// CHECK:   [[TMP6:%.*]] = bitcast i8* [[TMP2]] to { <16 x i8>, <16 x i8>, <16 x i8> }*
// CHECK:   store { <16 x i8>, <16 x i8>, <16 x i8> } [[VLD3_LANE]], { <16 x i8>, <16 x i8>, <16 x i8> }* [[TMP6]]
// CHECK:   [[TMP7:%.*]] = bitcast %struct.uint8x16x3_t* [[RETVAL]] to i8*
// CHECK:   [[TMP8:%.*]] = bitcast %struct.uint8x16x3_t* [[__RET]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP7]], i8* align 16 [[TMP8]], i64 48, i1 false)
// CHECK:   [[TMP9:%.*]] = load %struct.uint8x16x3_t, %struct.uint8x16x3_t* [[RETVAL]], align 16
// CHECK:   ret %struct.uint8x16x3_t [[TMP9]]
uint8x16x3_t test_vld3q_lane_u8(uint8_t const * ptr, uint8x16x3_t src) {
  return vld3q_lane_u8(ptr, src, 15);
}

// CHECK-LABEL: define{{.*}} %struct.uint16x8x2_t @test_vld2q_lane_u16(i16* noundef %a, [2 x <8 x i16>] %b.coerce) #2 {
// CHECK:   [[RETVAL:%.*]] = alloca %struct.uint16x8x2_t, align 16
// CHECK:   [[B:%.*]] = alloca %struct.uint16x8x2_t, align 16
// CHECK:   [[__RET:%.*]] = alloca %struct.uint16x8x2_t, align 16
// CHECK:   [[__S1:%.*]] = alloca %struct.uint16x8x2_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.uint16x8x2_t, %struct.uint16x8x2_t* [[B]], i32 0, i32 0
// CHECK:   store [2 x <8 x i16>] [[B]].coerce, [2 x <8 x i16>]* [[COERCE_DIVE]], align 16
// CHECK:   [[TMP0:%.*]] = bitcast %struct.uint16x8x2_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.uint16x8x2_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP0]], i8* align 16 [[TMP1]], i64 32, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast %struct.uint16x8x2_t* [[__RET]] to i8*
// CHECK:   [[TMP3:%.*]] = bitcast i16* %a to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.uint16x8x2_t, %struct.uint16x8x2_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <8 x i16>], [2 x <8 x i16>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP4:%.*]] = load <8 x i16>, <8 x i16>* [[ARRAYIDX]], align 16
// CHECK:   [[TMP5:%.*]] = bitcast <8 x i16> [[TMP4]] to <16 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.uint16x8x2_t, %struct.uint16x8x2_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <8 x i16>], [2 x <8 x i16>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP6:%.*]] = load <8 x i16>, <8 x i16>* [[ARRAYIDX2]], align 16
// CHECK:   [[TMP7:%.*]] = bitcast <8 x i16> [[TMP6]] to <16 x i8>
// CHECK:   [[TMP8:%.*]] = bitcast <16 x i8> [[TMP5]] to <8 x i16>
// CHECK:   [[TMP9:%.*]] = bitcast <16 x i8> [[TMP7]] to <8 x i16>
// CHECK:   [[VLD2_LANE:%.*]] = call { <8 x i16>, <8 x i16> } @llvm.aarch64.neon.ld2lane.v8i16.p0i8(<8 x i16> [[TMP8]], <8 x i16> [[TMP9]], i64 7, i8* [[TMP3]])
// CHECK:   [[TMP10:%.*]] = bitcast i8* [[TMP2]] to { <8 x i16>, <8 x i16> }*
// CHECK:   store { <8 x i16>, <8 x i16> } [[VLD2_LANE]], { <8 x i16>, <8 x i16> }* [[TMP10]]
// CHECK:   [[TMP11:%.*]] = bitcast %struct.uint16x8x2_t* [[RETVAL]] to i8*
// CHECK:   [[TMP12:%.*]] = bitcast %struct.uint16x8x2_t* [[__RET]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP11]], i8* align 16 [[TMP12]], i64 32, i1 false)
// CHECK:   [[TMP13:%.*]] = load %struct.uint16x8x2_t, %struct.uint16x8x2_t* [[RETVAL]], align 16
// CHECK:   ret %struct.uint16x8x2_t [[TMP13]]
uint16x8x2_t test_vld2q_lane_u16(uint16_t  *a, uint16x8x2_t b) {
  return vld2q_lane_u16(a, b, 7);
}

// CHECK-LABEL: define{{.*}} %struct.uint32x4x2_t @test_vld2q_lane_u32(i32* noundef %a, [2 x <4 x i32>] %b.coerce) #2 {
// CHECK:   [[RETVAL:%.*]] = alloca %struct.uint32x4x2_t, align 16
// CHECK:   [[B:%.*]] = alloca %struct.uint32x4x2_t, align 16
// CHECK:   [[__RET:%.*]] = alloca %struct.uint32x4x2_t, align 16
// CHECK:   [[__S1:%.*]] = alloca %struct.uint32x4x2_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.uint32x4x2_t, %struct.uint32x4x2_t* [[B]], i32 0, i32 0
// CHECK:   store [2 x <4 x i32>] [[B]].coerce, [2 x <4 x i32>]* [[COERCE_DIVE]], align 16
// CHECK:   [[TMP0:%.*]] = bitcast %struct.uint32x4x2_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.uint32x4x2_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP0]], i8* align 16 [[TMP1]], i64 32, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast %struct.uint32x4x2_t* [[__RET]] to i8*
// CHECK:   [[TMP3:%.*]] = bitcast i32* %a to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.uint32x4x2_t, %struct.uint32x4x2_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <4 x i32>], [2 x <4 x i32>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP4:%.*]] = load <4 x i32>, <4 x i32>* [[ARRAYIDX]], align 16
// CHECK:   [[TMP5:%.*]] = bitcast <4 x i32> [[TMP4]] to <16 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.uint32x4x2_t, %struct.uint32x4x2_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <4 x i32>], [2 x <4 x i32>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP6:%.*]] = load <4 x i32>, <4 x i32>* [[ARRAYIDX2]], align 16
// CHECK:   [[TMP7:%.*]] = bitcast <4 x i32> [[TMP6]] to <16 x i8>
// CHECK:   [[TMP8:%.*]] = bitcast <16 x i8> [[TMP5]] to <4 x i32>
// CHECK:   [[TMP9:%.*]] = bitcast <16 x i8> [[TMP7]] to <4 x i32>
// CHECK:   [[VLD2_LANE:%.*]] = call { <4 x i32>, <4 x i32> } @llvm.aarch64.neon.ld2lane.v4i32.p0i8(<4 x i32> [[TMP8]], <4 x i32> [[TMP9]], i64 3, i8* [[TMP3]])
// CHECK:   [[TMP10:%.*]] = bitcast i8* [[TMP2]] to { <4 x i32>, <4 x i32> }*
// CHECK:   store { <4 x i32>, <4 x i32> } [[VLD2_LANE]], { <4 x i32>, <4 x i32> }* [[TMP10]]
// CHECK:   [[TMP11:%.*]] = bitcast %struct.uint32x4x2_t* [[RETVAL]] to i8*
// CHECK:   [[TMP12:%.*]] = bitcast %struct.uint32x4x2_t* [[__RET]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP11]], i8* align 16 [[TMP12]], i64 32, i1 false)
// CHECK:   [[TMP13:%.*]] = load %struct.uint32x4x2_t, %struct.uint32x4x2_t* [[RETVAL]], align 16
// CHECK:   ret %struct.uint32x4x2_t [[TMP13]]
uint32x4x2_t test_vld2q_lane_u32(uint32_t  *a, uint32x4x2_t b) {
  return vld2q_lane_u32(a, b, 3);
}

// CHECK-LABEL: define{{.*}} %struct.uint64x2x2_t @test_vld2q_lane_u64(i64* noundef %a, [2 x <2 x i64>] %b.coerce) #2 {
// CHECK:   [[RETVAL:%.*]] = alloca %struct.uint64x2x2_t, align 16
// CHECK:   [[B:%.*]] = alloca %struct.uint64x2x2_t, align 16
// CHECK:   [[__RET:%.*]] = alloca %struct.uint64x2x2_t, align 16
// CHECK:   [[__S1:%.*]] = alloca %struct.uint64x2x2_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.uint64x2x2_t, %struct.uint64x2x2_t* [[B]], i32 0, i32 0
// CHECK:   store [2 x <2 x i64>] [[B]].coerce, [2 x <2 x i64>]* [[COERCE_DIVE]], align 16
// CHECK:   [[TMP0:%.*]] = bitcast %struct.uint64x2x2_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.uint64x2x2_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP0]], i8* align 16 [[TMP1]], i64 32, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast %struct.uint64x2x2_t* [[__RET]] to i8*
// CHECK:   [[TMP3:%.*]] = bitcast i64* %a to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.uint64x2x2_t, %struct.uint64x2x2_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <2 x i64>], [2 x <2 x i64>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP4:%.*]] = load <2 x i64>, <2 x i64>* [[ARRAYIDX]], align 16
// CHECK:   [[TMP5:%.*]] = bitcast <2 x i64> [[TMP4]] to <16 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.uint64x2x2_t, %struct.uint64x2x2_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <2 x i64>], [2 x <2 x i64>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP6:%.*]] = load <2 x i64>, <2 x i64>* [[ARRAYIDX2]], align 16
// CHECK:   [[TMP7:%.*]] = bitcast <2 x i64> [[TMP6]] to <16 x i8>
// CHECK:   [[TMP8:%.*]] = bitcast <16 x i8> [[TMP5]] to <2 x i64>
// CHECK:   [[TMP9:%.*]] = bitcast <16 x i8> [[TMP7]] to <2 x i64>
// CHECK:   [[VLD2_LANE:%.*]] = call { <2 x i64>, <2 x i64> } @llvm.aarch64.neon.ld2lane.v2i64.p0i8(<2 x i64> [[TMP8]], <2 x i64> [[TMP9]], i64 1, i8* [[TMP3]])
// CHECK:   [[TMP10:%.*]] = bitcast i8* [[TMP2]] to { <2 x i64>, <2 x i64> }*
// CHECK:   store { <2 x i64>, <2 x i64> } [[VLD2_LANE]], { <2 x i64>, <2 x i64> }* [[TMP10]]
// CHECK:   [[TMP11:%.*]] = bitcast %struct.uint64x2x2_t* [[RETVAL]] to i8*
// CHECK:   [[TMP12:%.*]] = bitcast %struct.uint64x2x2_t* [[__RET]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP11]], i8* align 16 [[TMP12]], i64 32, i1 false)
// CHECK:   [[TMP13:%.*]] = load %struct.uint64x2x2_t, %struct.uint64x2x2_t* [[RETVAL]], align 16
// CHECK:   ret %struct.uint64x2x2_t [[TMP13]]
uint64x2x2_t test_vld2q_lane_u64(uint64_t  *a, uint64x2x2_t b) {
  return vld2q_lane_u64(a, b, 1);
}

// CHECK-LABEL: define{{.*}} %struct.int16x8x2_t @test_vld2q_lane_s16(i16* noundef %a, [2 x <8 x i16>] %b.coerce) #2 {
// CHECK:   [[RETVAL:%.*]] = alloca %struct.int16x8x2_t, align 16
// CHECK:   [[B:%.*]] = alloca %struct.int16x8x2_t, align 16
// CHECK:   [[__RET:%.*]] = alloca %struct.int16x8x2_t, align 16
// CHECK:   [[__S1:%.*]] = alloca %struct.int16x8x2_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.int16x8x2_t, %struct.int16x8x2_t* [[B]], i32 0, i32 0
// CHECK:   store [2 x <8 x i16>] [[B]].coerce, [2 x <8 x i16>]* [[COERCE_DIVE]], align 16
// CHECK:   [[TMP0:%.*]] = bitcast %struct.int16x8x2_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.int16x8x2_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP0]], i8* align 16 [[TMP1]], i64 32, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast %struct.int16x8x2_t* [[__RET]] to i8*
// CHECK:   [[TMP3:%.*]] = bitcast i16* %a to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.int16x8x2_t, %struct.int16x8x2_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <8 x i16>], [2 x <8 x i16>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP4:%.*]] = load <8 x i16>, <8 x i16>* [[ARRAYIDX]], align 16
// CHECK:   [[TMP5:%.*]] = bitcast <8 x i16> [[TMP4]] to <16 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.int16x8x2_t, %struct.int16x8x2_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <8 x i16>], [2 x <8 x i16>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP6:%.*]] = load <8 x i16>, <8 x i16>* [[ARRAYIDX2]], align 16
// CHECK:   [[TMP7:%.*]] = bitcast <8 x i16> [[TMP6]] to <16 x i8>
// CHECK:   [[TMP8:%.*]] = bitcast <16 x i8> [[TMP5]] to <8 x i16>
// CHECK:   [[TMP9:%.*]] = bitcast <16 x i8> [[TMP7]] to <8 x i16>
// CHECK:   [[VLD2_LANE:%.*]] = call { <8 x i16>, <8 x i16> } @llvm.aarch64.neon.ld2lane.v8i16.p0i8(<8 x i16> [[TMP8]], <8 x i16> [[TMP9]], i64 7, i8* [[TMP3]])
// CHECK:   [[TMP10:%.*]] = bitcast i8* [[TMP2]] to { <8 x i16>, <8 x i16> }*
// CHECK:   store { <8 x i16>, <8 x i16> } [[VLD2_LANE]], { <8 x i16>, <8 x i16> }* [[TMP10]]
// CHECK:   [[TMP11:%.*]] = bitcast %struct.int16x8x2_t* [[RETVAL]] to i8*
// CHECK:   [[TMP12:%.*]] = bitcast %struct.int16x8x2_t* [[__RET]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP11]], i8* align 16 [[TMP12]], i64 32, i1 false)
// CHECK:   [[TMP13:%.*]] = load %struct.int16x8x2_t, %struct.int16x8x2_t* [[RETVAL]], align 16
// CHECK:   ret %struct.int16x8x2_t [[TMP13]]
int16x8x2_t test_vld2q_lane_s16(int16_t  *a, int16x8x2_t b) {
  return vld2q_lane_s16(a, b, 7);
}

// CHECK-LABEL: define{{.*}} %struct.int32x4x2_t @test_vld2q_lane_s32(i32* noundef %a, [2 x <4 x i32>] %b.coerce) #2 {
// CHECK:   [[RETVAL:%.*]] = alloca %struct.int32x4x2_t, align 16
// CHECK:   [[B:%.*]] = alloca %struct.int32x4x2_t, align 16
// CHECK:   [[__RET:%.*]] = alloca %struct.int32x4x2_t, align 16
// CHECK:   [[__S1:%.*]] = alloca %struct.int32x4x2_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.int32x4x2_t, %struct.int32x4x2_t* [[B]], i32 0, i32 0
// CHECK:   store [2 x <4 x i32>] [[B]].coerce, [2 x <4 x i32>]* [[COERCE_DIVE]], align 16
// CHECK:   [[TMP0:%.*]] = bitcast %struct.int32x4x2_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.int32x4x2_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP0]], i8* align 16 [[TMP1]], i64 32, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast %struct.int32x4x2_t* [[__RET]] to i8*
// CHECK:   [[TMP3:%.*]] = bitcast i32* %a to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.int32x4x2_t, %struct.int32x4x2_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <4 x i32>], [2 x <4 x i32>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP4:%.*]] = load <4 x i32>, <4 x i32>* [[ARRAYIDX]], align 16
// CHECK:   [[TMP5:%.*]] = bitcast <4 x i32> [[TMP4]] to <16 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.int32x4x2_t, %struct.int32x4x2_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <4 x i32>], [2 x <4 x i32>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP6:%.*]] = load <4 x i32>, <4 x i32>* [[ARRAYIDX2]], align 16
// CHECK:   [[TMP7:%.*]] = bitcast <4 x i32> [[TMP6]] to <16 x i8>
// CHECK:   [[TMP8:%.*]] = bitcast <16 x i8> [[TMP5]] to <4 x i32>
// CHECK:   [[TMP9:%.*]] = bitcast <16 x i8> [[TMP7]] to <4 x i32>
// CHECK:   [[VLD2_LANE:%.*]] = call { <4 x i32>, <4 x i32> } @llvm.aarch64.neon.ld2lane.v4i32.p0i8(<4 x i32> [[TMP8]], <4 x i32> [[TMP9]], i64 3, i8* [[TMP3]])
// CHECK:   [[TMP10:%.*]] = bitcast i8* [[TMP2]] to { <4 x i32>, <4 x i32> }*
// CHECK:   store { <4 x i32>, <4 x i32> } [[VLD2_LANE]], { <4 x i32>, <4 x i32> }* [[TMP10]]
// CHECK:   [[TMP11:%.*]] = bitcast %struct.int32x4x2_t* [[RETVAL]] to i8*
// CHECK:   [[TMP12:%.*]] = bitcast %struct.int32x4x2_t* [[__RET]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP11]], i8* align 16 [[TMP12]], i64 32, i1 false)
// CHECK:   [[TMP13:%.*]] = load %struct.int32x4x2_t, %struct.int32x4x2_t* [[RETVAL]], align 16
// CHECK:   ret %struct.int32x4x2_t [[TMP13]]
int32x4x2_t test_vld2q_lane_s32(int32_t  *a, int32x4x2_t b) {
  return vld2q_lane_s32(a, b, 3);
}

// CHECK-LABEL: define{{.*}} %struct.int64x2x2_t @test_vld2q_lane_s64(i64* noundef %a, [2 x <2 x i64>] %b.coerce) #2 {
// CHECK:   [[RETVAL:%.*]] = alloca %struct.int64x2x2_t, align 16
// CHECK:   [[B:%.*]] = alloca %struct.int64x2x2_t, align 16
// CHECK:   [[__RET:%.*]] = alloca %struct.int64x2x2_t, align 16
// CHECK:   [[__S1:%.*]] = alloca %struct.int64x2x2_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.int64x2x2_t, %struct.int64x2x2_t* [[B]], i32 0, i32 0
// CHECK:   store [2 x <2 x i64>] [[B]].coerce, [2 x <2 x i64>]* [[COERCE_DIVE]], align 16
// CHECK:   [[TMP0:%.*]] = bitcast %struct.int64x2x2_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.int64x2x2_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP0]], i8* align 16 [[TMP1]], i64 32, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast %struct.int64x2x2_t* [[__RET]] to i8*
// CHECK:   [[TMP3:%.*]] = bitcast i64* %a to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.int64x2x2_t, %struct.int64x2x2_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <2 x i64>], [2 x <2 x i64>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP4:%.*]] = load <2 x i64>, <2 x i64>* [[ARRAYIDX]], align 16
// CHECK:   [[TMP5:%.*]] = bitcast <2 x i64> [[TMP4]] to <16 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.int64x2x2_t, %struct.int64x2x2_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <2 x i64>], [2 x <2 x i64>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP6:%.*]] = load <2 x i64>, <2 x i64>* [[ARRAYIDX2]], align 16
// CHECK:   [[TMP7:%.*]] = bitcast <2 x i64> [[TMP6]] to <16 x i8>
// CHECK:   [[TMP8:%.*]] = bitcast <16 x i8> [[TMP5]] to <2 x i64>
// CHECK:   [[TMP9:%.*]] = bitcast <16 x i8> [[TMP7]] to <2 x i64>
// CHECK:   [[VLD2_LANE:%.*]] = call { <2 x i64>, <2 x i64> } @llvm.aarch64.neon.ld2lane.v2i64.p0i8(<2 x i64> [[TMP8]], <2 x i64> [[TMP9]], i64 1, i8* [[TMP3]])
// CHECK:   [[TMP10:%.*]] = bitcast i8* [[TMP2]] to { <2 x i64>, <2 x i64> }*
// CHECK:   store { <2 x i64>, <2 x i64> } [[VLD2_LANE]], { <2 x i64>, <2 x i64> }* [[TMP10]]
// CHECK:   [[TMP11:%.*]] = bitcast %struct.int64x2x2_t* [[RETVAL]] to i8*
// CHECK:   [[TMP12:%.*]] = bitcast %struct.int64x2x2_t* [[__RET]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP11]], i8* align 16 [[TMP12]], i64 32, i1 false)
// CHECK:   [[TMP13:%.*]] = load %struct.int64x2x2_t, %struct.int64x2x2_t* [[RETVAL]], align 16
// CHECK:   ret %struct.int64x2x2_t [[TMP13]]
int64x2x2_t test_vld2q_lane_s64(int64_t  *a, int64x2x2_t b) {
  return vld2q_lane_s64(a, b, 1);
}

// CHECK-LABEL: define{{.*}} %struct.float16x8x2_t @test_vld2q_lane_f16(half* noundef %a, [2 x <8 x half>] %b.coerce) #2 {
// CHECK:   [[RETVAL:%.*]] = alloca %struct.float16x8x2_t, align 16
// CHECK:   [[B:%.*]] = alloca %struct.float16x8x2_t, align 16
// CHECK:   [[__RET:%.*]] = alloca %struct.float16x8x2_t, align 16
// CHECK:   [[__S1:%.*]] = alloca %struct.float16x8x2_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.float16x8x2_t, %struct.float16x8x2_t* [[B]], i32 0, i32 0
// CHECK:   store [2 x <8 x half>] [[B]].coerce, [2 x <8 x half>]* [[COERCE_DIVE]], align 16
// CHECK:   [[TMP0:%.*]] = bitcast %struct.float16x8x2_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.float16x8x2_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP0]], i8* align 16 [[TMP1]], i64 32, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast %struct.float16x8x2_t* [[__RET]] to i8*
// CHECK:   [[TMP3:%.*]] = bitcast half* %a to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.float16x8x2_t, %struct.float16x8x2_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <8 x half>], [2 x <8 x half>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP4:%.*]] = load <8 x half>, <8 x half>* [[ARRAYIDX]], align 16
// CHECK:   [[TMP5:%.*]] = bitcast <8 x half> [[TMP4]] to <16 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.float16x8x2_t, %struct.float16x8x2_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <8 x half>], [2 x <8 x half>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP6:%.*]] = load <8 x half>, <8 x half>* [[ARRAYIDX2]], align 16
// CHECK:   [[TMP7:%.*]] = bitcast <8 x half> [[TMP6]] to <16 x i8>
// CHECK:   [[TMP8:%.*]] = bitcast <16 x i8> [[TMP5]] to <8 x half>
// CHECK:   [[TMP9:%.*]] = bitcast <16 x i8> [[TMP7]] to <8 x half>
// CHECK:   [[VLD2_LANE:%.*]] = call { <8 x half>, <8 x half> } @llvm.aarch64.neon.ld2lane.v8f16.p0i8(<8 x half> [[TMP8]], <8 x half> [[TMP9]], i64 7, i8* [[TMP3]])
// CHECK:   [[TMP10:%.*]] = bitcast i8* [[TMP2]] to { <8 x half>, <8 x half> }*
// CHECK:   store { <8 x half>, <8 x half> } [[VLD2_LANE]], { <8 x half>, <8 x half> }* [[TMP10]]
// CHECK:   [[TMP11:%.*]] = bitcast %struct.float16x8x2_t* [[RETVAL]] to i8*
// CHECK:   [[TMP12:%.*]] = bitcast %struct.float16x8x2_t* [[__RET]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP11]], i8* align 16 [[TMP12]], i64 32, i1 false)
// CHECK:   [[TMP13:%.*]] = load %struct.float16x8x2_t, %struct.float16x8x2_t* [[RETVAL]], align 16
// CHECK:   ret %struct.float16x8x2_t [[TMP13]]
float16x8x2_t test_vld2q_lane_f16(float16_t  *a, float16x8x2_t b) {
  return vld2q_lane_f16(a, b, 7);
}

// CHECK-LABEL: define{{.*}} %struct.float32x4x2_t @test_vld2q_lane_f32(float* noundef %a, [2 x <4 x float>] %b.coerce) #2 {
// CHECK:   [[RETVAL:%.*]] = alloca %struct.float32x4x2_t, align 16
// CHECK:   [[B:%.*]] = alloca %struct.float32x4x2_t, align 16
// CHECK:   [[__RET:%.*]] = alloca %struct.float32x4x2_t, align 16
// CHECK:   [[__S1:%.*]] = alloca %struct.float32x4x2_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.float32x4x2_t, %struct.float32x4x2_t* [[B]], i32 0, i32 0
// CHECK:   store [2 x <4 x float>] [[B]].coerce, [2 x <4 x float>]* [[COERCE_DIVE]], align 16
// CHECK:   [[TMP0:%.*]] = bitcast %struct.float32x4x2_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.float32x4x2_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP0]], i8* align 16 [[TMP1]], i64 32, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast %struct.float32x4x2_t* [[__RET]] to i8*
// CHECK:   [[TMP3:%.*]] = bitcast float* %a to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.float32x4x2_t, %struct.float32x4x2_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <4 x float>], [2 x <4 x float>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP4:%.*]] = load <4 x float>, <4 x float>* [[ARRAYIDX]], align 16
// CHECK:   [[TMP5:%.*]] = bitcast <4 x float> [[TMP4]] to <16 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.float32x4x2_t, %struct.float32x4x2_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <4 x float>], [2 x <4 x float>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP6:%.*]] = load <4 x float>, <4 x float>* [[ARRAYIDX2]], align 16
// CHECK:   [[TMP7:%.*]] = bitcast <4 x float> [[TMP6]] to <16 x i8>
// CHECK:   [[TMP8:%.*]] = bitcast <16 x i8> [[TMP5]] to <4 x float>
// CHECK:   [[TMP9:%.*]] = bitcast <16 x i8> [[TMP7]] to <4 x float>
// CHECK:   [[VLD2_LANE:%.*]] = call { <4 x float>, <4 x float> } @llvm.aarch64.neon.ld2lane.v4f32.p0i8(<4 x float> [[TMP8]], <4 x float> [[TMP9]], i64 3, i8* [[TMP3]])
// CHECK:   [[TMP10:%.*]] = bitcast i8* [[TMP2]] to { <4 x float>, <4 x float> }*
// CHECK:   store { <4 x float>, <4 x float> } [[VLD2_LANE]], { <4 x float>, <4 x float> }* [[TMP10]]
// CHECK:   [[TMP11:%.*]] = bitcast %struct.float32x4x2_t* [[RETVAL]] to i8*
// CHECK:   [[TMP12:%.*]] = bitcast %struct.float32x4x2_t* [[__RET]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP11]], i8* align 16 [[TMP12]], i64 32, i1 false)
// CHECK:   [[TMP13:%.*]] = load %struct.float32x4x2_t, %struct.float32x4x2_t* [[RETVAL]], align 16
// CHECK:   ret %struct.float32x4x2_t [[TMP13]]
float32x4x2_t test_vld2q_lane_f32(float32_t  *a, float32x4x2_t b) {
  return vld2q_lane_f32(a, b, 3);
}

// CHECK-LABEL: define{{.*}} %struct.float64x2x2_t @test_vld2q_lane_f64(double* noundef %a, [2 x <2 x double>] %b.coerce) #2 {
// CHECK:   [[RETVAL:%.*]] = alloca %struct.float64x2x2_t, align 16
// CHECK:   [[B:%.*]] = alloca %struct.float64x2x2_t, align 16
// CHECK:   [[__RET:%.*]] = alloca %struct.float64x2x2_t, align 16
// CHECK:   [[__S1:%.*]] = alloca %struct.float64x2x2_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.float64x2x2_t, %struct.float64x2x2_t* [[B]], i32 0, i32 0
// CHECK:   store [2 x <2 x double>] [[B]].coerce, [2 x <2 x double>]* [[COERCE_DIVE]], align 16
// CHECK:   [[TMP0:%.*]] = bitcast %struct.float64x2x2_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.float64x2x2_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP0]], i8* align 16 [[TMP1]], i64 32, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast %struct.float64x2x2_t* [[__RET]] to i8*
// CHECK:   [[TMP3:%.*]] = bitcast double* %a to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.float64x2x2_t, %struct.float64x2x2_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <2 x double>], [2 x <2 x double>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP4:%.*]] = load <2 x double>, <2 x double>* [[ARRAYIDX]], align 16
// CHECK:   [[TMP5:%.*]] = bitcast <2 x double> [[TMP4]] to <16 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.float64x2x2_t, %struct.float64x2x2_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <2 x double>], [2 x <2 x double>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP6:%.*]] = load <2 x double>, <2 x double>* [[ARRAYIDX2]], align 16
// CHECK:   [[TMP7:%.*]] = bitcast <2 x double> [[TMP6]] to <16 x i8>
// CHECK:   [[TMP8:%.*]] = bitcast <16 x i8> [[TMP5]] to <2 x double>
// CHECK:   [[TMP9:%.*]] = bitcast <16 x i8> [[TMP7]] to <2 x double>
// CHECK:   [[VLD2_LANE:%.*]] = call { <2 x double>, <2 x double> } @llvm.aarch64.neon.ld2lane.v2f64.p0i8(<2 x double> [[TMP8]], <2 x double> [[TMP9]], i64 1, i8* [[TMP3]])
// CHECK:   [[TMP10:%.*]] = bitcast i8* [[TMP2]] to { <2 x double>, <2 x double> }*
// CHECK:   store { <2 x double>, <2 x double> } [[VLD2_LANE]], { <2 x double>, <2 x double> }* [[TMP10]]
// CHECK:   [[TMP11:%.*]] = bitcast %struct.float64x2x2_t* [[RETVAL]] to i8*
// CHECK:   [[TMP12:%.*]] = bitcast %struct.float64x2x2_t* [[__RET]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP11]], i8* align 16 [[TMP12]], i64 32, i1 false)
// CHECK:   [[TMP13:%.*]] = load %struct.float64x2x2_t, %struct.float64x2x2_t* [[RETVAL]], align 16
// CHECK:   ret %struct.float64x2x2_t [[TMP13]]
float64x2x2_t test_vld2q_lane_f64(float64_t  *a, float64x2x2_t b) {
  return vld2q_lane_f64(a, b, 1);
}

// CHECK-LABEL: define{{.*}} %struct.poly16x8x2_t @test_vld2q_lane_p16(i16* noundef %a, [2 x <8 x i16>] %b.coerce) #2 {
// CHECK:   [[RETVAL:%.*]] = alloca %struct.poly16x8x2_t, align 16
// CHECK:   [[B:%.*]] = alloca %struct.poly16x8x2_t, align 16
// CHECK:   [[__RET:%.*]] = alloca %struct.poly16x8x2_t, align 16
// CHECK:   [[__S1:%.*]] = alloca %struct.poly16x8x2_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.poly16x8x2_t, %struct.poly16x8x2_t* [[B]], i32 0, i32 0
// CHECK:   store [2 x <8 x i16>] [[B]].coerce, [2 x <8 x i16>]* [[COERCE_DIVE]], align 16
// CHECK:   [[TMP0:%.*]] = bitcast %struct.poly16x8x2_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.poly16x8x2_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP0]], i8* align 16 [[TMP1]], i64 32, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast %struct.poly16x8x2_t* [[__RET]] to i8*
// CHECK:   [[TMP3:%.*]] = bitcast i16* %a to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.poly16x8x2_t, %struct.poly16x8x2_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <8 x i16>], [2 x <8 x i16>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP4:%.*]] = load <8 x i16>, <8 x i16>* [[ARRAYIDX]], align 16
// CHECK:   [[TMP5:%.*]] = bitcast <8 x i16> [[TMP4]] to <16 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.poly16x8x2_t, %struct.poly16x8x2_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <8 x i16>], [2 x <8 x i16>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP6:%.*]] = load <8 x i16>, <8 x i16>* [[ARRAYIDX2]], align 16
// CHECK:   [[TMP7:%.*]] = bitcast <8 x i16> [[TMP6]] to <16 x i8>
// CHECK:   [[TMP8:%.*]] = bitcast <16 x i8> [[TMP5]] to <8 x i16>
// CHECK:   [[TMP9:%.*]] = bitcast <16 x i8> [[TMP7]] to <8 x i16>
// CHECK:   [[VLD2_LANE:%.*]] = call { <8 x i16>, <8 x i16> } @llvm.aarch64.neon.ld2lane.v8i16.p0i8(<8 x i16> [[TMP8]], <8 x i16> [[TMP9]], i64 7, i8* [[TMP3]])
// CHECK:   [[TMP10:%.*]] = bitcast i8* [[TMP2]] to { <8 x i16>, <8 x i16> }*
// CHECK:   store { <8 x i16>, <8 x i16> } [[VLD2_LANE]], { <8 x i16>, <8 x i16> }* [[TMP10]]
// CHECK:   [[TMP11:%.*]] = bitcast %struct.poly16x8x2_t* [[RETVAL]] to i8*
// CHECK:   [[TMP12:%.*]] = bitcast %struct.poly16x8x2_t* [[__RET]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP11]], i8* align 16 [[TMP12]], i64 32, i1 false)
// CHECK:   [[TMP13:%.*]] = load %struct.poly16x8x2_t, %struct.poly16x8x2_t* [[RETVAL]], align 16
// CHECK:   ret %struct.poly16x8x2_t [[TMP13]]
poly16x8x2_t test_vld2q_lane_p16(poly16_t  *a, poly16x8x2_t b) {
  return vld2q_lane_p16(a, b, 7);
}

// CHECK-LABEL: define{{.*}} %struct.poly64x2x2_t @test_vld2q_lane_p64(i64* noundef %a, [2 x <2 x i64>] %b.coerce) #2 {
// CHECK:   [[RETVAL:%.*]] = alloca %struct.poly64x2x2_t, align 16
// CHECK:   [[B:%.*]] = alloca %struct.poly64x2x2_t, align 16
// CHECK:   [[__RET:%.*]] = alloca %struct.poly64x2x2_t, align 16
// CHECK:   [[__S1:%.*]] = alloca %struct.poly64x2x2_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.poly64x2x2_t, %struct.poly64x2x2_t* [[B]], i32 0, i32 0
// CHECK:   store [2 x <2 x i64>] [[B]].coerce, [2 x <2 x i64>]* [[COERCE_DIVE]], align 16
// CHECK:   [[TMP0:%.*]] = bitcast %struct.poly64x2x2_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.poly64x2x2_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP0]], i8* align 16 [[TMP1]], i64 32, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast %struct.poly64x2x2_t* [[__RET]] to i8*
// CHECK:   [[TMP3:%.*]] = bitcast i64* %a to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.poly64x2x2_t, %struct.poly64x2x2_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <2 x i64>], [2 x <2 x i64>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP4:%.*]] = load <2 x i64>, <2 x i64>* [[ARRAYIDX]], align 16
// CHECK:   [[TMP5:%.*]] = bitcast <2 x i64> [[TMP4]] to <16 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.poly64x2x2_t, %struct.poly64x2x2_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <2 x i64>], [2 x <2 x i64>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP6:%.*]] = load <2 x i64>, <2 x i64>* [[ARRAYIDX2]], align 16
// CHECK:   [[TMP7:%.*]] = bitcast <2 x i64> [[TMP6]] to <16 x i8>
// CHECK:   [[TMP8:%.*]] = bitcast <16 x i8> [[TMP5]] to <2 x i64>
// CHECK:   [[TMP9:%.*]] = bitcast <16 x i8> [[TMP7]] to <2 x i64>
// CHECK:   [[VLD2_LANE:%.*]] = call { <2 x i64>, <2 x i64> } @llvm.aarch64.neon.ld2lane.v2i64.p0i8(<2 x i64> [[TMP8]], <2 x i64> [[TMP9]], i64 1, i8* [[TMP3]])
// CHECK:   [[TMP10:%.*]] = bitcast i8* [[TMP2]] to { <2 x i64>, <2 x i64> }*
// CHECK:   store { <2 x i64>, <2 x i64> } [[VLD2_LANE]], { <2 x i64>, <2 x i64> }* [[TMP10]]
// CHECK:   [[TMP11:%.*]] = bitcast %struct.poly64x2x2_t* [[RETVAL]] to i8*
// CHECK:   [[TMP12:%.*]] = bitcast %struct.poly64x2x2_t* [[__RET]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP11]], i8* align 16 [[TMP12]], i64 32, i1 false)
// CHECK:   [[TMP13:%.*]] = load %struct.poly64x2x2_t, %struct.poly64x2x2_t* [[RETVAL]], align 16
// CHECK:   ret %struct.poly64x2x2_t [[TMP13]]
poly64x2x2_t test_vld2q_lane_p64(poly64_t  *a, poly64x2x2_t b) {
  return vld2q_lane_p64(a, b, 1);
}

// CHECK-LABEL: define{{.*}} %struct.uint8x8x2_t @test_vld2_lane_u8(i8* noundef %a, [2 x <8 x i8>] %b.coerce) #2 {
// CHECK:   [[RETVAL:%.*]] = alloca %struct.uint8x8x2_t, align 8
// CHECK:   [[B:%.*]] = alloca %struct.uint8x8x2_t, align 8
// CHECK:   [[__RET:%.*]] = alloca %struct.uint8x8x2_t, align 8
// CHECK:   [[__S1:%.*]] = alloca %struct.uint8x8x2_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.uint8x8x2_t, %struct.uint8x8x2_t* [[B]], i32 0, i32 0
// CHECK:   store [2 x <8 x i8>] [[B]].coerce, [2 x <8 x i8>]* [[COERCE_DIVE]], align 8
// CHECK:   [[TMP0:%.*]] = bitcast %struct.uint8x8x2_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.uint8x8x2_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[TMP0]], i8* align 8 [[TMP1]], i64 16, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast %struct.uint8x8x2_t* [[__RET]] to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.uint8x8x2_t, %struct.uint8x8x2_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <8 x i8>], [2 x <8 x i8>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <8 x i8>, <8 x i8>* [[ARRAYIDX]], align 8
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.uint8x8x2_t, %struct.uint8x8x2_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <8 x i8>], [2 x <8 x i8>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP4:%.*]] = load <8 x i8>, <8 x i8>* [[ARRAYIDX2]], align 8
// CHECK:   [[VLD2_LANE:%.*]] = call { <8 x i8>, <8 x i8> } @llvm.aarch64.neon.ld2lane.v8i8.p0i8(<8 x i8> [[TMP3]], <8 x i8> [[TMP4]], i64 7, i8* %a)
// CHECK:   [[TMP5:%.*]] = bitcast i8* [[TMP2]] to { <8 x i8>, <8 x i8> }*
// CHECK:   store { <8 x i8>, <8 x i8> } [[VLD2_LANE]], { <8 x i8>, <8 x i8> }* [[TMP5]]
// CHECK:   [[TMP6:%.*]] = bitcast %struct.uint8x8x2_t* [[RETVAL]] to i8*
// CHECK:   [[TMP7:%.*]] = bitcast %struct.uint8x8x2_t* [[__RET]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[TMP6]], i8* align 8 [[TMP7]], i64 16, i1 false)
// CHECK:   [[TMP8:%.*]] = load %struct.uint8x8x2_t, %struct.uint8x8x2_t* [[RETVAL]], align 8
// CHECK:   ret %struct.uint8x8x2_t [[TMP8]]
uint8x8x2_t test_vld2_lane_u8(uint8_t  *a, uint8x8x2_t b) {
  return vld2_lane_u8(a, b, 7);
}

// CHECK-LABEL: define{{.*}} %struct.uint16x4x2_t @test_vld2_lane_u16(i16* noundef %a, [2 x <4 x i16>] %b.coerce) #2 {
// CHECK:   [[RETVAL:%.*]] = alloca %struct.uint16x4x2_t, align 8
// CHECK:   [[B:%.*]] = alloca %struct.uint16x4x2_t, align 8
// CHECK:   [[__RET:%.*]] = alloca %struct.uint16x4x2_t, align 8
// CHECK:   [[__S1:%.*]] = alloca %struct.uint16x4x2_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.uint16x4x2_t, %struct.uint16x4x2_t* [[B]], i32 0, i32 0
// CHECK:   store [2 x <4 x i16>] [[B]].coerce, [2 x <4 x i16>]* [[COERCE_DIVE]], align 8
// CHECK:   [[TMP0:%.*]] = bitcast %struct.uint16x4x2_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.uint16x4x2_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[TMP0]], i8* align 8 [[TMP1]], i64 16, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast %struct.uint16x4x2_t* [[__RET]] to i8*
// CHECK:   [[TMP3:%.*]] = bitcast i16* %a to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.uint16x4x2_t, %struct.uint16x4x2_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <4 x i16>], [2 x <4 x i16>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP4:%.*]] = load <4 x i16>, <4 x i16>* [[ARRAYIDX]], align 8
// CHECK:   [[TMP5:%.*]] = bitcast <4 x i16> [[TMP4]] to <8 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.uint16x4x2_t, %struct.uint16x4x2_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <4 x i16>], [2 x <4 x i16>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP6:%.*]] = load <4 x i16>, <4 x i16>* [[ARRAYIDX2]], align 8
// CHECK:   [[TMP7:%.*]] = bitcast <4 x i16> [[TMP6]] to <8 x i8>
// CHECK:   [[TMP8:%.*]] = bitcast <8 x i8> [[TMP5]] to <4 x i16>
// CHECK:   [[TMP9:%.*]] = bitcast <8 x i8> [[TMP7]] to <4 x i16>
// CHECK:   [[VLD2_LANE:%.*]] = call { <4 x i16>, <4 x i16> } @llvm.aarch64.neon.ld2lane.v4i16.p0i8(<4 x i16> [[TMP8]], <4 x i16> [[TMP9]], i64 3, i8* [[TMP3]])
// CHECK:   [[TMP10:%.*]] = bitcast i8* [[TMP2]] to { <4 x i16>, <4 x i16> }*
// CHECK:   store { <4 x i16>, <4 x i16> } [[VLD2_LANE]], { <4 x i16>, <4 x i16> }* [[TMP10]]
// CHECK:   [[TMP11:%.*]] = bitcast %struct.uint16x4x2_t* [[RETVAL]] to i8*
// CHECK:   [[TMP12:%.*]] = bitcast %struct.uint16x4x2_t* [[__RET]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[TMP11]], i8* align 8 [[TMP12]], i64 16, i1 false)
// CHECK:   [[TMP13:%.*]] = load %struct.uint16x4x2_t, %struct.uint16x4x2_t* [[RETVAL]], align 8
// CHECK:   ret %struct.uint16x4x2_t [[TMP13]]
uint16x4x2_t test_vld2_lane_u16(uint16_t  *a, uint16x4x2_t b) {
  return vld2_lane_u16(a, b, 3);
}

// CHECK-LABEL: define{{.*}} %struct.uint32x2x2_t @test_vld2_lane_u32(i32* noundef %a, [2 x <2 x i32>] %b.coerce) #2 {
// CHECK:   [[RETVAL:%.*]] = alloca %struct.uint32x2x2_t, align 8
// CHECK:   [[B:%.*]] = alloca %struct.uint32x2x2_t, align 8
// CHECK:   [[__RET:%.*]] = alloca %struct.uint32x2x2_t, align 8
// CHECK:   [[__S1:%.*]] = alloca %struct.uint32x2x2_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.uint32x2x2_t, %struct.uint32x2x2_t* [[B]], i32 0, i32 0
// CHECK:   store [2 x <2 x i32>] [[B]].coerce, [2 x <2 x i32>]* [[COERCE_DIVE]], align 8
// CHECK:   [[TMP0:%.*]] = bitcast %struct.uint32x2x2_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.uint32x2x2_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[TMP0]], i8* align 8 [[TMP1]], i64 16, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast %struct.uint32x2x2_t* [[__RET]] to i8*
// CHECK:   [[TMP3:%.*]] = bitcast i32* %a to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.uint32x2x2_t, %struct.uint32x2x2_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <2 x i32>], [2 x <2 x i32>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP4:%.*]] = load <2 x i32>, <2 x i32>* [[ARRAYIDX]], align 8
// CHECK:   [[TMP5:%.*]] = bitcast <2 x i32> [[TMP4]] to <8 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.uint32x2x2_t, %struct.uint32x2x2_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <2 x i32>], [2 x <2 x i32>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP6:%.*]] = load <2 x i32>, <2 x i32>* [[ARRAYIDX2]], align 8
// CHECK:   [[TMP7:%.*]] = bitcast <2 x i32> [[TMP6]] to <8 x i8>
// CHECK:   [[TMP8:%.*]] = bitcast <8 x i8> [[TMP5]] to <2 x i32>
// CHECK:   [[TMP9:%.*]] = bitcast <8 x i8> [[TMP7]] to <2 x i32>
// CHECK:   [[VLD2_LANE:%.*]] = call { <2 x i32>, <2 x i32> } @llvm.aarch64.neon.ld2lane.v2i32.p0i8(<2 x i32> [[TMP8]], <2 x i32> [[TMP9]], i64 1, i8* [[TMP3]])
// CHECK:   [[TMP10:%.*]] = bitcast i8* [[TMP2]] to { <2 x i32>, <2 x i32> }*
// CHECK:   store { <2 x i32>, <2 x i32> } [[VLD2_LANE]], { <2 x i32>, <2 x i32> }* [[TMP10]]
// CHECK:   [[TMP11:%.*]] = bitcast %struct.uint32x2x2_t* [[RETVAL]] to i8*
// CHECK:   [[TMP12:%.*]] = bitcast %struct.uint32x2x2_t* [[__RET]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[TMP11]], i8* align 8 [[TMP12]], i64 16, i1 false)
// CHECK:   [[TMP13:%.*]] = load %struct.uint32x2x2_t, %struct.uint32x2x2_t* [[RETVAL]], align 8
// CHECK:   ret %struct.uint32x2x2_t [[TMP13]]
uint32x2x2_t test_vld2_lane_u32(uint32_t  *a, uint32x2x2_t b) {
  return vld2_lane_u32(a, b, 1);
}

// CHECK-LABEL: define{{.*}} %struct.uint64x1x2_t @test_vld2_lane_u64(i64* noundef %a, [2 x <1 x i64>] %b.coerce) #2 {
// CHECK:   [[RETVAL:%.*]] = alloca %struct.uint64x1x2_t, align 8
// CHECK:   [[B:%.*]] = alloca %struct.uint64x1x2_t, align 8
// CHECK:   [[__RET:%.*]] = alloca %struct.uint64x1x2_t, align 8
// CHECK:   [[__S1:%.*]] = alloca %struct.uint64x1x2_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.uint64x1x2_t, %struct.uint64x1x2_t* [[B]], i32 0, i32 0
// CHECK:   store [2 x <1 x i64>] [[B]].coerce, [2 x <1 x i64>]* [[COERCE_DIVE]], align 8
// CHECK:   [[TMP0:%.*]] = bitcast %struct.uint64x1x2_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.uint64x1x2_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[TMP0]], i8* align 8 [[TMP1]], i64 16, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast %struct.uint64x1x2_t* [[__RET]] to i8*
// CHECK:   [[TMP3:%.*]] = bitcast i64* %a to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.uint64x1x2_t, %struct.uint64x1x2_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <1 x i64>], [2 x <1 x i64>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP4:%.*]] = load <1 x i64>, <1 x i64>* [[ARRAYIDX]], align 8
// CHECK:   [[TMP5:%.*]] = bitcast <1 x i64> [[TMP4]] to <8 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.uint64x1x2_t, %struct.uint64x1x2_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <1 x i64>], [2 x <1 x i64>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP6:%.*]] = load <1 x i64>, <1 x i64>* [[ARRAYIDX2]], align 8
// CHECK:   [[TMP7:%.*]] = bitcast <1 x i64> [[TMP6]] to <8 x i8>
// CHECK:   [[TMP8:%.*]] = bitcast <8 x i8> [[TMP5]] to <1 x i64>
// CHECK:   [[TMP9:%.*]] = bitcast <8 x i8> [[TMP7]] to <1 x i64>
// CHECK:   [[VLD2_LANE:%.*]] = call { <1 x i64>, <1 x i64> } @llvm.aarch64.neon.ld2lane.v1i64.p0i8(<1 x i64> [[TMP8]], <1 x i64> [[TMP9]], i64 0, i8* [[TMP3]])
// CHECK:   [[TMP10:%.*]] = bitcast i8* [[TMP2]] to { <1 x i64>, <1 x i64> }*
// CHECK:   store { <1 x i64>, <1 x i64> } [[VLD2_LANE]], { <1 x i64>, <1 x i64> }* [[TMP10]]
// CHECK:   [[TMP11:%.*]] = bitcast %struct.uint64x1x2_t* [[RETVAL]] to i8*
// CHECK:   [[TMP12:%.*]] = bitcast %struct.uint64x1x2_t* [[__RET]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[TMP11]], i8* align 8 [[TMP12]], i64 16, i1 false)
// CHECK:   [[TMP13:%.*]] = load %struct.uint64x1x2_t, %struct.uint64x1x2_t* [[RETVAL]], align 8
// CHECK:   ret %struct.uint64x1x2_t [[TMP13]]
uint64x1x2_t test_vld2_lane_u64(uint64_t  *a, uint64x1x2_t b) {
  return vld2_lane_u64(a, b, 0);
}

// CHECK-LABEL: define{{.*}} %struct.int8x8x2_t @test_vld2_lane_s8(i8* noundef %a, [2 x <8 x i8>] %b.coerce) #2 {
// CHECK:   [[RETVAL:%.*]] = alloca %struct.int8x8x2_t, align 8
// CHECK:   [[B:%.*]] = alloca %struct.int8x8x2_t, align 8
// CHECK:   [[__RET:%.*]] = alloca %struct.int8x8x2_t, align 8
// CHECK:   [[__S1:%.*]] = alloca %struct.int8x8x2_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.int8x8x2_t, %struct.int8x8x2_t* [[B]], i32 0, i32 0
// CHECK:   store [2 x <8 x i8>] [[B]].coerce, [2 x <8 x i8>]* [[COERCE_DIVE]], align 8
// CHECK:   [[TMP0:%.*]] = bitcast %struct.int8x8x2_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.int8x8x2_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[TMP0]], i8* align 8 [[TMP1]], i64 16, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast %struct.int8x8x2_t* [[__RET]] to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.int8x8x2_t, %struct.int8x8x2_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <8 x i8>], [2 x <8 x i8>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <8 x i8>, <8 x i8>* [[ARRAYIDX]], align 8
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.int8x8x2_t, %struct.int8x8x2_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <8 x i8>], [2 x <8 x i8>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP4:%.*]] = load <8 x i8>, <8 x i8>* [[ARRAYIDX2]], align 8
// CHECK:   [[VLD2_LANE:%.*]] = call { <8 x i8>, <8 x i8> } @llvm.aarch64.neon.ld2lane.v8i8.p0i8(<8 x i8> [[TMP3]], <8 x i8> [[TMP4]], i64 7, i8* %a)
// CHECK:   [[TMP5:%.*]] = bitcast i8* [[TMP2]] to { <8 x i8>, <8 x i8> }*
// CHECK:   store { <8 x i8>, <8 x i8> } [[VLD2_LANE]], { <8 x i8>, <8 x i8> }* [[TMP5]]
// CHECK:   [[TMP6:%.*]] = bitcast %struct.int8x8x2_t* [[RETVAL]] to i8*
// CHECK:   [[TMP7:%.*]] = bitcast %struct.int8x8x2_t* [[__RET]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[TMP6]], i8* align 8 [[TMP7]], i64 16, i1 false)
// CHECK:   [[TMP8:%.*]] = load %struct.int8x8x2_t, %struct.int8x8x2_t* [[RETVAL]], align 8
// CHECK:   ret %struct.int8x8x2_t [[TMP8]]
int8x8x2_t test_vld2_lane_s8(int8_t  *a, int8x8x2_t b) {
  return vld2_lane_s8(a, b, 7);
}

// CHECK-LABEL: define{{.*}} %struct.int16x4x2_t @test_vld2_lane_s16(i16* noundef %a, [2 x <4 x i16>] %b.coerce) #2 {
// CHECK:   [[RETVAL:%.*]] = alloca %struct.int16x4x2_t, align 8
// CHECK:   [[B:%.*]] = alloca %struct.int16x4x2_t, align 8
// CHECK:   [[__RET:%.*]] = alloca %struct.int16x4x2_t, align 8
// CHECK:   [[__S1:%.*]] = alloca %struct.int16x4x2_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.int16x4x2_t, %struct.int16x4x2_t* [[B]], i32 0, i32 0
// CHECK:   store [2 x <4 x i16>] [[B]].coerce, [2 x <4 x i16>]* [[COERCE_DIVE]], align 8
// CHECK:   [[TMP0:%.*]] = bitcast %struct.int16x4x2_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.int16x4x2_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[TMP0]], i8* align 8 [[TMP1]], i64 16, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast %struct.int16x4x2_t* [[__RET]] to i8*
// CHECK:   [[TMP3:%.*]] = bitcast i16* %a to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.int16x4x2_t, %struct.int16x4x2_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <4 x i16>], [2 x <4 x i16>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP4:%.*]] = load <4 x i16>, <4 x i16>* [[ARRAYIDX]], align 8
// CHECK:   [[TMP5:%.*]] = bitcast <4 x i16> [[TMP4]] to <8 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.int16x4x2_t, %struct.int16x4x2_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <4 x i16>], [2 x <4 x i16>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP6:%.*]] = load <4 x i16>, <4 x i16>* [[ARRAYIDX2]], align 8
// CHECK:   [[TMP7:%.*]] = bitcast <4 x i16> [[TMP6]] to <8 x i8>
// CHECK:   [[TMP8:%.*]] = bitcast <8 x i8> [[TMP5]] to <4 x i16>
// CHECK:   [[TMP9:%.*]] = bitcast <8 x i8> [[TMP7]] to <4 x i16>
// CHECK:   [[VLD2_LANE:%.*]] = call { <4 x i16>, <4 x i16> } @llvm.aarch64.neon.ld2lane.v4i16.p0i8(<4 x i16> [[TMP8]], <4 x i16> [[TMP9]], i64 3, i8* [[TMP3]])
// CHECK:   [[TMP10:%.*]] = bitcast i8* [[TMP2]] to { <4 x i16>, <4 x i16> }*
// CHECK:   store { <4 x i16>, <4 x i16> } [[VLD2_LANE]], { <4 x i16>, <4 x i16> }* [[TMP10]]
// CHECK:   [[TMP11:%.*]] = bitcast %struct.int16x4x2_t* [[RETVAL]] to i8*
// CHECK:   [[TMP12:%.*]] = bitcast %struct.int16x4x2_t* [[__RET]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[TMP11]], i8* align 8 [[TMP12]], i64 16, i1 false)
// CHECK:   [[TMP13:%.*]] = load %struct.int16x4x2_t, %struct.int16x4x2_t* [[RETVAL]], align 8
// CHECK:   ret %struct.int16x4x2_t [[TMP13]]
int16x4x2_t test_vld2_lane_s16(int16_t  *a, int16x4x2_t b) {
  return vld2_lane_s16(a, b, 3);
}

// CHECK-LABEL: define{{.*}} %struct.int32x2x2_t @test_vld2_lane_s32(i32* noundef %a, [2 x <2 x i32>] %b.coerce) #2 {
// CHECK:   [[RETVAL:%.*]] = alloca %struct.int32x2x2_t, align 8
// CHECK:   [[B:%.*]] = alloca %struct.int32x2x2_t, align 8
// CHECK:   [[__RET:%.*]] = alloca %struct.int32x2x2_t, align 8
// CHECK:   [[__S1:%.*]] = alloca %struct.int32x2x2_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.int32x2x2_t, %struct.int32x2x2_t* [[B]], i32 0, i32 0
// CHECK:   store [2 x <2 x i32>] [[B]].coerce, [2 x <2 x i32>]* [[COERCE_DIVE]], align 8
// CHECK:   [[TMP0:%.*]] = bitcast %struct.int32x2x2_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.int32x2x2_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[TMP0]], i8* align 8 [[TMP1]], i64 16, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast %struct.int32x2x2_t* [[__RET]] to i8*
// CHECK:   [[TMP3:%.*]] = bitcast i32* %a to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.int32x2x2_t, %struct.int32x2x2_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <2 x i32>], [2 x <2 x i32>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP4:%.*]] = load <2 x i32>, <2 x i32>* [[ARRAYIDX]], align 8
// CHECK:   [[TMP5:%.*]] = bitcast <2 x i32> [[TMP4]] to <8 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.int32x2x2_t, %struct.int32x2x2_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <2 x i32>], [2 x <2 x i32>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP6:%.*]] = load <2 x i32>, <2 x i32>* [[ARRAYIDX2]], align 8
// CHECK:   [[TMP7:%.*]] = bitcast <2 x i32> [[TMP6]] to <8 x i8>
// CHECK:   [[TMP8:%.*]] = bitcast <8 x i8> [[TMP5]] to <2 x i32>
// CHECK:   [[TMP9:%.*]] = bitcast <8 x i8> [[TMP7]] to <2 x i32>
// CHECK:   [[VLD2_LANE:%.*]] = call { <2 x i32>, <2 x i32> } @llvm.aarch64.neon.ld2lane.v2i32.p0i8(<2 x i32> [[TMP8]], <2 x i32> [[TMP9]], i64 1, i8* [[TMP3]])
// CHECK:   [[TMP10:%.*]] = bitcast i8* [[TMP2]] to { <2 x i32>, <2 x i32> }*
// CHECK:   store { <2 x i32>, <2 x i32> } [[VLD2_LANE]], { <2 x i32>, <2 x i32> }* [[TMP10]]
// CHECK:   [[TMP11:%.*]] = bitcast %struct.int32x2x2_t* [[RETVAL]] to i8*
// CHECK:   [[TMP12:%.*]] = bitcast %struct.int32x2x2_t* [[__RET]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[TMP11]], i8* align 8 [[TMP12]], i64 16, i1 false)
// CHECK:   [[TMP13:%.*]] = load %struct.int32x2x2_t, %struct.int32x2x2_t* [[RETVAL]], align 8
// CHECK:   ret %struct.int32x2x2_t [[TMP13]]
int32x2x2_t test_vld2_lane_s32(int32_t  *a, int32x2x2_t b) {
  return vld2_lane_s32(a, b, 1);
}

// CHECK-LABEL: define{{.*}} %struct.int64x1x2_t @test_vld2_lane_s64(i64* noundef %a, [2 x <1 x i64>] %b.coerce) #2 {
// CHECK:   [[RETVAL:%.*]] = alloca %struct.int64x1x2_t, align 8
// CHECK:   [[B:%.*]] = alloca %struct.int64x1x2_t, align 8
// CHECK:   [[__RET:%.*]] = alloca %struct.int64x1x2_t, align 8
// CHECK:   [[__S1:%.*]] = alloca %struct.int64x1x2_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.int64x1x2_t, %struct.int64x1x2_t* [[B]], i32 0, i32 0
// CHECK:   store [2 x <1 x i64>] [[B]].coerce, [2 x <1 x i64>]* [[COERCE_DIVE]], align 8
// CHECK:   [[TMP0:%.*]] = bitcast %struct.int64x1x2_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.int64x1x2_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[TMP0]], i8* align 8 [[TMP1]], i64 16, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast %struct.int64x1x2_t* [[__RET]] to i8*
// CHECK:   [[TMP3:%.*]] = bitcast i64* %a to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.int64x1x2_t, %struct.int64x1x2_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <1 x i64>], [2 x <1 x i64>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP4:%.*]] = load <1 x i64>, <1 x i64>* [[ARRAYIDX]], align 8
// CHECK:   [[TMP5:%.*]] = bitcast <1 x i64> [[TMP4]] to <8 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.int64x1x2_t, %struct.int64x1x2_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <1 x i64>], [2 x <1 x i64>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP6:%.*]] = load <1 x i64>, <1 x i64>* [[ARRAYIDX2]], align 8
// CHECK:   [[TMP7:%.*]] = bitcast <1 x i64> [[TMP6]] to <8 x i8>
// CHECK:   [[TMP8:%.*]] = bitcast <8 x i8> [[TMP5]] to <1 x i64>
// CHECK:   [[TMP9:%.*]] = bitcast <8 x i8> [[TMP7]] to <1 x i64>
// CHECK:   [[VLD2_LANE:%.*]] = call { <1 x i64>, <1 x i64> } @llvm.aarch64.neon.ld2lane.v1i64.p0i8(<1 x i64> [[TMP8]], <1 x i64> [[TMP9]], i64 0, i8* [[TMP3]])
// CHECK:   [[TMP10:%.*]] = bitcast i8* [[TMP2]] to { <1 x i64>, <1 x i64> }*
// CHECK:   store { <1 x i64>, <1 x i64> } [[VLD2_LANE]], { <1 x i64>, <1 x i64> }* [[TMP10]]
// CHECK:   [[TMP11:%.*]] = bitcast %struct.int64x1x2_t* [[RETVAL]] to i8*
// CHECK:   [[TMP12:%.*]] = bitcast %struct.int64x1x2_t* [[__RET]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[TMP11]], i8* align 8 [[TMP12]], i64 16, i1 false)
// CHECK:   [[TMP13:%.*]] = load %struct.int64x1x2_t, %struct.int64x1x2_t* [[RETVAL]], align 8
// CHECK:   ret %struct.int64x1x2_t [[TMP13]]
int64x1x2_t test_vld2_lane_s64(int64_t  *a, int64x1x2_t b) {
  return vld2_lane_s64(a, b, 0);
}

// CHECK-LABEL: define{{.*}} %struct.float16x4x2_t @test_vld2_lane_f16(half* noundef %a, [2 x <4 x half>] %b.coerce) #2 {
// CHECK:   [[RETVAL:%.*]] = alloca %struct.float16x4x2_t, align 8
// CHECK:   [[B:%.*]] = alloca %struct.float16x4x2_t, align 8
// CHECK:   [[__RET:%.*]] = alloca %struct.float16x4x2_t, align 8
// CHECK:   [[__S1:%.*]] = alloca %struct.float16x4x2_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.float16x4x2_t, %struct.float16x4x2_t* [[B]], i32 0, i32 0
// CHECK:   store [2 x <4 x half>] [[B]].coerce, [2 x <4 x half>]* [[COERCE_DIVE]], align 8
// CHECK:   [[TMP0:%.*]] = bitcast %struct.float16x4x2_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.float16x4x2_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[TMP0]], i8* align 8 [[TMP1]], i64 16, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast %struct.float16x4x2_t* [[__RET]] to i8*
// CHECK:   [[TMP3:%.*]] = bitcast half* %a to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.float16x4x2_t, %struct.float16x4x2_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <4 x half>], [2 x <4 x half>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP4:%.*]] = load <4 x half>, <4 x half>* [[ARRAYIDX]], align 8
// CHECK:   [[TMP5:%.*]] = bitcast <4 x half> [[TMP4]] to <8 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.float16x4x2_t, %struct.float16x4x2_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <4 x half>], [2 x <4 x half>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP6:%.*]] = load <4 x half>, <4 x half>* [[ARRAYIDX2]], align 8
// CHECK:   [[TMP7:%.*]] = bitcast <4 x half> [[TMP6]] to <8 x i8>
// CHECK:   [[TMP8:%.*]] = bitcast <8 x i8> [[TMP5]] to <4 x half>
// CHECK:   [[TMP9:%.*]] = bitcast <8 x i8> [[TMP7]] to <4 x half>
// CHECK:   [[VLD2_LANE:%.*]] = call { <4 x half>, <4 x half> } @llvm.aarch64.neon.ld2lane.v4f16.p0i8(<4 x half> [[TMP8]], <4 x half> [[TMP9]], i64 3, i8* [[TMP3]])
// CHECK:   [[TMP10:%.*]] = bitcast i8* [[TMP2]] to { <4 x half>, <4 x half> }*
// CHECK:   store { <4 x half>, <4 x half> } [[VLD2_LANE]], { <4 x half>, <4 x half> }* [[TMP10]]
// CHECK:   [[TMP11:%.*]] = bitcast %struct.float16x4x2_t* [[RETVAL]] to i8*
// CHECK:   [[TMP12:%.*]] = bitcast %struct.float16x4x2_t* [[__RET]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[TMP11]], i8* align 8 [[TMP12]], i64 16, i1 false)
// CHECK:   [[TMP13:%.*]] = load %struct.float16x4x2_t, %struct.float16x4x2_t* [[RETVAL]], align 8
// CHECK:   ret %struct.float16x4x2_t [[TMP13]]
float16x4x2_t test_vld2_lane_f16(float16_t  *a, float16x4x2_t b) {
  return vld2_lane_f16(a, b, 3);
}

// CHECK-LABEL: define{{.*}} %struct.float32x2x2_t @test_vld2_lane_f32(float* noundef %a, [2 x <2 x float>] %b.coerce) #2 {
// CHECK:   [[RETVAL:%.*]] = alloca %struct.float32x2x2_t, align 8
// CHECK:   [[B:%.*]] = alloca %struct.float32x2x2_t, align 8
// CHECK:   [[__RET:%.*]] = alloca %struct.float32x2x2_t, align 8
// CHECK:   [[__S1:%.*]] = alloca %struct.float32x2x2_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.float32x2x2_t, %struct.float32x2x2_t* [[B]], i32 0, i32 0
// CHECK:   store [2 x <2 x float>] [[B]].coerce, [2 x <2 x float>]* [[COERCE_DIVE]], align 8
// CHECK:   [[TMP0:%.*]] = bitcast %struct.float32x2x2_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.float32x2x2_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[TMP0]], i8* align 8 [[TMP1]], i64 16, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast %struct.float32x2x2_t* [[__RET]] to i8*
// CHECK:   [[TMP3:%.*]] = bitcast float* %a to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.float32x2x2_t, %struct.float32x2x2_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <2 x float>], [2 x <2 x float>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP4:%.*]] = load <2 x float>, <2 x float>* [[ARRAYIDX]], align 8
// CHECK:   [[TMP5:%.*]] = bitcast <2 x float> [[TMP4]] to <8 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.float32x2x2_t, %struct.float32x2x2_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <2 x float>], [2 x <2 x float>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP6:%.*]] = load <2 x float>, <2 x float>* [[ARRAYIDX2]], align 8
// CHECK:   [[TMP7:%.*]] = bitcast <2 x float> [[TMP6]] to <8 x i8>
// CHECK:   [[TMP8:%.*]] = bitcast <8 x i8> [[TMP5]] to <2 x float>
// CHECK:   [[TMP9:%.*]] = bitcast <8 x i8> [[TMP7]] to <2 x float>
// CHECK:   [[VLD2_LANE:%.*]] = call { <2 x float>, <2 x float> } @llvm.aarch64.neon.ld2lane.v2f32.p0i8(<2 x float> [[TMP8]], <2 x float> [[TMP9]], i64 1, i8* [[TMP3]])
// CHECK:   [[TMP10:%.*]] = bitcast i8* [[TMP2]] to { <2 x float>, <2 x float> }*
// CHECK:   store { <2 x float>, <2 x float> } [[VLD2_LANE]], { <2 x float>, <2 x float> }* [[TMP10]]
// CHECK:   [[TMP11:%.*]] = bitcast %struct.float32x2x2_t* [[RETVAL]] to i8*
// CHECK:   [[TMP12:%.*]] = bitcast %struct.float32x2x2_t* [[__RET]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[TMP11]], i8* align 8 [[TMP12]], i64 16, i1 false)
// CHECK:   [[TMP13:%.*]] = load %struct.float32x2x2_t, %struct.float32x2x2_t* [[RETVAL]], align 8
// CHECK:   ret %struct.float32x2x2_t [[TMP13]]
float32x2x2_t test_vld2_lane_f32(float32_t  *a, float32x2x2_t b) {
  return vld2_lane_f32(a, b, 1);
}

// CHECK-LABEL: define{{.*}} %struct.float64x1x2_t @test_vld2_lane_f64(double* noundef %a, [2 x <1 x double>] %b.coerce) #2 {
// CHECK:   [[RETVAL:%.*]] = alloca %struct.float64x1x2_t, align 8
// CHECK:   [[B:%.*]] = alloca %struct.float64x1x2_t, align 8
// CHECK:   [[__RET:%.*]] = alloca %struct.float64x1x2_t, align 8
// CHECK:   [[__S1:%.*]] = alloca %struct.float64x1x2_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.float64x1x2_t, %struct.float64x1x2_t* [[B]], i32 0, i32 0
// CHECK:   store [2 x <1 x double>] [[B]].coerce, [2 x <1 x double>]* [[COERCE_DIVE]], align 8
// CHECK:   [[TMP0:%.*]] = bitcast %struct.float64x1x2_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.float64x1x2_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[TMP0]], i8* align 8 [[TMP1]], i64 16, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast %struct.float64x1x2_t* [[__RET]] to i8*
// CHECK:   [[TMP3:%.*]] = bitcast double* %a to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.float64x1x2_t, %struct.float64x1x2_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <1 x double>], [2 x <1 x double>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP4:%.*]] = load <1 x double>, <1 x double>* [[ARRAYIDX]], align 8
// CHECK:   [[TMP5:%.*]] = bitcast <1 x double> [[TMP4]] to <8 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.float64x1x2_t, %struct.float64x1x2_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <1 x double>], [2 x <1 x double>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP6:%.*]] = load <1 x double>, <1 x double>* [[ARRAYIDX2]], align 8
// CHECK:   [[TMP7:%.*]] = bitcast <1 x double> [[TMP6]] to <8 x i8>
// CHECK:   [[TMP8:%.*]] = bitcast <8 x i8> [[TMP5]] to <1 x double>
// CHECK:   [[TMP9:%.*]] = bitcast <8 x i8> [[TMP7]] to <1 x double>
// CHECK:   [[VLD2_LANE:%.*]] = call { <1 x double>, <1 x double> } @llvm.aarch64.neon.ld2lane.v1f64.p0i8(<1 x double> [[TMP8]], <1 x double> [[TMP9]], i64 0, i8* [[TMP3]])
// CHECK:   [[TMP10:%.*]] = bitcast i8* [[TMP2]] to { <1 x double>, <1 x double> }*
// CHECK:   store { <1 x double>, <1 x double> } [[VLD2_LANE]], { <1 x double>, <1 x double> }* [[TMP10]]
// CHECK:   [[TMP11:%.*]] = bitcast %struct.float64x1x2_t* [[RETVAL]] to i8*
// CHECK:   [[TMP12:%.*]] = bitcast %struct.float64x1x2_t* [[__RET]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[TMP11]], i8* align 8 [[TMP12]], i64 16, i1 false)
// CHECK:   [[TMP13:%.*]] = load %struct.float64x1x2_t, %struct.float64x1x2_t* [[RETVAL]], align 8
// CHECK:   ret %struct.float64x1x2_t [[TMP13]]
float64x1x2_t test_vld2_lane_f64(float64_t  *a, float64x1x2_t b) {
  return vld2_lane_f64(a, b, 0);
}

// CHECK-LABEL: define{{.*}} %struct.poly8x8x2_t @test_vld2_lane_p8(i8* noundef %a, [2 x <8 x i8>] %b.coerce) #2 {
// CHECK:   [[RETVAL:%.*]] = alloca %struct.poly8x8x2_t, align 8
// CHECK:   [[B:%.*]] = alloca %struct.poly8x8x2_t, align 8
// CHECK:   [[__RET:%.*]] = alloca %struct.poly8x8x2_t, align 8
// CHECK:   [[__S1:%.*]] = alloca %struct.poly8x8x2_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.poly8x8x2_t, %struct.poly8x8x2_t* [[B]], i32 0, i32 0
// CHECK:   store [2 x <8 x i8>] [[B]].coerce, [2 x <8 x i8>]* [[COERCE_DIVE]], align 8
// CHECK:   [[TMP0:%.*]] = bitcast %struct.poly8x8x2_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.poly8x8x2_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[TMP0]], i8* align 8 [[TMP1]], i64 16, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast %struct.poly8x8x2_t* [[__RET]] to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.poly8x8x2_t, %struct.poly8x8x2_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <8 x i8>], [2 x <8 x i8>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <8 x i8>, <8 x i8>* [[ARRAYIDX]], align 8
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.poly8x8x2_t, %struct.poly8x8x2_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <8 x i8>], [2 x <8 x i8>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP4:%.*]] = load <8 x i8>, <8 x i8>* [[ARRAYIDX2]], align 8
// CHECK:   [[VLD2_LANE:%.*]] = call { <8 x i8>, <8 x i8> } @llvm.aarch64.neon.ld2lane.v8i8.p0i8(<8 x i8> [[TMP3]], <8 x i8> [[TMP4]], i64 7, i8* %a)
// CHECK:   [[TMP5:%.*]] = bitcast i8* [[TMP2]] to { <8 x i8>, <8 x i8> }*
// CHECK:   store { <8 x i8>, <8 x i8> } [[VLD2_LANE]], { <8 x i8>, <8 x i8> }* [[TMP5]]
// CHECK:   [[TMP6:%.*]] = bitcast %struct.poly8x8x2_t* [[RETVAL]] to i8*
// CHECK:   [[TMP7:%.*]] = bitcast %struct.poly8x8x2_t* [[__RET]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[TMP6]], i8* align 8 [[TMP7]], i64 16, i1 false)
// CHECK:   [[TMP8:%.*]] = load %struct.poly8x8x2_t, %struct.poly8x8x2_t* [[RETVAL]], align 8
// CHECK:   ret %struct.poly8x8x2_t [[TMP8]]
poly8x8x2_t test_vld2_lane_p8(poly8_t  *a, poly8x8x2_t b) {
  return vld2_lane_p8(a, b, 7);
}

// CHECK-LABEL: define{{.*}} %struct.poly16x4x2_t @test_vld2_lane_p16(i16* noundef %a, [2 x <4 x i16>] %b.coerce) #2 {
// CHECK:   [[RETVAL:%.*]] = alloca %struct.poly16x4x2_t, align 8
// CHECK:   [[B:%.*]] = alloca %struct.poly16x4x2_t, align 8
// CHECK:   [[__RET:%.*]] = alloca %struct.poly16x4x2_t, align 8
// CHECK:   [[__S1:%.*]] = alloca %struct.poly16x4x2_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.poly16x4x2_t, %struct.poly16x4x2_t* [[B]], i32 0, i32 0
// CHECK:   store [2 x <4 x i16>] [[B]].coerce, [2 x <4 x i16>]* [[COERCE_DIVE]], align 8
// CHECK:   [[TMP0:%.*]] = bitcast %struct.poly16x4x2_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.poly16x4x2_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[TMP0]], i8* align 8 [[TMP1]], i64 16, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast %struct.poly16x4x2_t* [[__RET]] to i8*
// CHECK:   [[TMP3:%.*]] = bitcast i16* %a to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.poly16x4x2_t, %struct.poly16x4x2_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <4 x i16>], [2 x <4 x i16>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP4:%.*]] = load <4 x i16>, <4 x i16>* [[ARRAYIDX]], align 8
// CHECK:   [[TMP5:%.*]] = bitcast <4 x i16> [[TMP4]] to <8 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.poly16x4x2_t, %struct.poly16x4x2_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <4 x i16>], [2 x <4 x i16>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP6:%.*]] = load <4 x i16>, <4 x i16>* [[ARRAYIDX2]], align 8
// CHECK:   [[TMP7:%.*]] = bitcast <4 x i16> [[TMP6]] to <8 x i8>
// CHECK:   [[TMP8:%.*]] = bitcast <8 x i8> [[TMP5]] to <4 x i16>
// CHECK:   [[TMP9:%.*]] = bitcast <8 x i8> [[TMP7]] to <4 x i16>
// CHECK:   [[VLD2_LANE:%.*]] = call { <4 x i16>, <4 x i16> } @llvm.aarch64.neon.ld2lane.v4i16.p0i8(<4 x i16> [[TMP8]], <4 x i16> [[TMP9]], i64 3, i8* [[TMP3]])
// CHECK:   [[TMP10:%.*]] = bitcast i8* [[TMP2]] to { <4 x i16>, <4 x i16> }*
// CHECK:   store { <4 x i16>, <4 x i16> } [[VLD2_LANE]], { <4 x i16>, <4 x i16> }* [[TMP10]]
// CHECK:   [[TMP11:%.*]] = bitcast %struct.poly16x4x2_t* [[RETVAL]] to i8*
// CHECK:   [[TMP12:%.*]] = bitcast %struct.poly16x4x2_t* [[__RET]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[TMP11]], i8* align 8 [[TMP12]], i64 16, i1 false)
// CHECK:   [[TMP13:%.*]] = load %struct.poly16x4x2_t, %struct.poly16x4x2_t* [[RETVAL]], align 8
// CHECK:   ret %struct.poly16x4x2_t [[TMP13]]
poly16x4x2_t test_vld2_lane_p16(poly16_t  *a, poly16x4x2_t b) {
  return vld2_lane_p16(a, b, 3);
}

// CHECK-LABEL: define{{.*}} %struct.poly64x1x2_t @test_vld2_lane_p64(i64* noundef %a, [2 x <1 x i64>] %b.coerce) #2 {
// CHECK:   [[RETVAL:%.*]] = alloca %struct.poly64x1x2_t, align 8
// CHECK:   [[B:%.*]] = alloca %struct.poly64x1x2_t, align 8
// CHECK:   [[__RET:%.*]] = alloca %struct.poly64x1x2_t, align 8
// CHECK:   [[__S1:%.*]] = alloca %struct.poly64x1x2_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.poly64x1x2_t, %struct.poly64x1x2_t* [[B]], i32 0, i32 0
// CHECK:   store [2 x <1 x i64>] [[B]].coerce, [2 x <1 x i64>]* [[COERCE_DIVE]], align 8
// CHECK:   [[TMP0:%.*]] = bitcast %struct.poly64x1x2_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.poly64x1x2_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[TMP0]], i8* align 8 [[TMP1]], i64 16, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast %struct.poly64x1x2_t* [[__RET]] to i8*
// CHECK:   [[TMP3:%.*]] = bitcast i64* %a to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.poly64x1x2_t, %struct.poly64x1x2_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <1 x i64>], [2 x <1 x i64>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP4:%.*]] = load <1 x i64>, <1 x i64>* [[ARRAYIDX]], align 8
// CHECK:   [[TMP5:%.*]] = bitcast <1 x i64> [[TMP4]] to <8 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.poly64x1x2_t, %struct.poly64x1x2_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <1 x i64>], [2 x <1 x i64>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP6:%.*]] = load <1 x i64>, <1 x i64>* [[ARRAYIDX2]], align 8
// CHECK:   [[TMP7:%.*]] = bitcast <1 x i64> [[TMP6]] to <8 x i8>
// CHECK:   [[TMP8:%.*]] = bitcast <8 x i8> [[TMP5]] to <1 x i64>
// CHECK:   [[TMP9:%.*]] = bitcast <8 x i8> [[TMP7]] to <1 x i64>
// CHECK:   [[VLD2_LANE:%.*]] = call { <1 x i64>, <1 x i64> } @llvm.aarch64.neon.ld2lane.v1i64.p0i8(<1 x i64> [[TMP8]], <1 x i64> [[TMP9]], i64 0, i8* [[TMP3]])
// CHECK:   [[TMP10:%.*]] = bitcast i8* [[TMP2]] to { <1 x i64>, <1 x i64> }*
// CHECK:   store { <1 x i64>, <1 x i64> } [[VLD2_LANE]], { <1 x i64>, <1 x i64> }* [[TMP10]]
// CHECK:   [[TMP11:%.*]] = bitcast %struct.poly64x1x2_t* [[RETVAL]] to i8*
// CHECK:   [[TMP12:%.*]] = bitcast %struct.poly64x1x2_t* [[__RET]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[TMP11]], i8* align 8 [[TMP12]], i64 16, i1 false)
// CHECK:   [[TMP13:%.*]] = load %struct.poly64x1x2_t, %struct.poly64x1x2_t* [[RETVAL]], align 8
// CHECK:   ret %struct.poly64x1x2_t [[TMP13]]
poly64x1x2_t test_vld2_lane_p64(poly64_t  *a, poly64x1x2_t b) {
  return vld2_lane_p64(a, b, 0);
}

// CHECK-LABEL: define{{.*}} %struct.uint16x8x3_t @test_vld3q_lane_u16(i16* noundef %a, [3 x <8 x i16>] %b.coerce) #2 {
// CHECK:   [[RETVAL:%.*]] = alloca %struct.uint16x8x3_t, align 16
// CHECK:   [[B:%.*]] = alloca %struct.uint16x8x3_t, align 16
// CHECK:   [[__RET:%.*]] = alloca %struct.uint16x8x3_t, align 16
// CHECK:   [[__S1:%.*]] = alloca %struct.uint16x8x3_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.uint16x8x3_t, %struct.uint16x8x3_t* [[B]], i32 0, i32 0
// CHECK:   store [3 x <8 x i16>] [[B]].coerce, [3 x <8 x i16>]* [[COERCE_DIVE]], align 16
// CHECK:   [[TMP0:%.*]] = bitcast %struct.uint16x8x3_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.uint16x8x3_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP0]], i8* align 16 [[TMP1]], i64 48, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast %struct.uint16x8x3_t* [[__RET]] to i8*
// CHECK:   [[TMP3:%.*]] = bitcast i16* %a to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.uint16x8x3_t, %struct.uint16x8x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <8 x i16>], [3 x <8 x i16>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP4:%.*]] = load <8 x i16>, <8 x i16>* [[ARRAYIDX]], align 16
// CHECK:   [[TMP5:%.*]] = bitcast <8 x i16> [[TMP4]] to <16 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.uint16x8x3_t, %struct.uint16x8x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <8 x i16>], [3 x <8 x i16>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP6:%.*]] = load <8 x i16>, <8 x i16>* [[ARRAYIDX2]], align 16
// CHECK:   [[TMP7:%.*]] = bitcast <8 x i16> [[TMP6]] to <16 x i8>
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds %struct.uint16x8x3_t, %struct.uint16x8x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <8 x i16>], [3 x <8 x i16>]* [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP8:%.*]] = load <8 x i16>, <8 x i16>* [[ARRAYIDX4]], align 16
// CHECK:   [[TMP9:%.*]] = bitcast <8 x i16> [[TMP8]] to <16 x i8>
// CHECK:   [[TMP10:%.*]] = bitcast <16 x i8> [[TMP5]] to <8 x i16>
// CHECK:   [[TMP11:%.*]] = bitcast <16 x i8> [[TMP7]] to <8 x i16>
// CHECK:   [[TMP12:%.*]] = bitcast <16 x i8> [[TMP9]] to <8 x i16>
// CHECK:   [[VLD3_LANE:%.*]] = call { <8 x i16>, <8 x i16>, <8 x i16> } @llvm.aarch64.neon.ld3lane.v8i16.p0i8(<8 x i16> [[TMP10]], <8 x i16> [[TMP11]], <8 x i16> [[TMP12]], i64 7, i8* [[TMP3]])
// CHECK:   [[TMP13:%.*]] = bitcast i8* [[TMP2]] to { <8 x i16>, <8 x i16>, <8 x i16> }*
// CHECK:   store { <8 x i16>, <8 x i16>, <8 x i16> } [[VLD3_LANE]], { <8 x i16>, <8 x i16>, <8 x i16> }* [[TMP13]]
// CHECK:   [[TMP14:%.*]] = bitcast %struct.uint16x8x3_t* [[RETVAL]] to i8*
// CHECK:   [[TMP15:%.*]] = bitcast %struct.uint16x8x3_t* [[__RET]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP14]], i8* align 16 [[TMP15]], i64 48, i1 false)
// CHECK:   [[TMP16:%.*]] = load %struct.uint16x8x3_t, %struct.uint16x8x3_t* [[RETVAL]], align 16
// CHECK:   ret %struct.uint16x8x3_t [[TMP16]]
uint16x8x3_t test_vld3q_lane_u16(uint16_t  *a, uint16x8x3_t b) {
  return vld3q_lane_u16(a, b, 7);
}

// CHECK-LABEL: define{{.*}} %struct.uint32x4x3_t @test_vld3q_lane_u32(i32* noundef %a, [3 x <4 x i32>] %b.coerce) #2 {
// CHECK:   [[RETVAL:%.*]] = alloca %struct.uint32x4x3_t, align 16
// CHECK:   [[B:%.*]] = alloca %struct.uint32x4x3_t, align 16
// CHECK:   [[__RET:%.*]] = alloca %struct.uint32x4x3_t, align 16
// CHECK:   [[__S1:%.*]] = alloca %struct.uint32x4x3_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.uint32x4x3_t, %struct.uint32x4x3_t* [[B]], i32 0, i32 0
// CHECK:   store [3 x <4 x i32>] [[B]].coerce, [3 x <4 x i32>]* [[COERCE_DIVE]], align 16
// CHECK:   [[TMP0:%.*]] = bitcast %struct.uint32x4x3_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.uint32x4x3_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP0]], i8* align 16 [[TMP1]], i64 48, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast %struct.uint32x4x3_t* [[__RET]] to i8*
// CHECK:   [[TMP3:%.*]] = bitcast i32* %a to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.uint32x4x3_t, %struct.uint32x4x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <4 x i32>], [3 x <4 x i32>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP4:%.*]] = load <4 x i32>, <4 x i32>* [[ARRAYIDX]], align 16
// CHECK:   [[TMP5:%.*]] = bitcast <4 x i32> [[TMP4]] to <16 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.uint32x4x3_t, %struct.uint32x4x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <4 x i32>], [3 x <4 x i32>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP6:%.*]] = load <4 x i32>, <4 x i32>* [[ARRAYIDX2]], align 16
// CHECK:   [[TMP7:%.*]] = bitcast <4 x i32> [[TMP6]] to <16 x i8>
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds %struct.uint32x4x3_t, %struct.uint32x4x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <4 x i32>], [3 x <4 x i32>]* [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP8:%.*]] = load <4 x i32>, <4 x i32>* [[ARRAYIDX4]], align 16
// CHECK:   [[TMP9:%.*]] = bitcast <4 x i32> [[TMP8]] to <16 x i8>
// CHECK:   [[TMP10:%.*]] = bitcast <16 x i8> [[TMP5]] to <4 x i32>
// CHECK:   [[TMP11:%.*]] = bitcast <16 x i8> [[TMP7]] to <4 x i32>
// CHECK:   [[TMP12:%.*]] = bitcast <16 x i8> [[TMP9]] to <4 x i32>
// CHECK:   [[VLD3_LANE:%.*]] = call { <4 x i32>, <4 x i32>, <4 x i32> } @llvm.aarch64.neon.ld3lane.v4i32.p0i8(<4 x i32> [[TMP10]], <4 x i32> [[TMP11]], <4 x i32> [[TMP12]], i64 3, i8* [[TMP3]])
// CHECK:   [[TMP13:%.*]] = bitcast i8* [[TMP2]] to { <4 x i32>, <4 x i32>, <4 x i32> }*
// CHECK:   store { <4 x i32>, <4 x i32>, <4 x i32> } [[VLD3_LANE]], { <4 x i32>, <4 x i32>, <4 x i32> }* [[TMP13]]
// CHECK:   [[TMP14:%.*]] = bitcast %struct.uint32x4x3_t* [[RETVAL]] to i8*
// CHECK:   [[TMP15:%.*]] = bitcast %struct.uint32x4x3_t* [[__RET]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP14]], i8* align 16 [[TMP15]], i64 48, i1 false)
// CHECK:   [[TMP16:%.*]] = load %struct.uint32x4x3_t, %struct.uint32x4x3_t* [[RETVAL]], align 16
// CHECK:   ret %struct.uint32x4x3_t [[TMP16]]
uint32x4x3_t test_vld3q_lane_u32(uint32_t  *a, uint32x4x3_t b) {
  return vld3q_lane_u32(a, b, 3);
}

// CHECK-LABEL: define{{.*}} %struct.uint64x2x3_t @test_vld3q_lane_u64(i64* noundef %a, [3 x <2 x i64>] %b.coerce) #2 {
// CHECK:   [[RETVAL:%.*]] = alloca %struct.uint64x2x3_t, align 16
// CHECK:   [[B:%.*]] = alloca %struct.uint64x2x3_t, align 16
// CHECK:   [[__RET:%.*]] = alloca %struct.uint64x2x3_t, align 16
// CHECK:   [[__S1:%.*]] = alloca %struct.uint64x2x3_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.uint64x2x3_t, %struct.uint64x2x3_t* [[B]], i32 0, i32 0
// CHECK:   store [3 x <2 x i64>] [[B]].coerce, [3 x <2 x i64>]* [[COERCE_DIVE]], align 16
// CHECK:   [[TMP0:%.*]] = bitcast %struct.uint64x2x3_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.uint64x2x3_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP0]], i8* align 16 [[TMP1]], i64 48, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast %struct.uint64x2x3_t* [[__RET]] to i8*
// CHECK:   [[TMP3:%.*]] = bitcast i64* %a to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.uint64x2x3_t, %struct.uint64x2x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <2 x i64>], [3 x <2 x i64>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP4:%.*]] = load <2 x i64>, <2 x i64>* [[ARRAYIDX]], align 16
// CHECK:   [[TMP5:%.*]] = bitcast <2 x i64> [[TMP4]] to <16 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.uint64x2x3_t, %struct.uint64x2x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <2 x i64>], [3 x <2 x i64>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP6:%.*]] = load <2 x i64>, <2 x i64>* [[ARRAYIDX2]], align 16
// CHECK:   [[TMP7:%.*]] = bitcast <2 x i64> [[TMP6]] to <16 x i8>
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds %struct.uint64x2x3_t, %struct.uint64x2x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <2 x i64>], [3 x <2 x i64>]* [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP8:%.*]] = load <2 x i64>, <2 x i64>* [[ARRAYIDX4]], align 16
// CHECK:   [[TMP9:%.*]] = bitcast <2 x i64> [[TMP8]] to <16 x i8>
// CHECK:   [[TMP10:%.*]] = bitcast <16 x i8> [[TMP5]] to <2 x i64>
// CHECK:   [[TMP11:%.*]] = bitcast <16 x i8> [[TMP7]] to <2 x i64>
// CHECK:   [[TMP12:%.*]] = bitcast <16 x i8> [[TMP9]] to <2 x i64>
// CHECK:   [[VLD3_LANE:%.*]] = call { <2 x i64>, <2 x i64>, <2 x i64> } @llvm.aarch64.neon.ld3lane.v2i64.p0i8(<2 x i64> [[TMP10]], <2 x i64> [[TMP11]], <2 x i64> [[TMP12]], i64 1, i8* [[TMP3]])
// CHECK:   [[TMP13:%.*]] = bitcast i8* [[TMP2]] to { <2 x i64>, <2 x i64>, <2 x i64> }*
// CHECK:   store { <2 x i64>, <2 x i64>, <2 x i64> } [[VLD3_LANE]], { <2 x i64>, <2 x i64>, <2 x i64> }* [[TMP13]]
// CHECK:   [[TMP14:%.*]] = bitcast %struct.uint64x2x3_t* [[RETVAL]] to i8*
// CHECK:   [[TMP15:%.*]] = bitcast %struct.uint64x2x3_t* [[__RET]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP14]], i8* align 16 [[TMP15]], i64 48, i1 false)
// CHECK:   [[TMP16:%.*]] = load %struct.uint64x2x3_t, %struct.uint64x2x3_t* [[RETVAL]], align 16
// CHECK:   ret %struct.uint64x2x3_t [[TMP16]]
uint64x2x3_t test_vld3q_lane_u64(uint64_t  *a, uint64x2x3_t b) {
  return vld3q_lane_u64(a, b, 1);
}

// CHECK-LABEL: define{{.*}} %struct.int16x8x3_t @test_vld3q_lane_s16(i16* noundef %a, [3 x <8 x i16>] %b.coerce) #2 {
// CHECK:   [[RETVAL:%.*]] = alloca %struct.int16x8x3_t, align 16
// CHECK:   [[B:%.*]] = alloca %struct.int16x8x3_t, align 16
// CHECK:   [[__RET:%.*]] = alloca %struct.int16x8x3_t, align 16
// CHECK:   [[__S1:%.*]] = alloca %struct.int16x8x3_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.int16x8x3_t, %struct.int16x8x3_t* [[B]], i32 0, i32 0
// CHECK:   store [3 x <8 x i16>] [[B]].coerce, [3 x <8 x i16>]* [[COERCE_DIVE]], align 16
// CHECK:   [[TMP0:%.*]] = bitcast %struct.int16x8x3_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.int16x8x3_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP0]], i8* align 16 [[TMP1]], i64 48, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast %struct.int16x8x3_t* [[__RET]] to i8*
// CHECK:   [[TMP3:%.*]] = bitcast i16* %a to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.int16x8x3_t, %struct.int16x8x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <8 x i16>], [3 x <8 x i16>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP4:%.*]] = load <8 x i16>, <8 x i16>* [[ARRAYIDX]], align 16
// CHECK:   [[TMP5:%.*]] = bitcast <8 x i16> [[TMP4]] to <16 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.int16x8x3_t, %struct.int16x8x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <8 x i16>], [3 x <8 x i16>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP6:%.*]] = load <8 x i16>, <8 x i16>* [[ARRAYIDX2]], align 16
// CHECK:   [[TMP7:%.*]] = bitcast <8 x i16> [[TMP6]] to <16 x i8>
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds %struct.int16x8x3_t, %struct.int16x8x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <8 x i16>], [3 x <8 x i16>]* [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP8:%.*]] = load <8 x i16>, <8 x i16>* [[ARRAYIDX4]], align 16
// CHECK:   [[TMP9:%.*]] = bitcast <8 x i16> [[TMP8]] to <16 x i8>
// CHECK:   [[TMP10:%.*]] = bitcast <16 x i8> [[TMP5]] to <8 x i16>
// CHECK:   [[TMP11:%.*]] = bitcast <16 x i8> [[TMP7]] to <8 x i16>
// CHECK:   [[TMP12:%.*]] = bitcast <16 x i8> [[TMP9]] to <8 x i16>
// CHECK:   [[VLD3_LANE:%.*]] = call { <8 x i16>, <8 x i16>, <8 x i16> } @llvm.aarch64.neon.ld3lane.v8i16.p0i8(<8 x i16> [[TMP10]], <8 x i16> [[TMP11]], <8 x i16> [[TMP12]], i64 7, i8* [[TMP3]])
// CHECK:   [[TMP13:%.*]] = bitcast i8* [[TMP2]] to { <8 x i16>, <8 x i16>, <8 x i16> }*
// CHECK:   store { <8 x i16>, <8 x i16>, <8 x i16> } [[VLD3_LANE]], { <8 x i16>, <8 x i16>, <8 x i16> }* [[TMP13]]
// CHECK:   [[TMP14:%.*]] = bitcast %struct.int16x8x3_t* [[RETVAL]] to i8*
// CHECK:   [[TMP15:%.*]] = bitcast %struct.int16x8x3_t* [[__RET]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP14]], i8* align 16 [[TMP15]], i64 48, i1 false)
// CHECK:   [[TMP16:%.*]] = load %struct.int16x8x3_t, %struct.int16x8x3_t* [[RETVAL]], align 16
// CHECK:   ret %struct.int16x8x3_t [[TMP16]]
int16x8x3_t test_vld3q_lane_s16(int16_t  *a, int16x8x3_t b) {
  return vld3q_lane_s16(a, b, 7);
}

// CHECK-LABEL: define{{.*}} %struct.int32x4x3_t @test_vld3q_lane_s32(i32* noundef %a, [3 x <4 x i32>] %b.coerce) #2 {
// CHECK:   [[RETVAL:%.*]] = alloca %struct.int32x4x3_t, align 16
// CHECK:   [[B:%.*]] = alloca %struct.int32x4x3_t, align 16
// CHECK:   [[__RET:%.*]] = alloca %struct.int32x4x3_t, align 16
// CHECK:   [[__S1:%.*]] = alloca %struct.int32x4x3_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.int32x4x3_t, %struct.int32x4x3_t* [[B]], i32 0, i32 0
// CHECK:   store [3 x <4 x i32>] [[B]].coerce, [3 x <4 x i32>]* [[COERCE_DIVE]], align 16
// CHECK:   [[TMP0:%.*]] = bitcast %struct.int32x4x3_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.int32x4x3_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP0]], i8* align 16 [[TMP1]], i64 48, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast %struct.int32x4x3_t* [[__RET]] to i8*
// CHECK:   [[TMP3:%.*]] = bitcast i32* %a to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.int32x4x3_t, %struct.int32x4x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <4 x i32>], [3 x <4 x i32>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP4:%.*]] = load <4 x i32>, <4 x i32>* [[ARRAYIDX]], align 16
// CHECK:   [[TMP5:%.*]] = bitcast <4 x i32> [[TMP4]] to <16 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.int32x4x3_t, %struct.int32x4x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <4 x i32>], [3 x <4 x i32>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP6:%.*]] = load <4 x i32>, <4 x i32>* [[ARRAYIDX2]], align 16
// CHECK:   [[TMP7:%.*]] = bitcast <4 x i32> [[TMP6]] to <16 x i8>
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds %struct.int32x4x3_t, %struct.int32x4x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <4 x i32>], [3 x <4 x i32>]* [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP8:%.*]] = load <4 x i32>, <4 x i32>* [[ARRAYIDX4]], align 16
// CHECK:   [[TMP9:%.*]] = bitcast <4 x i32> [[TMP8]] to <16 x i8>
// CHECK:   [[TMP10:%.*]] = bitcast <16 x i8> [[TMP5]] to <4 x i32>
// CHECK:   [[TMP11:%.*]] = bitcast <16 x i8> [[TMP7]] to <4 x i32>
// CHECK:   [[TMP12:%.*]] = bitcast <16 x i8> [[TMP9]] to <4 x i32>
// CHECK:   [[VLD3_LANE:%.*]] = call { <4 x i32>, <4 x i32>, <4 x i32> } @llvm.aarch64.neon.ld3lane.v4i32.p0i8(<4 x i32> [[TMP10]], <4 x i32> [[TMP11]], <4 x i32> [[TMP12]], i64 3, i8* [[TMP3]])
// CHECK:   [[TMP13:%.*]] = bitcast i8* [[TMP2]] to { <4 x i32>, <4 x i32>, <4 x i32> }*
// CHECK:   store { <4 x i32>, <4 x i32>, <4 x i32> } [[VLD3_LANE]], { <4 x i32>, <4 x i32>, <4 x i32> }* [[TMP13]]
// CHECK:   [[TMP14:%.*]] = bitcast %struct.int32x4x3_t* [[RETVAL]] to i8*
// CHECK:   [[TMP15:%.*]] = bitcast %struct.int32x4x3_t* [[__RET]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP14]], i8* align 16 [[TMP15]], i64 48, i1 false)
// CHECK:   [[TMP16:%.*]] = load %struct.int32x4x3_t, %struct.int32x4x3_t* [[RETVAL]], align 16
// CHECK:   ret %struct.int32x4x3_t [[TMP16]]
int32x4x3_t test_vld3q_lane_s32(int32_t  *a, int32x4x3_t b) {
  return vld3q_lane_s32(a, b, 3);
}

// CHECK-LABEL: define{{.*}} %struct.int64x2x3_t @test_vld3q_lane_s64(i64* noundef %a, [3 x <2 x i64>] %b.coerce) #2 {
// CHECK:   [[RETVAL:%.*]] = alloca %struct.int64x2x3_t, align 16
// CHECK:   [[B:%.*]] = alloca %struct.int64x2x3_t, align 16
// CHECK:   [[__RET:%.*]] = alloca %struct.int64x2x3_t, align 16
// CHECK:   [[__S1:%.*]] = alloca %struct.int64x2x3_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.int64x2x3_t, %struct.int64x2x3_t* [[B]], i32 0, i32 0
// CHECK:   store [3 x <2 x i64>] [[B]].coerce, [3 x <2 x i64>]* [[COERCE_DIVE]], align 16
// CHECK:   [[TMP0:%.*]] = bitcast %struct.int64x2x3_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.int64x2x3_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP0]], i8* align 16 [[TMP1]], i64 48, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast %struct.int64x2x3_t* [[__RET]] to i8*
// CHECK:   [[TMP3:%.*]] = bitcast i64* %a to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.int64x2x3_t, %struct.int64x2x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <2 x i64>], [3 x <2 x i64>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP4:%.*]] = load <2 x i64>, <2 x i64>* [[ARRAYIDX]], align 16
// CHECK:   [[TMP5:%.*]] = bitcast <2 x i64> [[TMP4]] to <16 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.int64x2x3_t, %struct.int64x2x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <2 x i64>], [3 x <2 x i64>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP6:%.*]] = load <2 x i64>, <2 x i64>* [[ARRAYIDX2]], align 16
// CHECK:   [[TMP7:%.*]] = bitcast <2 x i64> [[TMP6]] to <16 x i8>
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds %struct.int64x2x3_t, %struct.int64x2x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <2 x i64>], [3 x <2 x i64>]* [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP8:%.*]] = load <2 x i64>, <2 x i64>* [[ARRAYIDX4]], align 16
// CHECK:   [[TMP9:%.*]] = bitcast <2 x i64> [[TMP8]] to <16 x i8>
// CHECK:   [[TMP10:%.*]] = bitcast <16 x i8> [[TMP5]] to <2 x i64>
// CHECK:   [[TMP11:%.*]] = bitcast <16 x i8> [[TMP7]] to <2 x i64>
// CHECK:   [[TMP12:%.*]] = bitcast <16 x i8> [[TMP9]] to <2 x i64>
// CHECK:   [[VLD3_LANE:%.*]] = call { <2 x i64>, <2 x i64>, <2 x i64> } @llvm.aarch64.neon.ld3lane.v2i64.p0i8(<2 x i64> [[TMP10]], <2 x i64> [[TMP11]], <2 x i64> [[TMP12]], i64 1, i8* [[TMP3]])
// CHECK:   [[TMP13:%.*]] = bitcast i8* [[TMP2]] to { <2 x i64>, <2 x i64>, <2 x i64> }*
// CHECK:   store { <2 x i64>, <2 x i64>, <2 x i64> } [[VLD3_LANE]], { <2 x i64>, <2 x i64>, <2 x i64> }* [[TMP13]]
// CHECK:   [[TMP14:%.*]] = bitcast %struct.int64x2x3_t* [[RETVAL]] to i8*
// CHECK:   [[TMP15:%.*]] = bitcast %struct.int64x2x3_t* [[__RET]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP14]], i8* align 16 [[TMP15]], i64 48, i1 false)
// CHECK:   [[TMP16:%.*]] = load %struct.int64x2x3_t, %struct.int64x2x3_t* [[RETVAL]], align 16
// CHECK:   ret %struct.int64x2x3_t [[TMP16]]
int64x2x3_t test_vld3q_lane_s64(int64_t  *a, int64x2x3_t b) {
  return vld3q_lane_s64(a, b, 1);
}

// CHECK-LABEL: define{{.*}} %struct.float16x8x3_t @test_vld3q_lane_f16(half* noundef %a, [3 x <8 x half>] %b.coerce) #2 {
// CHECK:   [[RETVAL:%.*]] = alloca %struct.float16x8x3_t, align 16
// CHECK:   [[B:%.*]] = alloca %struct.float16x8x3_t, align 16
// CHECK:   [[__RET:%.*]] = alloca %struct.float16x8x3_t, align 16
// CHECK:   [[__S1:%.*]] = alloca %struct.float16x8x3_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.float16x8x3_t, %struct.float16x8x3_t* [[B]], i32 0, i32 0
// CHECK:   store [3 x <8 x half>] [[B]].coerce, [3 x <8 x half>]* [[COERCE_DIVE]], align 16
// CHECK:   [[TMP0:%.*]] = bitcast %struct.float16x8x3_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.float16x8x3_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP0]], i8* align 16 [[TMP1]], i64 48, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast %struct.float16x8x3_t* [[__RET]] to i8*
// CHECK:   [[TMP3:%.*]] = bitcast half* %a to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.float16x8x3_t, %struct.float16x8x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <8 x half>], [3 x <8 x half>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP4:%.*]] = load <8 x half>, <8 x half>* [[ARRAYIDX]], align 16
// CHECK:   [[TMP5:%.*]] = bitcast <8 x half> [[TMP4]] to <16 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.float16x8x3_t, %struct.float16x8x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <8 x half>], [3 x <8 x half>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP6:%.*]] = load <8 x half>, <8 x half>* [[ARRAYIDX2]], align 16
// CHECK:   [[TMP7:%.*]] = bitcast <8 x half> [[TMP6]] to <16 x i8>
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds %struct.float16x8x3_t, %struct.float16x8x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <8 x half>], [3 x <8 x half>]* [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP8:%.*]] = load <8 x half>, <8 x half>* [[ARRAYIDX4]], align 16
// CHECK:   [[TMP9:%.*]] = bitcast <8 x half> [[TMP8]] to <16 x i8>
// CHECK:   [[TMP10:%.*]] = bitcast <16 x i8> [[TMP5]] to <8 x half>
// CHECK:   [[TMP11:%.*]] = bitcast <16 x i8> [[TMP7]] to <8 x half>
// CHECK:   [[TMP12:%.*]] = bitcast <16 x i8> [[TMP9]] to <8 x half>
// CHECK:   [[VLD3_LANE:%.*]] = call { <8 x half>, <8 x half>, <8 x half> } @llvm.aarch64.neon.ld3lane.v8f16.p0i8(<8 x half> [[TMP10]], <8 x half> [[TMP11]], <8 x half> [[TMP12]], i64 7, i8* [[TMP3]])
// CHECK:   [[TMP13:%.*]] = bitcast i8* [[TMP2]] to { <8 x half>, <8 x half>, <8 x half> }*
// CHECK:   store { <8 x half>, <8 x half>, <8 x half> } [[VLD3_LANE]], { <8 x half>, <8 x half>, <8 x half> }* [[TMP13]]
// CHECK:   [[TMP14:%.*]] = bitcast %struct.float16x8x3_t* [[RETVAL]] to i8*
// CHECK:   [[TMP15:%.*]] = bitcast %struct.float16x8x3_t* [[__RET]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP14]], i8* align 16 [[TMP15]], i64 48, i1 false)
// CHECK:   [[TMP16:%.*]] = load %struct.float16x8x3_t, %struct.float16x8x3_t* [[RETVAL]], align 16
// CHECK:   ret %struct.float16x8x3_t [[TMP16]]
float16x8x3_t test_vld3q_lane_f16(float16_t  *a, float16x8x3_t b) {
  return vld3q_lane_f16(a, b, 7);
}

// CHECK-LABEL: define{{.*}} %struct.float32x4x3_t @test_vld3q_lane_f32(float* noundef %a, [3 x <4 x float>] %b.coerce) #2 {
// CHECK:   [[RETVAL:%.*]] = alloca %struct.float32x4x3_t, align 16
// CHECK:   [[B:%.*]] = alloca %struct.float32x4x3_t, align 16
// CHECK:   [[__RET:%.*]] = alloca %struct.float32x4x3_t, align 16
// CHECK:   [[__S1:%.*]] = alloca %struct.float32x4x3_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.float32x4x3_t, %struct.float32x4x3_t* [[B]], i32 0, i32 0
// CHECK:   store [3 x <4 x float>] [[B]].coerce, [3 x <4 x float>]* [[COERCE_DIVE]], align 16
// CHECK:   [[TMP0:%.*]] = bitcast %struct.float32x4x3_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.float32x4x3_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP0]], i8* align 16 [[TMP1]], i64 48, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast %struct.float32x4x3_t* [[__RET]] to i8*
// CHECK:   [[TMP3:%.*]] = bitcast float* %a to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.float32x4x3_t, %struct.float32x4x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <4 x float>], [3 x <4 x float>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP4:%.*]] = load <4 x float>, <4 x float>* [[ARRAYIDX]], align 16
// CHECK:   [[TMP5:%.*]] = bitcast <4 x float> [[TMP4]] to <16 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.float32x4x3_t, %struct.float32x4x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <4 x float>], [3 x <4 x float>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP6:%.*]] = load <4 x float>, <4 x float>* [[ARRAYIDX2]], align 16
// CHECK:   [[TMP7:%.*]] = bitcast <4 x float> [[TMP6]] to <16 x i8>
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds %struct.float32x4x3_t, %struct.float32x4x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <4 x float>], [3 x <4 x float>]* [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP8:%.*]] = load <4 x float>, <4 x float>* [[ARRAYIDX4]], align 16
// CHECK:   [[TMP9:%.*]] = bitcast <4 x float> [[TMP8]] to <16 x i8>
// CHECK:   [[TMP10:%.*]] = bitcast <16 x i8> [[TMP5]] to <4 x float>
// CHECK:   [[TMP11:%.*]] = bitcast <16 x i8> [[TMP7]] to <4 x float>
// CHECK:   [[TMP12:%.*]] = bitcast <16 x i8> [[TMP9]] to <4 x float>
// CHECK:   [[VLD3_LANE:%.*]] = call { <4 x float>, <4 x float>, <4 x float> } @llvm.aarch64.neon.ld3lane.v4f32.p0i8(<4 x float> [[TMP10]], <4 x float> [[TMP11]], <4 x float> [[TMP12]], i64 3, i8* [[TMP3]])
// CHECK:   [[TMP13:%.*]] = bitcast i8* [[TMP2]] to { <4 x float>, <4 x float>, <4 x float> }*
// CHECK:   store { <4 x float>, <4 x float>, <4 x float> } [[VLD3_LANE]], { <4 x float>, <4 x float>, <4 x float> }* [[TMP13]]
// CHECK:   [[TMP14:%.*]] = bitcast %struct.float32x4x3_t* [[RETVAL]] to i8*
// CHECK:   [[TMP15:%.*]] = bitcast %struct.float32x4x3_t* [[__RET]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP14]], i8* align 16 [[TMP15]], i64 48, i1 false)
// CHECK:   [[TMP16:%.*]] = load %struct.float32x4x3_t, %struct.float32x4x3_t* [[RETVAL]], align 16
// CHECK:   ret %struct.float32x4x3_t [[TMP16]]
float32x4x3_t test_vld3q_lane_f32(float32_t  *a, float32x4x3_t b) {
  return vld3q_lane_f32(a, b, 3);
}

// CHECK-LABEL: define{{.*}} %struct.float64x2x3_t @test_vld3q_lane_f64(double* noundef %a, [3 x <2 x double>] %b.coerce) #2 {
// CHECK:   [[RETVAL:%.*]] = alloca %struct.float64x2x3_t, align 16
// CHECK:   [[B:%.*]] = alloca %struct.float64x2x3_t, align 16
// CHECK:   [[__RET:%.*]] = alloca %struct.float64x2x3_t, align 16
// CHECK:   [[__S1:%.*]] = alloca %struct.float64x2x3_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.float64x2x3_t, %struct.float64x2x3_t* [[B]], i32 0, i32 0
// CHECK:   store [3 x <2 x double>] [[B]].coerce, [3 x <2 x double>]* [[COERCE_DIVE]], align 16
// CHECK:   [[TMP0:%.*]] = bitcast %struct.float64x2x3_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.float64x2x3_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP0]], i8* align 16 [[TMP1]], i64 48, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast %struct.float64x2x3_t* [[__RET]] to i8*
// CHECK:   [[TMP3:%.*]] = bitcast double* %a to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.float64x2x3_t, %struct.float64x2x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <2 x double>], [3 x <2 x double>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP4:%.*]] = load <2 x double>, <2 x double>* [[ARRAYIDX]], align 16
// CHECK:   [[TMP5:%.*]] = bitcast <2 x double> [[TMP4]] to <16 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.float64x2x3_t, %struct.float64x2x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <2 x double>], [3 x <2 x double>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP6:%.*]] = load <2 x double>, <2 x double>* [[ARRAYIDX2]], align 16
// CHECK:   [[TMP7:%.*]] = bitcast <2 x double> [[TMP6]] to <16 x i8>
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds %struct.float64x2x3_t, %struct.float64x2x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <2 x double>], [3 x <2 x double>]* [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP8:%.*]] = load <2 x double>, <2 x double>* [[ARRAYIDX4]], align 16
// CHECK:   [[TMP9:%.*]] = bitcast <2 x double> [[TMP8]] to <16 x i8>
// CHECK:   [[TMP10:%.*]] = bitcast <16 x i8> [[TMP5]] to <2 x double>
// CHECK:   [[TMP11:%.*]] = bitcast <16 x i8> [[TMP7]] to <2 x double>
// CHECK:   [[TMP12:%.*]] = bitcast <16 x i8> [[TMP9]] to <2 x double>
// CHECK:   [[VLD3_LANE:%.*]] = call { <2 x double>, <2 x double>, <2 x double> } @llvm.aarch64.neon.ld3lane.v2f64.p0i8(<2 x double> [[TMP10]], <2 x double> [[TMP11]], <2 x double> [[TMP12]], i64 1, i8* [[TMP3]])
// CHECK:   [[TMP13:%.*]] = bitcast i8* [[TMP2]] to { <2 x double>, <2 x double>, <2 x double> }*
// CHECK:   store { <2 x double>, <2 x double>, <2 x double> } [[VLD3_LANE]], { <2 x double>, <2 x double>, <2 x double> }* [[TMP13]]
// CHECK:   [[TMP14:%.*]] = bitcast %struct.float64x2x3_t* [[RETVAL]] to i8*
// CHECK:   [[TMP15:%.*]] = bitcast %struct.float64x2x3_t* [[__RET]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP14]], i8* align 16 [[TMP15]], i64 48, i1 false)
// CHECK:   [[TMP16:%.*]] = load %struct.float64x2x3_t, %struct.float64x2x3_t* [[RETVAL]], align 16
// CHECK:   ret %struct.float64x2x3_t [[TMP16]]
float64x2x3_t test_vld3q_lane_f64(float64_t  *a, float64x2x3_t b) {
  return vld3q_lane_f64(a, b, 1);
}

// CHECK-LABEL: define{{.*}} %struct.poly8x16x3_t @test_vld3q_lane_p8(i8* noundef %a, [3 x <16 x i8>] %b.coerce) #2 {
// CHECK:   [[RETVAL:%.*]] = alloca %struct.poly8x16x3_t, align 16
// CHECK:   [[B:%.*]] = alloca %struct.poly8x16x3_t, align 16
// CHECK:   [[__RET:%.*]] = alloca %struct.poly8x16x3_t, align 16
// CHECK:   [[__S1:%.*]] = alloca %struct.poly8x16x3_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.poly8x16x3_t, %struct.poly8x16x3_t* [[B]], i32 0, i32 0
// CHECK:   store [3 x <16 x i8>] [[B]].coerce, [3 x <16 x i8>]* [[COERCE_DIVE]], align 16
// CHECK:   [[TMP0:%.*]] = bitcast %struct.poly8x16x3_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.poly8x16x3_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP0]], i8* align 16 [[TMP1]], i64 48, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast %struct.poly8x16x3_t* [[__RET]] to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.poly8x16x3_t, %struct.poly8x16x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <16 x i8>], [3 x <16 x i8>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX]], align 16
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.poly8x16x3_t, %struct.poly8x16x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <16 x i8>], [3 x <16 x i8>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP4:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX2]], align 16
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds %struct.poly8x16x3_t, %struct.poly8x16x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <16 x i8>], [3 x <16 x i8>]* [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP5:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX4]], align 16
// CHECK:   [[VLD3_LANE:%.*]] = call { <16 x i8>, <16 x i8>, <16 x i8> } @llvm.aarch64.neon.ld3lane.v16i8.p0i8(<16 x i8> [[TMP3]], <16 x i8> [[TMP4]], <16 x i8> [[TMP5]], i64 15, i8* %a)
// CHECK:   [[TMP6:%.*]] = bitcast i8* [[TMP2]] to { <16 x i8>, <16 x i8>, <16 x i8> }*
// CHECK:   store { <16 x i8>, <16 x i8>, <16 x i8> } [[VLD3_LANE]], { <16 x i8>, <16 x i8>, <16 x i8> }* [[TMP6]]
// CHECK:   [[TMP7:%.*]] = bitcast %struct.poly8x16x3_t* [[RETVAL]] to i8*
// CHECK:   [[TMP8:%.*]] = bitcast %struct.poly8x16x3_t* [[__RET]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP7]], i8* align 16 [[TMP8]], i64 48, i1 false)
// CHECK:   [[TMP9:%.*]] = load %struct.poly8x16x3_t, %struct.poly8x16x3_t* [[RETVAL]], align 16
// CHECK:   ret %struct.poly8x16x3_t [[TMP9]]
poly8x16x3_t test_vld3q_lane_p8(poly8_t  *a, poly8x16x3_t b) {
  return vld3q_lane_p8(a, b, 15);
}

// CHECK-LABEL: define{{.*}} %struct.poly16x8x3_t @test_vld3q_lane_p16(i16* noundef %a, [3 x <8 x i16>] %b.coerce) #2 {
// CHECK:   [[RETVAL:%.*]] = alloca %struct.poly16x8x3_t, align 16
// CHECK:   [[B:%.*]] = alloca %struct.poly16x8x3_t, align 16
// CHECK:   [[__RET:%.*]] = alloca %struct.poly16x8x3_t, align 16
// CHECK:   [[__S1:%.*]] = alloca %struct.poly16x8x3_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.poly16x8x3_t, %struct.poly16x8x3_t* [[B]], i32 0, i32 0
// CHECK:   store [3 x <8 x i16>] [[B]].coerce, [3 x <8 x i16>]* [[COERCE_DIVE]], align 16
// CHECK:   [[TMP0:%.*]] = bitcast %struct.poly16x8x3_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.poly16x8x3_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP0]], i8* align 16 [[TMP1]], i64 48, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast %struct.poly16x8x3_t* [[__RET]] to i8*
// CHECK:   [[TMP3:%.*]] = bitcast i16* %a to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.poly16x8x3_t, %struct.poly16x8x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <8 x i16>], [3 x <8 x i16>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP4:%.*]] = load <8 x i16>, <8 x i16>* [[ARRAYIDX]], align 16
// CHECK:   [[TMP5:%.*]] = bitcast <8 x i16> [[TMP4]] to <16 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.poly16x8x3_t, %struct.poly16x8x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <8 x i16>], [3 x <8 x i16>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP6:%.*]] = load <8 x i16>, <8 x i16>* [[ARRAYIDX2]], align 16
// CHECK:   [[TMP7:%.*]] = bitcast <8 x i16> [[TMP6]] to <16 x i8>
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds %struct.poly16x8x3_t, %struct.poly16x8x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <8 x i16>], [3 x <8 x i16>]* [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP8:%.*]] = load <8 x i16>, <8 x i16>* [[ARRAYIDX4]], align 16
// CHECK:   [[TMP9:%.*]] = bitcast <8 x i16> [[TMP8]] to <16 x i8>
// CHECK:   [[TMP10:%.*]] = bitcast <16 x i8> [[TMP5]] to <8 x i16>
// CHECK:   [[TMP11:%.*]] = bitcast <16 x i8> [[TMP7]] to <8 x i16>
// CHECK:   [[TMP12:%.*]] = bitcast <16 x i8> [[TMP9]] to <8 x i16>
// CHECK:   [[VLD3_LANE:%.*]] = call { <8 x i16>, <8 x i16>, <8 x i16> } @llvm.aarch64.neon.ld3lane.v8i16.p0i8(<8 x i16> [[TMP10]], <8 x i16> [[TMP11]], <8 x i16> [[TMP12]], i64 7, i8* [[TMP3]])
// CHECK:   [[TMP13:%.*]] = bitcast i8* [[TMP2]] to { <8 x i16>, <8 x i16>, <8 x i16> }*
// CHECK:   store { <8 x i16>, <8 x i16>, <8 x i16> } [[VLD3_LANE]], { <8 x i16>, <8 x i16>, <8 x i16> }* [[TMP13]]
// CHECK:   [[TMP14:%.*]] = bitcast %struct.poly16x8x3_t* [[RETVAL]] to i8*
// CHECK:   [[TMP15:%.*]] = bitcast %struct.poly16x8x3_t* [[__RET]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP14]], i8* align 16 [[TMP15]], i64 48, i1 false)
// CHECK:   [[TMP16:%.*]] = load %struct.poly16x8x3_t, %struct.poly16x8x3_t* [[RETVAL]], align 16
// CHECK:   ret %struct.poly16x8x3_t [[TMP16]]
poly16x8x3_t test_vld3q_lane_p16(poly16_t  *a, poly16x8x3_t b) {
  return vld3q_lane_p16(a, b, 7);
}

// CHECK-LABEL: define{{.*}} %struct.poly64x2x3_t @test_vld3q_lane_p64(i64* noundef %a, [3 x <2 x i64>] %b.coerce) #2 {
// CHECK:   [[RETVAL:%.*]] = alloca %struct.poly64x2x3_t, align 16
// CHECK:   [[B:%.*]] = alloca %struct.poly64x2x3_t, align 16
// CHECK:   [[__RET:%.*]] = alloca %struct.poly64x2x3_t, align 16
// CHECK:   [[__S1:%.*]] = alloca %struct.poly64x2x3_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.poly64x2x3_t, %struct.poly64x2x3_t* [[B]], i32 0, i32 0
// CHECK:   store [3 x <2 x i64>] [[B]].coerce, [3 x <2 x i64>]* [[COERCE_DIVE]], align 16
// CHECK:   [[TMP0:%.*]] = bitcast %struct.poly64x2x3_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.poly64x2x3_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP0]], i8* align 16 [[TMP1]], i64 48, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast %struct.poly64x2x3_t* [[__RET]] to i8*
// CHECK:   [[TMP3:%.*]] = bitcast i64* %a to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.poly64x2x3_t, %struct.poly64x2x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <2 x i64>], [3 x <2 x i64>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP4:%.*]] = load <2 x i64>, <2 x i64>* [[ARRAYIDX]], align 16
// CHECK:   [[TMP5:%.*]] = bitcast <2 x i64> [[TMP4]] to <16 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.poly64x2x3_t, %struct.poly64x2x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <2 x i64>], [3 x <2 x i64>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP6:%.*]] = load <2 x i64>, <2 x i64>* [[ARRAYIDX2]], align 16
// CHECK:   [[TMP7:%.*]] = bitcast <2 x i64> [[TMP6]] to <16 x i8>
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds %struct.poly64x2x3_t, %struct.poly64x2x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <2 x i64>], [3 x <2 x i64>]* [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP8:%.*]] = load <2 x i64>, <2 x i64>* [[ARRAYIDX4]], align 16
// CHECK:   [[TMP9:%.*]] = bitcast <2 x i64> [[TMP8]] to <16 x i8>
// CHECK:   [[TMP10:%.*]] = bitcast <16 x i8> [[TMP5]] to <2 x i64>
// CHECK:   [[TMP11:%.*]] = bitcast <16 x i8> [[TMP7]] to <2 x i64>
// CHECK:   [[TMP12:%.*]] = bitcast <16 x i8> [[TMP9]] to <2 x i64>
// CHECK:   [[VLD3_LANE:%.*]] = call { <2 x i64>, <2 x i64>, <2 x i64> } @llvm.aarch64.neon.ld3lane.v2i64.p0i8(<2 x i64> [[TMP10]], <2 x i64> [[TMP11]], <2 x i64> [[TMP12]], i64 1, i8* [[TMP3]])
// CHECK:   [[TMP13:%.*]] = bitcast i8* [[TMP2]] to { <2 x i64>, <2 x i64>, <2 x i64> }*
// CHECK:   store { <2 x i64>, <2 x i64>, <2 x i64> } [[VLD3_LANE]], { <2 x i64>, <2 x i64>, <2 x i64> }* [[TMP13]]
// CHECK:   [[TMP14:%.*]] = bitcast %struct.poly64x2x3_t* [[RETVAL]] to i8*
// CHECK:   [[TMP15:%.*]] = bitcast %struct.poly64x2x3_t* [[__RET]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP14]], i8* align 16 [[TMP15]], i64 48, i1 false)
// CHECK:   [[TMP16:%.*]] = load %struct.poly64x2x3_t, %struct.poly64x2x3_t* [[RETVAL]], align 16
// CHECK:   ret %struct.poly64x2x3_t [[TMP16]]
poly64x2x3_t test_vld3q_lane_p64(poly64_t  *a, poly64x2x3_t b) {
  return vld3q_lane_p64(a, b, 1);
}

// CHECK-LABEL: define{{.*}} %struct.uint8x8x3_t @test_vld3_lane_u8(i8* noundef %a, [3 x <8 x i8>] %b.coerce) #2 {
// CHECK:   [[RETVAL:%.*]] = alloca %struct.uint8x8x3_t, align 8
// CHECK:   [[B:%.*]] = alloca %struct.uint8x8x3_t, align 8
// CHECK:   [[__RET:%.*]] = alloca %struct.uint8x8x3_t, align 8
// CHECK:   [[__S1:%.*]] = alloca %struct.uint8x8x3_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.uint8x8x3_t, %struct.uint8x8x3_t* [[B]], i32 0, i32 0
// CHECK:   store [3 x <8 x i8>] [[B]].coerce, [3 x <8 x i8>]* [[COERCE_DIVE]], align 8
// CHECK:   [[TMP0:%.*]] = bitcast %struct.uint8x8x3_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.uint8x8x3_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[TMP0]], i8* align 8 [[TMP1]], i64 24, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast %struct.uint8x8x3_t* [[__RET]] to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.uint8x8x3_t, %struct.uint8x8x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <8 x i8>], [3 x <8 x i8>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <8 x i8>, <8 x i8>* [[ARRAYIDX]], align 8
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.uint8x8x3_t, %struct.uint8x8x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <8 x i8>], [3 x <8 x i8>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP4:%.*]] = load <8 x i8>, <8 x i8>* [[ARRAYIDX2]], align 8
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds %struct.uint8x8x3_t, %struct.uint8x8x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <8 x i8>], [3 x <8 x i8>]* [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP5:%.*]] = load <8 x i8>, <8 x i8>* [[ARRAYIDX4]], align 8
// CHECK:   [[VLD3_LANE:%.*]] = call { <8 x i8>, <8 x i8>, <8 x i8> } @llvm.aarch64.neon.ld3lane.v8i8.p0i8(<8 x i8> [[TMP3]], <8 x i8> [[TMP4]], <8 x i8> [[TMP5]], i64 7, i8* %a)
// CHECK:   [[TMP6:%.*]] = bitcast i8* [[TMP2]] to { <8 x i8>, <8 x i8>, <8 x i8> }*
// CHECK:   store { <8 x i8>, <8 x i8>, <8 x i8> } [[VLD3_LANE]], { <8 x i8>, <8 x i8>, <8 x i8> }* [[TMP6]]
// CHECK:   [[TMP7:%.*]] = bitcast %struct.uint8x8x3_t* [[RETVAL]] to i8*
// CHECK:   [[TMP8:%.*]] = bitcast %struct.uint8x8x3_t* [[__RET]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[TMP7]], i8* align 8 [[TMP8]], i64 24, i1 false)
// CHECK:   [[TMP9:%.*]] = load %struct.uint8x8x3_t, %struct.uint8x8x3_t* [[RETVAL]], align 8
// CHECK:   ret %struct.uint8x8x3_t [[TMP9]]
uint8x8x3_t test_vld3_lane_u8(uint8_t  *a, uint8x8x3_t b) {
  return vld3_lane_u8(a, b, 7);
}

// CHECK-LABEL: define{{.*}} %struct.uint16x4x3_t @test_vld3_lane_u16(i16* noundef %a, [3 x <4 x i16>] %b.coerce) #2 {
// CHECK:   [[RETVAL:%.*]] = alloca %struct.uint16x4x3_t, align 8
// CHECK:   [[B:%.*]] = alloca %struct.uint16x4x3_t, align 8
// CHECK:   [[__RET:%.*]] = alloca %struct.uint16x4x3_t, align 8
// CHECK:   [[__S1:%.*]] = alloca %struct.uint16x4x3_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.uint16x4x3_t, %struct.uint16x4x3_t* [[B]], i32 0, i32 0
// CHECK:   store [3 x <4 x i16>] [[B]].coerce, [3 x <4 x i16>]* [[COERCE_DIVE]], align 8
// CHECK:   [[TMP0:%.*]] = bitcast %struct.uint16x4x3_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.uint16x4x3_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[TMP0]], i8* align 8 [[TMP1]], i64 24, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast %struct.uint16x4x3_t* [[__RET]] to i8*
// CHECK:   [[TMP3:%.*]] = bitcast i16* %a to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.uint16x4x3_t, %struct.uint16x4x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <4 x i16>], [3 x <4 x i16>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP4:%.*]] = load <4 x i16>, <4 x i16>* [[ARRAYIDX]], align 8
// CHECK:   [[TMP5:%.*]] = bitcast <4 x i16> [[TMP4]] to <8 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.uint16x4x3_t, %struct.uint16x4x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <4 x i16>], [3 x <4 x i16>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP6:%.*]] = load <4 x i16>, <4 x i16>* [[ARRAYIDX2]], align 8
// CHECK:   [[TMP7:%.*]] = bitcast <4 x i16> [[TMP6]] to <8 x i8>
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds %struct.uint16x4x3_t, %struct.uint16x4x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <4 x i16>], [3 x <4 x i16>]* [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP8:%.*]] = load <4 x i16>, <4 x i16>* [[ARRAYIDX4]], align 8
// CHECK:   [[TMP9:%.*]] = bitcast <4 x i16> [[TMP8]] to <8 x i8>
// CHECK:   [[TMP10:%.*]] = bitcast <8 x i8> [[TMP5]] to <4 x i16>
// CHECK:   [[TMP11:%.*]] = bitcast <8 x i8> [[TMP7]] to <4 x i16>
// CHECK:   [[TMP12:%.*]] = bitcast <8 x i8> [[TMP9]] to <4 x i16>
// CHECK:   [[VLD3_LANE:%.*]] = call { <4 x i16>, <4 x i16>, <4 x i16> } @llvm.aarch64.neon.ld3lane.v4i16.p0i8(<4 x i16> [[TMP10]], <4 x i16> [[TMP11]], <4 x i16> [[TMP12]], i64 3, i8* [[TMP3]])
// CHECK:   [[TMP13:%.*]] = bitcast i8* [[TMP2]] to { <4 x i16>, <4 x i16>, <4 x i16> }*
// CHECK:   store { <4 x i16>, <4 x i16>, <4 x i16> } [[VLD3_LANE]], { <4 x i16>, <4 x i16>, <4 x i16> }* [[TMP13]]
// CHECK:   [[TMP14:%.*]] = bitcast %struct.uint16x4x3_t* [[RETVAL]] to i8*
// CHECK:   [[TMP15:%.*]] = bitcast %struct.uint16x4x3_t* [[__RET]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[TMP14]], i8* align 8 [[TMP15]], i64 24, i1 false)
// CHECK:   [[TMP16:%.*]] = load %struct.uint16x4x3_t, %struct.uint16x4x3_t* [[RETVAL]], align 8
// CHECK:   ret %struct.uint16x4x3_t [[TMP16]]
uint16x4x3_t test_vld3_lane_u16(uint16_t  *a, uint16x4x3_t b) {
  return vld3_lane_u16(a, b, 3);
}

// CHECK-LABEL: define{{.*}} %struct.uint32x2x3_t @test_vld3_lane_u32(i32* noundef %a, [3 x <2 x i32>] %b.coerce) #2 {
// CHECK:   [[RETVAL:%.*]] = alloca %struct.uint32x2x3_t, align 8
// CHECK:   [[B:%.*]] = alloca %struct.uint32x2x3_t, align 8
// CHECK:   [[__RET:%.*]] = alloca %struct.uint32x2x3_t, align 8
// CHECK:   [[__S1:%.*]] = alloca %struct.uint32x2x3_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.uint32x2x3_t, %struct.uint32x2x3_t* [[B]], i32 0, i32 0
// CHECK:   store [3 x <2 x i32>] [[B]].coerce, [3 x <2 x i32>]* [[COERCE_DIVE]], align 8
// CHECK:   [[TMP0:%.*]] = bitcast %struct.uint32x2x3_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.uint32x2x3_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[TMP0]], i8* align 8 [[TMP1]], i64 24, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast %struct.uint32x2x3_t* [[__RET]] to i8*
// CHECK:   [[TMP3:%.*]] = bitcast i32* %a to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.uint32x2x3_t, %struct.uint32x2x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <2 x i32>], [3 x <2 x i32>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP4:%.*]] = load <2 x i32>, <2 x i32>* [[ARRAYIDX]], align 8
// CHECK:   [[TMP5:%.*]] = bitcast <2 x i32> [[TMP4]] to <8 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.uint32x2x3_t, %struct.uint32x2x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <2 x i32>], [3 x <2 x i32>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP6:%.*]] = load <2 x i32>, <2 x i32>* [[ARRAYIDX2]], align 8
// CHECK:   [[TMP7:%.*]] = bitcast <2 x i32> [[TMP6]] to <8 x i8>
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds %struct.uint32x2x3_t, %struct.uint32x2x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <2 x i32>], [3 x <2 x i32>]* [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP8:%.*]] = load <2 x i32>, <2 x i32>* [[ARRAYIDX4]], align 8
// CHECK:   [[TMP9:%.*]] = bitcast <2 x i32> [[TMP8]] to <8 x i8>
// CHECK:   [[TMP10:%.*]] = bitcast <8 x i8> [[TMP5]] to <2 x i32>
// CHECK:   [[TMP11:%.*]] = bitcast <8 x i8> [[TMP7]] to <2 x i32>
// CHECK:   [[TMP12:%.*]] = bitcast <8 x i8> [[TMP9]] to <2 x i32>
// CHECK:   [[VLD3_LANE:%.*]] = call { <2 x i32>, <2 x i32>, <2 x i32> } @llvm.aarch64.neon.ld3lane.v2i32.p0i8(<2 x i32> [[TMP10]], <2 x i32> [[TMP11]], <2 x i32> [[TMP12]], i64 1, i8* [[TMP3]])
// CHECK:   [[TMP13:%.*]] = bitcast i8* [[TMP2]] to { <2 x i32>, <2 x i32>, <2 x i32> }*
// CHECK:   store { <2 x i32>, <2 x i32>, <2 x i32> } [[VLD3_LANE]], { <2 x i32>, <2 x i32>, <2 x i32> }* [[TMP13]]
// CHECK:   [[TMP14:%.*]] = bitcast %struct.uint32x2x3_t* [[RETVAL]] to i8*
// CHECK:   [[TMP15:%.*]] = bitcast %struct.uint32x2x3_t* [[__RET]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[TMP14]], i8* align 8 [[TMP15]], i64 24, i1 false)
// CHECK:   [[TMP16:%.*]] = load %struct.uint32x2x3_t, %struct.uint32x2x3_t* [[RETVAL]], align 8
// CHECK:   ret %struct.uint32x2x3_t [[TMP16]]
uint32x2x3_t test_vld3_lane_u32(uint32_t  *a, uint32x2x3_t b) {
  return vld3_lane_u32(a, b, 1);
}

// CHECK-LABEL: define{{.*}} %struct.uint64x1x3_t @test_vld3_lane_u64(i64* noundef %a, [3 x <1 x i64>] %b.coerce) #2 {
// CHECK:   [[RETVAL:%.*]] = alloca %struct.uint64x1x3_t, align 8
// CHECK:   [[B:%.*]] = alloca %struct.uint64x1x3_t, align 8
// CHECK:   [[__RET:%.*]] = alloca %struct.uint64x1x3_t, align 8
// CHECK:   [[__S1:%.*]] = alloca %struct.uint64x1x3_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.uint64x1x3_t, %struct.uint64x1x3_t* [[B]], i32 0, i32 0
// CHECK:   store [3 x <1 x i64>] [[B]].coerce, [3 x <1 x i64>]* [[COERCE_DIVE]], align 8
// CHECK:   [[TMP0:%.*]] = bitcast %struct.uint64x1x3_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.uint64x1x3_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[TMP0]], i8* align 8 [[TMP1]], i64 24, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast %struct.uint64x1x3_t* [[__RET]] to i8*
// CHECK:   [[TMP3:%.*]] = bitcast i64* %a to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.uint64x1x3_t, %struct.uint64x1x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <1 x i64>], [3 x <1 x i64>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP4:%.*]] = load <1 x i64>, <1 x i64>* [[ARRAYIDX]], align 8
// CHECK:   [[TMP5:%.*]] = bitcast <1 x i64> [[TMP4]] to <8 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.uint64x1x3_t, %struct.uint64x1x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <1 x i64>], [3 x <1 x i64>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP6:%.*]] = load <1 x i64>, <1 x i64>* [[ARRAYIDX2]], align 8
// CHECK:   [[TMP7:%.*]] = bitcast <1 x i64> [[TMP6]] to <8 x i8>
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds %struct.uint64x1x3_t, %struct.uint64x1x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <1 x i64>], [3 x <1 x i64>]* [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP8:%.*]] = load <1 x i64>, <1 x i64>* [[ARRAYIDX4]], align 8
// CHECK:   [[TMP9:%.*]] = bitcast <1 x i64> [[TMP8]] to <8 x i8>
// CHECK:   [[TMP10:%.*]] = bitcast <8 x i8> [[TMP5]] to <1 x i64>
// CHECK:   [[TMP11:%.*]] = bitcast <8 x i8> [[TMP7]] to <1 x i64>
// CHECK:   [[TMP12:%.*]] = bitcast <8 x i8> [[TMP9]] to <1 x i64>
// CHECK:   [[VLD3_LANE:%.*]] = call { <1 x i64>, <1 x i64>, <1 x i64> } @llvm.aarch64.neon.ld3lane.v1i64.p0i8(<1 x i64> [[TMP10]], <1 x i64> [[TMP11]], <1 x i64> [[TMP12]], i64 0, i8* [[TMP3]])
// CHECK:   [[TMP13:%.*]] = bitcast i8* [[TMP2]] to { <1 x i64>, <1 x i64>, <1 x i64> }*
// CHECK:   store { <1 x i64>, <1 x i64>, <1 x i64> } [[VLD3_LANE]], { <1 x i64>, <1 x i64>, <1 x i64> }* [[TMP13]]
// CHECK:   [[TMP14:%.*]] = bitcast %struct.uint64x1x3_t* [[RETVAL]] to i8*
// CHECK:   [[TMP15:%.*]] = bitcast %struct.uint64x1x3_t* [[__RET]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[TMP14]], i8* align 8 [[TMP15]], i64 24, i1 false)
// CHECK:   [[TMP16:%.*]] = load %struct.uint64x1x3_t, %struct.uint64x1x3_t* [[RETVAL]], align 8
// CHECK:   ret %struct.uint64x1x3_t [[TMP16]]
uint64x1x3_t test_vld3_lane_u64(uint64_t  *a, uint64x1x3_t b) {
  return vld3_lane_u64(a, b, 0);
}

// CHECK-LABEL: define{{.*}} %struct.int8x8x3_t @test_vld3_lane_s8(i8* noundef %a, [3 x <8 x i8>] %b.coerce) #2 {
// CHECK:   [[RETVAL:%.*]] = alloca %struct.int8x8x3_t, align 8
// CHECK:   [[B:%.*]] = alloca %struct.int8x8x3_t, align 8
// CHECK:   [[__RET:%.*]] = alloca %struct.int8x8x3_t, align 8
// CHECK:   [[__S1:%.*]] = alloca %struct.int8x8x3_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.int8x8x3_t, %struct.int8x8x3_t* [[B]], i32 0, i32 0
// CHECK:   store [3 x <8 x i8>] [[B]].coerce, [3 x <8 x i8>]* [[COERCE_DIVE]], align 8
// CHECK:   [[TMP0:%.*]] = bitcast %struct.int8x8x3_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.int8x8x3_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[TMP0]], i8* align 8 [[TMP1]], i64 24, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast %struct.int8x8x3_t* [[__RET]] to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.int8x8x3_t, %struct.int8x8x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <8 x i8>], [3 x <8 x i8>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <8 x i8>, <8 x i8>* [[ARRAYIDX]], align 8
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.int8x8x3_t, %struct.int8x8x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <8 x i8>], [3 x <8 x i8>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP4:%.*]] = load <8 x i8>, <8 x i8>* [[ARRAYIDX2]], align 8
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds %struct.int8x8x3_t, %struct.int8x8x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <8 x i8>], [3 x <8 x i8>]* [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP5:%.*]] = load <8 x i8>, <8 x i8>* [[ARRAYIDX4]], align 8
// CHECK:   [[VLD3_LANE:%.*]] = call { <8 x i8>, <8 x i8>, <8 x i8> } @llvm.aarch64.neon.ld3lane.v8i8.p0i8(<8 x i8> [[TMP3]], <8 x i8> [[TMP4]], <8 x i8> [[TMP5]], i64 7, i8* %a)
// CHECK:   [[TMP6:%.*]] = bitcast i8* [[TMP2]] to { <8 x i8>, <8 x i8>, <8 x i8> }*
// CHECK:   store { <8 x i8>, <8 x i8>, <8 x i8> } [[VLD3_LANE]], { <8 x i8>, <8 x i8>, <8 x i8> }* [[TMP6]]
// CHECK:   [[TMP7:%.*]] = bitcast %struct.int8x8x3_t* [[RETVAL]] to i8*
// CHECK:   [[TMP8:%.*]] = bitcast %struct.int8x8x3_t* [[__RET]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[TMP7]], i8* align 8 [[TMP8]], i64 24, i1 false)
// CHECK:   [[TMP9:%.*]] = load %struct.int8x8x3_t, %struct.int8x8x3_t* [[RETVAL]], align 8
// CHECK:   ret %struct.int8x8x3_t [[TMP9]]
int8x8x3_t test_vld3_lane_s8(int8_t  *a, int8x8x3_t b) {
  return vld3_lane_s8(a, b, 7);
}

// CHECK-LABEL: define{{.*}} %struct.int16x4x3_t @test_vld3_lane_s16(i16* noundef %a, [3 x <4 x i16>] %b.coerce) #2 {
// CHECK:   [[RETVAL:%.*]] = alloca %struct.int16x4x3_t, align 8
// CHECK:   [[B:%.*]] = alloca %struct.int16x4x3_t, align 8
// CHECK:   [[__RET:%.*]] = alloca %struct.int16x4x3_t, align 8
// CHECK:   [[__S1:%.*]] = alloca %struct.int16x4x3_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.int16x4x3_t, %struct.int16x4x3_t* [[B]], i32 0, i32 0
// CHECK:   store [3 x <4 x i16>] [[B]].coerce, [3 x <4 x i16>]* [[COERCE_DIVE]], align 8
// CHECK:   [[TMP0:%.*]] = bitcast %struct.int16x4x3_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.int16x4x3_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[TMP0]], i8* align 8 [[TMP1]], i64 24, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast %struct.int16x4x3_t* [[__RET]] to i8*
// CHECK:   [[TMP3:%.*]] = bitcast i16* %a to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.int16x4x3_t, %struct.int16x4x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <4 x i16>], [3 x <4 x i16>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP4:%.*]] = load <4 x i16>, <4 x i16>* [[ARRAYIDX]], align 8
// CHECK:   [[TMP5:%.*]] = bitcast <4 x i16> [[TMP4]] to <8 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.int16x4x3_t, %struct.int16x4x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <4 x i16>], [3 x <4 x i16>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP6:%.*]] = load <4 x i16>, <4 x i16>* [[ARRAYIDX2]], align 8
// CHECK:   [[TMP7:%.*]] = bitcast <4 x i16> [[TMP6]] to <8 x i8>
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds %struct.int16x4x3_t, %struct.int16x4x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <4 x i16>], [3 x <4 x i16>]* [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP8:%.*]] = load <4 x i16>, <4 x i16>* [[ARRAYIDX4]], align 8
// CHECK:   [[TMP9:%.*]] = bitcast <4 x i16> [[TMP8]] to <8 x i8>
// CHECK:   [[TMP10:%.*]] = bitcast <8 x i8> [[TMP5]] to <4 x i16>
// CHECK:   [[TMP11:%.*]] = bitcast <8 x i8> [[TMP7]] to <4 x i16>
// CHECK:   [[TMP12:%.*]] = bitcast <8 x i8> [[TMP9]] to <4 x i16>
// CHECK:   [[VLD3_LANE:%.*]] = call { <4 x i16>, <4 x i16>, <4 x i16> } @llvm.aarch64.neon.ld3lane.v4i16.p0i8(<4 x i16> [[TMP10]], <4 x i16> [[TMP11]], <4 x i16> [[TMP12]], i64 3, i8* [[TMP3]])
// CHECK:   [[TMP13:%.*]] = bitcast i8* [[TMP2]] to { <4 x i16>, <4 x i16>, <4 x i16> }*
// CHECK:   store { <4 x i16>, <4 x i16>, <4 x i16> } [[VLD3_LANE]], { <4 x i16>, <4 x i16>, <4 x i16> }* [[TMP13]]
// CHECK:   [[TMP14:%.*]] = bitcast %struct.int16x4x3_t* [[RETVAL]] to i8*
// CHECK:   [[TMP15:%.*]] = bitcast %struct.int16x4x3_t* [[__RET]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[TMP14]], i8* align 8 [[TMP15]], i64 24, i1 false)
// CHECK:   [[TMP16:%.*]] = load %struct.int16x4x3_t, %struct.int16x4x3_t* [[RETVAL]], align 8
// CHECK:   ret %struct.int16x4x3_t [[TMP16]]
int16x4x3_t test_vld3_lane_s16(int16_t  *a, int16x4x3_t b) {
  return vld3_lane_s16(a, b, 3);
}

// CHECK-LABEL: define{{.*}} %struct.int32x2x3_t @test_vld3_lane_s32(i32* noundef %a, [3 x <2 x i32>] %b.coerce) #2 {
// CHECK:   [[RETVAL:%.*]] = alloca %struct.int32x2x3_t, align 8
// CHECK:   [[B:%.*]] = alloca %struct.int32x2x3_t, align 8
// CHECK:   [[__RET:%.*]] = alloca %struct.int32x2x3_t, align 8
// CHECK:   [[__S1:%.*]] = alloca %struct.int32x2x3_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.int32x2x3_t, %struct.int32x2x3_t* [[B]], i32 0, i32 0
// CHECK:   store [3 x <2 x i32>] [[B]].coerce, [3 x <2 x i32>]* [[COERCE_DIVE]], align 8
// CHECK:   [[TMP0:%.*]] = bitcast %struct.int32x2x3_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.int32x2x3_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[TMP0]], i8* align 8 [[TMP1]], i64 24, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast %struct.int32x2x3_t* [[__RET]] to i8*
// CHECK:   [[TMP3:%.*]] = bitcast i32* %a to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.int32x2x3_t, %struct.int32x2x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <2 x i32>], [3 x <2 x i32>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP4:%.*]] = load <2 x i32>, <2 x i32>* [[ARRAYIDX]], align 8
// CHECK:   [[TMP5:%.*]] = bitcast <2 x i32> [[TMP4]] to <8 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.int32x2x3_t, %struct.int32x2x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <2 x i32>], [3 x <2 x i32>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP6:%.*]] = load <2 x i32>, <2 x i32>* [[ARRAYIDX2]], align 8
// CHECK:   [[TMP7:%.*]] = bitcast <2 x i32> [[TMP6]] to <8 x i8>
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds %struct.int32x2x3_t, %struct.int32x2x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <2 x i32>], [3 x <2 x i32>]* [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP8:%.*]] = load <2 x i32>, <2 x i32>* [[ARRAYIDX4]], align 8
// CHECK:   [[TMP9:%.*]] = bitcast <2 x i32> [[TMP8]] to <8 x i8>
// CHECK:   [[TMP10:%.*]] = bitcast <8 x i8> [[TMP5]] to <2 x i32>
// CHECK:   [[TMP11:%.*]] = bitcast <8 x i8> [[TMP7]] to <2 x i32>
// CHECK:   [[TMP12:%.*]] = bitcast <8 x i8> [[TMP9]] to <2 x i32>
// CHECK:   [[VLD3_LANE:%.*]] = call { <2 x i32>, <2 x i32>, <2 x i32> } @llvm.aarch64.neon.ld3lane.v2i32.p0i8(<2 x i32> [[TMP10]], <2 x i32> [[TMP11]], <2 x i32> [[TMP12]], i64 1, i8* [[TMP3]])
// CHECK:   [[TMP13:%.*]] = bitcast i8* [[TMP2]] to { <2 x i32>, <2 x i32>, <2 x i32> }*
// CHECK:   store { <2 x i32>, <2 x i32>, <2 x i32> } [[VLD3_LANE]], { <2 x i32>, <2 x i32>, <2 x i32> }* [[TMP13]]
// CHECK:   [[TMP14:%.*]] = bitcast %struct.int32x2x3_t* [[RETVAL]] to i8*
// CHECK:   [[TMP15:%.*]] = bitcast %struct.int32x2x3_t* [[__RET]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[TMP14]], i8* align 8 [[TMP15]], i64 24, i1 false)
// CHECK:   [[TMP16:%.*]] = load %struct.int32x2x3_t, %struct.int32x2x3_t* [[RETVAL]], align 8
// CHECK:   ret %struct.int32x2x3_t [[TMP16]]
int32x2x3_t test_vld3_lane_s32(int32_t  *a, int32x2x3_t b) {
  return vld3_lane_s32(a, b, 1);
}

// CHECK-LABEL: define{{.*}} %struct.int64x1x3_t @test_vld3_lane_s64(i64* noundef %a, [3 x <1 x i64>] %b.coerce) #2 {
// CHECK:   [[RETVAL:%.*]] = alloca %struct.int64x1x3_t, align 8
// CHECK:   [[B:%.*]] = alloca %struct.int64x1x3_t, align 8
// CHECK:   [[__RET:%.*]] = alloca %struct.int64x1x3_t, align 8
// CHECK:   [[__S1:%.*]] = alloca %struct.int64x1x3_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.int64x1x3_t, %struct.int64x1x3_t* [[B]], i32 0, i32 0
// CHECK:   store [3 x <1 x i64>] [[B]].coerce, [3 x <1 x i64>]* [[COERCE_DIVE]], align 8
// CHECK:   [[TMP0:%.*]] = bitcast %struct.int64x1x3_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.int64x1x3_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[TMP0]], i8* align 8 [[TMP1]], i64 24, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast %struct.int64x1x3_t* [[__RET]] to i8*
// CHECK:   [[TMP3:%.*]] = bitcast i64* %a to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.int64x1x3_t, %struct.int64x1x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <1 x i64>], [3 x <1 x i64>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP4:%.*]] = load <1 x i64>, <1 x i64>* [[ARRAYIDX]], align 8
// CHECK:   [[TMP5:%.*]] = bitcast <1 x i64> [[TMP4]] to <8 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.int64x1x3_t, %struct.int64x1x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <1 x i64>], [3 x <1 x i64>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP6:%.*]] = load <1 x i64>, <1 x i64>* [[ARRAYIDX2]], align 8
// CHECK:   [[TMP7:%.*]] = bitcast <1 x i64> [[TMP6]] to <8 x i8>
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds %struct.int64x1x3_t, %struct.int64x1x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <1 x i64>], [3 x <1 x i64>]* [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP8:%.*]] = load <1 x i64>, <1 x i64>* [[ARRAYIDX4]], align 8
// CHECK:   [[TMP9:%.*]] = bitcast <1 x i64> [[TMP8]] to <8 x i8>
// CHECK:   [[TMP10:%.*]] = bitcast <8 x i8> [[TMP5]] to <1 x i64>
// CHECK:   [[TMP11:%.*]] = bitcast <8 x i8> [[TMP7]] to <1 x i64>
// CHECK:   [[TMP12:%.*]] = bitcast <8 x i8> [[TMP9]] to <1 x i64>
// CHECK:   [[VLD3_LANE:%.*]] = call { <1 x i64>, <1 x i64>, <1 x i64> } @llvm.aarch64.neon.ld3lane.v1i64.p0i8(<1 x i64> [[TMP10]], <1 x i64> [[TMP11]], <1 x i64> [[TMP12]], i64 0, i8* [[TMP3]])
// CHECK:   [[TMP13:%.*]] = bitcast i8* [[TMP2]] to { <1 x i64>, <1 x i64>, <1 x i64> }*
// CHECK:   store { <1 x i64>, <1 x i64>, <1 x i64> } [[VLD3_LANE]], { <1 x i64>, <1 x i64>, <1 x i64> }* [[TMP13]]
// CHECK:   [[TMP14:%.*]] = bitcast %struct.int64x1x3_t* [[RETVAL]] to i8*
// CHECK:   [[TMP15:%.*]] = bitcast %struct.int64x1x3_t* [[__RET]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[TMP14]], i8* align 8 [[TMP15]], i64 24, i1 false)
// CHECK:   [[TMP16:%.*]] = load %struct.int64x1x3_t, %struct.int64x1x3_t* [[RETVAL]], align 8
// CHECK:   ret %struct.int64x1x3_t [[TMP16]]
int64x1x3_t test_vld3_lane_s64(int64_t  *a, int64x1x3_t b) {
  return vld3_lane_s64(a, b, 0);
}

// CHECK-LABEL: define{{.*}} %struct.float16x4x3_t @test_vld3_lane_f16(half* noundef %a, [3 x <4 x half>] %b.coerce) #2 {
// CHECK:   [[RETVAL:%.*]] = alloca %struct.float16x4x3_t, align 8
// CHECK:   [[B:%.*]] = alloca %struct.float16x4x3_t, align 8
// CHECK:   [[__RET:%.*]] = alloca %struct.float16x4x3_t, align 8
// CHECK:   [[__S1:%.*]] = alloca %struct.float16x4x3_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.float16x4x3_t, %struct.float16x4x3_t* [[B]], i32 0, i32 0
// CHECK:   store [3 x <4 x half>] [[B]].coerce, [3 x <4 x half>]* [[COERCE_DIVE]], align 8
// CHECK:   [[TMP0:%.*]] = bitcast %struct.float16x4x3_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.float16x4x3_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[TMP0]], i8* align 8 [[TMP1]], i64 24, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast %struct.float16x4x3_t* [[__RET]] to i8*
// CHECK:   [[TMP3:%.*]] = bitcast half* %a to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.float16x4x3_t, %struct.float16x4x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <4 x half>], [3 x <4 x half>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP4:%.*]] = load <4 x half>, <4 x half>* [[ARRAYIDX]], align 8
// CHECK:   [[TMP5:%.*]] = bitcast <4 x half> [[TMP4]] to <8 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.float16x4x3_t, %struct.float16x4x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <4 x half>], [3 x <4 x half>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP6:%.*]] = load <4 x half>, <4 x half>* [[ARRAYIDX2]], align 8
// CHECK:   [[TMP7:%.*]] = bitcast <4 x half> [[TMP6]] to <8 x i8>
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds %struct.float16x4x3_t, %struct.float16x4x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <4 x half>], [3 x <4 x half>]* [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP8:%.*]] = load <4 x half>, <4 x half>* [[ARRAYIDX4]], align 8
// CHECK:   [[TMP9:%.*]] = bitcast <4 x half> [[TMP8]] to <8 x i8>
// CHECK:   [[TMP10:%.*]] = bitcast <8 x i8> [[TMP5]] to <4 x half>
// CHECK:   [[TMP11:%.*]] = bitcast <8 x i8> [[TMP7]] to <4 x half>
// CHECK:   [[TMP12:%.*]] = bitcast <8 x i8> [[TMP9]] to <4 x half>
// CHECK:   [[VLD3_LANE:%.*]] = call { <4 x half>, <4 x half>, <4 x half> } @llvm.aarch64.neon.ld3lane.v4f16.p0i8(<4 x half> [[TMP10]], <4 x half> [[TMP11]], <4 x half> [[TMP12]], i64 3, i8* [[TMP3]])
// CHECK:   [[TMP13:%.*]] = bitcast i8* [[TMP2]] to { <4 x half>, <4 x half>, <4 x half> }*
// CHECK:   store { <4 x half>, <4 x half>, <4 x half> } [[VLD3_LANE]], { <4 x half>, <4 x half>, <4 x half> }* [[TMP13]]
// CHECK:   [[TMP14:%.*]] = bitcast %struct.float16x4x3_t* [[RETVAL]] to i8*
// CHECK:   [[TMP15:%.*]] = bitcast %struct.float16x4x3_t* [[__RET]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[TMP14]], i8* align 8 [[TMP15]], i64 24, i1 false)
// CHECK:   [[TMP16:%.*]] = load %struct.float16x4x3_t, %struct.float16x4x3_t* [[RETVAL]], align 8
// CHECK:   ret %struct.float16x4x3_t [[TMP16]]
float16x4x3_t test_vld3_lane_f16(float16_t  *a, float16x4x3_t b) {
  return vld3_lane_f16(a, b, 3);
}

// CHECK-LABEL: define{{.*}} %struct.float32x2x3_t @test_vld3_lane_f32(float* noundef %a, [3 x <2 x float>] %b.coerce) #2 {
// CHECK:   [[RETVAL:%.*]] = alloca %struct.float32x2x3_t, align 8
// CHECK:   [[B:%.*]] = alloca %struct.float32x2x3_t, align 8
// CHECK:   [[__RET:%.*]] = alloca %struct.float32x2x3_t, align 8
// CHECK:   [[__S1:%.*]] = alloca %struct.float32x2x3_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.float32x2x3_t, %struct.float32x2x3_t* [[B]], i32 0, i32 0
// CHECK:   store [3 x <2 x float>] [[B]].coerce, [3 x <2 x float>]* [[COERCE_DIVE]], align 8
// CHECK:   [[TMP0:%.*]] = bitcast %struct.float32x2x3_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.float32x2x3_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[TMP0]], i8* align 8 [[TMP1]], i64 24, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast %struct.float32x2x3_t* [[__RET]] to i8*
// CHECK:   [[TMP3:%.*]] = bitcast float* %a to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.float32x2x3_t, %struct.float32x2x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <2 x float>], [3 x <2 x float>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP4:%.*]] = load <2 x float>, <2 x float>* [[ARRAYIDX]], align 8
// CHECK:   [[TMP5:%.*]] = bitcast <2 x float> [[TMP4]] to <8 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.float32x2x3_t, %struct.float32x2x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <2 x float>], [3 x <2 x float>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP6:%.*]] = load <2 x float>, <2 x float>* [[ARRAYIDX2]], align 8
// CHECK:   [[TMP7:%.*]] = bitcast <2 x float> [[TMP6]] to <8 x i8>
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds %struct.float32x2x3_t, %struct.float32x2x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <2 x float>], [3 x <2 x float>]* [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP8:%.*]] = load <2 x float>, <2 x float>* [[ARRAYIDX4]], align 8
// CHECK:   [[TMP9:%.*]] = bitcast <2 x float> [[TMP8]] to <8 x i8>
// CHECK:   [[TMP10:%.*]] = bitcast <8 x i8> [[TMP5]] to <2 x float>
// CHECK:   [[TMP11:%.*]] = bitcast <8 x i8> [[TMP7]] to <2 x float>
// CHECK:   [[TMP12:%.*]] = bitcast <8 x i8> [[TMP9]] to <2 x float>
// CHECK:   [[VLD3_LANE:%.*]] = call { <2 x float>, <2 x float>, <2 x float> } @llvm.aarch64.neon.ld3lane.v2f32.p0i8(<2 x float> [[TMP10]], <2 x float> [[TMP11]], <2 x float> [[TMP12]], i64 1, i8* [[TMP3]])
// CHECK:   [[TMP13:%.*]] = bitcast i8* [[TMP2]] to { <2 x float>, <2 x float>, <2 x float> }*
// CHECK:   store { <2 x float>, <2 x float>, <2 x float> } [[VLD3_LANE]], { <2 x float>, <2 x float>, <2 x float> }* [[TMP13]]
// CHECK:   [[TMP14:%.*]] = bitcast %struct.float32x2x3_t* [[RETVAL]] to i8*
// CHECK:   [[TMP15:%.*]] = bitcast %struct.float32x2x3_t* [[__RET]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[TMP14]], i8* align 8 [[TMP15]], i64 24, i1 false)
// CHECK:   [[TMP16:%.*]] = load %struct.float32x2x3_t, %struct.float32x2x3_t* [[RETVAL]], align 8
// CHECK:   ret %struct.float32x2x3_t [[TMP16]]
float32x2x3_t test_vld3_lane_f32(float32_t  *a, float32x2x3_t b) {
  return vld3_lane_f32(a, b, 1);
}

// CHECK-LABEL: define{{.*}} %struct.float64x1x3_t @test_vld3_lane_f64(double* noundef %a, [3 x <1 x double>] %b.coerce) #2 {
// CHECK:   [[RETVAL:%.*]] = alloca %struct.float64x1x3_t, align 8
// CHECK:   [[B:%.*]] = alloca %struct.float64x1x3_t, align 8
// CHECK:   [[__RET:%.*]] = alloca %struct.float64x1x3_t, align 8
// CHECK:   [[__S1:%.*]] = alloca %struct.float64x1x3_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.float64x1x3_t, %struct.float64x1x3_t* [[B]], i32 0, i32 0
// CHECK:   store [3 x <1 x double>] [[B]].coerce, [3 x <1 x double>]* [[COERCE_DIVE]], align 8
// CHECK:   [[TMP0:%.*]] = bitcast %struct.float64x1x3_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.float64x1x3_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[TMP0]], i8* align 8 [[TMP1]], i64 24, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast %struct.float64x1x3_t* [[__RET]] to i8*
// CHECK:   [[TMP3:%.*]] = bitcast double* %a to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.float64x1x3_t, %struct.float64x1x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <1 x double>], [3 x <1 x double>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP4:%.*]] = load <1 x double>, <1 x double>* [[ARRAYIDX]], align 8
// CHECK:   [[TMP5:%.*]] = bitcast <1 x double> [[TMP4]] to <8 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.float64x1x3_t, %struct.float64x1x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <1 x double>], [3 x <1 x double>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP6:%.*]] = load <1 x double>, <1 x double>* [[ARRAYIDX2]], align 8
// CHECK:   [[TMP7:%.*]] = bitcast <1 x double> [[TMP6]] to <8 x i8>
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds %struct.float64x1x3_t, %struct.float64x1x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <1 x double>], [3 x <1 x double>]* [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP8:%.*]] = load <1 x double>, <1 x double>* [[ARRAYIDX4]], align 8
// CHECK:   [[TMP9:%.*]] = bitcast <1 x double> [[TMP8]] to <8 x i8>
// CHECK:   [[TMP10:%.*]] = bitcast <8 x i8> [[TMP5]] to <1 x double>
// CHECK:   [[TMP11:%.*]] = bitcast <8 x i8> [[TMP7]] to <1 x double>
// CHECK:   [[TMP12:%.*]] = bitcast <8 x i8> [[TMP9]] to <1 x double>
// CHECK:   [[VLD3_LANE:%.*]] = call { <1 x double>, <1 x double>, <1 x double> } @llvm.aarch64.neon.ld3lane.v1f64.p0i8(<1 x double> [[TMP10]], <1 x double> [[TMP11]], <1 x double> [[TMP12]], i64 0, i8* [[TMP3]])
// CHECK:   [[TMP13:%.*]] = bitcast i8* [[TMP2]] to { <1 x double>, <1 x double>, <1 x double> }*
// CHECK:   store { <1 x double>, <1 x double>, <1 x double> } [[VLD3_LANE]], { <1 x double>, <1 x double>, <1 x double> }* [[TMP13]]
// CHECK:   [[TMP14:%.*]] = bitcast %struct.float64x1x3_t* [[RETVAL]] to i8*
// CHECK:   [[TMP15:%.*]] = bitcast %struct.float64x1x3_t* [[__RET]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[TMP14]], i8* align 8 [[TMP15]], i64 24, i1 false)
// CHECK:   [[TMP16:%.*]] = load %struct.float64x1x3_t, %struct.float64x1x3_t* [[RETVAL]], align 8
// CHECK:   ret %struct.float64x1x3_t [[TMP16]]
float64x1x3_t test_vld3_lane_f64(float64_t  *a, float64x1x3_t b) {
  return vld3_lane_f64(a, b, 0);
}

// CHECK-LABEL: define{{.*}} %struct.poly8x8x3_t @test_vld3_lane_p8(i8* noundef %a, [3 x <8 x i8>] %b.coerce) #2 {
// CHECK:   [[RETVAL:%.*]] = alloca %struct.poly8x8x3_t, align 8
// CHECK:   [[B:%.*]] = alloca %struct.poly8x8x3_t, align 8
// CHECK:   [[__RET:%.*]] = alloca %struct.poly8x8x3_t, align 8
// CHECK:   [[__S1:%.*]] = alloca %struct.poly8x8x3_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.poly8x8x3_t, %struct.poly8x8x3_t* [[B]], i32 0, i32 0
// CHECK:   store [3 x <8 x i8>] [[B]].coerce, [3 x <8 x i8>]* [[COERCE_DIVE]], align 8
// CHECK:   [[TMP0:%.*]] = bitcast %struct.poly8x8x3_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.poly8x8x3_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[TMP0]], i8* align 8 [[TMP1]], i64 24, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast %struct.poly8x8x3_t* [[__RET]] to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.poly8x8x3_t, %struct.poly8x8x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <8 x i8>], [3 x <8 x i8>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <8 x i8>, <8 x i8>* [[ARRAYIDX]], align 8
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.poly8x8x3_t, %struct.poly8x8x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <8 x i8>], [3 x <8 x i8>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP4:%.*]] = load <8 x i8>, <8 x i8>* [[ARRAYIDX2]], align 8
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds %struct.poly8x8x3_t, %struct.poly8x8x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <8 x i8>], [3 x <8 x i8>]* [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP5:%.*]] = load <8 x i8>, <8 x i8>* [[ARRAYIDX4]], align 8
// CHECK:   [[VLD3_LANE:%.*]] = call { <8 x i8>, <8 x i8>, <8 x i8> } @llvm.aarch64.neon.ld3lane.v8i8.p0i8(<8 x i8> [[TMP3]], <8 x i8> [[TMP4]], <8 x i8> [[TMP5]], i64 7, i8* %a)
// CHECK:   [[TMP6:%.*]] = bitcast i8* [[TMP2]] to { <8 x i8>, <8 x i8>, <8 x i8> }*
// CHECK:   store { <8 x i8>, <8 x i8>, <8 x i8> } [[VLD3_LANE]], { <8 x i8>, <8 x i8>, <8 x i8> }* [[TMP6]]
// CHECK:   [[TMP7:%.*]] = bitcast %struct.poly8x8x3_t* [[RETVAL]] to i8*
// CHECK:   [[TMP8:%.*]] = bitcast %struct.poly8x8x3_t* [[__RET]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[TMP7]], i8* align 8 [[TMP8]], i64 24, i1 false)
// CHECK:   [[TMP9:%.*]] = load %struct.poly8x8x3_t, %struct.poly8x8x3_t* [[RETVAL]], align 8
// CHECK:   ret %struct.poly8x8x3_t [[TMP9]]
poly8x8x3_t test_vld3_lane_p8(poly8_t  *a, poly8x8x3_t b) {
  return vld3_lane_p8(a, b, 7);
}

// CHECK-LABEL: define{{.*}} %struct.poly16x4x3_t @test_vld3_lane_p16(i16* noundef %a, [3 x <4 x i16>] %b.coerce) #2 {
// CHECK:   [[RETVAL:%.*]] = alloca %struct.poly16x4x3_t, align 8
// CHECK:   [[B:%.*]] = alloca %struct.poly16x4x3_t, align 8
// CHECK:   [[__RET:%.*]] = alloca %struct.poly16x4x3_t, align 8
// CHECK:   [[__S1:%.*]] = alloca %struct.poly16x4x3_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.poly16x4x3_t, %struct.poly16x4x3_t* [[B]], i32 0, i32 0
// CHECK:   store [3 x <4 x i16>] [[B]].coerce, [3 x <4 x i16>]* [[COERCE_DIVE]], align 8
// CHECK:   [[TMP0:%.*]] = bitcast %struct.poly16x4x3_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.poly16x4x3_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[TMP0]], i8* align 8 [[TMP1]], i64 24, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast %struct.poly16x4x3_t* [[__RET]] to i8*
// CHECK:   [[TMP3:%.*]] = bitcast i16* %a to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.poly16x4x3_t, %struct.poly16x4x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <4 x i16>], [3 x <4 x i16>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP4:%.*]] = load <4 x i16>, <4 x i16>* [[ARRAYIDX]], align 8
// CHECK:   [[TMP5:%.*]] = bitcast <4 x i16> [[TMP4]] to <8 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.poly16x4x3_t, %struct.poly16x4x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <4 x i16>], [3 x <4 x i16>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP6:%.*]] = load <4 x i16>, <4 x i16>* [[ARRAYIDX2]], align 8
// CHECK:   [[TMP7:%.*]] = bitcast <4 x i16> [[TMP6]] to <8 x i8>
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds %struct.poly16x4x3_t, %struct.poly16x4x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <4 x i16>], [3 x <4 x i16>]* [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP8:%.*]] = load <4 x i16>, <4 x i16>* [[ARRAYIDX4]], align 8
// CHECK:   [[TMP9:%.*]] = bitcast <4 x i16> [[TMP8]] to <8 x i8>
// CHECK:   [[TMP10:%.*]] = bitcast <8 x i8> [[TMP5]] to <4 x i16>
// CHECK:   [[TMP11:%.*]] = bitcast <8 x i8> [[TMP7]] to <4 x i16>
// CHECK:   [[TMP12:%.*]] = bitcast <8 x i8> [[TMP9]] to <4 x i16>
// CHECK:   [[VLD3_LANE:%.*]] = call { <4 x i16>, <4 x i16>, <4 x i16> } @llvm.aarch64.neon.ld3lane.v4i16.p0i8(<4 x i16> [[TMP10]], <4 x i16> [[TMP11]], <4 x i16> [[TMP12]], i64 3, i8* [[TMP3]])
// CHECK:   [[TMP13:%.*]] = bitcast i8* [[TMP2]] to { <4 x i16>, <4 x i16>, <4 x i16> }*
// CHECK:   store { <4 x i16>, <4 x i16>, <4 x i16> } [[VLD3_LANE]], { <4 x i16>, <4 x i16>, <4 x i16> }* [[TMP13]]
// CHECK:   [[TMP14:%.*]] = bitcast %struct.poly16x4x3_t* [[RETVAL]] to i8*
// CHECK:   [[TMP15:%.*]] = bitcast %struct.poly16x4x3_t* [[__RET]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[TMP14]], i8* align 8 [[TMP15]], i64 24, i1 false)
// CHECK:   [[TMP16:%.*]] = load %struct.poly16x4x3_t, %struct.poly16x4x3_t* [[RETVAL]], align 8
// CHECK:   ret %struct.poly16x4x3_t [[TMP16]]
poly16x4x3_t test_vld3_lane_p16(poly16_t  *a, poly16x4x3_t b) {
  return vld3_lane_p16(a, b, 3);
}

// CHECK-LABEL: define{{.*}} %struct.poly64x1x3_t @test_vld3_lane_p64(i64* noundef %a, [3 x <1 x i64>] %b.coerce) #2 {
// CHECK:   [[RETVAL:%.*]] = alloca %struct.poly64x1x3_t, align 8
// CHECK:   [[B:%.*]] = alloca %struct.poly64x1x3_t, align 8
// CHECK:   [[__RET:%.*]] = alloca %struct.poly64x1x3_t, align 8
// CHECK:   [[__S1:%.*]] = alloca %struct.poly64x1x3_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.poly64x1x3_t, %struct.poly64x1x3_t* [[B]], i32 0, i32 0
// CHECK:   store [3 x <1 x i64>] [[B]].coerce, [3 x <1 x i64>]* [[COERCE_DIVE]], align 8
// CHECK:   [[TMP0:%.*]] = bitcast %struct.poly64x1x3_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.poly64x1x3_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[TMP0]], i8* align 8 [[TMP1]], i64 24, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast %struct.poly64x1x3_t* [[__RET]] to i8*
// CHECK:   [[TMP3:%.*]] = bitcast i64* %a to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.poly64x1x3_t, %struct.poly64x1x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <1 x i64>], [3 x <1 x i64>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP4:%.*]] = load <1 x i64>, <1 x i64>* [[ARRAYIDX]], align 8
// CHECK:   [[TMP5:%.*]] = bitcast <1 x i64> [[TMP4]] to <8 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.poly64x1x3_t, %struct.poly64x1x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <1 x i64>], [3 x <1 x i64>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP6:%.*]] = load <1 x i64>, <1 x i64>* [[ARRAYIDX2]], align 8
// CHECK:   [[TMP7:%.*]] = bitcast <1 x i64> [[TMP6]] to <8 x i8>
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds %struct.poly64x1x3_t, %struct.poly64x1x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <1 x i64>], [3 x <1 x i64>]* [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP8:%.*]] = load <1 x i64>, <1 x i64>* [[ARRAYIDX4]], align 8
// CHECK:   [[TMP9:%.*]] = bitcast <1 x i64> [[TMP8]] to <8 x i8>
// CHECK:   [[TMP10:%.*]] = bitcast <8 x i8> [[TMP5]] to <1 x i64>
// CHECK:   [[TMP11:%.*]] = bitcast <8 x i8> [[TMP7]] to <1 x i64>
// CHECK:   [[TMP12:%.*]] = bitcast <8 x i8> [[TMP9]] to <1 x i64>
// CHECK:   [[VLD3_LANE:%.*]] = call { <1 x i64>, <1 x i64>, <1 x i64> } @llvm.aarch64.neon.ld3lane.v1i64.p0i8(<1 x i64> [[TMP10]], <1 x i64> [[TMP11]], <1 x i64> [[TMP12]], i64 0, i8* [[TMP3]])
// CHECK:   [[TMP13:%.*]] = bitcast i8* [[TMP2]] to { <1 x i64>, <1 x i64>, <1 x i64> }*
// CHECK:   store { <1 x i64>, <1 x i64>, <1 x i64> } [[VLD3_LANE]], { <1 x i64>, <1 x i64>, <1 x i64> }* [[TMP13]]
// CHECK:   [[TMP14:%.*]] = bitcast %struct.poly64x1x3_t* [[RETVAL]] to i8*
// CHECK:   [[TMP15:%.*]] = bitcast %struct.poly64x1x3_t* [[__RET]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[TMP14]], i8* align 8 [[TMP15]], i64 24, i1 false)
// CHECK:   [[TMP16:%.*]] = load %struct.poly64x1x3_t, %struct.poly64x1x3_t* [[RETVAL]], align 8
// CHECK:   ret %struct.poly64x1x3_t [[TMP16]]
poly64x1x3_t test_vld3_lane_p64(poly64_t  *a, poly64x1x3_t b) {
  return vld3_lane_p64(a, b, 0);
}

// CHECK-LABEL: define{{.*}} %struct.uint8x16x4_t @test_vld4q_lane_u8(i8* noundef %a, [4 x <16 x i8>] %b.coerce) #2 {
// CHECK:   [[RETVAL:%.*]] = alloca %struct.uint8x16x4_t, align 16
// CHECK:   [[B:%.*]] = alloca %struct.uint8x16x4_t, align 16
// CHECK:   [[__RET:%.*]] = alloca %struct.uint8x16x4_t, align 16
// CHECK:   [[__S1:%.*]] = alloca %struct.uint8x16x4_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.uint8x16x4_t, %struct.uint8x16x4_t* [[B]], i32 0, i32 0
// CHECK:   store [4 x <16 x i8>] [[B]].coerce, [4 x <16 x i8>]* [[COERCE_DIVE]], align 16
// CHECK:   [[TMP0:%.*]] = bitcast %struct.uint8x16x4_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.uint8x16x4_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP0]], i8* align 16 [[TMP1]], i64 64, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast %struct.uint8x16x4_t* [[__RET]] to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.uint8x16x4_t, %struct.uint8x16x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <16 x i8>], [4 x <16 x i8>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX]], align 16
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.uint8x16x4_t, %struct.uint8x16x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <16 x i8>], [4 x <16 x i8>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP4:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX2]], align 16
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds %struct.uint8x16x4_t, %struct.uint8x16x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <16 x i8>], [4 x <16 x i8>]* [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP5:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX4]], align 16
// CHECK:   [[VAL5:%.*]] = getelementptr inbounds %struct.uint8x16x4_t, %struct.uint8x16x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <16 x i8>], [4 x <16 x i8>]* [[VAL5]], i64 0, i64 3
// CHECK:   [[TMP6:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX6]], align 16
// CHECK:   [[VLD4_LANE:%.*]] = call { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } @llvm.aarch64.neon.ld4lane.v16i8.p0i8(<16 x i8> [[TMP3]], <16 x i8> [[TMP4]], <16 x i8> [[TMP5]], <16 x i8> [[TMP6]], i64 15, i8* %a)
// CHECK:   [[TMP7:%.*]] = bitcast i8* [[TMP2]] to { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> }*
// CHECK:   store { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } [[VLD4_LANE]], { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> }* [[TMP7]]
// CHECK:   [[TMP8:%.*]] = bitcast %struct.uint8x16x4_t* [[RETVAL]] to i8*
// CHECK:   [[TMP9:%.*]] = bitcast %struct.uint8x16x4_t* [[__RET]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP8]], i8* align 16 [[TMP9]], i64 64, i1 false)
// CHECK:   [[TMP10:%.*]] = load %struct.uint8x16x4_t, %struct.uint8x16x4_t* [[RETVAL]], align 16
// CHECK:   ret %struct.uint8x16x4_t [[TMP10]]
uint8x16x4_t test_vld4q_lane_u8(uint8_t  *a, uint8x16x4_t b) {
  return vld4q_lane_u8(a, b, 15);
}

// CHECK-LABEL: define{{.*}} %struct.uint16x8x4_t @test_vld4q_lane_u16(i16* noundef %a, [4 x <8 x i16>] %b.coerce) #2 {
// CHECK:   [[RETVAL:%.*]] = alloca %struct.uint16x8x4_t, align 16
// CHECK:   [[B:%.*]] = alloca %struct.uint16x8x4_t, align 16
// CHECK:   [[__RET:%.*]] = alloca %struct.uint16x8x4_t, align 16
// CHECK:   [[__S1:%.*]] = alloca %struct.uint16x8x4_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.uint16x8x4_t, %struct.uint16x8x4_t* [[B]], i32 0, i32 0
// CHECK:   store [4 x <8 x i16>] [[B]].coerce, [4 x <8 x i16>]* [[COERCE_DIVE]], align 16
// CHECK:   [[TMP0:%.*]] = bitcast %struct.uint16x8x4_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.uint16x8x4_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP0]], i8* align 16 [[TMP1]], i64 64, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast %struct.uint16x8x4_t* [[__RET]] to i8*
// CHECK:   [[TMP3:%.*]] = bitcast i16* %a to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.uint16x8x4_t, %struct.uint16x8x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <8 x i16>], [4 x <8 x i16>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP4:%.*]] = load <8 x i16>, <8 x i16>* [[ARRAYIDX]], align 16
// CHECK:   [[TMP5:%.*]] = bitcast <8 x i16> [[TMP4]] to <16 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.uint16x8x4_t, %struct.uint16x8x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <8 x i16>], [4 x <8 x i16>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP6:%.*]] = load <8 x i16>, <8 x i16>* [[ARRAYIDX2]], align 16
// CHECK:   [[TMP7:%.*]] = bitcast <8 x i16> [[TMP6]] to <16 x i8>
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds %struct.uint16x8x4_t, %struct.uint16x8x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <8 x i16>], [4 x <8 x i16>]* [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP8:%.*]] = load <8 x i16>, <8 x i16>* [[ARRAYIDX4]], align 16
// CHECK:   [[TMP9:%.*]] = bitcast <8 x i16> [[TMP8]] to <16 x i8>
// CHECK:   [[VAL5:%.*]] = getelementptr inbounds %struct.uint16x8x4_t, %struct.uint16x8x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <8 x i16>], [4 x <8 x i16>]* [[VAL5]], i64 0, i64 3
// CHECK:   [[TMP10:%.*]] = load <8 x i16>, <8 x i16>* [[ARRAYIDX6]], align 16
// CHECK:   [[TMP11:%.*]] = bitcast <8 x i16> [[TMP10]] to <16 x i8>
// CHECK:   [[TMP12:%.*]] = bitcast <16 x i8> [[TMP5]] to <8 x i16>
// CHECK:   [[TMP13:%.*]] = bitcast <16 x i8> [[TMP7]] to <8 x i16>
// CHECK:   [[TMP14:%.*]] = bitcast <16 x i8> [[TMP9]] to <8 x i16>
// CHECK:   [[TMP15:%.*]] = bitcast <16 x i8> [[TMP11]] to <8 x i16>
// CHECK:   [[VLD4_LANE:%.*]] = call { <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16> } @llvm.aarch64.neon.ld4lane.v8i16.p0i8(<8 x i16> [[TMP12]], <8 x i16> [[TMP13]], <8 x i16> [[TMP14]], <8 x i16> [[TMP15]], i64 7, i8* [[TMP3]])
// CHECK:   [[TMP16:%.*]] = bitcast i8* [[TMP2]] to { <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16> }*
// CHECK:   store { <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16> } [[VLD4_LANE]], { <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16> }* [[TMP16]]
// CHECK:   [[TMP17:%.*]] = bitcast %struct.uint16x8x4_t* [[RETVAL]] to i8*
// CHECK:   [[TMP18:%.*]] = bitcast %struct.uint16x8x4_t* [[__RET]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP17]], i8* align 16 [[TMP18]], i64 64, i1 false)
// CHECK:   [[TMP19:%.*]] = load %struct.uint16x8x4_t, %struct.uint16x8x4_t* [[RETVAL]], align 16
// CHECK:   ret %struct.uint16x8x4_t [[TMP19]]
uint16x8x4_t test_vld4q_lane_u16(uint16_t  *a, uint16x8x4_t b) {
  return vld4q_lane_u16(a, b, 7);
}

// CHECK-LABEL: define{{.*}} %struct.uint32x4x4_t @test_vld4q_lane_u32(i32* noundef %a, [4 x <4 x i32>] %b.coerce) #2 {
// CHECK:   [[RETVAL:%.*]] = alloca %struct.uint32x4x4_t, align 16
// CHECK:   [[B:%.*]] = alloca %struct.uint32x4x4_t, align 16
// CHECK:   [[__RET:%.*]] = alloca %struct.uint32x4x4_t, align 16
// CHECK:   [[__S1:%.*]] = alloca %struct.uint32x4x4_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.uint32x4x4_t, %struct.uint32x4x4_t* [[B]], i32 0, i32 0
// CHECK:   store [4 x <4 x i32>] [[B]].coerce, [4 x <4 x i32>]* [[COERCE_DIVE]], align 16
// CHECK:   [[TMP0:%.*]] = bitcast %struct.uint32x4x4_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.uint32x4x4_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP0]], i8* align 16 [[TMP1]], i64 64, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast %struct.uint32x4x4_t* [[__RET]] to i8*
// CHECK:   [[TMP3:%.*]] = bitcast i32* %a to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.uint32x4x4_t, %struct.uint32x4x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <4 x i32>], [4 x <4 x i32>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP4:%.*]] = load <4 x i32>, <4 x i32>* [[ARRAYIDX]], align 16
// CHECK:   [[TMP5:%.*]] = bitcast <4 x i32> [[TMP4]] to <16 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.uint32x4x4_t, %struct.uint32x4x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <4 x i32>], [4 x <4 x i32>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP6:%.*]] = load <4 x i32>, <4 x i32>* [[ARRAYIDX2]], align 16
// CHECK:   [[TMP7:%.*]] = bitcast <4 x i32> [[TMP6]] to <16 x i8>
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds %struct.uint32x4x4_t, %struct.uint32x4x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <4 x i32>], [4 x <4 x i32>]* [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP8:%.*]] = load <4 x i32>, <4 x i32>* [[ARRAYIDX4]], align 16
// CHECK:   [[TMP9:%.*]] = bitcast <4 x i32> [[TMP8]] to <16 x i8>
// CHECK:   [[VAL5:%.*]] = getelementptr inbounds %struct.uint32x4x4_t, %struct.uint32x4x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <4 x i32>], [4 x <4 x i32>]* [[VAL5]], i64 0, i64 3
// CHECK:   [[TMP10:%.*]] = load <4 x i32>, <4 x i32>* [[ARRAYIDX6]], align 16
// CHECK:   [[TMP11:%.*]] = bitcast <4 x i32> [[TMP10]] to <16 x i8>
// CHECK:   [[TMP12:%.*]] = bitcast <16 x i8> [[TMP5]] to <4 x i32>
// CHECK:   [[TMP13:%.*]] = bitcast <16 x i8> [[TMP7]] to <4 x i32>
// CHECK:   [[TMP14:%.*]] = bitcast <16 x i8> [[TMP9]] to <4 x i32>
// CHECK:   [[TMP15:%.*]] = bitcast <16 x i8> [[TMP11]] to <4 x i32>
// CHECK:   [[VLD4_LANE:%.*]] = call { <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32> } @llvm.aarch64.neon.ld4lane.v4i32.p0i8(<4 x i32> [[TMP12]], <4 x i32> [[TMP13]], <4 x i32> [[TMP14]], <4 x i32> [[TMP15]], i64 3, i8* [[TMP3]])
// CHECK:   [[TMP16:%.*]] = bitcast i8* [[TMP2]] to { <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32> }*
// CHECK:   store { <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32> } [[VLD4_LANE]], { <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32> }* [[TMP16]]
// CHECK:   [[TMP17:%.*]] = bitcast %struct.uint32x4x4_t* [[RETVAL]] to i8*
// CHECK:   [[TMP18:%.*]] = bitcast %struct.uint32x4x4_t* [[__RET]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP17]], i8* align 16 [[TMP18]], i64 64, i1 false)
// CHECK:   [[TMP19:%.*]] = load %struct.uint32x4x4_t, %struct.uint32x4x4_t* [[RETVAL]], align 16
// CHECK:   ret %struct.uint32x4x4_t [[TMP19]]
uint32x4x4_t test_vld4q_lane_u32(uint32_t  *a, uint32x4x4_t b) {
  return vld4q_lane_u32(a, b, 3);
}

// CHECK-LABEL: define{{.*}} %struct.uint64x2x4_t @test_vld4q_lane_u64(i64* noundef %a, [4 x <2 x i64>] %b.coerce) #2 {
// CHECK:   [[RETVAL:%.*]] = alloca %struct.uint64x2x4_t, align 16
// CHECK:   [[B:%.*]] = alloca %struct.uint64x2x4_t, align 16
// CHECK:   [[__RET:%.*]] = alloca %struct.uint64x2x4_t, align 16
// CHECK:   [[__S1:%.*]] = alloca %struct.uint64x2x4_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.uint64x2x4_t, %struct.uint64x2x4_t* [[B]], i32 0, i32 0
// CHECK:   store [4 x <2 x i64>] [[B]].coerce, [4 x <2 x i64>]* [[COERCE_DIVE]], align 16
// CHECK:   [[TMP0:%.*]] = bitcast %struct.uint64x2x4_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.uint64x2x4_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP0]], i8* align 16 [[TMP1]], i64 64, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast %struct.uint64x2x4_t* [[__RET]] to i8*
// CHECK:   [[TMP3:%.*]] = bitcast i64* %a to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.uint64x2x4_t, %struct.uint64x2x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <2 x i64>], [4 x <2 x i64>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP4:%.*]] = load <2 x i64>, <2 x i64>* [[ARRAYIDX]], align 16
// CHECK:   [[TMP5:%.*]] = bitcast <2 x i64> [[TMP4]] to <16 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.uint64x2x4_t, %struct.uint64x2x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <2 x i64>], [4 x <2 x i64>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP6:%.*]] = load <2 x i64>, <2 x i64>* [[ARRAYIDX2]], align 16
// CHECK:   [[TMP7:%.*]] = bitcast <2 x i64> [[TMP6]] to <16 x i8>
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds %struct.uint64x2x4_t, %struct.uint64x2x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <2 x i64>], [4 x <2 x i64>]* [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP8:%.*]] = load <2 x i64>, <2 x i64>* [[ARRAYIDX4]], align 16
// CHECK:   [[TMP9:%.*]] = bitcast <2 x i64> [[TMP8]] to <16 x i8>
// CHECK:   [[VAL5:%.*]] = getelementptr inbounds %struct.uint64x2x4_t, %struct.uint64x2x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <2 x i64>], [4 x <2 x i64>]* [[VAL5]], i64 0, i64 3
// CHECK:   [[TMP10:%.*]] = load <2 x i64>, <2 x i64>* [[ARRAYIDX6]], align 16
// CHECK:   [[TMP11:%.*]] = bitcast <2 x i64> [[TMP10]] to <16 x i8>
// CHECK:   [[TMP12:%.*]] = bitcast <16 x i8> [[TMP5]] to <2 x i64>
// CHECK:   [[TMP13:%.*]] = bitcast <16 x i8> [[TMP7]] to <2 x i64>
// CHECK:   [[TMP14:%.*]] = bitcast <16 x i8> [[TMP9]] to <2 x i64>
// CHECK:   [[TMP15:%.*]] = bitcast <16 x i8> [[TMP11]] to <2 x i64>
// CHECK:   [[VLD4_LANE:%.*]] = call { <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } @llvm.aarch64.neon.ld4lane.v2i64.p0i8(<2 x i64> [[TMP12]], <2 x i64> [[TMP13]], <2 x i64> [[TMP14]], <2 x i64> [[TMP15]], i64 1, i8* [[TMP3]])
// CHECK:   [[TMP16:%.*]] = bitcast i8* [[TMP2]] to { <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> }*
// CHECK:   store { <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } [[VLD4_LANE]], { <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> }* [[TMP16]]
// CHECK:   [[TMP17:%.*]] = bitcast %struct.uint64x2x4_t* [[RETVAL]] to i8*
// CHECK:   [[TMP18:%.*]] = bitcast %struct.uint64x2x4_t* [[__RET]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP17]], i8* align 16 [[TMP18]], i64 64, i1 false)
// CHECK:   [[TMP19:%.*]] = load %struct.uint64x2x4_t, %struct.uint64x2x4_t* [[RETVAL]], align 16
// CHECK:   ret %struct.uint64x2x4_t [[TMP19]]
uint64x2x4_t test_vld4q_lane_u64(uint64_t  *a, uint64x2x4_t b) {
  return vld4q_lane_u64(a, b, 1);
}

// CHECK-LABEL: define{{.*}} %struct.int8x16x4_t @test_vld4q_lane_s8(i8* noundef %a, [4 x <16 x i8>] %b.coerce) #2 {
// CHECK:   [[RETVAL:%.*]] = alloca %struct.int8x16x4_t, align 16
// CHECK:   [[B:%.*]] = alloca %struct.int8x16x4_t, align 16
// CHECK:   [[__RET:%.*]] = alloca %struct.int8x16x4_t, align 16
// CHECK:   [[__S1:%.*]] = alloca %struct.int8x16x4_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.int8x16x4_t, %struct.int8x16x4_t* [[B]], i32 0, i32 0
// CHECK:   store [4 x <16 x i8>] [[B]].coerce, [4 x <16 x i8>]* [[COERCE_DIVE]], align 16
// CHECK:   [[TMP0:%.*]] = bitcast %struct.int8x16x4_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.int8x16x4_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP0]], i8* align 16 [[TMP1]], i64 64, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast %struct.int8x16x4_t* [[__RET]] to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.int8x16x4_t, %struct.int8x16x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <16 x i8>], [4 x <16 x i8>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX]], align 16
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.int8x16x4_t, %struct.int8x16x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <16 x i8>], [4 x <16 x i8>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP4:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX2]], align 16
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds %struct.int8x16x4_t, %struct.int8x16x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <16 x i8>], [4 x <16 x i8>]* [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP5:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX4]], align 16
// CHECK:   [[VAL5:%.*]] = getelementptr inbounds %struct.int8x16x4_t, %struct.int8x16x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <16 x i8>], [4 x <16 x i8>]* [[VAL5]], i64 0, i64 3
// CHECK:   [[TMP6:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX6]], align 16
// CHECK:   [[VLD4_LANE:%.*]] = call { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } @llvm.aarch64.neon.ld4lane.v16i8.p0i8(<16 x i8> [[TMP3]], <16 x i8> [[TMP4]], <16 x i8> [[TMP5]], <16 x i8> [[TMP6]], i64 15, i8* %a)
// CHECK:   [[TMP7:%.*]] = bitcast i8* [[TMP2]] to { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> }*
// CHECK:   store { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } [[VLD4_LANE]], { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> }* [[TMP7]]
// CHECK:   [[TMP8:%.*]] = bitcast %struct.int8x16x4_t* [[RETVAL]] to i8*
// CHECK:   [[TMP9:%.*]] = bitcast %struct.int8x16x4_t* [[__RET]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP8]], i8* align 16 [[TMP9]], i64 64, i1 false)
// CHECK:   [[TMP10:%.*]] = load %struct.int8x16x4_t, %struct.int8x16x4_t* [[RETVAL]], align 16
// CHECK:   ret %struct.int8x16x4_t [[TMP10]]
int8x16x4_t test_vld4q_lane_s8(int8_t  *a, int8x16x4_t b) {
  return vld4q_lane_s8(a, b, 15);
}

// CHECK-LABEL: define{{.*}} %struct.int16x8x4_t @test_vld4q_lane_s16(i16* noundef %a, [4 x <8 x i16>] %b.coerce) #2 {
// CHECK:   [[RETVAL:%.*]] = alloca %struct.int16x8x4_t, align 16
// CHECK:   [[B:%.*]] = alloca %struct.int16x8x4_t, align 16
// CHECK:   [[__RET:%.*]] = alloca %struct.int16x8x4_t, align 16
// CHECK:   [[__S1:%.*]] = alloca %struct.int16x8x4_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.int16x8x4_t, %struct.int16x8x4_t* [[B]], i32 0, i32 0
// CHECK:   store [4 x <8 x i16>] [[B]].coerce, [4 x <8 x i16>]* [[COERCE_DIVE]], align 16
// CHECK:   [[TMP0:%.*]] = bitcast %struct.int16x8x4_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.int16x8x4_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP0]], i8* align 16 [[TMP1]], i64 64, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast %struct.int16x8x4_t* [[__RET]] to i8*
// CHECK:   [[TMP3:%.*]] = bitcast i16* %a to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.int16x8x4_t, %struct.int16x8x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <8 x i16>], [4 x <8 x i16>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP4:%.*]] = load <8 x i16>, <8 x i16>* [[ARRAYIDX]], align 16
// CHECK:   [[TMP5:%.*]] = bitcast <8 x i16> [[TMP4]] to <16 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.int16x8x4_t, %struct.int16x8x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <8 x i16>], [4 x <8 x i16>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP6:%.*]] = load <8 x i16>, <8 x i16>* [[ARRAYIDX2]], align 16
// CHECK:   [[TMP7:%.*]] = bitcast <8 x i16> [[TMP6]] to <16 x i8>
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds %struct.int16x8x4_t, %struct.int16x8x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <8 x i16>], [4 x <8 x i16>]* [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP8:%.*]] = load <8 x i16>, <8 x i16>* [[ARRAYIDX4]], align 16
// CHECK:   [[TMP9:%.*]] = bitcast <8 x i16> [[TMP8]] to <16 x i8>
// CHECK:   [[VAL5:%.*]] = getelementptr inbounds %struct.int16x8x4_t, %struct.int16x8x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <8 x i16>], [4 x <8 x i16>]* [[VAL5]], i64 0, i64 3
// CHECK:   [[TMP10:%.*]] = load <8 x i16>, <8 x i16>* [[ARRAYIDX6]], align 16
// CHECK:   [[TMP11:%.*]] = bitcast <8 x i16> [[TMP10]] to <16 x i8>
// CHECK:   [[TMP12:%.*]] = bitcast <16 x i8> [[TMP5]] to <8 x i16>
// CHECK:   [[TMP13:%.*]] = bitcast <16 x i8> [[TMP7]] to <8 x i16>
// CHECK:   [[TMP14:%.*]] = bitcast <16 x i8> [[TMP9]] to <8 x i16>
// CHECK:   [[TMP15:%.*]] = bitcast <16 x i8> [[TMP11]] to <8 x i16>
// CHECK:   [[VLD4_LANE:%.*]] = call { <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16> } @llvm.aarch64.neon.ld4lane.v8i16.p0i8(<8 x i16> [[TMP12]], <8 x i16> [[TMP13]], <8 x i16> [[TMP14]], <8 x i16> [[TMP15]], i64 7, i8* [[TMP3]])
// CHECK:   [[TMP16:%.*]] = bitcast i8* [[TMP2]] to { <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16> }*
// CHECK:   store { <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16> } [[VLD4_LANE]], { <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16> }* [[TMP16]]
// CHECK:   [[TMP17:%.*]] = bitcast %struct.int16x8x4_t* [[RETVAL]] to i8*
// CHECK:   [[TMP18:%.*]] = bitcast %struct.int16x8x4_t* [[__RET]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP17]], i8* align 16 [[TMP18]], i64 64, i1 false)
// CHECK:   [[TMP19:%.*]] = load %struct.int16x8x4_t, %struct.int16x8x4_t* [[RETVAL]], align 16
// CHECK:   ret %struct.int16x8x4_t [[TMP19]]
int16x8x4_t test_vld4q_lane_s16(int16_t  *a, int16x8x4_t b) {
  return vld4q_lane_s16(a, b, 7);
}

// CHECK-LABEL: define{{.*}} %struct.int32x4x4_t @test_vld4q_lane_s32(i32* noundef %a, [4 x <4 x i32>] %b.coerce) #2 {
// CHECK:   [[RETVAL:%.*]] = alloca %struct.int32x4x4_t, align 16
// CHECK:   [[B:%.*]] = alloca %struct.int32x4x4_t, align 16
// CHECK:   [[__RET:%.*]] = alloca %struct.int32x4x4_t, align 16
// CHECK:   [[__S1:%.*]] = alloca %struct.int32x4x4_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.int32x4x4_t, %struct.int32x4x4_t* [[B]], i32 0, i32 0
// CHECK:   store [4 x <4 x i32>] [[B]].coerce, [4 x <4 x i32>]* [[COERCE_DIVE]], align 16
// CHECK:   [[TMP0:%.*]] = bitcast %struct.int32x4x4_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.int32x4x4_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP0]], i8* align 16 [[TMP1]], i64 64, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast %struct.int32x4x4_t* [[__RET]] to i8*
// CHECK:   [[TMP3:%.*]] = bitcast i32* %a to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.int32x4x4_t, %struct.int32x4x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <4 x i32>], [4 x <4 x i32>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP4:%.*]] = load <4 x i32>, <4 x i32>* [[ARRAYIDX]], align 16
// CHECK:   [[TMP5:%.*]] = bitcast <4 x i32> [[TMP4]] to <16 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.int32x4x4_t, %struct.int32x4x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <4 x i32>], [4 x <4 x i32>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP6:%.*]] = load <4 x i32>, <4 x i32>* [[ARRAYIDX2]], align 16
// CHECK:   [[TMP7:%.*]] = bitcast <4 x i32> [[TMP6]] to <16 x i8>
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds %struct.int32x4x4_t, %struct.int32x4x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <4 x i32>], [4 x <4 x i32>]* [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP8:%.*]] = load <4 x i32>, <4 x i32>* [[ARRAYIDX4]], align 16
// CHECK:   [[TMP9:%.*]] = bitcast <4 x i32> [[TMP8]] to <16 x i8>
// CHECK:   [[VAL5:%.*]] = getelementptr inbounds %struct.int32x4x4_t, %struct.int32x4x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <4 x i32>], [4 x <4 x i32>]* [[VAL5]], i64 0, i64 3
// CHECK:   [[TMP10:%.*]] = load <4 x i32>, <4 x i32>* [[ARRAYIDX6]], align 16
// CHECK:   [[TMP11:%.*]] = bitcast <4 x i32> [[TMP10]] to <16 x i8>
// CHECK:   [[TMP12:%.*]] = bitcast <16 x i8> [[TMP5]] to <4 x i32>
// CHECK:   [[TMP13:%.*]] = bitcast <16 x i8> [[TMP7]] to <4 x i32>
// CHECK:   [[TMP14:%.*]] = bitcast <16 x i8> [[TMP9]] to <4 x i32>
// CHECK:   [[TMP15:%.*]] = bitcast <16 x i8> [[TMP11]] to <4 x i32>
// CHECK:   [[VLD4_LANE:%.*]] = call { <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32> } @llvm.aarch64.neon.ld4lane.v4i32.p0i8(<4 x i32> [[TMP12]], <4 x i32> [[TMP13]], <4 x i32> [[TMP14]], <4 x i32> [[TMP15]], i64 3, i8* [[TMP3]])
// CHECK:   [[TMP16:%.*]] = bitcast i8* [[TMP2]] to { <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32> }*
// CHECK:   store { <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32> } [[VLD4_LANE]], { <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32> }* [[TMP16]]
// CHECK:   [[TMP17:%.*]] = bitcast %struct.int32x4x4_t* [[RETVAL]] to i8*
// CHECK:   [[TMP18:%.*]] = bitcast %struct.int32x4x4_t* [[__RET]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP17]], i8* align 16 [[TMP18]], i64 64, i1 false)
// CHECK:   [[TMP19:%.*]] = load %struct.int32x4x4_t, %struct.int32x4x4_t* [[RETVAL]], align 16
// CHECK:   ret %struct.int32x4x4_t [[TMP19]]
int32x4x4_t test_vld4q_lane_s32(int32_t  *a, int32x4x4_t b) {
  return vld4q_lane_s32(a, b, 3);
}

// CHECK-LABEL: define{{.*}} %struct.int64x2x4_t @test_vld4q_lane_s64(i64* noundef %a, [4 x <2 x i64>] %b.coerce) #2 {
// CHECK:   [[RETVAL:%.*]] = alloca %struct.int64x2x4_t, align 16
// CHECK:   [[B:%.*]] = alloca %struct.int64x2x4_t, align 16
// CHECK:   [[__RET:%.*]] = alloca %struct.int64x2x4_t, align 16
// CHECK:   [[__S1:%.*]] = alloca %struct.int64x2x4_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.int64x2x4_t, %struct.int64x2x4_t* [[B]], i32 0, i32 0
// CHECK:   store [4 x <2 x i64>] [[B]].coerce, [4 x <2 x i64>]* [[COERCE_DIVE]], align 16
// CHECK:   [[TMP0:%.*]] = bitcast %struct.int64x2x4_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.int64x2x4_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP0]], i8* align 16 [[TMP1]], i64 64, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast %struct.int64x2x4_t* [[__RET]] to i8*
// CHECK:   [[TMP3:%.*]] = bitcast i64* %a to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.int64x2x4_t, %struct.int64x2x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <2 x i64>], [4 x <2 x i64>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP4:%.*]] = load <2 x i64>, <2 x i64>* [[ARRAYIDX]], align 16
// CHECK:   [[TMP5:%.*]] = bitcast <2 x i64> [[TMP4]] to <16 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.int64x2x4_t, %struct.int64x2x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <2 x i64>], [4 x <2 x i64>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP6:%.*]] = load <2 x i64>, <2 x i64>* [[ARRAYIDX2]], align 16
// CHECK:   [[TMP7:%.*]] = bitcast <2 x i64> [[TMP6]] to <16 x i8>
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds %struct.int64x2x4_t, %struct.int64x2x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <2 x i64>], [4 x <2 x i64>]* [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP8:%.*]] = load <2 x i64>, <2 x i64>* [[ARRAYIDX4]], align 16
// CHECK:   [[TMP9:%.*]] = bitcast <2 x i64> [[TMP8]] to <16 x i8>
// CHECK:   [[VAL5:%.*]] = getelementptr inbounds %struct.int64x2x4_t, %struct.int64x2x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <2 x i64>], [4 x <2 x i64>]* [[VAL5]], i64 0, i64 3
// CHECK:   [[TMP10:%.*]] = load <2 x i64>, <2 x i64>* [[ARRAYIDX6]], align 16
// CHECK:   [[TMP11:%.*]] = bitcast <2 x i64> [[TMP10]] to <16 x i8>
// CHECK:   [[TMP12:%.*]] = bitcast <16 x i8> [[TMP5]] to <2 x i64>
// CHECK:   [[TMP13:%.*]] = bitcast <16 x i8> [[TMP7]] to <2 x i64>
// CHECK:   [[TMP14:%.*]] = bitcast <16 x i8> [[TMP9]] to <2 x i64>
// CHECK:   [[TMP15:%.*]] = bitcast <16 x i8> [[TMP11]] to <2 x i64>
// CHECK:   [[VLD4_LANE:%.*]] = call { <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } @llvm.aarch64.neon.ld4lane.v2i64.p0i8(<2 x i64> [[TMP12]], <2 x i64> [[TMP13]], <2 x i64> [[TMP14]], <2 x i64> [[TMP15]], i64 1, i8* [[TMP3]])
// CHECK:   [[TMP16:%.*]] = bitcast i8* [[TMP2]] to { <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> }*
// CHECK:   store { <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } [[VLD4_LANE]], { <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> }* [[TMP16]]
// CHECK:   [[TMP17:%.*]] = bitcast %struct.int64x2x4_t* [[RETVAL]] to i8*
// CHECK:   [[TMP18:%.*]] = bitcast %struct.int64x2x4_t* [[__RET]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP17]], i8* align 16 [[TMP18]], i64 64, i1 false)
// CHECK:   [[TMP19:%.*]] = load %struct.int64x2x4_t, %struct.int64x2x4_t* [[RETVAL]], align 16
// CHECK:   ret %struct.int64x2x4_t [[TMP19]]
int64x2x4_t test_vld4q_lane_s64(int64_t  *a, int64x2x4_t b) {
  return vld4q_lane_s64(a, b, 1);
}

// CHECK-LABEL: define{{.*}} %struct.float16x8x4_t @test_vld4q_lane_f16(half* noundef %a, [4 x <8 x half>] %b.coerce) #2 {
// CHECK:   [[RETVAL:%.*]] = alloca %struct.float16x8x4_t, align 16
// CHECK:   [[B:%.*]] = alloca %struct.float16x8x4_t, align 16
// CHECK:   [[__RET:%.*]] = alloca %struct.float16x8x4_t, align 16
// CHECK:   [[__S1:%.*]] = alloca %struct.float16x8x4_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.float16x8x4_t, %struct.float16x8x4_t* [[B]], i32 0, i32 0
// CHECK:   store [4 x <8 x half>] [[B]].coerce, [4 x <8 x half>]* [[COERCE_DIVE]], align 16
// CHECK:   [[TMP0:%.*]] = bitcast %struct.float16x8x4_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.float16x8x4_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP0]], i8* align 16 [[TMP1]], i64 64, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast %struct.float16x8x4_t* [[__RET]] to i8*
// CHECK:   [[TMP3:%.*]] = bitcast half* %a to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.float16x8x4_t, %struct.float16x8x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <8 x half>], [4 x <8 x half>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP4:%.*]] = load <8 x half>, <8 x half>* [[ARRAYIDX]], align 16
// CHECK:   [[TMP5:%.*]] = bitcast <8 x half> [[TMP4]] to <16 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.float16x8x4_t, %struct.float16x8x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <8 x half>], [4 x <8 x half>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP6:%.*]] = load <8 x half>, <8 x half>* [[ARRAYIDX2]], align 16
// CHECK:   [[TMP7:%.*]] = bitcast <8 x half> [[TMP6]] to <16 x i8>
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds %struct.float16x8x4_t, %struct.float16x8x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <8 x half>], [4 x <8 x half>]* [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP8:%.*]] = load <8 x half>, <8 x half>* [[ARRAYIDX4]], align 16
// CHECK:   [[TMP9:%.*]] = bitcast <8 x half> [[TMP8]] to <16 x i8>
// CHECK:   [[VAL5:%.*]] = getelementptr inbounds %struct.float16x8x4_t, %struct.float16x8x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <8 x half>], [4 x <8 x half>]* [[VAL5]], i64 0, i64 3
// CHECK:   [[TMP10:%.*]] = load <8 x half>, <8 x half>* [[ARRAYIDX6]], align 16
// CHECK:   [[TMP11:%.*]] = bitcast <8 x half> [[TMP10]] to <16 x i8>
// CHECK:   [[TMP12:%.*]] = bitcast <16 x i8> [[TMP5]] to <8 x half>
// CHECK:   [[TMP13:%.*]] = bitcast <16 x i8> [[TMP7]] to <8 x half>
// CHECK:   [[TMP14:%.*]] = bitcast <16 x i8> [[TMP9]] to <8 x half>
// CHECK:   [[TMP15:%.*]] = bitcast <16 x i8> [[TMP11]] to <8 x half>
// CHECK:   [[VLD4_LANE:%.*]] = call { <8 x half>, <8 x half>, <8 x half>, <8 x half> } @llvm.aarch64.neon.ld4lane.v8f16.p0i8(<8 x half> [[TMP12]], <8 x half> [[TMP13]], <8 x half> [[TMP14]], <8 x half> [[TMP15]], i64 7, i8* [[TMP3]])
// CHECK:   [[TMP16:%.*]] = bitcast i8* [[TMP2]] to { <8 x half>, <8 x half>, <8 x half>, <8 x half> }*
// CHECK:   store { <8 x half>, <8 x half>, <8 x half>, <8 x half> } [[VLD4_LANE]], { <8 x half>, <8 x half>, <8 x half>, <8 x half> }* [[TMP16]]
// CHECK:   [[TMP17:%.*]] = bitcast %struct.float16x8x4_t* [[RETVAL]] to i8*
// CHECK:   [[TMP18:%.*]] = bitcast %struct.float16x8x4_t* [[__RET]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP17]], i8* align 16 [[TMP18]], i64 64, i1 false)
// CHECK:   [[TMP19:%.*]] = load %struct.float16x8x4_t, %struct.float16x8x4_t* [[RETVAL]], align 16
// CHECK:   ret %struct.float16x8x4_t [[TMP19]]
float16x8x4_t test_vld4q_lane_f16(float16_t  *a, float16x8x4_t b) {
  return vld4q_lane_f16(a, b, 7);
}

// CHECK-LABEL: define{{.*}} %struct.float32x4x4_t @test_vld4q_lane_f32(float* noundef %a, [4 x <4 x float>] %b.coerce) #2 {
// CHECK:   [[RETVAL:%.*]] = alloca %struct.float32x4x4_t, align 16
// CHECK:   [[B:%.*]] = alloca %struct.float32x4x4_t, align 16
// CHECK:   [[__RET:%.*]] = alloca %struct.float32x4x4_t, align 16
// CHECK:   [[__S1:%.*]] = alloca %struct.float32x4x4_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.float32x4x4_t, %struct.float32x4x4_t* [[B]], i32 0, i32 0
// CHECK:   store [4 x <4 x float>] [[B]].coerce, [4 x <4 x float>]* [[COERCE_DIVE]], align 16
// CHECK:   [[TMP0:%.*]] = bitcast %struct.float32x4x4_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.float32x4x4_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP0]], i8* align 16 [[TMP1]], i64 64, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast %struct.float32x4x4_t* [[__RET]] to i8*
// CHECK:   [[TMP3:%.*]] = bitcast float* %a to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.float32x4x4_t, %struct.float32x4x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <4 x float>], [4 x <4 x float>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP4:%.*]] = load <4 x float>, <4 x float>* [[ARRAYIDX]], align 16
// CHECK:   [[TMP5:%.*]] = bitcast <4 x float> [[TMP4]] to <16 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.float32x4x4_t, %struct.float32x4x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <4 x float>], [4 x <4 x float>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP6:%.*]] = load <4 x float>, <4 x float>* [[ARRAYIDX2]], align 16
// CHECK:   [[TMP7:%.*]] = bitcast <4 x float> [[TMP6]] to <16 x i8>
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds %struct.float32x4x4_t, %struct.float32x4x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <4 x float>], [4 x <4 x float>]* [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP8:%.*]] = load <4 x float>, <4 x float>* [[ARRAYIDX4]], align 16
// CHECK:   [[TMP9:%.*]] = bitcast <4 x float> [[TMP8]] to <16 x i8>
// CHECK:   [[VAL5:%.*]] = getelementptr inbounds %struct.float32x4x4_t, %struct.float32x4x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <4 x float>], [4 x <4 x float>]* [[VAL5]], i64 0, i64 3
// CHECK:   [[TMP10:%.*]] = load <4 x float>, <4 x float>* [[ARRAYIDX6]], align 16
// CHECK:   [[TMP11:%.*]] = bitcast <4 x float> [[TMP10]] to <16 x i8>
// CHECK:   [[TMP12:%.*]] = bitcast <16 x i8> [[TMP5]] to <4 x float>
// CHECK:   [[TMP13:%.*]] = bitcast <16 x i8> [[TMP7]] to <4 x float>
// CHECK:   [[TMP14:%.*]] = bitcast <16 x i8> [[TMP9]] to <4 x float>
// CHECK:   [[TMP15:%.*]] = bitcast <16 x i8> [[TMP11]] to <4 x float>
// CHECK:   [[VLD4_LANE:%.*]] = call { <4 x float>, <4 x float>, <4 x float>, <4 x float> } @llvm.aarch64.neon.ld4lane.v4f32.p0i8(<4 x float> [[TMP12]], <4 x float> [[TMP13]], <4 x float> [[TMP14]], <4 x float> [[TMP15]], i64 3, i8* [[TMP3]])
// CHECK:   [[TMP16:%.*]] = bitcast i8* [[TMP2]] to { <4 x float>, <4 x float>, <4 x float>, <4 x float> }*
// CHECK:   store { <4 x float>, <4 x float>, <4 x float>, <4 x float> } [[VLD4_LANE]], { <4 x float>, <4 x float>, <4 x float>, <4 x float> }* [[TMP16]]
// CHECK:   [[TMP17:%.*]] = bitcast %struct.float32x4x4_t* [[RETVAL]] to i8*
// CHECK:   [[TMP18:%.*]] = bitcast %struct.float32x4x4_t* [[__RET]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP17]], i8* align 16 [[TMP18]], i64 64, i1 false)
// CHECK:   [[TMP19:%.*]] = load %struct.float32x4x4_t, %struct.float32x4x4_t* [[RETVAL]], align 16
// CHECK:   ret %struct.float32x4x4_t [[TMP19]]
float32x4x4_t test_vld4q_lane_f32(float32_t  *a, float32x4x4_t b) {
  return vld4q_lane_f32(a, b, 3);
}

// CHECK-LABEL: define{{.*}} %struct.float64x2x4_t @test_vld4q_lane_f64(double* noundef %a, [4 x <2 x double>] %b.coerce) #2 {
// CHECK:   [[RETVAL:%.*]] = alloca %struct.float64x2x4_t, align 16
// CHECK:   [[B:%.*]] = alloca %struct.float64x2x4_t, align 16
// CHECK:   [[__RET:%.*]] = alloca %struct.float64x2x4_t, align 16
// CHECK:   [[__S1:%.*]] = alloca %struct.float64x2x4_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.float64x2x4_t, %struct.float64x2x4_t* [[B]], i32 0, i32 0
// CHECK:   store [4 x <2 x double>] [[B]].coerce, [4 x <2 x double>]* [[COERCE_DIVE]], align 16
// CHECK:   [[TMP0:%.*]] = bitcast %struct.float64x2x4_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.float64x2x4_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP0]], i8* align 16 [[TMP1]], i64 64, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast %struct.float64x2x4_t* [[__RET]] to i8*
// CHECK:   [[TMP3:%.*]] = bitcast double* %a to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.float64x2x4_t, %struct.float64x2x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <2 x double>], [4 x <2 x double>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP4:%.*]] = load <2 x double>, <2 x double>* [[ARRAYIDX]], align 16
// CHECK:   [[TMP5:%.*]] = bitcast <2 x double> [[TMP4]] to <16 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.float64x2x4_t, %struct.float64x2x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <2 x double>], [4 x <2 x double>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP6:%.*]] = load <2 x double>, <2 x double>* [[ARRAYIDX2]], align 16
// CHECK:   [[TMP7:%.*]] = bitcast <2 x double> [[TMP6]] to <16 x i8>
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds %struct.float64x2x4_t, %struct.float64x2x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <2 x double>], [4 x <2 x double>]* [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP8:%.*]] = load <2 x double>, <2 x double>* [[ARRAYIDX4]], align 16
// CHECK:   [[TMP9:%.*]] = bitcast <2 x double> [[TMP8]] to <16 x i8>
// CHECK:   [[VAL5:%.*]] = getelementptr inbounds %struct.float64x2x4_t, %struct.float64x2x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <2 x double>], [4 x <2 x double>]* [[VAL5]], i64 0, i64 3
// CHECK:   [[TMP10:%.*]] = load <2 x double>, <2 x double>* [[ARRAYIDX6]], align 16
// CHECK:   [[TMP11:%.*]] = bitcast <2 x double> [[TMP10]] to <16 x i8>
// CHECK:   [[TMP12:%.*]] = bitcast <16 x i8> [[TMP5]] to <2 x double>
// CHECK:   [[TMP13:%.*]] = bitcast <16 x i8> [[TMP7]] to <2 x double>
// CHECK:   [[TMP14:%.*]] = bitcast <16 x i8> [[TMP9]] to <2 x double>
// CHECK:   [[TMP15:%.*]] = bitcast <16 x i8> [[TMP11]] to <2 x double>
// CHECK:   [[VLD4_LANE:%.*]] = call { <2 x double>, <2 x double>, <2 x double>, <2 x double> } @llvm.aarch64.neon.ld4lane.v2f64.p0i8(<2 x double> [[TMP12]], <2 x double> [[TMP13]], <2 x double> [[TMP14]], <2 x double> [[TMP15]], i64 1, i8* [[TMP3]])
// CHECK:   [[TMP16:%.*]] = bitcast i8* [[TMP2]] to { <2 x double>, <2 x double>, <2 x double>, <2 x double> }*
// CHECK:   store { <2 x double>, <2 x double>, <2 x double>, <2 x double> } [[VLD4_LANE]], { <2 x double>, <2 x double>, <2 x double>, <2 x double> }* [[TMP16]]
// CHECK:   [[TMP17:%.*]] = bitcast %struct.float64x2x4_t* [[RETVAL]] to i8*
// CHECK:   [[TMP18:%.*]] = bitcast %struct.float64x2x4_t* [[__RET]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP17]], i8* align 16 [[TMP18]], i64 64, i1 false)
// CHECK:   [[TMP19:%.*]] = load %struct.float64x2x4_t, %struct.float64x2x4_t* [[RETVAL]], align 16
// CHECK:   ret %struct.float64x2x4_t [[TMP19]]
float64x2x4_t test_vld4q_lane_f64(float64_t  *a, float64x2x4_t b) {
  return vld4q_lane_f64(a, b, 1);
}

// CHECK-LABEL: define{{.*}} %struct.poly8x16x4_t @test_vld4q_lane_p8(i8* noundef %a, [4 x <16 x i8>] %b.coerce) #2 {
// CHECK:   [[RETVAL:%.*]] = alloca %struct.poly8x16x4_t, align 16
// CHECK:   [[B:%.*]] = alloca %struct.poly8x16x4_t, align 16
// CHECK:   [[__RET:%.*]] = alloca %struct.poly8x16x4_t, align 16
// CHECK:   [[__S1:%.*]] = alloca %struct.poly8x16x4_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.poly8x16x4_t, %struct.poly8x16x4_t* [[B]], i32 0, i32 0
// CHECK:   store [4 x <16 x i8>] [[B]].coerce, [4 x <16 x i8>]* [[COERCE_DIVE]], align 16
// CHECK:   [[TMP0:%.*]] = bitcast %struct.poly8x16x4_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.poly8x16x4_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP0]], i8* align 16 [[TMP1]], i64 64, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast %struct.poly8x16x4_t* [[__RET]] to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.poly8x16x4_t, %struct.poly8x16x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <16 x i8>], [4 x <16 x i8>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX]], align 16
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.poly8x16x4_t, %struct.poly8x16x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <16 x i8>], [4 x <16 x i8>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP4:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX2]], align 16
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds %struct.poly8x16x4_t, %struct.poly8x16x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <16 x i8>], [4 x <16 x i8>]* [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP5:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX4]], align 16
// CHECK:   [[VAL5:%.*]] = getelementptr inbounds %struct.poly8x16x4_t, %struct.poly8x16x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <16 x i8>], [4 x <16 x i8>]* [[VAL5]], i64 0, i64 3
// CHECK:   [[TMP6:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX6]], align 16
// CHECK:   [[VLD4_LANE:%.*]] = call { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } @llvm.aarch64.neon.ld4lane.v16i8.p0i8(<16 x i8> [[TMP3]], <16 x i8> [[TMP4]], <16 x i8> [[TMP5]], <16 x i8> [[TMP6]], i64 15, i8* %a)
// CHECK:   [[TMP7:%.*]] = bitcast i8* [[TMP2]] to { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> }*
// CHECK:   store { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } [[VLD4_LANE]], { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> }* [[TMP7]]
// CHECK:   [[TMP8:%.*]] = bitcast %struct.poly8x16x4_t* [[RETVAL]] to i8*
// CHECK:   [[TMP9:%.*]] = bitcast %struct.poly8x16x4_t* [[__RET]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP8]], i8* align 16 [[TMP9]], i64 64, i1 false)
// CHECK:   [[TMP10:%.*]] = load %struct.poly8x16x4_t, %struct.poly8x16x4_t* [[RETVAL]], align 16
// CHECK:   ret %struct.poly8x16x4_t [[TMP10]]
poly8x16x4_t test_vld4q_lane_p8(poly8_t  *a, poly8x16x4_t b) {
  return vld4q_lane_p8(a, b, 15);
}

// CHECK-LABEL: define{{.*}} %struct.poly16x8x4_t @test_vld4q_lane_p16(i16* noundef %a, [4 x <8 x i16>] %b.coerce) #2 {
// CHECK:   [[RETVAL:%.*]] = alloca %struct.poly16x8x4_t, align 16
// CHECK:   [[B:%.*]] = alloca %struct.poly16x8x4_t, align 16
// CHECK:   [[__RET:%.*]] = alloca %struct.poly16x8x4_t, align 16
// CHECK:   [[__S1:%.*]] = alloca %struct.poly16x8x4_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.poly16x8x4_t, %struct.poly16x8x4_t* [[B]], i32 0, i32 0
// CHECK:   store [4 x <8 x i16>] [[B]].coerce, [4 x <8 x i16>]* [[COERCE_DIVE]], align 16
// CHECK:   [[TMP0:%.*]] = bitcast %struct.poly16x8x4_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.poly16x8x4_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP0]], i8* align 16 [[TMP1]], i64 64, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast %struct.poly16x8x4_t* [[__RET]] to i8*
// CHECK:   [[TMP3:%.*]] = bitcast i16* %a to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.poly16x8x4_t, %struct.poly16x8x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <8 x i16>], [4 x <8 x i16>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP4:%.*]] = load <8 x i16>, <8 x i16>* [[ARRAYIDX]], align 16
// CHECK:   [[TMP5:%.*]] = bitcast <8 x i16> [[TMP4]] to <16 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.poly16x8x4_t, %struct.poly16x8x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <8 x i16>], [4 x <8 x i16>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP6:%.*]] = load <8 x i16>, <8 x i16>* [[ARRAYIDX2]], align 16
// CHECK:   [[TMP7:%.*]] = bitcast <8 x i16> [[TMP6]] to <16 x i8>
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds %struct.poly16x8x4_t, %struct.poly16x8x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <8 x i16>], [4 x <8 x i16>]* [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP8:%.*]] = load <8 x i16>, <8 x i16>* [[ARRAYIDX4]], align 16
// CHECK:   [[TMP9:%.*]] = bitcast <8 x i16> [[TMP8]] to <16 x i8>
// CHECK:   [[VAL5:%.*]] = getelementptr inbounds %struct.poly16x8x4_t, %struct.poly16x8x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <8 x i16>], [4 x <8 x i16>]* [[VAL5]], i64 0, i64 3
// CHECK:   [[TMP10:%.*]] = load <8 x i16>, <8 x i16>* [[ARRAYIDX6]], align 16
// CHECK:   [[TMP11:%.*]] = bitcast <8 x i16> [[TMP10]] to <16 x i8>
// CHECK:   [[TMP12:%.*]] = bitcast <16 x i8> [[TMP5]] to <8 x i16>
// CHECK:   [[TMP13:%.*]] = bitcast <16 x i8> [[TMP7]] to <8 x i16>
// CHECK:   [[TMP14:%.*]] = bitcast <16 x i8> [[TMP9]] to <8 x i16>
// CHECK:   [[TMP15:%.*]] = bitcast <16 x i8> [[TMP11]] to <8 x i16>
// CHECK:   [[VLD4_LANE:%.*]] = call { <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16> } @llvm.aarch64.neon.ld4lane.v8i16.p0i8(<8 x i16> [[TMP12]], <8 x i16> [[TMP13]], <8 x i16> [[TMP14]], <8 x i16> [[TMP15]], i64 7, i8* [[TMP3]])
// CHECK:   [[TMP16:%.*]] = bitcast i8* [[TMP2]] to { <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16> }*
// CHECK:   store { <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16> } [[VLD4_LANE]], { <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16> }* [[TMP16]]
// CHECK:   [[TMP17:%.*]] = bitcast %struct.poly16x8x4_t* [[RETVAL]] to i8*
// CHECK:   [[TMP18:%.*]] = bitcast %struct.poly16x8x4_t* [[__RET]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP17]], i8* align 16 [[TMP18]], i64 64, i1 false)
// CHECK:   [[TMP19:%.*]] = load %struct.poly16x8x4_t, %struct.poly16x8x4_t* [[RETVAL]], align 16
// CHECK:   ret %struct.poly16x8x4_t [[TMP19]]
poly16x8x4_t test_vld4q_lane_p16(poly16_t  *a, poly16x8x4_t b) {
  return vld4q_lane_p16(a, b, 7);
}

// CHECK-LABEL: define{{.*}} %struct.poly64x2x4_t @test_vld4q_lane_p64(i64* noundef %a, [4 x <2 x i64>] %b.coerce) #2 {
// CHECK:   [[RETVAL:%.*]] = alloca %struct.poly64x2x4_t, align 16
// CHECK:   [[B:%.*]] = alloca %struct.poly64x2x4_t, align 16
// CHECK:   [[__RET:%.*]] = alloca %struct.poly64x2x4_t, align 16
// CHECK:   [[__S1:%.*]] = alloca %struct.poly64x2x4_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.poly64x2x4_t, %struct.poly64x2x4_t* [[B]], i32 0, i32 0
// CHECK:   store [4 x <2 x i64>] [[B]].coerce, [4 x <2 x i64>]* [[COERCE_DIVE]], align 16
// CHECK:   [[TMP0:%.*]] = bitcast %struct.poly64x2x4_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.poly64x2x4_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP0]], i8* align 16 [[TMP1]], i64 64, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast %struct.poly64x2x4_t* [[__RET]] to i8*
// CHECK:   [[TMP3:%.*]] = bitcast i64* %a to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.poly64x2x4_t, %struct.poly64x2x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <2 x i64>], [4 x <2 x i64>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP4:%.*]] = load <2 x i64>, <2 x i64>* [[ARRAYIDX]], align 16
// CHECK:   [[TMP5:%.*]] = bitcast <2 x i64> [[TMP4]] to <16 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.poly64x2x4_t, %struct.poly64x2x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <2 x i64>], [4 x <2 x i64>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP6:%.*]] = load <2 x i64>, <2 x i64>* [[ARRAYIDX2]], align 16
// CHECK:   [[TMP7:%.*]] = bitcast <2 x i64> [[TMP6]] to <16 x i8>
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds %struct.poly64x2x4_t, %struct.poly64x2x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <2 x i64>], [4 x <2 x i64>]* [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP8:%.*]] = load <2 x i64>, <2 x i64>* [[ARRAYIDX4]], align 16
// CHECK:   [[TMP9:%.*]] = bitcast <2 x i64> [[TMP8]] to <16 x i8>
// CHECK:   [[VAL5:%.*]] = getelementptr inbounds %struct.poly64x2x4_t, %struct.poly64x2x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <2 x i64>], [4 x <2 x i64>]* [[VAL5]], i64 0, i64 3
// CHECK:   [[TMP10:%.*]] = load <2 x i64>, <2 x i64>* [[ARRAYIDX6]], align 16
// CHECK:   [[TMP11:%.*]] = bitcast <2 x i64> [[TMP10]] to <16 x i8>
// CHECK:   [[TMP12:%.*]] = bitcast <16 x i8> [[TMP5]] to <2 x i64>
// CHECK:   [[TMP13:%.*]] = bitcast <16 x i8> [[TMP7]] to <2 x i64>
// CHECK:   [[TMP14:%.*]] = bitcast <16 x i8> [[TMP9]] to <2 x i64>
// CHECK:   [[TMP15:%.*]] = bitcast <16 x i8> [[TMP11]] to <2 x i64>
// CHECK:   [[VLD4_LANE:%.*]] = call { <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } @llvm.aarch64.neon.ld4lane.v2i64.p0i8(<2 x i64> [[TMP12]], <2 x i64> [[TMP13]], <2 x i64> [[TMP14]], <2 x i64> [[TMP15]], i64 1, i8* [[TMP3]])
// CHECK:   [[TMP16:%.*]] = bitcast i8* [[TMP2]] to { <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> }*
// CHECK:   store { <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } [[VLD4_LANE]], { <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> }* [[TMP16]]
// CHECK:   [[TMP17:%.*]] = bitcast %struct.poly64x2x4_t* [[RETVAL]] to i8*
// CHECK:   [[TMP18:%.*]] = bitcast %struct.poly64x2x4_t* [[__RET]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP17]], i8* align 16 [[TMP18]], i64 64, i1 false)
// CHECK:   [[TMP19:%.*]] = load %struct.poly64x2x4_t, %struct.poly64x2x4_t* [[RETVAL]], align 16
// CHECK:   ret %struct.poly64x2x4_t [[TMP19]]
poly64x2x4_t test_vld4q_lane_p64(poly64_t  *a, poly64x2x4_t b) {
  return vld4q_lane_p64(a, b, 1);
}

// CHECK-LABEL: define{{.*}} %struct.uint8x8x4_t @test_vld4_lane_u8(i8* noundef %a, [4 x <8 x i8>] %b.coerce) #2 {
// CHECK:   [[RETVAL:%.*]] = alloca %struct.uint8x8x4_t, align 8
// CHECK:   [[B:%.*]] = alloca %struct.uint8x8x4_t, align 8
// CHECK:   [[__RET:%.*]] = alloca %struct.uint8x8x4_t, align 8
// CHECK:   [[__S1:%.*]] = alloca %struct.uint8x8x4_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.uint8x8x4_t, %struct.uint8x8x4_t* [[B]], i32 0, i32 0
// CHECK:   store [4 x <8 x i8>] [[B]].coerce, [4 x <8 x i8>]* [[COERCE_DIVE]], align 8
// CHECK:   [[TMP0:%.*]] = bitcast %struct.uint8x8x4_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.uint8x8x4_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[TMP0]], i8* align 8 [[TMP1]], i64 32, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast %struct.uint8x8x4_t* [[__RET]] to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.uint8x8x4_t, %struct.uint8x8x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <8 x i8>], [4 x <8 x i8>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <8 x i8>, <8 x i8>* [[ARRAYIDX]], align 8
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.uint8x8x4_t, %struct.uint8x8x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <8 x i8>], [4 x <8 x i8>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP4:%.*]] = load <8 x i8>, <8 x i8>* [[ARRAYIDX2]], align 8
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds %struct.uint8x8x4_t, %struct.uint8x8x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <8 x i8>], [4 x <8 x i8>]* [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP5:%.*]] = load <8 x i8>, <8 x i8>* [[ARRAYIDX4]], align 8
// CHECK:   [[VAL5:%.*]] = getelementptr inbounds %struct.uint8x8x4_t, %struct.uint8x8x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <8 x i8>], [4 x <8 x i8>]* [[VAL5]], i64 0, i64 3
// CHECK:   [[TMP6:%.*]] = load <8 x i8>, <8 x i8>* [[ARRAYIDX6]], align 8
// CHECK:   [[VLD4_LANE:%.*]] = call { <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8> } @llvm.aarch64.neon.ld4lane.v8i8.p0i8(<8 x i8> [[TMP3]], <8 x i8> [[TMP4]], <8 x i8> [[TMP5]], <8 x i8> [[TMP6]], i64 7, i8* %a)
// CHECK:   [[TMP7:%.*]] = bitcast i8* [[TMP2]] to { <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8> }*
// CHECK:   store { <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8> } [[VLD4_LANE]], { <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8> }* [[TMP7]]
// CHECK:   [[TMP8:%.*]] = bitcast %struct.uint8x8x4_t* [[RETVAL]] to i8*
// CHECK:   [[TMP9:%.*]] = bitcast %struct.uint8x8x4_t* [[__RET]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[TMP8]], i8* align 8 [[TMP9]], i64 32, i1 false)
// CHECK:   [[TMP10:%.*]] = load %struct.uint8x8x4_t, %struct.uint8x8x4_t* [[RETVAL]], align 8
// CHECK:   ret %struct.uint8x8x4_t [[TMP10]]
uint8x8x4_t test_vld4_lane_u8(uint8_t  *a, uint8x8x4_t b) {
  return vld4_lane_u8(a, b, 7);
}

// CHECK-LABEL: define{{.*}} %struct.uint16x4x4_t @test_vld4_lane_u16(i16* noundef %a, [4 x <4 x i16>] %b.coerce) #2 {
// CHECK:   [[RETVAL:%.*]] = alloca %struct.uint16x4x4_t, align 8
// CHECK:   [[B:%.*]] = alloca %struct.uint16x4x4_t, align 8
// CHECK:   [[__RET:%.*]] = alloca %struct.uint16x4x4_t, align 8
// CHECK:   [[__S1:%.*]] = alloca %struct.uint16x4x4_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.uint16x4x4_t, %struct.uint16x4x4_t* [[B]], i32 0, i32 0
// CHECK:   store [4 x <4 x i16>] [[B]].coerce, [4 x <4 x i16>]* [[COERCE_DIVE]], align 8
// CHECK:   [[TMP0:%.*]] = bitcast %struct.uint16x4x4_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.uint16x4x4_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[TMP0]], i8* align 8 [[TMP1]], i64 32, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast %struct.uint16x4x4_t* [[__RET]] to i8*
// CHECK:   [[TMP3:%.*]] = bitcast i16* %a to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.uint16x4x4_t, %struct.uint16x4x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <4 x i16>], [4 x <4 x i16>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP4:%.*]] = load <4 x i16>, <4 x i16>* [[ARRAYIDX]], align 8
// CHECK:   [[TMP5:%.*]] = bitcast <4 x i16> [[TMP4]] to <8 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.uint16x4x4_t, %struct.uint16x4x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <4 x i16>], [4 x <4 x i16>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP6:%.*]] = load <4 x i16>, <4 x i16>* [[ARRAYIDX2]], align 8
// CHECK:   [[TMP7:%.*]] = bitcast <4 x i16> [[TMP6]] to <8 x i8>
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds %struct.uint16x4x4_t, %struct.uint16x4x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <4 x i16>], [4 x <4 x i16>]* [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP8:%.*]] = load <4 x i16>, <4 x i16>* [[ARRAYIDX4]], align 8
// CHECK:   [[TMP9:%.*]] = bitcast <4 x i16> [[TMP8]] to <8 x i8>
// CHECK:   [[VAL5:%.*]] = getelementptr inbounds %struct.uint16x4x4_t, %struct.uint16x4x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <4 x i16>], [4 x <4 x i16>]* [[VAL5]], i64 0, i64 3
// CHECK:   [[TMP10:%.*]] = load <4 x i16>, <4 x i16>* [[ARRAYIDX6]], align 8
// CHECK:   [[TMP11:%.*]] = bitcast <4 x i16> [[TMP10]] to <8 x i8>
// CHECK:   [[TMP12:%.*]] = bitcast <8 x i8> [[TMP5]] to <4 x i16>
// CHECK:   [[TMP13:%.*]] = bitcast <8 x i8> [[TMP7]] to <4 x i16>
// CHECK:   [[TMP14:%.*]] = bitcast <8 x i8> [[TMP9]] to <4 x i16>
// CHECK:   [[TMP15:%.*]] = bitcast <8 x i8> [[TMP11]] to <4 x i16>
// CHECK:   [[VLD4_LANE:%.*]] = call { <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16> } @llvm.aarch64.neon.ld4lane.v4i16.p0i8(<4 x i16> [[TMP12]], <4 x i16> [[TMP13]], <4 x i16> [[TMP14]], <4 x i16> [[TMP15]], i64 3, i8* [[TMP3]])
// CHECK:   [[TMP16:%.*]] = bitcast i8* [[TMP2]] to { <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16> }*
// CHECK:   store { <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16> } [[VLD4_LANE]], { <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16> }* [[TMP16]]
// CHECK:   [[TMP17:%.*]] = bitcast %struct.uint16x4x4_t* [[RETVAL]] to i8*
// CHECK:   [[TMP18:%.*]] = bitcast %struct.uint16x4x4_t* [[__RET]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[TMP17]], i8* align 8 [[TMP18]], i64 32, i1 false)
// CHECK:   [[TMP19:%.*]] = load %struct.uint16x4x4_t, %struct.uint16x4x4_t* [[RETVAL]], align 8
// CHECK:   ret %struct.uint16x4x4_t [[TMP19]]
uint16x4x4_t test_vld4_lane_u16(uint16_t  *a, uint16x4x4_t b) {
  return vld4_lane_u16(a, b, 3);
}

// CHECK-LABEL: define{{.*}} %struct.uint32x2x4_t @test_vld4_lane_u32(i32* noundef %a, [4 x <2 x i32>] %b.coerce) #2 {
// CHECK:   [[RETVAL:%.*]] = alloca %struct.uint32x2x4_t, align 8
// CHECK:   [[B:%.*]] = alloca %struct.uint32x2x4_t, align 8
// CHECK:   [[__RET:%.*]] = alloca %struct.uint32x2x4_t, align 8
// CHECK:   [[__S1:%.*]] = alloca %struct.uint32x2x4_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.uint32x2x4_t, %struct.uint32x2x4_t* [[B]], i32 0, i32 0
// CHECK:   store [4 x <2 x i32>] [[B]].coerce, [4 x <2 x i32>]* [[COERCE_DIVE]], align 8
// CHECK:   [[TMP0:%.*]] = bitcast %struct.uint32x2x4_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.uint32x2x4_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[TMP0]], i8* align 8 [[TMP1]], i64 32, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast %struct.uint32x2x4_t* [[__RET]] to i8*
// CHECK:   [[TMP3:%.*]] = bitcast i32* %a to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.uint32x2x4_t, %struct.uint32x2x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <2 x i32>], [4 x <2 x i32>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP4:%.*]] = load <2 x i32>, <2 x i32>* [[ARRAYIDX]], align 8
// CHECK:   [[TMP5:%.*]] = bitcast <2 x i32> [[TMP4]] to <8 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.uint32x2x4_t, %struct.uint32x2x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <2 x i32>], [4 x <2 x i32>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP6:%.*]] = load <2 x i32>, <2 x i32>* [[ARRAYIDX2]], align 8
// CHECK:   [[TMP7:%.*]] = bitcast <2 x i32> [[TMP6]] to <8 x i8>
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds %struct.uint32x2x4_t, %struct.uint32x2x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <2 x i32>], [4 x <2 x i32>]* [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP8:%.*]] = load <2 x i32>, <2 x i32>* [[ARRAYIDX4]], align 8
// CHECK:   [[TMP9:%.*]] = bitcast <2 x i32> [[TMP8]] to <8 x i8>
// CHECK:   [[VAL5:%.*]] = getelementptr inbounds %struct.uint32x2x4_t, %struct.uint32x2x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <2 x i32>], [4 x <2 x i32>]* [[VAL5]], i64 0, i64 3
// CHECK:   [[TMP10:%.*]] = load <2 x i32>, <2 x i32>* [[ARRAYIDX6]], align 8
// CHECK:   [[TMP11:%.*]] = bitcast <2 x i32> [[TMP10]] to <8 x i8>
// CHECK:   [[TMP12:%.*]] = bitcast <8 x i8> [[TMP5]] to <2 x i32>
// CHECK:   [[TMP13:%.*]] = bitcast <8 x i8> [[TMP7]] to <2 x i32>
// CHECK:   [[TMP14:%.*]] = bitcast <8 x i8> [[TMP9]] to <2 x i32>
// CHECK:   [[TMP15:%.*]] = bitcast <8 x i8> [[TMP11]] to <2 x i32>
// CHECK:   [[VLD4_LANE:%.*]] = call { <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32> } @llvm.aarch64.neon.ld4lane.v2i32.p0i8(<2 x i32> [[TMP12]], <2 x i32> [[TMP13]], <2 x i32> [[TMP14]], <2 x i32> [[TMP15]], i64 1, i8* [[TMP3]])
// CHECK:   [[TMP16:%.*]] = bitcast i8* [[TMP2]] to { <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32> }*
// CHECK:   store { <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32> } [[VLD4_LANE]], { <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32> }* [[TMP16]]
// CHECK:   [[TMP17:%.*]] = bitcast %struct.uint32x2x4_t* [[RETVAL]] to i8*
// CHECK:   [[TMP18:%.*]] = bitcast %struct.uint32x2x4_t* [[__RET]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[TMP17]], i8* align 8 [[TMP18]], i64 32, i1 false)
// CHECK:   [[TMP19:%.*]] = load %struct.uint32x2x4_t, %struct.uint32x2x4_t* [[RETVAL]], align 8
// CHECK:   ret %struct.uint32x2x4_t [[TMP19]]
uint32x2x4_t test_vld4_lane_u32(uint32_t  *a, uint32x2x4_t b) {
  return vld4_lane_u32(a, b, 1);
}

// CHECK-LABEL: define{{.*}} %struct.uint64x1x4_t @test_vld4_lane_u64(i64* noundef %a, [4 x <1 x i64>] %b.coerce) #2 {
// CHECK:   [[RETVAL:%.*]] = alloca %struct.uint64x1x4_t, align 8
// CHECK:   [[B:%.*]] = alloca %struct.uint64x1x4_t, align 8
// CHECK:   [[__RET:%.*]] = alloca %struct.uint64x1x4_t, align 8
// CHECK:   [[__S1:%.*]] = alloca %struct.uint64x1x4_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.uint64x1x4_t, %struct.uint64x1x4_t* [[B]], i32 0, i32 0
// CHECK:   store [4 x <1 x i64>] [[B]].coerce, [4 x <1 x i64>]* [[COERCE_DIVE]], align 8
// CHECK:   [[TMP0:%.*]] = bitcast %struct.uint64x1x4_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.uint64x1x4_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[TMP0]], i8* align 8 [[TMP1]], i64 32, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast %struct.uint64x1x4_t* [[__RET]] to i8*
// CHECK:   [[TMP3:%.*]] = bitcast i64* %a to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.uint64x1x4_t, %struct.uint64x1x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <1 x i64>], [4 x <1 x i64>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP4:%.*]] = load <1 x i64>, <1 x i64>* [[ARRAYIDX]], align 8
// CHECK:   [[TMP5:%.*]] = bitcast <1 x i64> [[TMP4]] to <8 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.uint64x1x4_t, %struct.uint64x1x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <1 x i64>], [4 x <1 x i64>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP6:%.*]] = load <1 x i64>, <1 x i64>* [[ARRAYIDX2]], align 8
// CHECK:   [[TMP7:%.*]] = bitcast <1 x i64> [[TMP6]] to <8 x i8>
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds %struct.uint64x1x4_t, %struct.uint64x1x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <1 x i64>], [4 x <1 x i64>]* [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP8:%.*]] = load <1 x i64>, <1 x i64>* [[ARRAYIDX4]], align 8
// CHECK:   [[TMP9:%.*]] = bitcast <1 x i64> [[TMP8]] to <8 x i8>
// CHECK:   [[VAL5:%.*]] = getelementptr inbounds %struct.uint64x1x4_t, %struct.uint64x1x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <1 x i64>], [4 x <1 x i64>]* [[VAL5]], i64 0, i64 3
// CHECK:   [[TMP10:%.*]] = load <1 x i64>, <1 x i64>* [[ARRAYIDX6]], align 8
// CHECK:   [[TMP11:%.*]] = bitcast <1 x i64> [[TMP10]] to <8 x i8>
// CHECK:   [[TMP12:%.*]] = bitcast <8 x i8> [[TMP5]] to <1 x i64>
// CHECK:   [[TMP13:%.*]] = bitcast <8 x i8> [[TMP7]] to <1 x i64>
// CHECK:   [[TMP14:%.*]] = bitcast <8 x i8> [[TMP9]] to <1 x i64>
// CHECK:   [[TMP15:%.*]] = bitcast <8 x i8> [[TMP11]] to <1 x i64>
// CHECK:   [[VLD4_LANE:%.*]] = call { <1 x i64>, <1 x i64>, <1 x i64>, <1 x i64> } @llvm.aarch64.neon.ld4lane.v1i64.p0i8(<1 x i64> [[TMP12]], <1 x i64> [[TMP13]], <1 x i64> [[TMP14]], <1 x i64> [[TMP15]], i64 0, i8* [[TMP3]])
// CHECK:   [[TMP16:%.*]] = bitcast i8* [[TMP2]] to { <1 x i64>, <1 x i64>, <1 x i64>, <1 x i64> }*
// CHECK:   store { <1 x i64>, <1 x i64>, <1 x i64>, <1 x i64> } [[VLD4_LANE]], { <1 x i64>, <1 x i64>, <1 x i64>, <1 x i64> }* [[TMP16]]
// CHECK:   [[TMP17:%.*]] = bitcast %struct.uint64x1x4_t* [[RETVAL]] to i8*
// CHECK:   [[TMP18:%.*]] = bitcast %struct.uint64x1x4_t* [[__RET]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[TMP17]], i8* align 8 [[TMP18]], i64 32, i1 false)
// CHECK:   [[TMP19:%.*]] = load %struct.uint64x1x4_t, %struct.uint64x1x4_t* [[RETVAL]], align 8
// CHECK:   ret %struct.uint64x1x4_t [[TMP19]]
uint64x1x4_t test_vld4_lane_u64(uint64_t  *a, uint64x1x4_t b) {
  return vld4_lane_u64(a, b, 0);
}

// CHECK-LABEL: define{{.*}} %struct.int8x8x4_t @test_vld4_lane_s8(i8* noundef %a, [4 x <8 x i8>] %b.coerce) #2 {
// CHECK:   [[RETVAL:%.*]] = alloca %struct.int8x8x4_t, align 8
// CHECK:   [[B:%.*]] = alloca %struct.int8x8x4_t, align 8
// CHECK:   [[__RET:%.*]] = alloca %struct.int8x8x4_t, align 8
// CHECK:   [[__S1:%.*]] = alloca %struct.int8x8x4_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.int8x8x4_t, %struct.int8x8x4_t* [[B]], i32 0, i32 0
// CHECK:   store [4 x <8 x i8>] [[B]].coerce, [4 x <8 x i8>]* [[COERCE_DIVE]], align 8
// CHECK:   [[TMP0:%.*]] = bitcast %struct.int8x8x4_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.int8x8x4_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[TMP0]], i8* align 8 [[TMP1]], i64 32, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast %struct.int8x8x4_t* [[__RET]] to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.int8x8x4_t, %struct.int8x8x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <8 x i8>], [4 x <8 x i8>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <8 x i8>, <8 x i8>* [[ARRAYIDX]], align 8
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.int8x8x4_t, %struct.int8x8x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <8 x i8>], [4 x <8 x i8>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP4:%.*]] = load <8 x i8>, <8 x i8>* [[ARRAYIDX2]], align 8
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds %struct.int8x8x4_t, %struct.int8x8x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <8 x i8>], [4 x <8 x i8>]* [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP5:%.*]] = load <8 x i8>, <8 x i8>* [[ARRAYIDX4]], align 8
// CHECK:   [[VAL5:%.*]] = getelementptr inbounds %struct.int8x8x4_t, %struct.int8x8x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <8 x i8>], [4 x <8 x i8>]* [[VAL5]], i64 0, i64 3
// CHECK:   [[TMP6:%.*]] = load <8 x i8>, <8 x i8>* [[ARRAYIDX6]], align 8
// CHECK:   [[VLD4_LANE:%.*]] = call { <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8> } @llvm.aarch64.neon.ld4lane.v8i8.p0i8(<8 x i8> [[TMP3]], <8 x i8> [[TMP4]], <8 x i8> [[TMP5]], <8 x i8> [[TMP6]], i64 7, i8* %a)
// CHECK:   [[TMP7:%.*]] = bitcast i8* [[TMP2]] to { <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8> }*
// CHECK:   store { <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8> } [[VLD4_LANE]], { <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8> }* [[TMP7]]
// CHECK:   [[TMP8:%.*]] = bitcast %struct.int8x8x4_t* [[RETVAL]] to i8*
// CHECK:   [[TMP9:%.*]] = bitcast %struct.int8x8x4_t* [[__RET]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[TMP8]], i8* align 8 [[TMP9]], i64 32, i1 false)
// CHECK:   [[TMP10:%.*]] = load %struct.int8x8x4_t, %struct.int8x8x4_t* [[RETVAL]], align 8
// CHECK:   ret %struct.int8x8x4_t [[TMP10]]
int8x8x4_t test_vld4_lane_s8(int8_t  *a, int8x8x4_t b) {
  return vld4_lane_s8(a, b, 7);
}

// CHECK-LABEL: define{{.*}} %struct.int16x4x4_t @test_vld4_lane_s16(i16* noundef %a, [4 x <4 x i16>] %b.coerce) #2 {
// CHECK:   [[RETVAL:%.*]] = alloca %struct.int16x4x4_t, align 8
// CHECK:   [[B:%.*]] = alloca %struct.int16x4x4_t, align 8
// CHECK:   [[__RET:%.*]] = alloca %struct.int16x4x4_t, align 8
// CHECK:   [[__S1:%.*]] = alloca %struct.int16x4x4_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.int16x4x4_t, %struct.int16x4x4_t* [[B]], i32 0, i32 0
// CHECK:   store [4 x <4 x i16>] [[B]].coerce, [4 x <4 x i16>]* [[COERCE_DIVE]], align 8
// CHECK:   [[TMP0:%.*]] = bitcast %struct.int16x4x4_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.int16x4x4_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[TMP0]], i8* align 8 [[TMP1]], i64 32, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast %struct.int16x4x4_t* [[__RET]] to i8*
// CHECK:   [[TMP3:%.*]] = bitcast i16* %a to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.int16x4x4_t, %struct.int16x4x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <4 x i16>], [4 x <4 x i16>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP4:%.*]] = load <4 x i16>, <4 x i16>* [[ARRAYIDX]], align 8
// CHECK:   [[TMP5:%.*]] = bitcast <4 x i16> [[TMP4]] to <8 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.int16x4x4_t, %struct.int16x4x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <4 x i16>], [4 x <4 x i16>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP6:%.*]] = load <4 x i16>, <4 x i16>* [[ARRAYIDX2]], align 8
// CHECK:   [[TMP7:%.*]] = bitcast <4 x i16> [[TMP6]] to <8 x i8>
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds %struct.int16x4x4_t, %struct.int16x4x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <4 x i16>], [4 x <4 x i16>]* [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP8:%.*]] = load <4 x i16>, <4 x i16>* [[ARRAYIDX4]], align 8
// CHECK:   [[TMP9:%.*]] = bitcast <4 x i16> [[TMP8]] to <8 x i8>
// CHECK:   [[VAL5:%.*]] = getelementptr inbounds %struct.int16x4x4_t, %struct.int16x4x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <4 x i16>], [4 x <4 x i16>]* [[VAL5]], i64 0, i64 3
// CHECK:   [[TMP10:%.*]] = load <4 x i16>, <4 x i16>* [[ARRAYIDX6]], align 8
// CHECK:   [[TMP11:%.*]] = bitcast <4 x i16> [[TMP10]] to <8 x i8>
// CHECK:   [[TMP12:%.*]] = bitcast <8 x i8> [[TMP5]] to <4 x i16>
// CHECK:   [[TMP13:%.*]] = bitcast <8 x i8> [[TMP7]] to <4 x i16>
// CHECK:   [[TMP14:%.*]] = bitcast <8 x i8> [[TMP9]] to <4 x i16>
// CHECK:   [[TMP15:%.*]] = bitcast <8 x i8> [[TMP11]] to <4 x i16>
// CHECK:   [[VLD4_LANE:%.*]] = call { <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16> } @llvm.aarch64.neon.ld4lane.v4i16.p0i8(<4 x i16> [[TMP12]], <4 x i16> [[TMP13]], <4 x i16> [[TMP14]], <4 x i16> [[TMP15]], i64 3, i8* [[TMP3]])
// CHECK:   [[TMP16:%.*]] = bitcast i8* [[TMP2]] to { <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16> }*
// CHECK:   store { <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16> } [[VLD4_LANE]], { <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16> }* [[TMP16]]
// CHECK:   [[TMP17:%.*]] = bitcast %struct.int16x4x4_t* [[RETVAL]] to i8*
// CHECK:   [[TMP18:%.*]] = bitcast %struct.int16x4x4_t* [[__RET]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[TMP17]], i8* align 8 [[TMP18]], i64 32, i1 false)
// CHECK:   [[TMP19:%.*]] = load %struct.int16x4x4_t, %struct.int16x4x4_t* [[RETVAL]], align 8
// CHECK:   ret %struct.int16x4x4_t [[TMP19]]
int16x4x4_t test_vld4_lane_s16(int16_t  *a, int16x4x4_t b) {
  return vld4_lane_s16(a, b, 3);
}

// CHECK-LABEL: define{{.*}} %struct.int32x2x4_t @test_vld4_lane_s32(i32* noundef %a, [4 x <2 x i32>] %b.coerce) #2 {
// CHECK:   [[RETVAL:%.*]] = alloca %struct.int32x2x4_t, align 8
// CHECK:   [[B:%.*]] = alloca %struct.int32x2x4_t, align 8
// CHECK:   [[__RET:%.*]] = alloca %struct.int32x2x4_t, align 8
// CHECK:   [[__S1:%.*]] = alloca %struct.int32x2x4_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.int32x2x4_t, %struct.int32x2x4_t* [[B]], i32 0, i32 0
// CHECK:   store [4 x <2 x i32>] [[B]].coerce, [4 x <2 x i32>]* [[COERCE_DIVE]], align 8
// CHECK:   [[TMP0:%.*]] = bitcast %struct.int32x2x4_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.int32x2x4_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[TMP0]], i8* align 8 [[TMP1]], i64 32, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast %struct.int32x2x4_t* [[__RET]] to i8*
// CHECK:   [[TMP3:%.*]] = bitcast i32* %a to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.int32x2x4_t, %struct.int32x2x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <2 x i32>], [4 x <2 x i32>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP4:%.*]] = load <2 x i32>, <2 x i32>* [[ARRAYIDX]], align 8
// CHECK:   [[TMP5:%.*]] = bitcast <2 x i32> [[TMP4]] to <8 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.int32x2x4_t, %struct.int32x2x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <2 x i32>], [4 x <2 x i32>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP6:%.*]] = load <2 x i32>, <2 x i32>* [[ARRAYIDX2]], align 8
// CHECK:   [[TMP7:%.*]] = bitcast <2 x i32> [[TMP6]] to <8 x i8>
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds %struct.int32x2x4_t, %struct.int32x2x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <2 x i32>], [4 x <2 x i32>]* [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP8:%.*]] = load <2 x i32>, <2 x i32>* [[ARRAYIDX4]], align 8
// CHECK:   [[TMP9:%.*]] = bitcast <2 x i32> [[TMP8]] to <8 x i8>
// CHECK:   [[VAL5:%.*]] = getelementptr inbounds %struct.int32x2x4_t, %struct.int32x2x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <2 x i32>], [4 x <2 x i32>]* [[VAL5]], i64 0, i64 3
// CHECK:   [[TMP10:%.*]] = load <2 x i32>, <2 x i32>* [[ARRAYIDX6]], align 8
// CHECK:   [[TMP11:%.*]] = bitcast <2 x i32> [[TMP10]] to <8 x i8>
// CHECK:   [[TMP12:%.*]] = bitcast <8 x i8> [[TMP5]] to <2 x i32>
// CHECK:   [[TMP13:%.*]] = bitcast <8 x i8> [[TMP7]] to <2 x i32>
// CHECK:   [[TMP14:%.*]] = bitcast <8 x i8> [[TMP9]] to <2 x i32>
// CHECK:   [[TMP15:%.*]] = bitcast <8 x i8> [[TMP11]] to <2 x i32>
// CHECK:   [[VLD4_LANE:%.*]] = call { <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32> } @llvm.aarch64.neon.ld4lane.v2i32.p0i8(<2 x i32> [[TMP12]], <2 x i32> [[TMP13]], <2 x i32> [[TMP14]], <2 x i32> [[TMP15]], i64 1, i8* [[TMP3]])
// CHECK:   [[TMP16:%.*]] = bitcast i8* [[TMP2]] to { <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32> }*
// CHECK:   store { <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32> } [[VLD4_LANE]], { <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32> }* [[TMP16]]
// CHECK:   [[TMP17:%.*]] = bitcast %struct.int32x2x4_t* [[RETVAL]] to i8*
// CHECK:   [[TMP18:%.*]] = bitcast %struct.int32x2x4_t* [[__RET]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[TMP17]], i8* align 8 [[TMP18]], i64 32, i1 false)
// CHECK:   [[TMP19:%.*]] = load %struct.int32x2x4_t, %struct.int32x2x4_t* [[RETVAL]], align 8
// CHECK:   ret %struct.int32x2x4_t [[TMP19]]
int32x2x4_t test_vld4_lane_s32(int32_t  *a, int32x2x4_t b) {
  return vld4_lane_s32(a, b, 1);
}

// CHECK-LABEL: define{{.*}} %struct.int64x1x4_t @test_vld4_lane_s64(i64* noundef %a, [4 x <1 x i64>] %b.coerce) #2 {
// CHECK:   [[RETVAL:%.*]] = alloca %struct.int64x1x4_t, align 8
// CHECK:   [[B:%.*]] = alloca %struct.int64x1x4_t, align 8
// CHECK:   [[__RET:%.*]] = alloca %struct.int64x1x4_t, align 8
// CHECK:   [[__S1:%.*]] = alloca %struct.int64x1x4_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.int64x1x4_t, %struct.int64x1x4_t* [[B]], i32 0, i32 0
// CHECK:   store [4 x <1 x i64>] [[B]].coerce, [4 x <1 x i64>]* [[COERCE_DIVE]], align 8
// CHECK:   [[TMP0:%.*]] = bitcast %struct.int64x1x4_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.int64x1x4_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[TMP0]], i8* align 8 [[TMP1]], i64 32, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast %struct.int64x1x4_t* [[__RET]] to i8*
// CHECK:   [[TMP3:%.*]] = bitcast i64* %a to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.int64x1x4_t, %struct.int64x1x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <1 x i64>], [4 x <1 x i64>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP4:%.*]] = load <1 x i64>, <1 x i64>* [[ARRAYIDX]], align 8
// CHECK:   [[TMP5:%.*]] = bitcast <1 x i64> [[TMP4]] to <8 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.int64x1x4_t, %struct.int64x1x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <1 x i64>], [4 x <1 x i64>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP6:%.*]] = load <1 x i64>, <1 x i64>* [[ARRAYIDX2]], align 8
// CHECK:   [[TMP7:%.*]] = bitcast <1 x i64> [[TMP6]] to <8 x i8>
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds %struct.int64x1x4_t, %struct.int64x1x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <1 x i64>], [4 x <1 x i64>]* [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP8:%.*]] = load <1 x i64>, <1 x i64>* [[ARRAYIDX4]], align 8
// CHECK:   [[TMP9:%.*]] = bitcast <1 x i64> [[TMP8]] to <8 x i8>
// CHECK:   [[VAL5:%.*]] = getelementptr inbounds %struct.int64x1x4_t, %struct.int64x1x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <1 x i64>], [4 x <1 x i64>]* [[VAL5]], i64 0, i64 3
// CHECK:   [[TMP10:%.*]] = load <1 x i64>, <1 x i64>* [[ARRAYIDX6]], align 8
// CHECK:   [[TMP11:%.*]] = bitcast <1 x i64> [[TMP10]] to <8 x i8>
// CHECK:   [[TMP12:%.*]] = bitcast <8 x i8> [[TMP5]] to <1 x i64>
// CHECK:   [[TMP13:%.*]] = bitcast <8 x i8> [[TMP7]] to <1 x i64>
// CHECK:   [[TMP14:%.*]] = bitcast <8 x i8> [[TMP9]] to <1 x i64>
// CHECK:   [[TMP15:%.*]] = bitcast <8 x i8> [[TMP11]] to <1 x i64>
// CHECK:   [[VLD4_LANE:%.*]] = call { <1 x i64>, <1 x i64>, <1 x i64>, <1 x i64> } @llvm.aarch64.neon.ld4lane.v1i64.p0i8(<1 x i64> [[TMP12]], <1 x i64> [[TMP13]], <1 x i64> [[TMP14]], <1 x i64> [[TMP15]], i64 0, i8* [[TMP3]])
// CHECK:   [[TMP16:%.*]] = bitcast i8* [[TMP2]] to { <1 x i64>, <1 x i64>, <1 x i64>, <1 x i64> }*
// CHECK:   store { <1 x i64>, <1 x i64>, <1 x i64>, <1 x i64> } [[VLD4_LANE]], { <1 x i64>, <1 x i64>, <1 x i64>, <1 x i64> }* [[TMP16]]
// CHECK:   [[TMP17:%.*]] = bitcast %struct.int64x1x4_t* [[RETVAL]] to i8*
// CHECK:   [[TMP18:%.*]] = bitcast %struct.int64x1x4_t* [[__RET]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[TMP17]], i8* align 8 [[TMP18]], i64 32, i1 false)
// CHECK:   [[TMP19:%.*]] = load %struct.int64x1x4_t, %struct.int64x1x4_t* [[RETVAL]], align 8
// CHECK:   ret %struct.int64x1x4_t [[TMP19]]
int64x1x4_t test_vld4_lane_s64(int64_t  *a, int64x1x4_t b) {
  return vld4_lane_s64(a, b, 0);
}

// CHECK-LABEL: define{{.*}} %struct.float16x4x4_t @test_vld4_lane_f16(half* noundef %a, [4 x <4 x half>] %b.coerce) #2 {
// CHECK:   [[RETVAL:%.*]] = alloca %struct.float16x4x4_t, align 8
// CHECK:   [[B:%.*]] = alloca %struct.float16x4x4_t, align 8
// CHECK:   [[__RET:%.*]] = alloca %struct.float16x4x4_t, align 8
// CHECK:   [[__S1:%.*]] = alloca %struct.float16x4x4_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.float16x4x4_t, %struct.float16x4x4_t* [[B]], i32 0, i32 0
// CHECK:   store [4 x <4 x half>] [[B]].coerce, [4 x <4 x half>]* [[COERCE_DIVE]], align 8
// CHECK:   [[TMP0:%.*]] = bitcast %struct.float16x4x4_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.float16x4x4_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[TMP0]], i8* align 8 [[TMP1]], i64 32, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast %struct.float16x4x4_t* [[__RET]] to i8*
// CHECK:   [[TMP3:%.*]] = bitcast half* %a to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.float16x4x4_t, %struct.float16x4x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <4 x half>], [4 x <4 x half>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP4:%.*]] = load <4 x half>, <4 x half>* [[ARRAYIDX]], align 8
// CHECK:   [[TMP5:%.*]] = bitcast <4 x half> [[TMP4]] to <8 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.float16x4x4_t, %struct.float16x4x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <4 x half>], [4 x <4 x half>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP6:%.*]] = load <4 x half>, <4 x half>* [[ARRAYIDX2]], align 8
// CHECK:   [[TMP7:%.*]] = bitcast <4 x half> [[TMP6]] to <8 x i8>
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds %struct.float16x4x4_t, %struct.float16x4x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <4 x half>], [4 x <4 x half>]* [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP8:%.*]] = load <4 x half>, <4 x half>* [[ARRAYIDX4]], align 8
// CHECK:   [[TMP9:%.*]] = bitcast <4 x half> [[TMP8]] to <8 x i8>
// CHECK:   [[VAL5:%.*]] = getelementptr inbounds %struct.float16x4x4_t, %struct.float16x4x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <4 x half>], [4 x <4 x half>]* [[VAL5]], i64 0, i64 3
// CHECK:   [[TMP10:%.*]] = load <4 x half>, <4 x half>* [[ARRAYIDX6]], align 8
// CHECK:   [[TMP11:%.*]] = bitcast <4 x half> [[TMP10]] to <8 x i8>
// CHECK:   [[TMP12:%.*]] = bitcast <8 x i8> [[TMP5]] to <4 x half>
// CHECK:   [[TMP13:%.*]] = bitcast <8 x i8> [[TMP7]] to <4 x half>
// CHECK:   [[TMP14:%.*]] = bitcast <8 x i8> [[TMP9]] to <4 x half>
// CHECK:   [[TMP15:%.*]] = bitcast <8 x i8> [[TMP11]] to <4 x half>
// CHECK:   [[VLD4_LANE:%.*]] = call { <4 x half>, <4 x half>, <4 x half>, <4 x half> } @llvm.aarch64.neon.ld4lane.v4f16.p0i8(<4 x half> [[TMP12]], <4 x half> [[TMP13]], <4 x half> [[TMP14]], <4 x half> [[TMP15]], i64 3, i8* [[TMP3]])
// CHECK:   [[TMP16:%.*]] = bitcast i8* [[TMP2]] to { <4 x half>, <4 x half>, <4 x half>, <4 x half> }*
// CHECK:   store { <4 x half>, <4 x half>, <4 x half>, <4 x half> } [[VLD4_LANE]], { <4 x half>, <4 x half>, <4 x half>, <4 x half> }* [[TMP16]]
// CHECK:   [[TMP17:%.*]] = bitcast %struct.float16x4x4_t* [[RETVAL]] to i8*
// CHECK:   [[TMP18:%.*]] = bitcast %struct.float16x4x4_t* [[__RET]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[TMP17]], i8* align 8 [[TMP18]], i64 32, i1 false)
// CHECK:   [[TMP19:%.*]] = load %struct.float16x4x4_t, %struct.float16x4x4_t* [[RETVAL]], align 8
// CHECK:   ret %struct.float16x4x4_t [[TMP19]]
float16x4x4_t test_vld4_lane_f16(float16_t  *a, float16x4x4_t b) {
  return vld4_lane_f16(a, b, 3);
}

// CHECK-LABEL: define{{.*}} %struct.float32x2x4_t @test_vld4_lane_f32(float* noundef %a, [4 x <2 x float>] %b.coerce) #2 {
// CHECK:   [[RETVAL:%.*]] = alloca %struct.float32x2x4_t, align 8
// CHECK:   [[B:%.*]] = alloca %struct.float32x2x4_t, align 8
// CHECK:   [[__RET:%.*]] = alloca %struct.float32x2x4_t, align 8
// CHECK:   [[__S1:%.*]] = alloca %struct.float32x2x4_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.float32x2x4_t, %struct.float32x2x4_t* [[B]], i32 0, i32 0
// CHECK:   store [4 x <2 x float>] [[B]].coerce, [4 x <2 x float>]* [[COERCE_DIVE]], align 8
// CHECK:   [[TMP0:%.*]] = bitcast %struct.float32x2x4_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.float32x2x4_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[TMP0]], i8* align 8 [[TMP1]], i64 32, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast %struct.float32x2x4_t* [[__RET]] to i8*
// CHECK:   [[TMP3:%.*]] = bitcast float* %a to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.float32x2x4_t, %struct.float32x2x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <2 x float>], [4 x <2 x float>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP4:%.*]] = load <2 x float>, <2 x float>* [[ARRAYIDX]], align 8
// CHECK:   [[TMP5:%.*]] = bitcast <2 x float> [[TMP4]] to <8 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.float32x2x4_t, %struct.float32x2x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <2 x float>], [4 x <2 x float>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP6:%.*]] = load <2 x float>, <2 x float>* [[ARRAYIDX2]], align 8
// CHECK:   [[TMP7:%.*]] = bitcast <2 x float> [[TMP6]] to <8 x i8>
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds %struct.float32x2x4_t, %struct.float32x2x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <2 x float>], [4 x <2 x float>]* [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP8:%.*]] = load <2 x float>, <2 x float>* [[ARRAYIDX4]], align 8
// CHECK:   [[TMP9:%.*]] = bitcast <2 x float> [[TMP8]] to <8 x i8>
// CHECK:   [[VAL5:%.*]] = getelementptr inbounds %struct.float32x2x4_t, %struct.float32x2x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <2 x float>], [4 x <2 x float>]* [[VAL5]], i64 0, i64 3
// CHECK:   [[TMP10:%.*]] = load <2 x float>, <2 x float>* [[ARRAYIDX6]], align 8
// CHECK:   [[TMP11:%.*]] = bitcast <2 x float> [[TMP10]] to <8 x i8>
// CHECK:   [[TMP12:%.*]] = bitcast <8 x i8> [[TMP5]] to <2 x float>
// CHECK:   [[TMP13:%.*]] = bitcast <8 x i8> [[TMP7]] to <2 x float>
// CHECK:   [[TMP14:%.*]] = bitcast <8 x i8> [[TMP9]] to <2 x float>
// CHECK:   [[TMP15:%.*]] = bitcast <8 x i8> [[TMP11]] to <2 x float>
// CHECK:   [[VLD4_LANE:%.*]] = call { <2 x float>, <2 x float>, <2 x float>, <2 x float> } @llvm.aarch64.neon.ld4lane.v2f32.p0i8(<2 x float> [[TMP12]], <2 x float> [[TMP13]], <2 x float> [[TMP14]], <2 x float> [[TMP15]], i64 1, i8* [[TMP3]])
// CHECK:   [[TMP16:%.*]] = bitcast i8* [[TMP2]] to { <2 x float>, <2 x float>, <2 x float>, <2 x float> }*
// CHECK:   store { <2 x float>, <2 x float>, <2 x float>, <2 x float> } [[VLD4_LANE]], { <2 x float>, <2 x float>, <2 x float>, <2 x float> }* [[TMP16]]
// CHECK:   [[TMP17:%.*]] = bitcast %struct.float32x2x4_t* [[RETVAL]] to i8*
// CHECK:   [[TMP18:%.*]] = bitcast %struct.float32x2x4_t* [[__RET]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[TMP17]], i8* align 8 [[TMP18]], i64 32, i1 false)
// CHECK:   [[TMP19:%.*]] = load %struct.float32x2x4_t, %struct.float32x2x4_t* [[RETVAL]], align 8
// CHECK:   ret %struct.float32x2x4_t [[TMP19]]
float32x2x4_t test_vld4_lane_f32(float32_t  *a, float32x2x4_t b) {
  return vld4_lane_f32(a, b, 1);
}

// CHECK-LABEL: define{{.*}} %struct.float64x1x4_t @test_vld4_lane_f64(double* noundef %a, [4 x <1 x double>] %b.coerce) #2 {
// CHECK:   [[RETVAL:%.*]] = alloca %struct.float64x1x4_t, align 8
// CHECK:   [[B:%.*]] = alloca %struct.float64x1x4_t, align 8
// CHECK:   [[__RET:%.*]] = alloca %struct.float64x1x4_t, align 8
// CHECK:   [[__S1:%.*]] = alloca %struct.float64x1x4_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.float64x1x4_t, %struct.float64x1x4_t* [[B]], i32 0, i32 0
// CHECK:   store [4 x <1 x double>] [[B]].coerce, [4 x <1 x double>]* [[COERCE_DIVE]], align 8
// CHECK:   [[TMP0:%.*]] = bitcast %struct.float64x1x4_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.float64x1x4_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[TMP0]], i8* align 8 [[TMP1]], i64 32, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast %struct.float64x1x4_t* [[__RET]] to i8*
// CHECK:   [[TMP3:%.*]] = bitcast double* %a to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.float64x1x4_t, %struct.float64x1x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <1 x double>], [4 x <1 x double>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP4:%.*]] = load <1 x double>, <1 x double>* [[ARRAYIDX]], align 8
// CHECK:   [[TMP5:%.*]] = bitcast <1 x double> [[TMP4]] to <8 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.float64x1x4_t, %struct.float64x1x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <1 x double>], [4 x <1 x double>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP6:%.*]] = load <1 x double>, <1 x double>* [[ARRAYIDX2]], align 8
// CHECK:   [[TMP7:%.*]] = bitcast <1 x double> [[TMP6]] to <8 x i8>
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds %struct.float64x1x4_t, %struct.float64x1x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <1 x double>], [4 x <1 x double>]* [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP8:%.*]] = load <1 x double>, <1 x double>* [[ARRAYIDX4]], align 8
// CHECK:   [[TMP9:%.*]] = bitcast <1 x double> [[TMP8]] to <8 x i8>
// CHECK:   [[VAL5:%.*]] = getelementptr inbounds %struct.float64x1x4_t, %struct.float64x1x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <1 x double>], [4 x <1 x double>]* [[VAL5]], i64 0, i64 3
// CHECK:   [[TMP10:%.*]] = load <1 x double>, <1 x double>* [[ARRAYIDX6]], align 8
// CHECK:   [[TMP11:%.*]] = bitcast <1 x double> [[TMP10]] to <8 x i8>
// CHECK:   [[TMP12:%.*]] = bitcast <8 x i8> [[TMP5]] to <1 x double>
// CHECK:   [[TMP13:%.*]] = bitcast <8 x i8> [[TMP7]] to <1 x double>
// CHECK:   [[TMP14:%.*]] = bitcast <8 x i8> [[TMP9]] to <1 x double>
// CHECK:   [[TMP15:%.*]] = bitcast <8 x i8> [[TMP11]] to <1 x double>
// CHECK:   [[VLD4_LANE:%.*]] = call { <1 x double>, <1 x double>, <1 x double>, <1 x double> } @llvm.aarch64.neon.ld4lane.v1f64.p0i8(<1 x double> [[TMP12]], <1 x double> [[TMP13]], <1 x double> [[TMP14]], <1 x double> [[TMP15]], i64 0, i8* [[TMP3]])
// CHECK:   [[TMP16:%.*]] = bitcast i8* [[TMP2]] to { <1 x double>, <1 x double>, <1 x double>, <1 x double> }*
// CHECK:   store { <1 x double>, <1 x double>, <1 x double>, <1 x double> } [[VLD4_LANE]], { <1 x double>, <1 x double>, <1 x double>, <1 x double> }* [[TMP16]]
// CHECK:   [[TMP17:%.*]] = bitcast %struct.float64x1x4_t* [[RETVAL]] to i8*
// CHECK:   [[TMP18:%.*]] = bitcast %struct.float64x1x4_t* [[__RET]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[TMP17]], i8* align 8 [[TMP18]], i64 32, i1 false)
// CHECK:   [[TMP19:%.*]] = load %struct.float64x1x4_t, %struct.float64x1x4_t* [[RETVAL]], align 8
// CHECK:   ret %struct.float64x1x4_t [[TMP19]]
float64x1x4_t test_vld4_lane_f64(float64_t  *a, float64x1x4_t b) {
  return vld4_lane_f64(a, b, 0);
}

// CHECK-LABEL: define{{.*}} %struct.poly8x8x4_t @test_vld4_lane_p8(i8* noundef %a, [4 x <8 x i8>] %b.coerce) #2 {
// CHECK:   [[RETVAL:%.*]] = alloca %struct.poly8x8x4_t, align 8
// CHECK:   [[B:%.*]] = alloca %struct.poly8x8x4_t, align 8
// CHECK:   [[__RET:%.*]] = alloca %struct.poly8x8x4_t, align 8
// CHECK:   [[__S1:%.*]] = alloca %struct.poly8x8x4_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.poly8x8x4_t, %struct.poly8x8x4_t* [[B]], i32 0, i32 0
// CHECK:   store [4 x <8 x i8>] [[B]].coerce, [4 x <8 x i8>]* [[COERCE_DIVE]], align 8
// CHECK:   [[TMP0:%.*]] = bitcast %struct.poly8x8x4_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.poly8x8x4_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[TMP0]], i8* align 8 [[TMP1]], i64 32, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast %struct.poly8x8x4_t* [[__RET]] to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.poly8x8x4_t, %struct.poly8x8x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <8 x i8>], [4 x <8 x i8>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <8 x i8>, <8 x i8>* [[ARRAYIDX]], align 8
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.poly8x8x4_t, %struct.poly8x8x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <8 x i8>], [4 x <8 x i8>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP4:%.*]] = load <8 x i8>, <8 x i8>* [[ARRAYIDX2]], align 8
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds %struct.poly8x8x4_t, %struct.poly8x8x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <8 x i8>], [4 x <8 x i8>]* [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP5:%.*]] = load <8 x i8>, <8 x i8>* [[ARRAYIDX4]], align 8
// CHECK:   [[VAL5:%.*]] = getelementptr inbounds %struct.poly8x8x4_t, %struct.poly8x8x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <8 x i8>], [4 x <8 x i8>]* [[VAL5]], i64 0, i64 3
// CHECK:   [[TMP6:%.*]] = load <8 x i8>, <8 x i8>* [[ARRAYIDX6]], align 8
// CHECK:   [[VLD4_LANE:%.*]] = call { <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8> } @llvm.aarch64.neon.ld4lane.v8i8.p0i8(<8 x i8> [[TMP3]], <8 x i8> [[TMP4]], <8 x i8> [[TMP5]], <8 x i8> [[TMP6]], i64 7, i8* %a)
// CHECK:   [[TMP7:%.*]] = bitcast i8* [[TMP2]] to { <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8> }*
// CHECK:   store { <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8> } [[VLD4_LANE]], { <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8> }* [[TMP7]]
// CHECK:   [[TMP8:%.*]] = bitcast %struct.poly8x8x4_t* [[RETVAL]] to i8*
// CHECK:   [[TMP9:%.*]] = bitcast %struct.poly8x8x4_t* [[__RET]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[TMP8]], i8* align 8 [[TMP9]], i64 32, i1 false)
// CHECK:   [[TMP10:%.*]] = load %struct.poly8x8x4_t, %struct.poly8x8x4_t* [[RETVAL]], align 8
// CHECK:   ret %struct.poly8x8x4_t [[TMP10]]
poly8x8x4_t test_vld4_lane_p8(poly8_t  *a, poly8x8x4_t b) {
  return vld4_lane_p8(a, b, 7);
}

// CHECK-LABEL: define{{.*}} %struct.poly16x4x4_t @test_vld4_lane_p16(i16* noundef %a, [4 x <4 x i16>] %b.coerce) #2 {
// CHECK:   [[RETVAL:%.*]] = alloca %struct.poly16x4x4_t, align 8
// CHECK:   [[B:%.*]] = alloca %struct.poly16x4x4_t, align 8
// CHECK:   [[__RET:%.*]] = alloca %struct.poly16x4x4_t, align 8
// CHECK:   [[__S1:%.*]] = alloca %struct.poly16x4x4_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.poly16x4x4_t, %struct.poly16x4x4_t* [[B]], i32 0, i32 0
// CHECK:   store [4 x <4 x i16>] [[B]].coerce, [4 x <4 x i16>]* [[COERCE_DIVE]], align 8
// CHECK:   [[TMP0:%.*]] = bitcast %struct.poly16x4x4_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.poly16x4x4_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[TMP0]], i8* align 8 [[TMP1]], i64 32, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast %struct.poly16x4x4_t* [[__RET]] to i8*
// CHECK:   [[TMP3:%.*]] = bitcast i16* %a to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.poly16x4x4_t, %struct.poly16x4x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <4 x i16>], [4 x <4 x i16>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP4:%.*]] = load <4 x i16>, <4 x i16>* [[ARRAYIDX]], align 8
// CHECK:   [[TMP5:%.*]] = bitcast <4 x i16> [[TMP4]] to <8 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.poly16x4x4_t, %struct.poly16x4x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <4 x i16>], [4 x <4 x i16>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP6:%.*]] = load <4 x i16>, <4 x i16>* [[ARRAYIDX2]], align 8
// CHECK:   [[TMP7:%.*]] = bitcast <4 x i16> [[TMP6]] to <8 x i8>
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds %struct.poly16x4x4_t, %struct.poly16x4x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <4 x i16>], [4 x <4 x i16>]* [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP8:%.*]] = load <4 x i16>, <4 x i16>* [[ARRAYIDX4]], align 8
// CHECK:   [[TMP9:%.*]] = bitcast <4 x i16> [[TMP8]] to <8 x i8>
// CHECK:   [[VAL5:%.*]] = getelementptr inbounds %struct.poly16x4x4_t, %struct.poly16x4x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <4 x i16>], [4 x <4 x i16>]* [[VAL5]], i64 0, i64 3
// CHECK:   [[TMP10:%.*]] = load <4 x i16>, <4 x i16>* [[ARRAYIDX6]], align 8
// CHECK:   [[TMP11:%.*]] = bitcast <4 x i16> [[TMP10]] to <8 x i8>
// CHECK:   [[TMP12:%.*]] = bitcast <8 x i8> [[TMP5]] to <4 x i16>
// CHECK:   [[TMP13:%.*]] = bitcast <8 x i8> [[TMP7]] to <4 x i16>
// CHECK:   [[TMP14:%.*]] = bitcast <8 x i8> [[TMP9]] to <4 x i16>
// CHECK:   [[TMP15:%.*]] = bitcast <8 x i8> [[TMP11]] to <4 x i16>
// CHECK:   [[VLD4_LANE:%.*]] = call { <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16> } @llvm.aarch64.neon.ld4lane.v4i16.p0i8(<4 x i16> [[TMP12]], <4 x i16> [[TMP13]], <4 x i16> [[TMP14]], <4 x i16> [[TMP15]], i64 3, i8* [[TMP3]])
// CHECK:   [[TMP16:%.*]] = bitcast i8* [[TMP2]] to { <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16> }*
// CHECK:   store { <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16> } [[VLD4_LANE]], { <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16> }* [[TMP16]]
// CHECK:   [[TMP17:%.*]] = bitcast %struct.poly16x4x4_t* [[RETVAL]] to i8*
// CHECK:   [[TMP18:%.*]] = bitcast %struct.poly16x4x4_t* [[__RET]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[TMP17]], i8* align 8 [[TMP18]], i64 32, i1 false)
// CHECK:   [[TMP19:%.*]] = load %struct.poly16x4x4_t, %struct.poly16x4x4_t* [[RETVAL]], align 8
// CHECK:   ret %struct.poly16x4x4_t [[TMP19]]
poly16x4x4_t test_vld4_lane_p16(poly16_t  *a, poly16x4x4_t b) {
  return vld4_lane_p16(a, b, 3);
}

// CHECK-LABEL: define{{.*}} %struct.poly64x1x4_t @test_vld4_lane_p64(i64* noundef %a, [4 x <1 x i64>] %b.coerce) #2 {
// CHECK:   [[RETVAL:%.*]] = alloca %struct.poly64x1x4_t, align 8
// CHECK:   [[B:%.*]] = alloca %struct.poly64x1x4_t, align 8
// CHECK:   [[__RET:%.*]] = alloca %struct.poly64x1x4_t, align 8
// CHECK:   [[__S1:%.*]] = alloca %struct.poly64x1x4_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.poly64x1x4_t, %struct.poly64x1x4_t* [[B]], i32 0, i32 0
// CHECK:   store [4 x <1 x i64>] [[B]].coerce, [4 x <1 x i64>]* [[COERCE_DIVE]], align 8
// CHECK:   [[TMP0:%.*]] = bitcast %struct.poly64x1x4_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.poly64x1x4_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[TMP0]], i8* align 8 [[TMP1]], i64 32, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast %struct.poly64x1x4_t* [[__RET]] to i8*
// CHECK:   [[TMP3:%.*]] = bitcast i64* %a to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.poly64x1x4_t, %struct.poly64x1x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <1 x i64>], [4 x <1 x i64>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP4:%.*]] = load <1 x i64>, <1 x i64>* [[ARRAYIDX]], align 8
// CHECK:   [[TMP5:%.*]] = bitcast <1 x i64> [[TMP4]] to <8 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.poly64x1x4_t, %struct.poly64x1x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <1 x i64>], [4 x <1 x i64>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP6:%.*]] = load <1 x i64>, <1 x i64>* [[ARRAYIDX2]], align 8
// CHECK:   [[TMP7:%.*]] = bitcast <1 x i64> [[TMP6]] to <8 x i8>
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds %struct.poly64x1x4_t, %struct.poly64x1x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <1 x i64>], [4 x <1 x i64>]* [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP8:%.*]] = load <1 x i64>, <1 x i64>* [[ARRAYIDX4]], align 8
// CHECK:   [[TMP9:%.*]] = bitcast <1 x i64> [[TMP8]] to <8 x i8>
// CHECK:   [[VAL5:%.*]] = getelementptr inbounds %struct.poly64x1x4_t, %struct.poly64x1x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <1 x i64>], [4 x <1 x i64>]* [[VAL5]], i64 0, i64 3
// CHECK:   [[TMP10:%.*]] = load <1 x i64>, <1 x i64>* [[ARRAYIDX6]], align 8
// CHECK:   [[TMP11:%.*]] = bitcast <1 x i64> [[TMP10]] to <8 x i8>
// CHECK:   [[TMP12:%.*]] = bitcast <8 x i8> [[TMP5]] to <1 x i64>
// CHECK:   [[TMP13:%.*]] = bitcast <8 x i8> [[TMP7]] to <1 x i64>
// CHECK:   [[TMP14:%.*]] = bitcast <8 x i8> [[TMP9]] to <1 x i64>
// CHECK:   [[TMP15:%.*]] = bitcast <8 x i8> [[TMP11]] to <1 x i64>
// CHECK:   [[VLD4_LANE:%.*]] = call { <1 x i64>, <1 x i64>, <1 x i64>, <1 x i64> } @llvm.aarch64.neon.ld4lane.v1i64.p0i8(<1 x i64> [[TMP12]], <1 x i64> [[TMP13]], <1 x i64> [[TMP14]], <1 x i64> [[TMP15]], i64 0, i8* [[TMP3]])
// CHECK:   [[TMP16:%.*]] = bitcast i8* [[TMP2]] to { <1 x i64>, <1 x i64>, <1 x i64>, <1 x i64> }*
// CHECK:   store { <1 x i64>, <1 x i64>, <1 x i64>, <1 x i64> } [[VLD4_LANE]], { <1 x i64>, <1 x i64>, <1 x i64>, <1 x i64> }* [[TMP16]]
// CHECK:   [[TMP17:%.*]] = bitcast %struct.poly64x1x4_t* [[RETVAL]] to i8*
// CHECK:   [[TMP18:%.*]] = bitcast %struct.poly64x1x4_t* [[__RET]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[TMP17]], i8* align 8 [[TMP18]], i64 32, i1 false)
// CHECK:   [[TMP19:%.*]] = load %struct.poly64x1x4_t, %struct.poly64x1x4_t* [[RETVAL]], align 8
// CHECK:   ret %struct.poly64x1x4_t [[TMP19]]
poly64x1x4_t test_vld4_lane_p64(poly64_t  *a, poly64x1x4_t b) {
  return vld4_lane_p64(a, b, 0);
}

// CHECK-LABEL: define{{.*}} void @test_vst1q_lane_u8(i8* noundef %a, <16 x i8> noundef %b) #0 {
// CHECK:   [[TMP0:%.*]] = extractelement <16 x i8> %b, i32 15
// CHECK:   store i8 [[TMP0]], i8* %a
// CHECK:   ret void
void test_vst1q_lane_u8(uint8_t  *a, uint8x16_t b) {
  vst1q_lane_u8(a, b, 15);
}

// CHECK-LABEL: define{{.*}} void @test_vst1q_lane_u16(i16* noundef %a, <8 x i16> noundef %b) #0 {
// CHECK:   [[TMP0:%.*]] = bitcast i16* %a to i8*
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <16 x i8> [[TMP1]] to <8 x i16>
// CHECK:   [[TMP3:%.*]] = extractelement <8 x i16> [[TMP2]], i32 7
// CHECK:   [[TMP4:%.*]] = bitcast i8* [[TMP0]] to i16*
// CHECK:   store i16 [[TMP3]], i16* [[TMP4]]
// CHECK:   ret void
void test_vst1q_lane_u16(uint16_t  *a, uint16x8_t b) {
  vst1q_lane_u16(a, b, 7);
}

// CHECK-LABEL: define{{.*}} void @test_vst1q_lane_u32(i32* noundef %a, <4 x i32> noundef %b) #0 {
// CHECK:   [[TMP0:%.*]] = bitcast i32* %a to i8*
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i32> %b to <16 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <16 x i8> [[TMP1]] to <4 x i32>
// CHECK:   [[TMP3:%.*]] = extractelement <4 x i32> [[TMP2]], i32 3
// CHECK:   [[TMP4:%.*]] = bitcast i8* [[TMP0]] to i32*
// CHECK:   store i32 [[TMP3]], i32* [[TMP4]]
// CHECK:   ret void
void test_vst1q_lane_u32(uint32_t  *a, uint32x4_t b) {
  vst1q_lane_u32(a, b, 3);
}

// CHECK-LABEL: define{{.*}} void @test_vst1q_lane_u64(i64* noundef %a, <2 x i64> noundef %b) #0 {
// CHECK:   [[TMP0:%.*]] = bitcast i64* %a to i8*
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i64> %b to <16 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <16 x i8> [[TMP1]] to <2 x i64>
// CHECK:   [[TMP3:%.*]] = extractelement <2 x i64> [[TMP2]], i32 1
// CHECK:   [[TMP4:%.*]] = bitcast i8* [[TMP0]] to i64*
// CHECK:   store i64 [[TMP3]], i64* [[TMP4]]
// CHECK:   ret void
void test_vst1q_lane_u64(uint64_t  *a, uint64x2_t b) {
  vst1q_lane_u64(a, b, 1);
}

// CHECK-LABEL: define{{.*}} void @test_vst1q_lane_s8(i8* noundef %a, <16 x i8> noundef %b) #0 {
// CHECK:   [[TMP0:%.*]] = extractelement <16 x i8> %b, i32 15
// CHECK:   store i8 [[TMP0]], i8* %a
// CHECK:   ret void
void test_vst1q_lane_s8(int8_t  *a, int8x16_t b) {
  vst1q_lane_s8(a, b, 15);
}

// CHECK-LABEL: define{{.*}} void @test_vst1q_lane_s16(i16* noundef %a, <8 x i16> noundef %b) #0 {
// CHECK:   [[TMP0:%.*]] = bitcast i16* %a to i8*
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <16 x i8> [[TMP1]] to <8 x i16>
// CHECK:   [[TMP3:%.*]] = extractelement <8 x i16> [[TMP2]], i32 7
// CHECK:   [[TMP4:%.*]] = bitcast i8* [[TMP0]] to i16*
// CHECK:   store i16 [[TMP3]], i16* [[TMP4]]
// CHECK:   ret void
void test_vst1q_lane_s16(int16_t  *a, int16x8_t b) {
  vst1q_lane_s16(a, b, 7);
}

// CHECK-LABEL: define{{.*}} void @test_vst1q_lane_s32(i32* noundef %a, <4 x i32> noundef %b) #0 {
// CHECK:   [[TMP0:%.*]] = bitcast i32* %a to i8*
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i32> %b to <16 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <16 x i8> [[TMP1]] to <4 x i32>
// CHECK:   [[TMP3:%.*]] = extractelement <4 x i32> [[TMP2]], i32 3
// CHECK:   [[TMP4:%.*]] = bitcast i8* [[TMP0]] to i32*
// CHECK:   store i32 [[TMP3]], i32* [[TMP4]]
// CHECK:   ret void
void test_vst1q_lane_s32(int32_t  *a, int32x4_t b) {
  vst1q_lane_s32(a, b, 3);
}

// CHECK-LABEL: define{{.*}} void @test_vst1q_lane_s64(i64* noundef %a, <2 x i64> noundef %b) #0 {
// CHECK:   [[TMP0:%.*]] = bitcast i64* %a to i8*
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i64> %b to <16 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <16 x i8> [[TMP1]] to <2 x i64>
// CHECK:   [[TMP3:%.*]] = extractelement <2 x i64> [[TMP2]], i32 1
// CHECK:   [[TMP4:%.*]] = bitcast i8* [[TMP0]] to i64*
// CHECK:   store i64 [[TMP3]], i64* [[TMP4]]
// CHECK:   ret void
void test_vst1q_lane_s64(int64_t  *a, int64x2_t b) {
  vst1q_lane_s64(a, b, 1);
}

// CHECK-LABEL: define{{.*}} void @test_vst1q_lane_f16(half* noundef %a, <8 x half> noundef %b) #0 {
// CHECK:   [[TMP0:%.*]] = bitcast half* %a to i8*
// CHECK:   [[TMP1:%.*]] = bitcast <8 x half> %b to <16 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <16 x i8> [[TMP1]] to <8 x half>
// CHECK:   [[TMP3:%.*]] = extractelement <8 x half> [[TMP2]], i32 7
// CHECK:   [[TMP4:%.*]] = bitcast i8* [[TMP0]] to half*
// CHECK:   store half [[TMP3]], half* [[TMP4]]
// CHECK:   ret void
void test_vst1q_lane_f16(float16_t  *a, float16x8_t b) {
  vst1q_lane_f16(a, b, 7);
}

// CHECK-LABEL: define{{.*}} void @test_vst1q_lane_f32(float* noundef %a, <4 x float> noundef %b) #0 {
// CHECK:   [[TMP0:%.*]] = bitcast float* %a to i8*
// CHECK:   [[TMP1:%.*]] = bitcast <4 x float> %b to <16 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <16 x i8> [[TMP1]] to <4 x float>
// CHECK:   [[TMP3:%.*]] = extractelement <4 x float> [[TMP2]], i32 3
// CHECK:   [[TMP4:%.*]] = bitcast i8* [[TMP0]] to float*
// CHECK:   store float [[TMP3]], float* [[TMP4]]
// CHECK:   ret void
void test_vst1q_lane_f32(float32_t  *a, float32x4_t b) {
  vst1q_lane_f32(a, b, 3);
}

// CHECK-LABEL: define{{.*}} void @test_vst1q_lane_f64(double* noundef %a, <2 x double> noundef %b) #0 {
// CHECK:   [[TMP0:%.*]] = bitcast double* %a to i8*
// CHECK:   [[TMP1:%.*]] = bitcast <2 x double> %b to <16 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <16 x i8> [[TMP1]] to <2 x double>
// CHECK:   [[TMP3:%.*]] = extractelement <2 x double> [[TMP2]], i32 1
// CHECK:   [[TMP4:%.*]] = bitcast i8* [[TMP0]] to double*
// CHECK:   store double [[TMP3]], double* [[TMP4]]
// CHECK:   ret void
void test_vst1q_lane_f64(float64_t  *a, float64x2_t b) {
  vst1q_lane_f64(a, b, 1);
}

// CHECK-LABEL: define{{.*}} void @test_vst1q_lane_p8(i8* noundef %a, <16 x i8> noundef %b) #0 {
// CHECK:   [[TMP0:%.*]] = extractelement <16 x i8> %b, i32 15
// CHECK:   store i8 [[TMP0]], i8* %a
// CHECK:   ret void
void test_vst1q_lane_p8(poly8_t  *a, poly8x16_t b) {
  vst1q_lane_p8(a, b, 15);
}

// CHECK-LABEL: define{{.*}} void @test_vst1q_lane_p16(i16* noundef %a, <8 x i16> noundef %b) #0 {
// CHECK:   [[TMP0:%.*]] = bitcast i16* %a to i8*
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <16 x i8> [[TMP1]] to <8 x i16>
// CHECK:   [[TMP3:%.*]] = extractelement <8 x i16> [[TMP2]], i32 7
// CHECK:   [[TMP4:%.*]] = bitcast i8* [[TMP0]] to i16*
// CHECK:   store i16 [[TMP3]], i16* [[TMP4]]
// CHECK:   ret void
void test_vst1q_lane_p16(poly16_t  *a, poly16x8_t b) {
  vst1q_lane_p16(a, b, 7);
}

// CHECK-LABEL: define{{.*}} void @test_vst1q_lane_p64(i64* noundef %a, <2 x i64> noundef %b) #0 {
// CHECK:   [[TMP0:%.*]] = bitcast i64* %a to i8*
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i64> %b to <16 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <16 x i8> [[TMP1]] to <2 x i64>
// CHECK:   [[TMP3:%.*]] = extractelement <2 x i64> [[TMP2]], i32 1
// CHECK:   [[TMP4:%.*]] = bitcast i8* [[TMP0]] to i64*
// CHECK:   store i64 [[TMP3]], i64* [[TMP4]]
// CHECK:   ret void
void test_vst1q_lane_p64(poly64_t  *a, poly64x2_t b) {
  vst1q_lane_p64(a, b, 1);
}

// CHECK-LABEL: define{{.*}} void @test_vst1_lane_u8(i8* noundef %a, <8 x i8> noundef %b) #1 {
// CHECK:   [[TMP0:%.*]] = extractelement <8 x i8> %b, i32 7
// CHECK:   store i8 [[TMP0]], i8* %a
// CHECK:   ret void
void test_vst1_lane_u8(uint8_t  *a, uint8x8_t b) {
  vst1_lane_u8(a, b, 7);
}

// CHECK-LABEL: define{{.*}} void @test_vst1_lane_u16(i16* noundef %a, <4 x i16> noundef %b) #1 {
// CHECK:   [[TMP0:%.*]] = bitcast i16* %a to i8*
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <8 x i8> [[TMP1]] to <4 x i16>
// CHECK:   [[TMP3:%.*]] = extractelement <4 x i16> [[TMP2]], i32 3
// CHECK:   [[TMP4:%.*]] = bitcast i8* [[TMP0]] to i16*
// CHECK:   store i16 [[TMP3]], i16* [[TMP4]]
// CHECK:   ret void
void test_vst1_lane_u16(uint16_t  *a, uint16x4_t b) {
  vst1_lane_u16(a, b, 3);
}

// CHECK-LABEL: define{{.*}} void @test_vst1_lane_u32(i32* noundef %a, <2 x i32> noundef %b) #1 {
// CHECK:   [[TMP0:%.*]] = bitcast i32* %a to i8*
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <8 x i8> [[TMP1]] to <2 x i32>
// CHECK:   [[TMP3:%.*]] = extractelement <2 x i32> [[TMP2]], i32 1
// CHECK:   [[TMP4:%.*]] = bitcast i8* [[TMP0]] to i32*
// CHECK:   store i32 [[TMP3]], i32* [[TMP4]]
// CHECK:   ret void
void test_vst1_lane_u32(uint32_t  *a, uint32x2_t b) {
  vst1_lane_u32(a, b, 1);
}

// CHECK-LABEL: define{{.*}} void @test_vst1_lane_u64(i64* noundef %a, <1 x i64> noundef %b) #1 {
// CHECK:   [[TMP0:%.*]] = bitcast i64* %a to i8*
// CHECK:   [[TMP1:%.*]] = bitcast <1 x i64> %b to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <8 x i8> [[TMP1]] to <1 x i64>
// CHECK:   [[TMP3:%.*]] = extractelement <1 x i64> [[TMP2]], i32 0
// CHECK:   [[TMP4:%.*]] = bitcast i8* [[TMP0]] to i64*
// CHECK:   store i64 [[TMP3]], i64* [[TMP4]]
// CHECK:   ret void
void test_vst1_lane_u64(uint64_t  *a, uint64x1_t b) {
  vst1_lane_u64(a, b, 0);
}

// CHECK-LABEL: define{{.*}} void @test_vst1_lane_s8(i8* noundef %a, <8 x i8> noundef %b) #1 {
// CHECK:   [[TMP0:%.*]] = extractelement <8 x i8> %b, i32 7
// CHECK:   store i8 [[TMP0]], i8* %a
// CHECK:   ret void
void test_vst1_lane_s8(int8_t  *a, int8x8_t b) {
  vst1_lane_s8(a, b, 7);
}

// CHECK-LABEL: define{{.*}} void @test_vst1_lane_s16(i16* noundef %a, <4 x i16> noundef %b) #1 {
// CHECK:   [[TMP0:%.*]] = bitcast i16* %a to i8*
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <8 x i8> [[TMP1]] to <4 x i16>
// CHECK:   [[TMP3:%.*]] = extractelement <4 x i16> [[TMP2]], i32 3
// CHECK:   [[TMP4:%.*]] = bitcast i8* [[TMP0]] to i16*
// CHECK:   store i16 [[TMP3]], i16* [[TMP4]]
// CHECK:   ret void
void test_vst1_lane_s16(int16_t  *a, int16x4_t b) {
  vst1_lane_s16(a, b, 3);
}

// CHECK-LABEL: define{{.*}} void @test_vst1_lane_s32(i32* noundef %a, <2 x i32> noundef %b) #1 {
// CHECK:   [[TMP0:%.*]] = bitcast i32* %a to i8*
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <8 x i8> [[TMP1]] to <2 x i32>
// CHECK:   [[TMP3:%.*]] = extractelement <2 x i32> [[TMP2]], i32 1
// CHECK:   [[TMP4:%.*]] = bitcast i8* [[TMP0]] to i32*
// CHECK:   store i32 [[TMP3]], i32* [[TMP4]]
// CHECK:   ret void
void test_vst1_lane_s32(int32_t  *a, int32x2_t b) {
  vst1_lane_s32(a, b, 1);
}

// CHECK-LABEL: define{{.*}} void @test_vst1_lane_s64(i64* noundef %a, <1 x i64> noundef %b) #1 {
// CHECK:   [[TMP0:%.*]] = bitcast i64* %a to i8*
// CHECK:   [[TMP1:%.*]] = bitcast <1 x i64> %b to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <8 x i8> [[TMP1]] to <1 x i64>
// CHECK:   [[TMP3:%.*]] = extractelement <1 x i64> [[TMP2]], i32 0
// CHECK:   [[TMP4:%.*]] = bitcast i8* [[TMP0]] to i64*
// CHECK:   store i64 [[TMP3]], i64* [[TMP4]]
// CHECK:   ret void
void test_vst1_lane_s64(int64_t  *a, int64x1_t b) {
  vst1_lane_s64(a, b, 0);
}

// CHECK-LABEL: define{{.*}} void @test_vst1_lane_f16(half* noundef %a, <4 x half> noundef %b) #1 {
// CHECK:   [[TMP0:%.*]] = bitcast half* %a to i8*
// CHECK:   [[TMP1:%.*]] = bitcast <4 x half> %b to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <8 x i8> [[TMP1]] to <4 x half>
// CHECK:   [[TMP3:%.*]] = extractelement <4 x half> [[TMP2]], i32 3
// CHECK:   [[TMP4:%.*]] = bitcast i8* [[TMP0]] to half*
// CHECK:   store half [[TMP3]], half* [[TMP4]]
// CHECK:   ret void
void test_vst1_lane_f16(float16_t  *a, float16x4_t b) {
  vst1_lane_f16(a, b, 3);
}

// CHECK-LABEL: define{{.*}} void @test_vst1_lane_f32(float* noundef %a, <2 x float> noundef %b) #1 {
// CHECK:   [[TMP0:%.*]] = bitcast float* %a to i8*
// CHECK:   [[TMP1:%.*]] = bitcast <2 x float> %b to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <8 x i8> [[TMP1]] to <2 x float>
// CHECK:   [[TMP3:%.*]] = extractelement <2 x float> [[TMP2]], i32 1
// CHECK:   [[TMP4:%.*]] = bitcast i8* [[TMP0]] to float*
// CHECK:   store float [[TMP3]], float* [[TMP4]]
// CHECK:   ret void
void test_vst1_lane_f32(float32_t  *a, float32x2_t b) {
  vst1_lane_f32(a, b, 1);
}

// CHECK-LABEL: define{{.*}} void @test_vst1_lane_f64(double* noundef %a, <1 x double> noundef %b) #1 {
// CHECK:   [[TMP0:%.*]] = bitcast double* %a to i8*
// CHECK:   [[TMP1:%.*]] = bitcast <1 x double> %b to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <8 x i8> [[TMP1]] to <1 x double>
// CHECK:   [[TMP3:%.*]] = extractelement <1 x double> [[TMP2]], i32 0
// CHECK:   [[TMP4:%.*]] = bitcast i8* [[TMP0]] to double*
// CHECK:   store double [[TMP3]], double* [[TMP4]]
// CHECK:   ret void
void test_vst1_lane_f64(float64_t  *a, float64x1_t b) {
  vst1_lane_f64(a, b, 0);
}

// CHECK-LABEL: define{{.*}} void @test_vst1_lane_p8(i8* noundef %a, <8 x i8> noundef %b) #1 {
// CHECK:   [[TMP0:%.*]] = extractelement <8 x i8> %b, i32 7
// CHECK:   store i8 [[TMP0]], i8* %a
// CHECK:   ret void
void test_vst1_lane_p8(poly8_t  *a, poly8x8_t b) {
  vst1_lane_p8(a, b, 7);
}

// CHECK-LABEL: define{{.*}} void @test_vst1_lane_p16(i16* noundef %a, <4 x i16> noundef %b) #1 {
// CHECK:   [[TMP0:%.*]] = bitcast i16* %a to i8*
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <8 x i8> [[TMP1]] to <4 x i16>
// CHECK:   [[TMP3:%.*]] = extractelement <4 x i16> [[TMP2]], i32 3
// CHECK:   [[TMP4:%.*]] = bitcast i8* [[TMP0]] to i16*
// CHECK:   store i16 [[TMP3]], i16* [[TMP4]]
// CHECK:   ret void
void test_vst1_lane_p16(poly16_t  *a, poly16x4_t b) {
  vst1_lane_p16(a, b, 3);
}

// CHECK-LABEL: define{{.*}} void @test_vst1_lane_p64(i64* noundef %a, <1 x i64> noundef %b) #1 {
// CHECK:   [[TMP0:%.*]] = bitcast i64* %a to i8*
// CHECK:   [[TMP1:%.*]] = bitcast <1 x i64> %b to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <8 x i8> [[TMP1]] to <1 x i64>
// CHECK:   [[TMP3:%.*]] = extractelement <1 x i64> [[TMP2]], i32 0
// CHECK:   [[TMP4:%.*]] = bitcast i8* [[TMP0]] to i64*
// CHECK:   store i64 [[TMP3]], i64* [[TMP4]]
// CHECK:   ret void
void test_vst1_lane_p64(poly64_t  *a, poly64x1_t b) {
  vst1_lane_p64(a, b, 0);
}

// CHECK-LABEL: define{{.*}} void @test_vst2q_lane_u8(i8* noundef %a, [2 x <16 x i8>] %b.coerce) #2 {
// CHECK:   [[B:%.*]] = alloca %struct.uint8x16x2_t, align 16
// CHECK:   [[__S1:%.*]] = alloca %struct.uint8x16x2_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.uint8x16x2_t, %struct.uint8x16x2_t* [[B]], i32 0, i32 0
// CHECK:   store [2 x <16 x i8>] [[B]].coerce, [2 x <16 x i8>]* [[COERCE_DIVE]], align 16
// CHECK:   [[TMP0:%.*]] = bitcast %struct.uint8x16x2_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.uint8x16x2_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP0]], i8* align 16 [[TMP1]], i64 32, i1 false)
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.uint8x16x2_t, %struct.uint8x16x2_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <16 x i8>], [2 x <16 x i8>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP2:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX]], align 16
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.uint8x16x2_t, %struct.uint8x16x2_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <16 x i8>], [2 x <16 x i8>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP3:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX2]], align 16
// CHECK:   call void @llvm.aarch64.neon.st2lane.v16i8.p0i8(<16 x i8> [[TMP2]], <16 x i8> [[TMP3]], i64 15, i8* %a)
// CHECK:   ret void
void test_vst2q_lane_u8(uint8_t  *a, uint8x16x2_t b) {
  vst2q_lane_u8(a, b, 15);
}

// CHECK-LABEL: define{{.*}} void @test_vst2q_lane_u16(i16* noundef %a, [2 x <8 x i16>] %b.coerce) #2 {
// CHECK:   [[B:%.*]] = alloca %struct.uint16x8x2_t, align 16
// CHECK:   [[__S1:%.*]] = alloca %struct.uint16x8x2_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.uint16x8x2_t, %struct.uint16x8x2_t* [[B]], i32 0, i32 0
// CHECK:   store [2 x <8 x i16>] [[B]].coerce, [2 x <8 x i16>]* [[COERCE_DIVE]], align 16
// CHECK:   [[TMP0:%.*]] = bitcast %struct.uint16x8x2_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.uint16x8x2_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP0]], i8* align 16 [[TMP1]], i64 32, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast i16* %a to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.uint16x8x2_t, %struct.uint16x8x2_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <8 x i16>], [2 x <8 x i16>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <8 x i16>, <8 x i16>* [[ARRAYIDX]], align 16
// CHECK:   [[TMP4:%.*]] = bitcast <8 x i16> [[TMP3]] to <16 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.uint16x8x2_t, %struct.uint16x8x2_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <8 x i16>], [2 x <8 x i16>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <8 x i16>, <8 x i16>* [[ARRAYIDX2]], align 16
// CHECK:   [[TMP6:%.*]] = bitcast <8 x i16> [[TMP5]] to <16 x i8>
// CHECK:   [[TMP7:%.*]] = bitcast <16 x i8> [[TMP4]] to <8 x i16>
// CHECK:   [[TMP8:%.*]] = bitcast <16 x i8> [[TMP6]] to <8 x i16>
// CHECK:   call void @llvm.aarch64.neon.st2lane.v8i16.p0i8(<8 x i16> [[TMP7]], <8 x i16> [[TMP8]], i64 7, i8* [[TMP2]])
// CHECK:   ret void
void test_vst2q_lane_u16(uint16_t  *a, uint16x8x2_t b) {
  vst2q_lane_u16(a, b, 7);
}

// CHECK-LABEL: define{{.*}} void @test_vst2q_lane_u32(i32* noundef %a, [2 x <4 x i32>] %b.coerce) #2 {
// CHECK:   [[B:%.*]] = alloca %struct.uint32x4x2_t, align 16
// CHECK:   [[__S1:%.*]] = alloca %struct.uint32x4x2_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.uint32x4x2_t, %struct.uint32x4x2_t* [[B]], i32 0, i32 0
// CHECK:   store [2 x <4 x i32>] [[B]].coerce, [2 x <4 x i32>]* [[COERCE_DIVE]], align 16
// CHECK:   [[TMP0:%.*]] = bitcast %struct.uint32x4x2_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.uint32x4x2_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP0]], i8* align 16 [[TMP1]], i64 32, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast i32* %a to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.uint32x4x2_t, %struct.uint32x4x2_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <4 x i32>], [2 x <4 x i32>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <4 x i32>, <4 x i32>* [[ARRAYIDX]], align 16
// CHECK:   [[TMP4:%.*]] = bitcast <4 x i32> [[TMP3]] to <16 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.uint32x4x2_t, %struct.uint32x4x2_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <4 x i32>], [2 x <4 x i32>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <4 x i32>, <4 x i32>* [[ARRAYIDX2]], align 16
// CHECK:   [[TMP6:%.*]] = bitcast <4 x i32> [[TMP5]] to <16 x i8>
// CHECK:   [[TMP7:%.*]] = bitcast <16 x i8> [[TMP4]] to <4 x i32>
// CHECK:   [[TMP8:%.*]] = bitcast <16 x i8> [[TMP6]] to <4 x i32>
// CHECK:   call void @llvm.aarch64.neon.st2lane.v4i32.p0i8(<4 x i32> [[TMP7]], <4 x i32> [[TMP8]], i64 3, i8* [[TMP2]])
// CHECK:   ret void
void test_vst2q_lane_u32(uint32_t  *a, uint32x4x2_t b) {
  vst2q_lane_u32(a, b, 3);
}

// CHECK-LABEL: define{{.*}} void @test_vst2q_lane_u64(i64* noundef %a, [2 x <2 x i64>] %b.coerce) #2 {
// CHECK:   [[B:%.*]] = alloca %struct.uint64x2x2_t, align 16
// CHECK:   [[__S1:%.*]] = alloca %struct.uint64x2x2_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.uint64x2x2_t, %struct.uint64x2x2_t* [[B]], i32 0, i32 0
// CHECK:   store [2 x <2 x i64>] [[B]].coerce, [2 x <2 x i64>]* [[COERCE_DIVE]], align 16
// CHECK:   [[TMP0:%.*]] = bitcast %struct.uint64x2x2_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.uint64x2x2_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP0]], i8* align 16 [[TMP1]], i64 32, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast i64* %a to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.uint64x2x2_t, %struct.uint64x2x2_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <2 x i64>], [2 x <2 x i64>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <2 x i64>, <2 x i64>* [[ARRAYIDX]], align 16
// CHECK:   [[TMP4:%.*]] = bitcast <2 x i64> [[TMP3]] to <16 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.uint64x2x2_t, %struct.uint64x2x2_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <2 x i64>], [2 x <2 x i64>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <2 x i64>, <2 x i64>* [[ARRAYIDX2]], align 16
// CHECK:   [[TMP6:%.*]] = bitcast <2 x i64> [[TMP5]] to <16 x i8>
// CHECK:   [[TMP7:%.*]] = bitcast <16 x i8> [[TMP4]] to <2 x i64>
// CHECK:   [[TMP8:%.*]] = bitcast <16 x i8> [[TMP6]] to <2 x i64>
// CHECK:   call void @llvm.aarch64.neon.st2lane.v2i64.p0i8(<2 x i64> [[TMP7]], <2 x i64> [[TMP8]], i64 1, i8* [[TMP2]])
// CHECK:   ret void
void test_vst2q_lane_u64(uint64_t  *a, uint64x2x2_t b) {
  vst2q_lane_u64(a, b, 1);
}

// CHECK-LABEL: define{{.*}} void @test_vst2q_lane_s8(i8* noundef %a, [2 x <16 x i8>] %b.coerce) #2 {
// CHECK:   [[B:%.*]] = alloca %struct.int8x16x2_t, align 16
// CHECK:   [[__S1:%.*]] = alloca %struct.int8x16x2_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.int8x16x2_t, %struct.int8x16x2_t* [[B]], i32 0, i32 0
// CHECK:   store [2 x <16 x i8>] [[B]].coerce, [2 x <16 x i8>]* [[COERCE_DIVE]], align 16
// CHECK:   [[TMP0:%.*]] = bitcast %struct.int8x16x2_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.int8x16x2_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP0]], i8* align 16 [[TMP1]], i64 32, i1 false)
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.int8x16x2_t, %struct.int8x16x2_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <16 x i8>], [2 x <16 x i8>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP2:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX]], align 16
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.int8x16x2_t, %struct.int8x16x2_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <16 x i8>], [2 x <16 x i8>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP3:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX2]], align 16
// CHECK:   call void @llvm.aarch64.neon.st2lane.v16i8.p0i8(<16 x i8> [[TMP2]], <16 x i8> [[TMP3]], i64 15, i8* %a)
// CHECK:   ret void
void test_vst2q_lane_s8(int8_t  *a, int8x16x2_t b) {
  vst2q_lane_s8(a, b, 15);
}

// CHECK-LABEL: define{{.*}} void @test_vst2q_lane_s16(i16* noundef %a, [2 x <8 x i16>] %b.coerce) #2 {
// CHECK:   [[B:%.*]] = alloca %struct.int16x8x2_t, align 16
// CHECK:   [[__S1:%.*]] = alloca %struct.int16x8x2_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.int16x8x2_t, %struct.int16x8x2_t* [[B]], i32 0, i32 0
// CHECK:   store [2 x <8 x i16>] [[B]].coerce, [2 x <8 x i16>]* [[COERCE_DIVE]], align 16
// CHECK:   [[TMP0:%.*]] = bitcast %struct.int16x8x2_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.int16x8x2_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP0]], i8* align 16 [[TMP1]], i64 32, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast i16* %a to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.int16x8x2_t, %struct.int16x8x2_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <8 x i16>], [2 x <8 x i16>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <8 x i16>, <8 x i16>* [[ARRAYIDX]], align 16
// CHECK:   [[TMP4:%.*]] = bitcast <8 x i16> [[TMP3]] to <16 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.int16x8x2_t, %struct.int16x8x2_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <8 x i16>], [2 x <8 x i16>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <8 x i16>, <8 x i16>* [[ARRAYIDX2]], align 16
// CHECK:   [[TMP6:%.*]] = bitcast <8 x i16> [[TMP5]] to <16 x i8>
// CHECK:   [[TMP7:%.*]] = bitcast <16 x i8> [[TMP4]] to <8 x i16>
// CHECK:   [[TMP8:%.*]] = bitcast <16 x i8> [[TMP6]] to <8 x i16>
// CHECK:   call void @llvm.aarch64.neon.st2lane.v8i16.p0i8(<8 x i16> [[TMP7]], <8 x i16> [[TMP8]], i64 7, i8* [[TMP2]])
// CHECK:   ret void
void test_vst2q_lane_s16(int16_t  *a, int16x8x2_t b) {
  vst2q_lane_s16(a, b, 7);
}

// CHECK-LABEL: define{{.*}} void @test_vst2q_lane_s32(i32* noundef %a, [2 x <4 x i32>] %b.coerce) #2 {
// CHECK:   [[B:%.*]] = alloca %struct.int32x4x2_t, align 16
// CHECK:   [[__S1:%.*]] = alloca %struct.int32x4x2_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.int32x4x2_t, %struct.int32x4x2_t* [[B]], i32 0, i32 0
// CHECK:   store [2 x <4 x i32>] [[B]].coerce, [2 x <4 x i32>]* [[COERCE_DIVE]], align 16
// CHECK:   [[TMP0:%.*]] = bitcast %struct.int32x4x2_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.int32x4x2_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP0]], i8* align 16 [[TMP1]], i64 32, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast i32* %a to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.int32x4x2_t, %struct.int32x4x2_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <4 x i32>], [2 x <4 x i32>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <4 x i32>, <4 x i32>* [[ARRAYIDX]], align 16
// CHECK:   [[TMP4:%.*]] = bitcast <4 x i32> [[TMP3]] to <16 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.int32x4x2_t, %struct.int32x4x2_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <4 x i32>], [2 x <4 x i32>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <4 x i32>, <4 x i32>* [[ARRAYIDX2]], align 16
// CHECK:   [[TMP6:%.*]] = bitcast <4 x i32> [[TMP5]] to <16 x i8>
// CHECK:   [[TMP7:%.*]] = bitcast <16 x i8> [[TMP4]] to <4 x i32>
// CHECK:   [[TMP8:%.*]] = bitcast <16 x i8> [[TMP6]] to <4 x i32>
// CHECK:   call void @llvm.aarch64.neon.st2lane.v4i32.p0i8(<4 x i32> [[TMP7]], <4 x i32> [[TMP8]], i64 3, i8* [[TMP2]])
// CHECK:   ret void
void test_vst2q_lane_s32(int32_t  *a, int32x4x2_t b) {
  vst2q_lane_s32(a, b, 3);
}

// CHECK-LABEL: define{{.*}} void @test_vst2q_lane_s64(i64* noundef %a, [2 x <2 x i64>] %b.coerce) #2 {
// CHECK:   [[B:%.*]] = alloca %struct.int64x2x2_t, align 16
// CHECK:   [[__S1:%.*]] = alloca %struct.int64x2x2_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.int64x2x2_t, %struct.int64x2x2_t* [[B]], i32 0, i32 0
// CHECK:   store [2 x <2 x i64>] [[B]].coerce, [2 x <2 x i64>]* [[COERCE_DIVE]], align 16
// CHECK:   [[TMP0:%.*]] = bitcast %struct.int64x2x2_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.int64x2x2_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP0]], i8* align 16 [[TMP1]], i64 32, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast i64* %a to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.int64x2x2_t, %struct.int64x2x2_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <2 x i64>], [2 x <2 x i64>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <2 x i64>, <2 x i64>* [[ARRAYIDX]], align 16
// CHECK:   [[TMP4:%.*]] = bitcast <2 x i64> [[TMP3]] to <16 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.int64x2x2_t, %struct.int64x2x2_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <2 x i64>], [2 x <2 x i64>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <2 x i64>, <2 x i64>* [[ARRAYIDX2]], align 16
// CHECK:   [[TMP6:%.*]] = bitcast <2 x i64> [[TMP5]] to <16 x i8>
// CHECK:   [[TMP7:%.*]] = bitcast <16 x i8> [[TMP4]] to <2 x i64>
// CHECK:   [[TMP8:%.*]] = bitcast <16 x i8> [[TMP6]] to <2 x i64>
// CHECK:   call void @llvm.aarch64.neon.st2lane.v2i64.p0i8(<2 x i64> [[TMP7]], <2 x i64> [[TMP8]], i64 1, i8* [[TMP2]])
// CHECK:   ret void
void test_vst2q_lane_s64(int64_t  *a, int64x2x2_t b) {
  vst2q_lane_s64(a, b, 1);
}

// CHECK-LABEL: define{{.*}} void @test_vst2q_lane_f16(half* noundef %a, [2 x <8 x half>] %b.coerce) #2 {
// CHECK:   [[B:%.*]] = alloca %struct.float16x8x2_t, align 16
// CHECK:   [[__S1:%.*]] = alloca %struct.float16x8x2_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.float16x8x2_t, %struct.float16x8x2_t* [[B]], i32 0, i32 0
// CHECK:   store [2 x <8 x half>] [[B]].coerce, [2 x <8 x half>]* [[COERCE_DIVE]], align 16
// CHECK:   [[TMP0:%.*]] = bitcast %struct.float16x8x2_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.float16x8x2_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP0]], i8* align 16 [[TMP1]], i64 32, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast half* %a to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.float16x8x2_t, %struct.float16x8x2_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <8 x half>], [2 x <8 x half>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <8 x half>, <8 x half>* [[ARRAYIDX]], align 16
// CHECK:   [[TMP4:%.*]] = bitcast <8 x half> [[TMP3]] to <16 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.float16x8x2_t, %struct.float16x8x2_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <8 x half>], [2 x <8 x half>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <8 x half>, <8 x half>* [[ARRAYIDX2]], align 16
// CHECK:   [[TMP6:%.*]] = bitcast <8 x half> [[TMP5]] to <16 x i8>
// CHECK:   [[TMP7:%.*]] = bitcast <16 x i8> [[TMP4]] to <8 x half>
// CHECK:   [[TMP8:%.*]] = bitcast <16 x i8> [[TMP6]] to <8 x half>
// CHECK:   call void @llvm.aarch64.neon.st2lane.v8f16.p0i8(<8 x half> [[TMP7]], <8 x half> [[TMP8]], i64 7, i8* [[TMP2]])
// CHECK:   ret void
void test_vst2q_lane_f16(float16_t  *a, float16x8x2_t b) {
  vst2q_lane_f16(a, b, 7);
}

// CHECK-LABEL: define{{.*}} void @test_vst2q_lane_f32(float* noundef %a, [2 x <4 x float>] %b.coerce) #2 {
// CHECK:   [[B:%.*]] = alloca %struct.float32x4x2_t, align 16
// CHECK:   [[__S1:%.*]] = alloca %struct.float32x4x2_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.float32x4x2_t, %struct.float32x4x2_t* [[B]], i32 0, i32 0
// CHECK:   store [2 x <4 x float>] [[B]].coerce, [2 x <4 x float>]* [[COERCE_DIVE]], align 16
// CHECK:   [[TMP0:%.*]] = bitcast %struct.float32x4x2_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.float32x4x2_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP0]], i8* align 16 [[TMP1]], i64 32, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast float* %a to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.float32x4x2_t, %struct.float32x4x2_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <4 x float>], [2 x <4 x float>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <4 x float>, <4 x float>* [[ARRAYIDX]], align 16
// CHECK:   [[TMP4:%.*]] = bitcast <4 x float> [[TMP3]] to <16 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.float32x4x2_t, %struct.float32x4x2_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <4 x float>], [2 x <4 x float>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <4 x float>, <4 x float>* [[ARRAYIDX2]], align 16
// CHECK:   [[TMP6:%.*]] = bitcast <4 x float> [[TMP5]] to <16 x i8>
// CHECK:   [[TMP7:%.*]] = bitcast <16 x i8> [[TMP4]] to <4 x float>
// CHECK:   [[TMP8:%.*]] = bitcast <16 x i8> [[TMP6]] to <4 x float>
// CHECK:   call void @llvm.aarch64.neon.st2lane.v4f32.p0i8(<4 x float> [[TMP7]], <4 x float> [[TMP8]], i64 3, i8* [[TMP2]])
// CHECK:   ret void
void test_vst2q_lane_f32(float32_t  *a, float32x4x2_t b) {
  vst2q_lane_f32(a, b, 3);
}

// CHECK-LABEL: define{{.*}} void @test_vst2q_lane_f64(double* noundef %a, [2 x <2 x double>] %b.coerce) #2 {
// CHECK:   [[B:%.*]] = alloca %struct.float64x2x2_t, align 16
// CHECK:   [[__S1:%.*]] = alloca %struct.float64x2x2_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.float64x2x2_t, %struct.float64x2x2_t* [[B]], i32 0, i32 0
// CHECK:   store [2 x <2 x double>] [[B]].coerce, [2 x <2 x double>]* [[COERCE_DIVE]], align 16
// CHECK:   [[TMP0:%.*]] = bitcast %struct.float64x2x2_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.float64x2x2_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP0]], i8* align 16 [[TMP1]], i64 32, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast double* %a to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.float64x2x2_t, %struct.float64x2x2_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <2 x double>], [2 x <2 x double>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <2 x double>, <2 x double>* [[ARRAYIDX]], align 16
// CHECK:   [[TMP4:%.*]] = bitcast <2 x double> [[TMP3]] to <16 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.float64x2x2_t, %struct.float64x2x2_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <2 x double>], [2 x <2 x double>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <2 x double>, <2 x double>* [[ARRAYIDX2]], align 16
// CHECK:   [[TMP6:%.*]] = bitcast <2 x double> [[TMP5]] to <16 x i8>
// CHECK:   [[TMP7:%.*]] = bitcast <16 x i8> [[TMP4]] to <2 x double>
// CHECK:   [[TMP8:%.*]] = bitcast <16 x i8> [[TMP6]] to <2 x double>
// CHECK:   call void @llvm.aarch64.neon.st2lane.v2f64.p0i8(<2 x double> [[TMP7]], <2 x double> [[TMP8]], i64 1, i8* [[TMP2]])
// CHECK:   ret void
void test_vst2q_lane_f64(float64_t  *a, float64x2x2_t b) {
  vst2q_lane_f64(a, b, 1);
}

// CHECK-LABEL: define{{.*}} void @test_vst2q_lane_p8(i8* noundef %a, [2 x <16 x i8>] %b.coerce) #2 {
// CHECK:   [[B:%.*]] = alloca %struct.poly8x16x2_t, align 16
// CHECK:   [[__S1:%.*]] = alloca %struct.poly8x16x2_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.poly8x16x2_t, %struct.poly8x16x2_t* [[B]], i32 0, i32 0
// CHECK:   store [2 x <16 x i8>] [[B]].coerce, [2 x <16 x i8>]* [[COERCE_DIVE]], align 16
// CHECK:   [[TMP0:%.*]] = bitcast %struct.poly8x16x2_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.poly8x16x2_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP0]], i8* align 16 [[TMP1]], i64 32, i1 false)
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.poly8x16x2_t, %struct.poly8x16x2_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <16 x i8>], [2 x <16 x i8>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP2:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX]], align 16
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.poly8x16x2_t, %struct.poly8x16x2_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <16 x i8>], [2 x <16 x i8>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP3:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX2]], align 16
// CHECK:   call void @llvm.aarch64.neon.st2lane.v16i8.p0i8(<16 x i8> [[TMP2]], <16 x i8> [[TMP3]], i64 15, i8* %a)
// CHECK:   ret void
void test_vst2q_lane_p8(poly8_t  *a, poly8x16x2_t b) {
  vst2q_lane_p8(a, b, 15);
}

// CHECK-LABEL: define{{.*}} void @test_vst2q_lane_p16(i16* noundef %a, [2 x <8 x i16>] %b.coerce) #2 {
// CHECK:   [[B:%.*]] = alloca %struct.poly16x8x2_t, align 16
// CHECK:   [[__S1:%.*]] = alloca %struct.poly16x8x2_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.poly16x8x2_t, %struct.poly16x8x2_t* [[B]], i32 0, i32 0
// CHECK:   store [2 x <8 x i16>] [[B]].coerce, [2 x <8 x i16>]* [[COERCE_DIVE]], align 16
// CHECK:   [[TMP0:%.*]] = bitcast %struct.poly16x8x2_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.poly16x8x2_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP0]], i8* align 16 [[TMP1]], i64 32, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast i16* %a to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.poly16x8x2_t, %struct.poly16x8x2_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <8 x i16>], [2 x <8 x i16>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <8 x i16>, <8 x i16>* [[ARRAYIDX]], align 16
// CHECK:   [[TMP4:%.*]] = bitcast <8 x i16> [[TMP3]] to <16 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.poly16x8x2_t, %struct.poly16x8x2_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <8 x i16>], [2 x <8 x i16>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <8 x i16>, <8 x i16>* [[ARRAYIDX2]], align 16
// CHECK:   [[TMP6:%.*]] = bitcast <8 x i16> [[TMP5]] to <16 x i8>
// CHECK:   [[TMP7:%.*]] = bitcast <16 x i8> [[TMP4]] to <8 x i16>
// CHECK:   [[TMP8:%.*]] = bitcast <16 x i8> [[TMP6]] to <8 x i16>
// CHECK:   call void @llvm.aarch64.neon.st2lane.v8i16.p0i8(<8 x i16> [[TMP7]], <8 x i16> [[TMP8]], i64 7, i8* [[TMP2]])
// CHECK:   ret void
void test_vst2q_lane_p16(poly16_t  *a, poly16x8x2_t b) {
  vst2q_lane_p16(a, b, 7);
}

// CHECK-LABEL: define{{.*}} void @test_vst2q_lane_p64(i64* noundef %a, [2 x <2 x i64>] %b.coerce) #2 {
// CHECK:   [[B:%.*]] = alloca %struct.poly64x2x2_t, align 16
// CHECK:   [[__S1:%.*]] = alloca %struct.poly64x2x2_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.poly64x2x2_t, %struct.poly64x2x2_t* [[B]], i32 0, i32 0
// CHECK:   store [2 x <2 x i64>] [[B]].coerce, [2 x <2 x i64>]* [[COERCE_DIVE]], align 16
// CHECK:   [[TMP0:%.*]] = bitcast %struct.poly64x2x2_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.poly64x2x2_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP0]], i8* align 16 [[TMP1]], i64 32, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast i64* %a to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.poly64x2x2_t, %struct.poly64x2x2_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <2 x i64>], [2 x <2 x i64>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <2 x i64>, <2 x i64>* [[ARRAYIDX]], align 16
// CHECK:   [[TMP4:%.*]] = bitcast <2 x i64> [[TMP3]] to <16 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.poly64x2x2_t, %struct.poly64x2x2_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <2 x i64>], [2 x <2 x i64>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <2 x i64>, <2 x i64>* [[ARRAYIDX2]], align 16
// CHECK:   [[TMP6:%.*]] = bitcast <2 x i64> [[TMP5]] to <16 x i8>
// CHECK:   [[TMP7:%.*]] = bitcast <16 x i8> [[TMP4]] to <2 x i64>
// CHECK:   [[TMP8:%.*]] = bitcast <16 x i8> [[TMP6]] to <2 x i64>
// CHECK:   call void @llvm.aarch64.neon.st2lane.v2i64.p0i8(<2 x i64> [[TMP7]], <2 x i64> [[TMP8]], i64 1, i8* [[TMP2]])
// CHECK:   ret void
void test_vst2q_lane_p64(poly64_t  *a, poly64x2x2_t b) {
  vst2q_lane_p64(a, b, 1);
}

// CHECK-LABEL: define{{.*}} void @test_vst2_lane_u8(i8* noundef %a, [2 x <8 x i8>] %b.coerce) #2 {
// CHECK:   [[B:%.*]] = alloca %struct.uint8x8x2_t, align 8
// CHECK:   [[__S1:%.*]] = alloca %struct.uint8x8x2_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.uint8x8x2_t, %struct.uint8x8x2_t* [[B]], i32 0, i32 0
// CHECK:   store [2 x <8 x i8>] [[B]].coerce, [2 x <8 x i8>]* [[COERCE_DIVE]], align 8
// CHECK:   [[TMP0:%.*]] = bitcast %struct.uint8x8x2_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.uint8x8x2_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[TMP0]], i8* align 8 [[TMP1]], i64 16, i1 false)
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.uint8x8x2_t, %struct.uint8x8x2_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <8 x i8>], [2 x <8 x i8>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP2:%.*]] = load <8 x i8>, <8 x i8>* [[ARRAYIDX]], align 8
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.uint8x8x2_t, %struct.uint8x8x2_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <8 x i8>], [2 x <8 x i8>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP3:%.*]] = load <8 x i8>, <8 x i8>* [[ARRAYIDX2]], align 8
// CHECK:   call void @llvm.aarch64.neon.st2lane.v8i8.p0i8(<8 x i8> [[TMP2]], <8 x i8> [[TMP3]], i64 7, i8* %a)
// CHECK:   ret void
void test_vst2_lane_u8(uint8_t  *a, uint8x8x2_t b) {
  vst2_lane_u8(a, b, 7);
}

// CHECK-LABEL: define{{.*}} void @test_vst2_lane_u16(i16* noundef %a, [2 x <4 x i16>] %b.coerce) #2 {
// CHECK:   [[B:%.*]] = alloca %struct.uint16x4x2_t, align 8
// CHECK:   [[__S1:%.*]] = alloca %struct.uint16x4x2_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.uint16x4x2_t, %struct.uint16x4x2_t* [[B]], i32 0, i32 0
// CHECK:   store [2 x <4 x i16>] [[B]].coerce, [2 x <4 x i16>]* [[COERCE_DIVE]], align 8
// CHECK:   [[TMP0:%.*]] = bitcast %struct.uint16x4x2_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.uint16x4x2_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[TMP0]], i8* align 8 [[TMP1]], i64 16, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast i16* %a to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.uint16x4x2_t, %struct.uint16x4x2_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <4 x i16>], [2 x <4 x i16>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <4 x i16>, <4 x i16>* [[ARRAYIDX]], align 8
// CHECK:   [[TMP4:%.*]] = bitcast <4 x i16> [[TMP3]] to <8 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.uint16x4x2_t, %struct.uint16x4x2_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <4 x i16>], [2 x <4 x i16>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <4 x i16>, <4 x i16>* [[ARRAYIDX2]], align 8
// CHECK:   [[TMP6:%.*]] = bitcast <4 x i16> [[TMP5]] to <8 x i8>
// CHECK:   [[TMP7:%.*]] = bitcast <8 x i8> [[TMP4]] to <4 x i16>
// CHECK:   [[TMP8:%.*]] = bitcast <8 x i8> [[TMP6]] to <4 x i16>
// CHECK:   call void @llvm.aarch64.neon.st2lane.v4i16.p0i8(<4 x i16> [[TMP7]], <4 x i16> [[TMP8]], i64 3, i8* [[TMP2]])
// CHECK:   ret void
void test_vst2_lane_u16(uint16_t  *a, uint16x4x2_t b) {
  vst2_lane_u16(a, b, 3);
}

// CHECK-LABEL: define{{.*}} void @test_vst2_lane_u32(i32* noundef %a, [2 x <2 x i32>] %b.coerce) #2 {
// CHECK:   [[B:%.*]] = alloca %struct.uint32x2x2_t, align 8
// CHECK:   [[__S1:%.*]] = alloca %struct.uint32x2x2_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.uint32x2x2_t, %struct.uint32x2x2_t* [[B]], i32 0, i32 0
// CHECK:   store [2 x <2 x i32>] [[B]].coerce, [2 x <2 x i32>]* [[COERCE_DIVE]], align 8
// CHECK:   [[TMP0:%.*]] = bitcast %struct.uint32x2x2_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.uint32x2x2_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[TMP0]], i8* align 8 [[TMP1]], i64 16, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast i32* %a to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.uint32x2x2_t, %struct.uint32x2x2_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <2 x i32>], [2 x <2 x i32>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <2 x i32>, <2 x i32>* [[ARRAYIDX]], align 8
// CHECK:   [[TMP4:%.*]] = bitcast <2 x i32> [[TMP3]] to <8 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.uint32x2x2_t, %struct.uint32x2x2_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <2 x i32>], [2 x <2 x i32>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <2 x i32>, <2 x i32>* [[ARRAYIDX2]], align 8
// CHECK:   [[TMP6:%.*]] = bitcast <2 x i32> [[TMP5]] to <8 x i8>
// CHECK:   [[TMP7:%.*]] = bitcast <8 x i8> [[TMP4]] to <2 x i32>
// CHECK:   [[TMP8:%.*]] = bitcast <8 x i8> [[TMP6]] to <2 x i32>
// CHECK:   call void @llvm.aarch64.neon.st2lane.v2i32.p0i8(<2 x i32> [[TMP7]], <2 x i32> [[TMP8]], i64 1, i8* [[TMP2]])
// CHECK:   ret void
void test_vst2_lane_u32(uint32_t  *a, uint32x2x2_t b) {
  vst2_lane_u32(a, b, 1);
}

// CHECK-LABEL: define{{.*}} void @test_vst2_lane_u64(i64* noundef %a, [2 x <1 x i64>] %b.coerce) #2 {
// CHECK:   [[B:%.*]] = alloca %struct.uint64x1x2_t, align 8
// CHECK:   [[__S1:%.*]] = alloca %struct.uint64x1x2_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.uint64x1x2_t, %struct.uint64x1x2_t* [[B]], i32 0, i32 0
// CHECK:   store [2 x <1 x i64>] [[B]].coerce, [2 x <1 x i64>]* [[COERCE_DIVE]], align 8
// CHECK:   [[TMP0:%.*]] = bitcast %struct.uint64x1x2_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.uint64x1x2_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[TMP0]], i8* align 8 [[TMP1]], i64 16, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast i64* %a to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.uint64x1x2_t, %struct.uint64x1x2_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <1 x i64>], [2 x <1 x i64>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <1 x i64>, <1 x i64>* [[ARRAYIDX]], align 8
// CHECK:   [[TMP4:%.*]] = bitcast <1 x i64> [[TMP3]] to <8 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.uint64x1x2_t, %struct.uint64x1x2_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <1 x i64>], [2 x <1 x i64>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <1 x i64>, <1 x i64>* [[ARRAYIDX2]], align 8
// CHECK:   [[TMP6:%.*]] = bitcast <1 x i64> [[TMP5]] to <8 x i8>
// CHECK:   [[TMP7:%.*]] = bitcast <8 x i8> [[TMP4]] to <1 x i64>
// CHECK:   [[TMP8:%.*]] = bitcast <8 x i8> [[TMP6]] to <1 x i64>
// CHECK:   call void @llvm.aarch64.neon.st2lane.v1i64.p0i8(<1 x i64> [[TMP7]], <1 x i64> [[TMP8]], i64 0, i8* [[TMP2]])
// CHECK:   ret void
void test_vst2_lane_u64(uint64_t  *a, uint64x1x2_t b) {
  vst2_lane_u64(a, b, 0);
}

// CHECK-LABEL: define{{.*}} void @test_vst2_lane_s8(i8* noundef %a, [2 x <8 x i8>] %b.coerce) #2 {
// CHECK:   [[B:%.*]] = alloca %struct.int8x8x2_t, align 8
// CHECK:   [[__S1:%.*]] = alloca %struct.int8x8x2_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.int8x8x2_t, %struct.int8x8x2_t* [[B]], i32 0, i32 0
// CHECK:   store [2 x <8 x i8>] [[B]].coerce, [2 x <8 x i8>]* [[COERCE_DIVE]], align 8
// CHECK:   [[TMP0:%.*]] = bitcast %struct.int8x8x2_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.int8x8x2_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[TMP0]], i8* align 8 [[TMP1]], i64 16, i1 false)
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.int8x8x2_t, %struct.int8x8x2_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <8 x i8>], [2 x <8 x i8>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP2:%.*]] = load <8 x i8>, <8 x i8>* [[ARRAYIDX]], align 8
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.int8x8x2_t, %struct.int8x8x2_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <8 x i8>], [2 x <8 x i8>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP3:%.*]] = load <8 x i8>, <8 x i8>* [[ARRAYIDX2]], align 8
// CHECK:   call void @llvm.aarch64.neon.st2lane.v8i8.p0i8(<8 x i8> [[TMP2]], <8 x i8> [[TMP3]], i64 7, i8* %a)
// CHECK:   ret void
void test_vst2_lane_s8(int8_t  *a, int8x8x2_t b) {
  vst2_lane_s8(a, b, 7);
}

// CHECK-LABEL: define{{.*}} void @test_vst2_lane_s16(i16* noundef %a, [2 x <4 x i16>] %b.coerce) #2 {
// CHECK:   [[B:%.*]] = alloca %struct.int16x4x2_t, align 8
// CHECK:   [[__S1:%.*]] = alloca %struct.int16x4x2_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.int16x4x2_t, %struct.int16x4x2_t* [[B]], i32 0, i32 0
// CHECK:   store [2 x <4 x i16>] [[B]].coerce, [2 x <4 x i16>]* [[COERCE_DIVE]], align 8
// CHECK:   [[TMP0:%.*]] = bitcast %struct.int16x4x2_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.int16x4x2_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[TMP0]], i8* align 8 [[TMP1]], i64 16, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast i16* %a to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.int16x4x2_t, %struct.int16x4x2_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <4 x i16>], [2 x <4 x i16>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <4 x i16>, <4 x i16>* [[ARRAYIDX]], align 8
// CHECK:   [[TMP4:%.*]] = bitcast <4 x i16> [[TMP3]] to <8 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.int16x4x2_t, %struct.int16x4x2_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <4 x i16>], [2 x <4 x i16>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <4 x i16>, <4 x i16>* [[ARRAYIDX2]], align 8
// CHECK:   [[TMP6:%.*]] = bitcast <4 x i16> [[TMP5]] to <8 x i8>
// CHECK:   [[TMP7:%.*]] = bitcast <8 x i8> [[TMP4]] to <4 x i16>
// CHECK:   [[TMP8:%.*]] = bitcast <8 x i8> [[TMP6]] to <4 x i16>
// CHECK:   call void @llvm.aarch64.neon.st2lane.v4i16.p0i8(<4 x i16> [[TMP7]], <4 x i16> [[TMP8]], i64 3, i8* [[TMP2]])
// CHECK:   ret void
void test_vst2_lane_s16(int16_t  *a, int16x4x2_t b) {
  vst2_lane_s16(a, b, 3);
}

// CHECK-LABEL: define{{.*}} void @test_vst2_lane_s32(i32* noundef %a, [2 x <2 x i32>] %b.coerce) #2 {
// CHECK:   [[B:%.*]] = alloca %struct.int32x2x2_t, align 8
// CHECK:   [[__S1:%.*]] = alloca %struct.int32x2x2_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.int32x2x2_t, %struct.int32x2x2_t* [[B]], i32 0, i32 0
// CHECK:   store [2 x <2 x i32>] [[B]].coerce, [2 x <2 x i32>]* [[COERCE_DIVE]], align 8
// CHECK:   [[TMP0:%.*]] = bitcast %struct.int32x2x2_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.int32x2x2_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[TMP0]], i8* align 8 [[TMP1]], i64 16, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast i32* %a to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.int32x2x2_t, %struct.int32x2x2_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <2 x i32>], [2 x <2 x i32>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <2 x i32>, <2 x i32>* [[ARRAYIDX]], align 8
// CHECK:   [[TMP4:%.*]] = bitcast <2 x i32> [[TMP3]] to <8 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.int32x2x2_t, %struct.int32x2x2_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <2 x i32>], [2 x <2 x i32>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <2 x i32>, <2 x i32>* [[ARRAYIDX2]], align 8
// CHECK:   [[TMP6:%.*]] = bitcast <2 x i32> [[TMP5]] to <8 x i8>
// CHECK:   [[TMP7:%.*]] = bitcast <8 x i8> [[TMP4]] to <2 x i32>
// CHECK:   [[TMP8:%.*]] = bitcast <8 x i8> [[TMP6]] to <2 x i32>
// CHECK:   call void @llvm.aarch64.neon.st2lane.v2i32.p0i8(<2 x i32> [[TMP7]], <2 x i32> [[TMP8]], i64 1, i8* [[TMP2]])
// CHECK:   ret void
void test_vst2_lane_s32(int32_t  *a, int32x2x2_t b) {
  vst2_lane_s32(a, b, 1);
}

// CHECK-LABEL: define{{.*}} void @test_vst2_lane_s64(i64* noundef %a, [2 x <1 x i64>] %b.coerce) #2 {
// CHECK:   [[B:%.*]] = alloca %struct.int64x1x2_t, align 8
// CHECK:   [[__S1:%.*]] = alloca %struct.int64x1x2_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.int64x1x2_t, %struct.int64x1x2_t* [[B]], i32 0, i32 0
// CHECK:   store [2 x <1 x i64>] [[B]].coerce, [2 x <1 x i64>]* [[COERCE_DIVE]], align 8
// CHECK:   [[TMP0:%.*]] = bitcast %struct.int64x1x2_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.int64x1x2_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[TMP0]], i8* align 8 [[TMP1]], i64 16, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast i64* %a to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.int64x1x2_t, %struct.int64x1x2_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <1 x i64>], [2 x <1 x i64>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <1 x i64>, <1 x i64>* [[ARRAYIDX]], align 8
// CHECK:   [[TMP4:%.*]] = bitcast <1 x i64> [[TMP3]] to <8 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.int64x1x2_t, %struct.int64x1x2_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <1 x i64>], [2 x <1 x i64>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <1 x i64>, <1 x i64>* [[ARRAYIDX2]], align 8
// CHECK:   [[TMP6:%.*]] = bitcast <1 x i64> [[TMP5]] to <8 x i8>
// CHECK:   [[TMP7:%.*]] = bitcast <8 x i8> [[TMP4]] to <1 x i64>
// CHECK:   [[TMP8:%.*]] = bitcast <8 x i8> [[TMP6]] to <1 x i64>
// CHECK:   call void @llvm.aarch64.neon.st2lane.v1i64.p0i8(<1 x i64> [[TMP7]], <1 x i64> [[TMP8]], i64 0, i8* [[TMP2]])
// CHECK:   ret void
void test_vst2_lane_s64(int64_t  *a, int64x1x2_t b) {
  vst2_lane_s64(a, b, 0);
}

// CHECK-LABEL: define{{.*}} void @test_vst2_lane_f16(half* noundef %a, [2 x <4 x half>] %b.coerce) #2 {
// CHECK:   [[B:%.*]] = alloca %struct.float16x4x2_t, align 8
// CHECK:   [[__S1:%.*]] = alloca %struct.float16x4x2_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.float16x4x2_t, %struct.float16x4x2_t* [[B]], i32 0, i32 0
// CHECK:   store [2 x <4 x half>] [[B]].coerce, [2 x <4 x half>]* [[COERCE_DIVE]], align 8
// CHECK:   [[TMP0:%.*]] = bitcast %struct.float16x4x2_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.float16x4x2_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[TMP0]], i8* align 8 [[TMP1]], i64 16, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast half* %a to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.float16x4x2_t, %struct.float16x4x2_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <4 x half>], [2 x <4 x half>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <4 x half>, <4 x half>* [[ARRAYIDX]], align 8
// CHECK:   [[TMP4:%.*]] = bitcast <4 x half> [[TMP3]] to <8 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.float16x4x2_t, %struct.float16x4x2_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <4 x half>], [2 x <4 x half>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <4 x half>, <4 x half>* [[ARRAYIDX2]], align 8
// CHECK:   [[TMP6:%.*]] = bitcast <4 x half> [[TMP5]] to <8 x i8>
// CHECK:   [[TMP7:%.*]] = bitcast <8 x i8> [[TMP4]] to <4 x half>
// CHECK:   [[TMP8:%.*]] = bitcast <8 x i8> [[TMP6]] to <4 x half>
// CHECK:   call void @llvm.aarch64.neon.st2lane.v4f16.p0i8(<4 x half> [[TMP7]], <4 x half> [[TMP8]], i64 3, i8* [[TMP2]])
// CHECK:   ret void
void test_vst2_lane_f16(float16_t  *a, float16x4x2_t b) {
  vst2_lane_f16(a, b, 3);
}

// CHECK-LABEL: define{{.*}} void @test_vst2_lane_f32(float* noundef %a, [2 x <2 x float>] %b.coerce) #2 {
// CHECK:   [[B:%.*]] = alloca %struct.float32x2x2_t, align 8
// CHECK:   [[__S1:%.*]] = alloca %struct.float32x2x2_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.float32x2x2_t, %struct.float32x2x2_t* [[B]], i32 0, i32 0
// CHECK:   store [2 x <2 x float>] [[B]].coerce, [2 x <2 x float>]* [[COERCE_DIVE]], align 8
// CHECK:   [[TMP0:%.*]] = bitcast %struct.float32x2x2_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.float32x2x2_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[TMP0]], i8* align 8 [[TMP1]], i64 16, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast float* %a to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.float32x2x2_t, %struct.float32x2x2_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <2 x float>], [2 x <2 x float>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <2 x float>, <2 x float>* [[ARRAYIDX]], align 8
// CHECK:   [[TMP4:%.*]] = bitcast <2 x float> [[TMP3]] to <8 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.float32x2x2_t, %struct.float32x2x2_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <2 x float>], [2 x <2 x float>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <2 x float>, <2 x float>* [[ARRAYIDX2]], align 8
// CHECK:   [[TMP6:%.*]] = bitcast <2 x float> [[TMP5]] to <8 x i8>
// CHECK:   [[TMP7:%.*]] = bitcast <8 x i8> [[TMP4]] to <2 x float>
// CHECK:   [[TMP8:%.*]] = bitcast <8 x i8> [[TMP6]] to <2 x float>
// CHECK:   call void @llvm.aarch64.neon.st2lane.v2f32.p0i8(<2 x float> [[TMP7]], <2 x float> [[TMP8]], i64 1, i8* [[TMP2]])
// CHECK:   ret void
void test_vst2_lane_f32(float32_t  *a, float32x2x2_t b) {
  vst2_lane_f32(a, b, 1);
}

// CHECK-LABEL: define{{.*}} void @test_vst2_lane_f64(double* noundef %a, [2 x <1 x double>] %b.coerce) #2 {
// CHECK:   [[B:%.*]] = alloca %struct.float64x1x2_t, align 8
// CHECK:   [[__S1:%.*]] = alloca %struct.float64x1x2_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.float64x1x2_t, %struct.float64x1x2_t* [[B]], i32 0, i32 0
// CHECK:   store [2 x <1 x double>] [[B]].coerce, [2 x <1 x double>]* [[COERCE_DIVE]], align 8
// CHECK:   [[TMP0:%.*]] = bitcast %struct.float64x1x2_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.float64x1x2_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[TMP0]], i8* align 8 [[TMP1]], i64 16, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast double* %a to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.float64x1x2_t, %struct.float64x1x2_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <1 x double>], [2 x <1 x double>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <1 x double>, <1 x double>* [[ARRAYIDX]], align 8
// CHECK:   [[TMP4:%.*]] = bitcast <1 x double> [[TMP3]] to <8 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.float64x1x2_t, %struct.float64x1x2_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <1 x double>], [2 x <1 x double>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <1 x double>, <1 x double>* [[ARRAYIDX2]], align 8
// CHECK:   [[TMP6:%.*]] = bitcast <1 x double> [[TMP5]] to <8 x i8>
// CHECK:   [[TMP7:%.*]] = bitcast <8 x i8> [[TMP4]] to <1 x double>
// CHECK:   [[TMP8:%.*]] = bitcast <8 x i8> [[TMP6]] to <1 x double>
// CHECK:   call void @llvm.aarch64.neon.st2lane.v1f64.p0i8(<1 x double> [[TMP7]], <1 x double> [[TMP8]], i64 0, i8* [[TMP2]])
// CHECK:   ret void
void test_vst2_lane_f64(float64_t  *a, float64x1x2_t b) {
  vst2_lane_f64(a, b, 0);
}

// CHECK-LABEL: define{{.*}} void @test_vst2_lane_p8(i8* noundef %a, [2 x <8 x i8>] %b.coerce) #2 {
// CHECK:   [[B:%.*]] = alloca %struct.poly8x8x2_t, align 8
// CHECK:   [[__S1:%.*]] = alloca %struct.poly8x8x2_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.poly8x8x2_t, %struct.poly8x8x2_t* [[B]], i32 0, i32 0
// CHECK:   store [2 x <8 x i8>] [[B]].coerce, [2 x <8 x i8>]* [[COERCE_DIVE]], align 8
// CHECK:   [[TMP0:%.*]] = bitcast %struct.poly8x8x2_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.poly8x8x2_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[TMP0]], i8* align 8 [[TMP1]], i64 16, i1 false)
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.poly8x8x2_t, %struct.poly8x8x2_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <8 x i8>], [2 x <8 x i8>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP2:%.*]] = load <8 x i8>, <8 x i8>* [[ARRAYIDX]], align 8
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.poly8x8x2_t, %struct.poly8x8x2_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <8 x i8>], [2 x <8 x i8>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP3:%.*]] = load <8 x i8>, <8 x i8>* [[ARRAYIDX2]], align 8
// CHECK:   call void @llvm.aarch64.neon.st2lane.v8i8.p0i8(<8 x i8> [[TMP2]], <8 x i8> [[TMP3]], i64 7, i8* %a)
// CHECK:   ret void
void test_vst2_lane_p8(poly8_t  *a, poly8x8x2_t b) {
  vst2_lane_p8(a, b, 7);
}

// CHECK-LABEL: define{{.*}} void @test_vst2_lane_p16(i16* noundef %a, [2 x <4 x i16>] %b.coerce) #2 {
// CHECK:   [[B:%.*]] = alloca %struct.poly16x4x2_t, align 8
// CHECK:   [[__S1:%.*]] = alloca %struct.poly16x4x2_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.poly16x4x2_t, %struct.poly16x4x2_t* [[B]], i32 0, i32 0
// CHECK:   store [2 x <4 x i16>] [[B]].coerce, [2 x <4 x i16>]* [[COERCE_DIVE]], align 8
// CHECK:   [[TMP0:%.*]] = bitcast %struct.poly16x4x2_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.poly16x4x2_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[TMP0]], i8* align 8 [[TMP1]], i64 16, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast i16* %a to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.poly16x4x2_t, %struct.poly16x4x2_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <4 x i16>], [2 x <4 x i16>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <4 x i16>, <4 x i16>* [[ARRAYIDX]], align 8
// CHECK:   [[TMP4:%.*]] = bitcast <4 x i16> [[TMP3]] to <8 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.poly16x4x2_t, %struct.poly16x4x2_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <4 x i16>], [2 x <4 x i16>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <4 x i16>, <4 x i16>* [[ARRAYIDX2]], align 8
// CHECK:   [[TMP6:%.*]] = bitcast <4 x i16> [[TMP5]] to <8 x i8>
// CHECK:   [[TMP7:%.*]] = bitcast <8 x i8> [[TMP4]] to <4 x i16>
// CHECK:   [[TMP8:%.*]] = bitcast <8 x i8> [[TMP6]] to <4 x i16>
// CHECK:   call void @llvm.aarch64.neon.st2lane.v4i16.p0i8(<4 x i16> [[TMP7]], <4 x i16> [[TMP8]], i64 3, i8* [[TMP2]])
// CHECK:   ret void
void test_vst2_lane_p16(poly16_t  *a, poly16x4x2_t b) {
  vst2_lane_p16(a, b, 3);
}

// CHECK-LABEL: define{{.*}} void @test_vst2_lane_p64(i64* noundef %a, [2 x <1 x i64>] %b.coerce) #2 {
// CHECK:   [[B:%.*]] = alloca %struct.poly64x1x2_t, align 8
// CHECK:   [[__S1:%.*]] = alloca %struct.poly64x1x2_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.poly64x1x2_t, %struct.poly64x1x2_t* [[B]], i32 0, i32 0
// CHECK:   store [2 x <1 x i64>] [[B]].coerce, [2 x <1 x i64>]* [[COERCE_DIVE]], align 8
// CHECK:   [[TMP0:%.*]] = bitcast %struct.poly64x1x2_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.poly64x1x2_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[TMP0]], i8* align 8 [[TMP1]], i64 16, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast i64* %a to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.poly64x1x2_t, %struct.poly64x1x2_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <1 x i64>], [2 x <1 x i64>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <1 x i64>, <1 x i64>* [[ARRAYIDX]], align 8
// CHECK:   [[TMP4:%.*]] = bitcast <1 x i64> [[TMP3]] to <8 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.poly64x1x2_t, %struct.poly64x1x2_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <1 x i64>], [2 x <1 x i64>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <1 x i64>, <1 x i64>* [[ARRAYIDX2]], align 8
// CHECK:   [[TMP6:%.*]] = bitcast <1 x i64> [[TMP5]] to <8 x i8>
// CHECK:   [[TMP7:%.*]] = bitcast <8 x i8> [[TMP4]] to <1 x i64>
// CHECK:   [[TMP8:%.*]] = bitcast <8 x i8> [[TMP6]] to <1 x i64>
// CHECK:   call void @llvm.aarch64.neon.st2lane.v1i64.p0i8(<1 x i64> [[TMP7]], <1 x i64> [[TMP8]], i64 0, i8* [[TMP2]])
// CHECK:   ret void
void test_vst2_lane_p64(poly64_t  *a, poly64x1x2_t b) {
  vst2_lane_p64(a, b, 0);
}

// CHECK-LABEL: define{{.*}} void @test_vst3q_lane_u8(i8* noundef %a, [3 x <16 x i8>] %b.coerce) #2 {
// CHECK:   [[B:%.*]] = alloca %struct.uint8x16x3_t, align 16
// CHECK:   [[__S1:%.*]] = alloca %struct.uint8x16x3_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.uint8x16x3_t, %struct.uint8x16x3_t* [[B]], i32 0, i32 0
// CHECK:   store [3 x <16 x i8>] [[B]].coerce, [3 x <16 x i8>]* [[COERCE_DIVE]], align 16
// CHECK:   [[TMP0:%.*]] = bitcast %struct.uint8x16x3_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.uint8x16x3_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP0]], i8* align 16 [[TMP1]], i64 48, i1 false)
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.uint8x16x3_t, %struct.uint8x16x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <16 x i8>], [3 x <16 x i8>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP2:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX]], align 16
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.uint8x16x3_t, %struct.uint8x16x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <16 x i8>], [3 x <16 x i8>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP3:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX2]], align 16
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds %struct.uint8x16x3_t, %struct.uint8x16x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <16 x i8>], [3 x <16 x i8>]* [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP4:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX4]], align 16
// CHECK:   call void @llvm.aarch64.neon.st3lane.v16i8.p0i8(<16 x i8> [[TMP2]], <16 x i8> [[TMP3]], <16 x i8> [[TMP4]], i64 15, i8* %a)
// CHECK:   ret void
void test_vst3q_lane_u8(uint8_t  *a, uint8x16x3_t b) {
  vst3q_lane_u8(a, b, 15);
}

// CHECK-LABEL: define{{.*}} void @test_vst3q_lane_u16(i16* noundef %a, [3 x <8 x i16>] %b.coerce) #2 {
// CHECK:   [[B:%.*]] = alloca %struct.uint16x8x3_t, align 16
// CHECK:   [[__S1:%.*]] = alloca %struct.uint16x8x3_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.uint16x8x3_t, %struct.uint16x8x3_t* [[B]], i32 0, i32 0
// CHECK:   store [3 x <8 x i16>] [[B]].coerce, [3 x <8 x i16>]* [[COERCE_DIVE]], align 16
// CHECK:   [[TMP0:%.*]] = bitcast %struct.uint16x8x3_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.uint16x8x3_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP0]], i8* align 16 [[TMP1]], i64 48, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast i16* %a to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.uint16x8x3_t, %struct.uint16x8x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <8 x i16>], [3 x <8 x i16>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <8 x i16>, <8 x i16>* [[ARRAYIDX]], align 16
// CHECK:   [[TMP4:%.*]] = bitcast <8 x i16> [[TMP3]] to <16 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.uint16x8x3_t, %struct.uint16x8x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <8 x i16>], [3 x <8 x i16>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <8 x i16>, <8 x i16>* [[ARRAYIDX2]], align 16
// CHECK:   [[TMP6:%.*]] = bitcast <8 x i16> [[TMP5]] to <16 x i8>
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds %struct.uint16x8x3_t, %struct.uint16x8x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <8 x i16>], [3 x <8 x i16>]* [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP7:%.*]] = load <8 x i16>, <8 x i16>* [[ARRAYIDX4]], align 16
// CHECK:   [[TMP8:%.*]] = bitcast <8 x i16> [[TMP7]] to <16 x i8>
// CHECK:   [[TMP9:%.*]] = bitcast <16 x i8> [[TMP4]] to <8 x i16>
// CHECK:   [[TMP10:%.*]] = bitcast <16 x i8> [[TMP6]] to <8 x i16>
// CHECK:   [[TMP11:%.*]] = bitcast <16 x i8> [[TMP8]] to <8 x i16>
// CHECK:   call void @llvm.aarch64.neon.st3lane.v8i16.p0i8(<8 x i16> [[TMP9]], <8 x i16> [[TMP10]], <8 x i16> [[TMP11]], i64 7, i8* [[TMP2]])
// CHECK:   ret void
void test_vst3q_lane_u16(uint16_t  *a, uint16x8x3_t b) {
  vst3q_lane_u16(a, b, 7);
}

// CHECK-LABEL: define{{.*}} void @test_vst3q_lane_u32(i32* noundef %a, [3 x <4 x i32>] %b.coerce) #2 {
// CHECK:   [[B:%.*]] = alloca %struct.uint32x4x3_t, align 16
// CHECK:   [[__S1:%.*]] = alloca %struct.uint32x4x3_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.uint32x4x3_t, %struct.uint32x4x3_t* [[B]], i32 0, i32 0
// CHECK:   store [3 x <4 x i32>] [[B]].coerce, [3 x <4 x i32>]* [[COERCE_DIVE]], align 16
// CHECK:   [[TMP0:%.*]] = bitcast %struct.uint32x4x3_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.uint32x4x3_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP0]], i8* align 16 [[TMP1]], i64 48, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast i32* %a to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.uint32x4x3_t, %struct.uint32x4x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <4 x i32>], [3 x <4 x i32>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <4 x i32>, <4 x i32>* [[ARRAYIDX]], align 16
// CHECK:   [[TMP4:%.*]] = bitcast <4 x i32> [[TMP3]] to <16 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.uint32x4x3_t, %struct.uint32x4x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <4 x i32>], [3 x <4 x i32>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <4 x i32>, <4 x i32>* [[ARRAYIDX2]], align 16
// CHECK:   [[TMP6:%.*]] = bitcast <4 x i32> [[TMP5]] to <16 x i8>
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds %struct.uint32x4x3_t, %struct.uint32x4x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <4 x i32>], [3 x <4 x i32>]* [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP7:%.*]] = load <4 x i32>, <4 x i32>* [[ARRAYIDX4]], align 16
// CHECK:   [[TMP8:%.*]] = bitcast <4 x i32> [[TMP7]] to <16 x i8>
// CHECK:   [[TMP9:%.*]] = bitcast <16 x i8> [[TMP4]] to <4 x i32>
// CHECK:   [[TMP10:%.*]] = bitcast <16 x i8> [[TMP6]] to <4 x i32>
// CHECK:   [[TMP11:%.*]] = bitcast <16 x i8> [[TMP8]] to <4 x i32>
// CHECK:   call void @llvm.aarch64.neon.st3lane.v4i32.p0i8(<4 x i32> [[TMP9]], <4 x i32> [[TMP10]], <4 x i32> [[TMP11]], i64 3, i8* [[TMP2]])
// CHECK:   ret void
void test_vst3q_lane_u32(uint32_t  *a, uint32x4x3_t b) {
  vst3q_lane_u32(a, b, 3);
}

// CHECK-LABEL: define{{.*}} void @test_vst3q_lane_u64(i64* noundef %a, [3 x <2 x i64>] %b.coerce) #2 {
// CHECK:   [[B:%.*]] = alloca %struct.uint64x2x3_t, align 16
// CHECK:   [[__S1:%.*]] = alloca %struct.uint64x2x3_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.uint64x2x3_t, %struct.uint64x2x3_t* [[B]], i32 0, i32 0
// CHECK:   store [3 x <2 x i64>] [[B]].coerce, [3 x <2 x i64>]* [[COERCE_DIVE]], align 16
// CHECK:   [[TMP0:%.*]] = bitcast %struct.uint64x2x3_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.uint64x2x3_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP0]], i8* align 16 [[TMP1]], i64 48, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast i64* %a to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.uint64x2x3_t, %struct.uint64x2x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <2 x i64>], [3 x <2 x i64>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <2 x i64>, <2 x i64>* [[ARRAYIDX]], align 16
// CHECK:   [[TMP4:%.*]] = bitcast <2 x i64> [[TMP3]] to <16 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.uint64x2x3_t, %struct.uint64x2x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <2 x i64>], [3 x <2 x i64>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <2 x i64>, <2 x i64>* [[ARRAYIDX2]], align 16
// CHECK:   [[TMP6:%.*]] = bitcast <2 x i64> [[TMP5]] to <16 x i8>
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds %struct.uint64x2x3_t, %struct.uint64x2x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <2 x i64>], [3 x <2 x i64>]* [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP7:%.*]] = load <2 x i64>, <2 x i64>* [[ARRAYIDX4]], align 16
// CHECK:   [[TMP8:%.*]] = bitcast <2 x i64> [[TMP7]] to <16 x i8>
// CHECK:   [[TMP9:%.*]] = bitcast <16 x i8> [[TMP4]] to <2 x i64>
// CHECK:   [[TMP10:%.*]] = bitcast <16 x i8> [[TMP6]] to <2 x i64>
// CHECK:   [[TMP11:%.*]] = bitcast <16 x i8> [[TMP8]] to <2 x i64>
// CHECK:   call void @llvm.aarch64.neon.st3lane.v2i64.p0i8(<2 x i64> [[TMP9]], <2 x i64> [[TMP10]], <2 x i64> [[TMP11]], i64 1, i8* [[TMP2]])
// CHECK:   ret void
void test_vst3q_lane_u64(uint64_t  *a, uint64x2x3_t b) {
  vst3q_lane_u64(a, b, 1);
}

// CHECK-LABEL: define{{.*}} void @test_vst3q_lane_s8(i8* noundef %a, [3 x <16 x i8>] %b.coerce) #2 {
// CHECK:   [[B:%.*]] = alloca %struct.int8x16x3_t, align 16
// CHECK:   [[__S1:%.*]] = alloca %struct.int8x16x3_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.int8x16x3_t, %struct.int8x16x3_t* [[B]], i32 0, i32 0
// CHECK:   store [3 x <16 x i8>] [[B]].coerce, [3 x <16 x i8>]* [[COERCE_DIVE]], align 16
// CHECK:   [[TMP0:%.*]] = bitcast %struct.int8x16x3_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.int8x16x3_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP0]], i8* align 16 [[TMP1]], i64 48, i1 false)
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.int8x16x3_t, %struct.int8x16x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <16 x i8>], [3 x <16 x i8>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP2:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX]], align 16
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.int8x16x3_t, %struct.int8x16x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <16 x i8>], [3 x <16 x i8>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP3:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX2]], align 16
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds %struct.int8x16x3_t, %struct.int8x16x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <16 x i8>], [3 x <16 x i8>]* [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP4:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX4]], align 16
// CHECK:   call void @llvm.aarch64.neon.st3lane.v16i8.p0i8(<16 x i8> [[TMP2]], <16 x i8> [[TMP3]], <16 x i8> [[TMP4]], i64 15, i8* %a)
// CHECK:   ret void
void test_vst3q_lane_s8(int8_t  *a, int8x16x3_t b) {
  vst3q_lane_s8(a, b, 15);
}

// CHECK-LABEL: define{{.*}} void @test_vst3q_lane_s16(i16* noundef %a, [3 x <8 x i16>] %b.coerce) #2 {
// CHECK:   [[B:%.*]] = alloca %struct.int16x8x3_t, align 16
// CHECK:   [[__S1:%.*]] = alloca %struct.int16x8x3_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.int16x8x3_t, %struct.int16x8x3_t* [[B]], i32 0, i32 0
// CHECK:   store [3 x <8 x i16>] [[B]].coerce, [3 x <8 x i16>]* [[COERCE_DIVE]], align 16
// CHECK:   [[TMP0:%.*]] = bitcast %struct.int16x8x3_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.int16x8x3_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP0]], i8* align 16 [[TMP1]], i64 48, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast i16* %a to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.int16x8x3_t, %struct.int16x8x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <8 x i16>], [3 x <8 x i16>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <8 x i16>, <8 x i16>* [[ARRAYIDX]], align 16
// CHECK:   [[TMP4:%.*]] = bitcast <8 x i16> [[TMP3]] to <16 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.int16x8x3_t, %struct.int16x8x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <8 x i16>], [3 x <8 x i16>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <8 x i16>, <8 x i16>* [[ARRAYIDX2]], align 16
// CHECK:   [[TMP6:%.*]] = bitcast <8 x i16> [[TMP5]] to <16 x i8>
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds %struct.int16x8x3_t, %struct.int16x8x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <8 x i16>], [3 x <8 x i16>]* [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP7:%.*]] = load <8 x i16>, <8 x i16>* [[ARRAYIDX4]], align 16
// CHECK:   [[TMP8:%.*]] = bitcast <8 x i16> [[TMP7]] to <16 x i8>
// CHECK:   [[TMP9:%.*]] = bitcast <16 x i8> [[TMP4]] to <8 x i16>
// CHECK:   [[TMP10:%.*]] = bitcast <16 x i8> [[TMP6]] to <8 x i16>
// CHECK:   [[TMP11:%.*]] = bitcast <16 x i8> [[TMP8]] to <8 x i16>
// CHECK:   call void @llvm.aarch64.neon.st3lane.v8i16.p0i8(<8 x i16> [[TMP9]], <8 x i16> [[TMP10]], <8 x i16> [[TMP11]], i64 7, i8* [[TMP2]])
// CHECK:   ret void
void test_vst3q_lane_s16(int16_t  *a, int16x8x3_t b) {
  vst3q_lane_s16(a, b, 7);
}

// CHECK-LABEL: define{{.*}} void @test_vst3q_lane_s32(i32* noundef %a, [3 x <4 x i32>] %b.coerce) #2 {
// CHECK:   [[B:%.*]] = alloca %struct.int32x4x3_t, align 16
// CHECK:   [[__S1:%.*]] = alloca %struct.int32x4x3_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.int32x4x3_t, %struct.int32x4x3_t* [[B]], i32 0, i32 0
// CHECK:   store [3 x <4 x i32>] [[B]].coerce, [3 x <4 x i32>]* [[COERCE_DIVE]], align 16
// CHECK:   [[TMP0:%.*]] = bitcast %struct.int32x4x3_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.int32x4x3_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP0]], i8* align 16 [[TMP1]], i64 48, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast i32* %a to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.int32x4x3_t, %struct.int32x4x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <4 x i32>], [3 x <4 x i32>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <4 x i32>, <4 x i32>* [[ARRAYIDX]], align 16
// CHECK:   [[TMP4:%.*]] = bitcast <4 x i32> [[TMP3]] to <16 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.int32x4x3_t, %struct.int32x4x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <4 x i32>], [3 x <4 x i32>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <4 x i32>, <4 x i32>* [[ARRAYIDX2]], align 16
// CHECK:   [[TMP6:%.*]] = bitcast <4 x i32> [[TMP5]] to <16 x i8>
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds %struct.int32x4x3_t, %struct.int32x4x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <4 x i32>], [3 x <4 x i32>]* [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP7:%.*]] = load <4 x i32>, <4 x i32>* [[ARRAYIDX4]], align 16
// CHECK:   [[TMP8:%.*]] = bitcast <4 x i32> [[TMP7]] to <16 x i8>
// CHECK:   [[TMP9:%.*]] = bitcast <16 x i8> [[TMP4]] to <4 x i32>
// CHECK:   [[TMP10:%.*]] = bitcast <16 x i8> [[TMP6]] to <4 x i32>
// CHECK:   [[TMP11:%.*]] = bitcast <16 x i8> [[TMP8]] to <4 x i32>
// CHECK:   call void @llvm.aarch64.neon.st3lane.v4i32.p0i8(<4 x i32> [[TMP9]], <4 x i32> [[TMP10]], <4 x i32> [[TMP11]], i64 3, i8* [[TMP2]])
// CHECK:   ret void
void test_vst3q_lane_s32(int32_t  *a, int32x4x3_t b) {
  vst3q_lane_s32(a, b, 3);
}

// CHECK-LABEL: define{{.*}} void @test_vst3q_lane_s64(i64* noundef %a, [3 x <2 x i64>] %b.coerce) #2 {
// CHECK:   [[B:%.*]] = alloca %struct.int64x2x3_t, align 16
// CHECK:   [[__S1:%.*]] = alloca %struct.int64x2x3_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.int64x2x3_t, %struct.int64x2x3_t* [[B]], i32 0, i32 0
// CHECK:   store [3 x <2 x i64>] [[B]].coerce, [3 x <2 x i64>]* [[COERCE_DIVE]], align 16
// CHECK:   [[TMP0:%.*]] = bitcast %struct.int64x2x3_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.int64x2x3_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP0]], i8* align 16 [[TMP1]], i64 48, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast i64* %a to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.int64x2x3_t, %struct.int64x2x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <2 x i64>], [3 x <2 x i64>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <2 x i64>, <2 x i64>* [[ARRAYIDX]], align 16
// CHECK:   [[TMP4:%.*]] = bitcast <2 x i64> [[TMP3]] to <16 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.int64x2x3_t, %struct.int64x2x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <2 x i64>], [3 x <2 x i64>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <2 x i64>, <2 x i64>* [[ARRAYIDX2]], align 16
// CHECK:   [[TMP6:%.*]] = bitcast <2 x i64> [[TMP5]] to <16 x i8>
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds %struct.int64x2x3_t, %struct.int64x2x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <2 x i64>], [3 x <2 x i64>]* [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP7:%.*]] = load <2 x i64>, <2 x i64>* [[ARRAYIDX4]], align 16
// CHECK:   [[TMP8:%.*]] = bitcast <2 x i64> [[TMP7]] to <16 x i8>
// CHECK:   [[TMP9:%.*]] = bitcast <16 x i8> [[TMP4]] to <2 x i64>
// CHECK:   [[TMP10:%.*]] = bitcast <16 x i8> [[TMP6]] to <2 x i64>
// CHECK:   [[TMP11:%.*]] = bitcast <16 x i8> [[TMP8]] to <2 x i64>
// CHECK:   call void @llvm.aarch64.neon.st3lane.v2i64.p0i8(<2 x i64> [[TMP9]], <2 x i64> [[TMP10]], <2 x i64> [[TMP11]], i64 1, i8* [[TMP2]])
// CHECK:   ret void
void test_vst3q_lane_s64(int64_t  *a, int64x2x3_t b) {
  vst3q_lane_s64(a, b, 1);
}

// CHECK-LABEL: define{{.*}} void @test_vst3q_lane_f16(half* noundef %a, [3 x <8 x half>] %b.coerce) #2 {
// CHECK:   [[B:%.*]] = alloca %struct.float16x8x3_t, align 16
// CHECK:   [[__S1:%.*]] = alloca %struct.float16x8x3_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.float16x8x3_t, %struct.float16x8x3_t* [[B]], i32 0, i32 0
// CHECK:   store [3 x <8 x half>] [[B]].coerce, [3 x <8 x half>]* [[COERCE_DIVE]], align 16
// CHECK:   [[TMP0:%.*]] = bitcast %struct.float16x8x3_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.float16x8x3_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP0]], i8* align 16 [[TMP1]], i64 48, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast half* %a to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.float16x8x3_t, %struct.float16x8x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <8 x half>], [3 x <8 x half>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <8 x half>, <8 x half>* [[ARRAYIDX]], align 16
// CHECK:   [[TMP4:%.*]] = bitcast <8 x half> [[TMP3]] to <16 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.float16x8x3_t, %struct.float16x8x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <8 x half>], [3 x <8 x half>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <8 x half>, <8 x half>* [[ARRAYIDX2]], align 16
// CHECK:   [[TMP6:%.*]] = bitcast <8 x half> [[TMP5]] to <16 x i8>
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds %struct.float16x8x3_t, %struct.float16x8x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <8 x half>], [3 x <8 x half>]* [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP7:%.*]] = load <8 x half>, <8 x half>* [[ARRAYIDX4]], align 16
// CHECK:   [[TMP8:%.*]] = bitcast <8 x half> [[TMP7]] to <16 x i8>
// CHECK:   [[TMP9:%.*]] = bitcast <16 x i8> [[TMP4]] to <8 x half>
// CHECK:   [[TMP10:%.*]] = bitcast <16 x i8> [[TMP6]] to <8 x half>
// CHECK:   [[TMP11:%.*]] = bitcast <16 x i8> [[TMP8]] to <8 x half>
// CHECK:   call void @llvm.aarch64.neon.st3lane.v8f16.p0i8(<8 x half> [[TMP9]], <8 x half> [[TMP10]], <8 x half> [[TMP11]], i64 7, i8* [[TMP2]])
// CHECK:   ret void
void test_vst3q_lane_f16(float16_t  *a, float16x8x3_t b) {
  vst3q_lane_f16(a, b, 7);
}

// CHECK-LABEL: define{{.*}} void @test_vst3q_lane_f32(float* noundef %a, [3 x <4 x float>] %b.coerce) #2 {
// CHECK:   [[B:%.*]] = alloca %struct.float32x4x3_t, align 16
// CHECK:   [[__S1:%.*]] = alloca %struct.float32x4x3_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.float32x4x3_t, %struct.float32x4x3_t* [[B]], i32 0, i32 0
// CHECK:   store [3 x <4 x float>] [[B]].coerce, [3 x <4 x float>]* [[COERCE_DIVE]], align 16
// CHECK:   [[TMP0:%.*]] = bitcast %struct.float32x4x3_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.float32x4x3_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP0]], i8* align 16 [[TMP1]], i64 48, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast float* %a to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.float32x4x3_t, %struct.float32x4x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <4 x float>], [3 x <4 x float>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <4 x float>, <4 x float>* [[ARRAYIDX]], align 16
// CHECK:   [[TMP4:%.*]] = bitcast <4 x float> [[TMP3]] to <16 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.float32x4x3_t, %struct.float32x4x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <4 x float>], [3 x <4 x float>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <4 x float>, <4 x float>* [[ARRAYIDX2]], align 16
// CHECK:   [[TMP6:%.*]] = bitcast <4 x float> [[TMP5]] to <16 x i8>
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds %struct.float32x4x3_t, %struct.float32x4x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <4 x float>], [3 x <4 x float>]* [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP7:%.*]] = load <4 x float>, <4 x float>* [[ARRAYIDX4]], align 16
// CHECK:   [[TMP8:%.*]] = bitcast <4 x float> [[TMP7]] to <16 x i8>
// CHECK:   [[TMP9:%.*]] = bitcast <16 x i8> [[TMP4]] to <4 x float>
// CHECK:   [[TMP10:%.*]] = bitcast <16 x i8> [[TMP6]] to <4 x float>
// CHECK:   [[TMP11:%.*]] = bitcast <16 x i8> [[TMP8]] to <4 x float>
// CHECK:   call void @llvm.aarch64.neon.st3lane.v4f32.p0i8(<4 x float> [[TMP9]], <4 x float> [[TMP10]], <4 x float> [[TMP11]], i64 3, i8* [[TMP2]])
// CHECK:   ret void
void test_vst3q_lane_f32(float32_t  *a, float32x4x3_t b) {
  vst3q_lane_f32(a, b, 3);
}

// CHECK-LABEL: define{{.*}} void @test_vst3q_lane_f64(double* noundef %a, [3 x <2 x double>] %b.coerce) #2 {
// CHECK:   [[B:%.*]] = alloca %struct.float64x2x3_t, align 16
// CHECK:   [[__S1:%.*]] = alloca %struct.float64x2x3_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.float64x2x3_t, %struct.float64x2x3_t* [[B]], i32 0, i32 0
// CHECK:   store [3 x <2 x double>] [[B]].coerce, [3 x <2 x double>]* [[COERCE_DIVE]], align 16
// CHECK:   [[TMP0:%.*]] = bitcast %struct.float64x2x3_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.float64x2x3_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP0]], i8* align 16 [[TMP1]], i64 48, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast double* %a to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.float64x2x3_t, %struct.float64x2x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <2 x double>], [3 x <2 x double>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <2 x double>, <2 x double>* [[ARRAYIDX]], align 16
// CHECK:   [[TMP4:%.*]] = bitcast <2 x double> [[TMP3]] to <16 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.float64x2x3_t, %struct.float64x2x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <2 x double>], [3 x <2 x double>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <2 x double>, <2 x double>* [[ARRAYIDX2]], align 16
// CHECK:   [[TMP6:%.*]] = bitcast <2 x double> [[TMP5]] to <16 x i8>
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds %struct.float64x2x3_t, %struct.float64x2x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <2 x double>], [3 x <2 x double>]* [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP7:%.*]] = load <2 x double>, <2 x double>* [[ARRAYIDX4]], align 16
// CHECK:   [[TMP8:%.*]] = bitcast <2 x double> [[TMP7]] to <16 x i8>
// CHECK:   [[TMP9:%.*]] = bitcast <16 x i8> [[TMP4]] to <2 x double>
// CHECK:   [[TMP10:%.*]] = bitcast <16 x i8> [[TMP6]] to <2 x double>
// CHECK:   [[TMP11:%.*]] = bitcast <16 x i8> [[TMP8]] to <2 x double>
// CHECK:   call void @llvm.aarch64.neon.st3lane.v2f64.p0i8(<2 x double> [[TMP9]], <2 x double> [[TMP10]], <2 x double> [[TMP11]], i64 1, i8* [[TMP2]])
// CHECK:   ret void
void test_vst3q_lane_f64(float64_t  *a, float64x2x3_t b) {
  vst3q_lane_f64(a, b, 1);
}

// CHECK-LABEL: define{{.*}} void @test_vst3q_lane_p8(i8* noundef %a, [3 x <16 x i8>] %b.coerce) #2 {
// CHECK:   [[B:%.*]] = alloca %struct.poly8x16x3_t, align 16
// CHECK:   [[__S1:%.*]] = alloca %struct.poly8x16x3_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.poly8x16x3_t, %struct.poly8x16x3_t* [[B]], i32 0, i32 0
// CHECK:   store [3 x <16 x i8>] [[B]].coerce, [3 x <16 x i8>]* [[COERCE_DIVE]], align 16
// CHECK:   [[TMP0:%.*]] = bitcast %struct.poly8x16x3_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.poly8x16x3_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP0]], i8* align 16 [[TMP1]], i64 48, i1 false)
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.poly8x16x3_t, %struct.poly8x16x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <16 x i8>], [3 x <16 x i8>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP2:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX]], align 16
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.poly8x16x3_t, %struct.poly8x16x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <16 x i8>], [3 x <16 x i8>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP3:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX2]], align 16
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds %struct.poly8x16x3_t, %struct.poly8x16x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <16 x i8>], [3 x <16 x i8>]* [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP4:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX4]], align 16
// CHECK:   call void @llvm.aarch64.neon.st3lane.v16i8.p0i8(<16 x i8> [[TMP2]], <16 x i8> [[TMP3]], <16 x i8> [[TMP4]], i64 15, i8* %a)
// CHECK:   ret void
void test_vst3q_lane_p8(poly8_t  *a, poly8x16x3_t b) {
  vst3q_lane_p8(a, b, 15);
}

// CHECK-LABEL: define{{.*}} void @test_vst3q_lane_p16(i16* noundef %a, [3 x <8 x i16>] %b.coerce) #2 {
// CHECK:   [[B:%.*]] = alloca %struct.poly16x8x3_t, align 16
// CHECK:   [[__S1:%.*]] = alloca %struct.poly16x8x3_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.poly16x8x3_t, %struct.poly16x8x3_t* [[B]], i32 0, i32 0
// CHECK:   store [3 x <8 x i16>] [[B]].coerce, [3 x <8 x i16>]* [[COERCE_DIVE]], align 16
// CHECK:   [[TMP0:%.*]] = bitcast %struct.poly16x8x3_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.poly16x8x3_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP0]], i8* align 16 [[TMP1]], i64 48, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast i16* %a to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.poly16x8x3_t, %struct.poly16x8x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <8 x i16>], [3 x <8 x i16>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <8 x i16>, <8 x i16>* [[ARRAYIDX]], align 16
// CHECK:   [[TMP4:%.*]] = bitcast <8 x i16> [[TMP3]] to <16 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.poly16x8x3_t, %struct.poly16x8x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <8 x i16>], [3 x <8 x i16>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <8 x i16>, <8 x i16>* [[ARRAYIDX2]], align 16
// CHECK:   [[TMP6:%.*]] = bitcast <8 x i16> [[TMP5]] to <16 x i8>
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds %struct.poly16x8x3_t, %struct.poly16x8x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <8 x i16>], [3 x <8 x i16>]* [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP7:%.*]] = load <8 x i16>, <8 x i16>* [[ARRAYIDX4]], align 16
// CHECK:   [[TMP8:%.*]] = bitcast <8 x i16> [[TMP7]] to <16 x i8>
// CHECK:   [[TMP9:%.*]] = bitcast <16 x i8> [[TMP4]] to <8 x i16>
// CHECK:   [[TMP10:%.*]] = bitcast <16 x i8> [[TMP6]] to <8 x i16>
// CHECK:   [[TMP11:%.*]] = bitcast <16 x i8> [[TMP8]] to <8 x i16>
// CHECK:   call void @llvm.aarch64.neon.st3lane.v8i16.p0i8(<8 x i16> [[TMP9]], <8 x i16> [[TMP10]], <8 x i16> [[TMP11]], i64 7, i8* [[TMP2]])
// CHECK:   ret void
void test_vst3q_lane_p16(poly16_t  *a, poly16x8x3_t b) {
  vst3q_lane_p16(a, b, 7);
}

// CHECK-LABEL: define{{.*}} void @test_vst3q_lane_p64(i64* noundef %a, [3 x <2 x i64>] %b.coerce) #2 {
// CHECK:   [[B:%.*]] = alloca %struct.poly64x2x3_t, align 16
// CHECK:   [[__S1:%.*]] = alloca %struct.poly64x2x3_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.poly64x2x3_t, %struct.poly64x2x3_t* [[B]], i32 0, i32 0
// CHECK:   store [3 x <2 x i64>] [[B]].coerce, [3 x <2 x i64>]* [[COERCE_DIVE]], align 16
// CHECK:   [[TMP0:%.*]] = bitcast %struct.poly64x2x3_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.poly64x2x3_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP0]], i8* align 16 [[TMP1]], i64 48, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast i64* %a to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.poly64x2x3_t, %struct.poly64x2x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <2 x i64>], [3 x <2 x i64>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <2 x i64>, <2 x i64>* [[ARRAYIDX]], align 16
// CHECK:   [[TMP4:%.*]] = bitcast <2 x i64> [[TMP3]] to <16 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.poly64x2x3_t, %struct.poly64x2x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <2 x i64>], [3 x <2 x i64>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <2 x i64>, <2 x i64>* [[ARRAYIDX2]], align 16
// CHECK:   [[TMP6:%.*]] = bitcast <2 x i64> [[TMP5]] to <16 x i8>
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds %struct.poly64x2x3_t, %struct.poly64x2x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <2 x i64>], [3 x <2 x i64>]* [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP7:%.*]] = load <2 x i64>, <2 x i64>* [[ARRAYIDX4]], align 16
// CHECK:   [[TMP8:%.*]] = bitcast <2 x i64> [[TMP7]] to <16 x i8>
// CHECK:   [[TMP9:%.*]] = bitcast <16 x i8> [[TMP4]] to <2 x i64>
// CHECK:   [[TMP10:%.*]] = bitcast <16 x i8> [[TMP6]] to <2 x i64>
// CHECK:   [[TMP11:%.*]] = bitcast <16 x i8> [[TMP8]] to <2 x i64>
// CHECK:   call void @llvm.aarch64.neon.st3lane.v2i64.p0i8(<2 x i64> [[TMP9]], <2 x i64> [[TMP10]], <2 x i64> [[TMP11]], i64 1, i8* [[TMP2]])
// CHECK:   ret void
void test_vst3q_lane_p64(poly64_t  *a, poly64x2x3_t b) {
  vst3q_lane_p64(a, b, 1);
}

// CHECK-LABEL: define{{.*}} void @test_vst3_lane_u8(i8* noundef %a, [3 x <8 x i8>] %b.coerce) #2 {
// CHECK:   [[B:%.*]] = alloca %struct.uint8x8x3_t, align 8
// CHECK:   [[__S1:%.*]] = alloca %struct.uint8x8x3_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.uint8x8x3_t, %struct.uint8x8x3_t* [[B]], i32 0, i32 0
// CHECK:   store [3 x <8 x i8>] [[B]].coerce, [3 x <8 x i8>]* [[COERCE_DIVE]], align 8
// CHECK:   [[TMP0:%.*]] = bitcast %struct.uint8x8x3_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.uint8x8x3_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[TMP0]], i8* align 8 [[TMP1]], i64 24, i1 false)
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.uint8x8x3_t, %struct.uint8x8x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <8 x i8>], [3 x <8 x i8>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP2:%.*]] = load <8 x i8>, <8 x i8>* [[ARRAYIDX]], align 8
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.uint8x8x3_t, %struct.uint8x8x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <8 x i8>], [3 x <8 x i8>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP3:%.*]] = load <8 x i8>, <8 x i8>* [[ARRAYIDX2]], align 8
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds %struct.uint8x8x3_t, %struct.uint8x8x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <8 x i8>], [3 x <8 x i8>]* [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP4:%.*]] = load <8 x i8>, <8 x i8>* [[ARRAYIDX4]], align 8
// CHECK:   call void @llvm.aarch64.neon.st3lane.v8i8.p0i8(<8 x i8> [[TMP2]], <8 x i8> [[TMP3]], <8 x i8> [[TMP4]], i64 7, i8* %a)
// CHECK:   ret void
void test_vst3_lane_u8(uint8_t  *a, uint8x8x3_t b) {
  vst3_lane_u8(a, b, 7);
}

// CHECK-LABEL: define{{.*}} void @test_vst3_lane_u16(i16* noundef %a, [3 x <4 x i16>] %b.coerce) #2 {
// CHECK:   [[B:%.*]] = alloca %struct.uint16x4x3_t, align 8
// CHECK:   [[__S1:%.*]] = alloca %struct.uint16x4x3_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.uint16x4x3_t, %struct.uint16x4x3_t* [[B]], i32 0, i32 0
// CHECK:   store [3 x <4 x i16>] [[B]].coerce, [3 x <4 x i16>]* [[COERCE_DIVE]], align 8
// CHECK:   [[TMP0:%.*]] = bitcast %struct.uint16x4x3_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.uint16x4x3_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[TMP0]], i8* align 8 [[TMP1]], i64 24, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast i16* %a to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.uint16x4x3_t, %struct.uint16x4x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <4 x i16>], [3 x <4 x i16>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <4 x i16>, <4 x i16>* [[ARRAYIDX]], align 8
// CHECK:   [[TMP4:%.*]] = bitcast <4 x i16> [[TMP3]] to <8 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.uint16x4x3_t, %struct.uint16x4x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <4 x i16>], [3 x <4 x i16>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <4 x i16>, <4 x i16>* [[ARRAYIDX2]], align 8
// CHECK:   [[TMP6:%.*]] = bitcast <4 x i16> [[TMP5]] to <8 x i8>
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds %struct.uint16x4x3_t, %struct.uint16x4x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <4 x i16>], [3 x <4 x i16>]* [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP7:%.*]] = load <4 x i16>, <4 x i16>* [[ARRAYIDX4]], align 8
// CHECK:   [[TMP8:%.*]] = bitcast <4 x i16> [[TMP7]] to <8 x i8>
// CHECK:   [[TMP9:%.*]] = bitcast <8 x i8> [[TMP4]] to <4 x i16>
// CHECK:   [[TMP10:%.*]] = bitcast <8 x i8> [[TMP6]] to <4 x i16>
// CHECK:   [[TMP11:%.*]] = bitcast <8 x i8> [[TMP8]] to <4 x i16>
// CHECK:   call void @llvm.aarch64.neon.st3lane.v4i16.p0i8(<4 x i16> [[TMP9]], <4 x i16> [[TMP10]], <4 x i16> [[TMP11]], i64 3, i8* [[TMP2]])
// CHECK:   ret void
void test_vst3_lane_u16(uint16_t  *a, uint16x4x3_t b) {
  vst3_lane_u16(a, b, 3);
}

// CHECK-LABEL: define{{.*}} void @test_vst3_lane_u32(i32* noundef %a, [3 x <2 x i32>] %b.coerce) #2 {
// CHECK:   [[B:%.*]] = alloca %struct.uint32x2x3_t, align 8
// CHECK:   [[__S1:%.*]] = alloca %struct.uint32x2x3_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.uint32x2x3_t, %struct.uint32x2x3_t* [[B]], i32 0, i32 0
// CHECK:   store [3 x <2 x i32>] [[B]].coerce, [3 x <2 x i32>]* [[COERCE_DIVE]], align 8
// CHECK:   [[TMP0:%.*]] = bitcast %struct.uint32x2x3_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.uint32x2x3_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[TMP0]], i8* align 8 [[TMP1]], i64 24, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast i32* %a to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.uint32x2x3_t, %struct.uint32x2x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <2 x i32>], [3 x <2 x i32>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <2 x i32>, <2 x i32>* [[ARRAYIDX]], align 8
// CHECK:   [[TMP4:%.*]] = bitcast <2 x i32> [[TMP3]] to <8 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.uint32x2x3_t, %struct.uint32x2x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <2 x i32>], [3 x <2 x i32>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <2 x i32>, <2 x i32>* [[ARRAYIDX2]], align 8
// CHECK:   [[TMP6:%.*]] = bitcast <2 x i32> [[TMP5]] to <8 x i8>
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds %struct.uint32x2x3_t, %struct.uint32x2x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <2 x i32>], [3 x <2 x i32>]* [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP7:%.*]] = load <2 x i32>, <2 x i32>* [[ARRAYIDX4]], align 8
// CHECK:   [[TMP8:%.*]] = bitcast <2 x i32> [[TMP7]] to <8 x i8>
// CHECK:   [[TMP9:%.*]] = bitcast <8 x i8> [[TMP4]] to <2 x i32>
// CHECK:   [[TMP10:%.*]] = bitcast <8 x i8> [[TMP6]] to <2 x i32>
// CHECK:   [[TMP11:%.*]] = bitcast <8 x i8> [[TMP8]] to <2 x i32>
// CHECK:   call void @llvm.aarch64.neon.st3lane.v2i32.p0i8(<2 x i32> [[TMP9]], <2 x i32> [[TMP10]], <2 x i32> [[TMP11]], i64 1, i8* [[TMP2]])
// CHECK:   ret void
void test_vst3_lane_u32(uint32_t  *a, uint32x2x3_t b) {
  vst3_lane_u32(a, b, 1);
}

// CHECK-LABEL: define{{.*}} void @test_vst3_lane_u64(i64* noundef %a, [3 x <1 x i64>] %b.coerce) #2 {
// CHECK:   [[B:%.*]] = alloca %struct.uint64x1x3_t, align 8
// CHECK:   [[__S1:%.*]] = alloca %struct.uint64x1x3_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.uint64x1x3_t, %struct.uint64x1x3_t* [[B]], i32 0, i32 0
// CHECK:   store [3 x <1 x i64>] [[B]].coerce, [3 x <1 x i64>]* [[COERCE_DIVE]], align 8
// CHECK:   [[TMP0:%.*]] = bitcast %struct.uint64x1x3_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.uint64x1x3_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[TMP0]], i8* align 8 [[TMP1]], i64 24, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast i64* %a to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.uint64x1x3_t, %struct.uint64x1x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <1 x i64>], [3 x <1 x i64>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <1 x i64>, <1 x i64>* [[ARRAYIDX]], align 8
// CHECK:   [[TMP4:%.*]] = bitcast <1 x i64> [[TMP3]] to <8 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.uint64x1x3_t, %struct.uint64x1x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <1 x i64>], [3 x <1 x i64>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <1 x i64>, <1 x i64>* [[ARRAYIDX2]], align 8
// CHECK:   [[TMP6:%.*]] = bitcast <1 x i64> [[TMP5]] to <8 x i8>
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds %struct.uint64x1x3_t, %struct.uint64x1x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <1 x i64>], [3 x <1 x i64>]* [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP7:%.*]] = load <1 x i64>, <1 x i64>* [[ARRAYIDX4]], align 8
// CHECK:   [[TMP8:%.*]] = bitcast <1 x i64> [[TMP7]] to <8 x i8>
// CHECK:   [[TMP9:%.*]] = bitcast <8 x i8> [[TMP4]] to <1 x i64>
// CHECK:   [[TMP10:%.*]] = bitcast <8 x i8> [[TMP6]] to <1 x i64>
// CHECK:   [[TMP11:%.*]] = bitcast <8 x i8> [[TMP8]] to <1 x i64>
// CHECK:   call void @llvm.aarch64.neon.st3lane.v1i64.p0i8(<1 x i64> [[TMP9]], <1 x i64> [[TMP10]], <1 x i64> [[TMP11]], i64 0, i8* [[TMP2]])
// CHECK:   ret void
void test_vst3_lane_u64(uint64_t  *a, uint64x1x3_t b) {
  vst3_lane_u64(a, b, 0);
}

// CHECK-LABEL: define{{.*}} void @test_vst3_lane_s8(i8* noundef %a, [3 x <8 x i8>] %b.coerce) #2 {
// CHECK:   [[B:%.*]] = alloca %struct.int8x8x3_t, align 8
// CHECK:   [[__S1:%.*]] = alloca %struct.int8x8x3_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.int8x8x3_t, %struct.int8x8x3_t* [[B]], i32 0, i32 0
// CHECK:   store [3 x <8 x i8>] [[B]].coerce, [3 x <8 x i8>]* [[COERCE_DIVE]], align 8
// CHECK:   [[TMP0:%.*]] = bitcast %struct.int8x8x3_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.int8x8x3_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[TMP0]], i8* align 8 [[TMP1]], i64 24, i1 false)
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.int8x8x3_t, %struct.int8x8x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <8 x i8>], [3 x <8 x i8>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP2:%.*]] = load <8 x i8>, <8 x i8>* [[ARRAYIDX]], align 8
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.int8x8x3_t, %struct.int8x8x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <8 x i8>], [3 x <8 x i8>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP3:%.*]] = load <8 x i8>, <8 x i8>* [[ARRAYIDX2]], align 8
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds %struct.int8x8x3_t, %struct.int8x8x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <8 x i8>], [3 x <8 x i8>]* [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP4:%.*]] = load <8 x i8>, <8 x i8>* [[ARRAYIDX4]], align 8
// CHECK:   call void @llvm.aarch64.neon.st3lane.v8i8.p0i8(<8 x i8> [[TMP2]], <8 x i8> [[TMP3]], <8 x i8> [[TMP4]], i64 7, i8* %a)
// CHECK:   ret void
void test_vst3_lane_s8(int8_t  *a, int8x8x3_t b) {
  vst3_lane_s8(a, b, 7);
}

// CHECK-LABEL: define{{.*}} void @test_vst3_lane_s16(i16* noundef %a, [3 x <4 x i16>] %b.coerce) #2 {
// CHECK:   [[B:%.*]] = alloca %struct.int16x4x3_t, align 8
// CHECK:   [[__S1:%.*]] = alloca %struct.int16x4x3_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.int16x4x3_t, %struct.int16x4x3_t* [[B]], i32 0, i32 0
// CHECK:   store [3 x <4 x i16>] [[B]].coerce, [3 x <4 x i16>]* [[COERCE_DIVE]], align 8
// CHECK:   [[TMP0:%.*]] = bitcast %struct.int16x4x3_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.int16x4x3_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[TMP0]], i8* align 8 [[TMP1]], i64 24, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast i16* %a to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.int16x4x3_t, %struct.int16x4x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <4 x i16>], [3 x <4 x i16>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <4 x i16>, <4 x i16>* [[ARRAYIDX]], align 8
// CHECK:   [[TMP4:%.*]] = bitcast <4 x i16> [[TMP3]] to <8 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.int16x4x3_t, %struct.int16x4x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <4 x i16>], [3 x <4 x i16>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <4 x i16>, <4 x i16>* [[ARRAYIDX2]], align 8
// CHECK:   [[TMP6:%.*]] = bitcast <4 x i16> [[TMP5]] to <8 x i8>
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds %struct.int16x4x3_t, %struct.int16x4x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <4 x i16>], [3 x <4 x i16>]* [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP7:%.*]] = load <4 x i16>, <4 x i16>* [[ARRAYIDX4]], align 8
// CHECK:   [[TMP8:%.*]] = bitcast <4 x i16> [[TMP7]] to <8 x i8>
// CHECK:   [[TMP9:%.*]] = bitcast <8 x i8> [[TMP4]] to <4 x i16>
// CHECK:   [[TMP10:%.*]] = bitcast <8 x i8> [[TMP6]] to <4 x i16>
// CHECK:   [[TMP11:%.*]] = bitcast <8 x i8> [[TMP8]] to <4 x i16>
// CHECK:   call void @llvm.aarch64.neon.st3lane.v4i16.p0i8(<4 x i16> [[TMP9]], <4 x i16> [[TMP10]], <4 x i16> [[TMP11]], i64 3, i8* [[TMP2]])
// CHECK:   ret void
void test_vst3_lane_s16(int16_t  *a, int16x4x3_t b) {
  vst3_lane_s16(a, b, 3);
}

// CHECK-LABEL: define{{.*}} void @test_vst3_lane_s32(i32* noundef %a, [3 x <2 x i32>] %b.coerce) #2 {
// CHECK:   [[B:%.*]] = alloca %struct.int32x2x3_t, align 8
// CHECK:   [[__S1:%.*]] = alloca %struct.int32x2x3_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.int32x2x3_t, %struct.int32x2x3_t* [[B]], i32 0, i32 0
// CHECK:   store [3 x <2 x i32>] [[B]].coerce, [3 x <2 x i32>]* [[COERCE_DIVE]], align 8
// CHECK:   [[TMP0:%.*]] = bitcast %struct.int32x2x3_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.int32x2x3_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[TMP0]], i8* align 8 [[TMP1]], i64 24, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast i32* %a to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.int32x2x3_t, %struct.int32x2x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <2 x i32>], [3 x <2 x i32>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <2 x i32>, <2 x i32>* [[ARRAYIDX]], align 8
// CHECK:   [[TMP4:%.*]] = bitcast <2 x i32> [[TMP3]] to <8 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.int32x2x3_t, %struct.int32x2x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <2 x i32>], [3 x <2 x i32>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <2 x i32>, <2 x i32>* [[ARRAYIDX2]], align 8
// CHECK:   [[TMP6:%.*]] = bitcast <2 x i32> [[TMP5]] to <8 x i8>
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds %struct.int32x2x3_t, %struct.int32x2x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <2 x i32>], [3 x <2 x i32>]* [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP7:%.*]] = load <2 x i32>, <2 x i32>* [[ARRAYIDX4]], align 8
// CHECK:   [[TMP8:%.*]] = bitcast <2 x i32> [[TMP7]] to <8 x i8>
// CHECK:   [[TMP9:%.*]] = bitcast <8 x i8> [[TMP4]] to <2 x i32>
// CHECK:   [[TMP10:%.*]] = bitcast <8 x i8> [[TMP6]] to <2 x i32>
// CHECK:   [[TMP11:%.*]] = bitcast <8 x i8> [[TMP8]] to <2 x i32>
// CHECK:   call void @llvm.aarch64.neon.st3lane.v2i32.p0i8(<2 x i32> [[TMP9]], <2 x i32> [[TMP10]], <2 x i32> [[TMP11]], i64 1, i8* [[TMP2]])
// CHECK:   ret void
void test_vst3_lane_s32(int32_t  *a, int32x2x3_t b) {
  vst3_lane_s32(a, b, 1);
}

// CHECK-LABEL: define{{.*}} void @test_vst3_lane_s64(i64* noundef %a, [3 x <1 x i64>] %b.coerce) #2 {
// CHECK:   [[B:%.*]] = alloca %struct.int64x1x3_t, align 8
// CHECK:   [[__S1:%.*]] = alloca %struct.int64x1x3_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.int64x1x3_t, %struct.int64x1x3_t* [[B]], i32 0, i32 0
// CHECK:   store [3 x <1 x i64>] [[B]].coerce, [3 x <1 x i64>]* [[COERCE_DIVE]], align 8
// CHECK:   [[TMP0:%.*]] = bitcast %struct.int64x1x3_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.int64x1x3_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[TMP0]], i8* align 8 [[TMP1]], i64 24, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast i64* %a to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.int64x1x3_t, %struct.int64x1x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <1 x i64>], [3 x <1 x i64>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <1 x i64>, <1 x i64>* [[ARRAYIDX]], align 8
// CHECK:   [[TMP4:%.*]] = bitcast <1 x i64> [[TMP3]] to <8 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.int64x1x3_t, %struct.int64x1x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <1 x i64>], [3 x <1 x i64>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <1 x i64>, <1 x i64>* [[ARRAYIDX2]], align 8
// CHECK:   [[TMP6:%.*]] = bitcast <1 x i64> [[TMP5]] to <8 x i8>
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds %struct.int64x1x3_t, %struct.int64x1x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <1 x i64>], [3 x <1 x i64>]* [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP7:%.*]] = load <1 x i64>, <1 x i64>* [[ARRAYIDX4]], align 8
// CHECK:   [[TMP8:%.*]] = bitcast <1 x i64> [[TMP7]] to <8 x i8>
// CHECK:   [[TMP9:%.*]] = bitcast <8 x i8> [[TMP4]] to <1 x i64>
// CHECK:   [[TMP10:%.*]] = bitcast <8 x i8> [[TMP6]] to <1 x i64>
// CHECK:   [[TMP11:%.*]] = bitcast <8 x i8> [[TMP8]] to <1 x i64>
// CHECK:   call void @llvm.aarch64.neon.st3lane.v1i64.p0i8(<1 x i64> [[TMP9]], <1 x i64> [[TMP10]], <1 x i64> [[TMP11]], i64 0, i8* [[TMP2]])
// CHECK:   ret void
void test_vst3_lane_s64(int64_t  *a, int64x1x3_t b) {
  vst3_lane_s64(a, b, 0);
}

// CHECK-LABEL: define{{.*}} void @test_vst3_lane_f16(half* noundef %a, [3 x <4 x half>] %b.coerce) #2 {
// CHECK:   [[B:%.*]] = alloca %struct.float16x4x3_t, align 8
// CHECK:   [[__S1:%.*]] = alloca %struct.float16x4x3_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.float16x4x3_t, %struct.float16x4x3_t* [[B]], i32 0, i32 0
// CHECK:   store [3 x <4 x half>] [[B]].coerce, [3 x <4 x half>]* [[COERCE_DIVE]], align 8
// CHECK:   [[TMP0:%.*]] = bitcast %struct.float16x4x3_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.float16x4x3_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[TMP0]], i8* align 8 [[TMP1]], i64 24, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast half* %a to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.float16x4x3_t, %struct.float16x4x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <4 x half>], [3 x <4 x half>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <4 x half>, <4 x half>* [[ARRAYIDX]], align 8
// CHECK:   [[TMP4:%.*]] = bitcast <4 x half> [[TMP3]] to <8 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.float16x4x3_t, %struct.float16x4x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <4 x half>], [3 x <4 x half>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <4 x half>, <4 x half>* [[ARRAYIDX2]], align 8
// CHECK:   [[TMP6:%.*]] = bitcast <4 x half> [[TMP5]] to <8 x i8>
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds %struct.float16x4x3_t, %struct.float16x4x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <4 x half>], [3 x <4 x half>]* [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP7:%.*]] = load <4 x half>, <4 x half>* [[ARRAYIDX4]], align 8
// CHECK:   [[TMP8:%.*]] = bitcast <4 x half> [[TMP7]] to <8 x i8>
// CHECK:   [[TMP9:%.*]] = bitcast <8 x i8> [[TMP4]] to <4 x half>
// CHECK:   [[TMP10:%.*]] = bitcast <8 x i8> [[TMP6]] to <4 x half>
// CHECK:   [[TMP11:%.*]] = bitcast <8 x i8> [[TMP8]] to <4 x half>
// CHECK:   call void @llvm.aarch64.neon.st3lane.v4f16.p0i8(<4 x half> [[TMP9]], <4 x half> [[TMP10]], <4 x half> [[TMP11]], i64 3, i8* [[TMP2]])
// CHECK:   ret void
void test_vst3_lane_f16(float16_t  *a, float16x4x3_t b) {
  vst3_lane_f16(a, b, 3);
}

// CHECK-LABEL: define{{.*}} void @test_vst3_lane_f32(float* noundef %a, [3 x <2 x float>] %b.coerce) #2 {
// CHECK:   [[B:%.*]] = alloca %struct.float32x2x3_t, align 8
// CHECK:   [[__S1:%.*]] = alloca %struct.float32x2x3_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.float32x2x3_t, %struct.float32x2x3_t* [[B]], i32 0, i32 0
// CHECK:   store [3 x <2 x float>] [[B]].coerce, [3 x <2 x float>]* [[COERCE_DIVE]], align 8
// CHECK:   [[TMP0:%.*]] = bitcast %struct.float32x2x3_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.float32x2x3_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[TMP0]], i8* align 8 [[TMP1]], i64 24, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast float* %a to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.float32x2x3_t, %struct.float32x2x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <2 x float>], [3 x <2 x float>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <2 x float>, <2 x float>* [[ARRAYIDX]], align 8
// CHECK:   [[TMP4:%.*]] = bitcast <2 x float> [[TMP3]] to <8 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.float32x2x3_t, %struct.float32x2x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <2 x float>], [3 x <2 x float>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <2 x float>, <2 x float>* [[ARRAYIDX2]], align 8
// CHECK:   [[TMP6:%.*]] = bitcast <2 x float> [[TMP5]] to <8 x i8>
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds %struct.float32x2x3_t, %struct.float32x2x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <2 x float>], [3 x <2 x float>]* [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP7:%.*]] = load <2 x float>, <2 x float>* [[ARRAYIDX4]], align 8
// CHECK:   [[TMP8:%.*]] = bitcast <2 x float> [[TMP7]] to <8 x i8>
// CHECK:   [[TMP9:%.*]] = bitcast <8 x i8> [[TMP4]] to <2 x float>
// CHECK:   [[TMP10:%.*]] = bitcast <8 x i8> [[TMP6]] to <2 x float>
// CHECK:   [[TMP11:%.*]] = bitcast <8 x i8> [[TMP8]] to <2 x float>
// CHECK:   call void @llvm.aarch64.neon.st3lane.v2f32.p0i8(<2 x float> [[TMP9]], <2 x float> [[TMP10]], <2 x float> [[TMP11]], i64 1, i8* [[TMP2]])
// CHECK:   ret void
void test_vst3_lane_f32(float32_t  *a, float32x2x3_t b) {
  vst3_lane_f32(a, b, 1);
}

// CHECK-LABEL: define{{.*}} void @test_vst3_lane_f64(double* noundef %a, [3 x <1 x double>] %b.coerce) #2 {
// CHECK:   [[B:%.*]] = alloca %struct.float64x1x3_t, align 8
// CHECK:   [[__S1:%.*]] = alloca %struct.float64x1x3_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.float64x1x3_t, %struct.float64x1x3_t* [[B]], i32 0, i32 0
// CHECK:   store [3 x <1 x double>] [[B]].coerce, [3 x <1 x double>]* [[COERCE_DIVE]], align 8
// CHECK:   [[TMP0:%.*]] = bitcast %struct.float64x1x3_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.float64x1x3_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[TMP0]], i8* align 8 [[TMP1]], i64 24, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast double* %a to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.float64x1x3_t, %struct.float64x1x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <1 x double>], [3 x <1 x double>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <1 x double>, <1 x double>* [[ARRAYIDX]], align 8
// CHECK:   [[TMP4:%.*]] = bitcast <1 x double> [[TMP3]] to <8 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.float64x1x3_t, %struct.float64x1x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <1 x double>], [3 x <1 x double>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <1 x double>, <1 x double>* [[ARRAYIDX2]], align 8
// CHECK:   [[TMP6:%.*]] = bitcast <1 x double> [[TMP5]] to <8 x i8>
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds %struct.float64x1x3_t, %struct.float64x1x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <1 x double>], [3 x <1 x double>]* [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP7:%.*]] = load <1 x double>, <1 x double>* [[ARRAYIDX4]], align 8
// CHECK:   [[TMP8:%.*]] = bitcast <1 x double> [[TMP7]] to <8 x i8>
// CHECK:   [[TMP9:%.*]] = bitcast <8 x i8> [[TMP4]] to <1 x double>
// CHECK:   [[TMP10:%.*]] = bitcast <8 x i8> [[TMP6]] to <1 x double>
// CHECK:   [[TMP11:%.*]] = bitcast <8 x i8> [[TMP8]] to <1 x double>
// CHECK:   call void @llvm.aarch64.neon.st3lane.v1f64.p0i8(<1 x double> [[TMP9]], <1 x double> [[TMP10]], <1 x double> [[TMP11]], i64 0, i8* [[TMP2]])
// CHECK:   ret void
void test_vst3_lane_f64(float64_t  *a, float64x1x3_t b) {
  vst3_lane_f64(a, b, 0);
}

// CHECK-LABEL: define{{.*}} void @test_vst3_lane_p8(i8* noundef %a, [3 x <8 x i8>] %b.coerce) #2 {
// CHECK:   [[B:%.*]] = alloca %struct.poly8x8x3_t, align 8
// CHECK:   [[__S1:%.*]] = alloca %struct.poly8x8x3_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.poly8x8x3_t, %struct.poly8x8x3_t* [[B]], i32 0, i32 0
// CHECK:   store [3 x <8 x i8>] [[B]].coerce, [3 x <8 x i8>]* [[COERCE_DIVE]], align 8
// CHECK:   [[TMP0:%.*]] = bitcast %struct.poly8x8x3_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.poly8x8x3_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[TMP0]], i8* align 8 [[TMP1]], i64 24, i1 false)
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.poly8x8x3_t, %struct.poly8x8x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <8 x i8>], [3 x <8 x i8>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP2:%.*]] = load <8 x i8>, <8 x i8>* [[ARRAYIDX]], align 8
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.poly8x8x3_t, %struct.poly8x8x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <8 x i8>], [3 x <8 x i8>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP3:%.*]] = load <8 x i8>, <8 x i8>* [[ARRAYIDX2]], align 8
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds %struct.poly8x8x3_t, %struct.poly8x8x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <8 x i8>], [3 x <8 x i8>]* [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP4:%.*]] = load <8 x i8>, <8 x i8>* [[ARRAYIDX4]], align 8
// CHECK:   call void @llvm.aarch64.neon.st3lane.v8i8.p0i8(<8 x i8> [[TMP2]], <8 x i8> [[TMP3]], <8 x i8> [[TMP4]], i64 7, i8* %a)
// CHECK:   ret void
void test_vst3_lane_p8(poly8_t  *a, poly8x8x3_t b) {
  vst3_lane_p8(a, b, 7);
}

// CHECK-LABEL: define{{.*}} void @test_vst3_lane_p16(i16* noundef %a, [3 x <4 x i16>] %b.coerce) #2 {
// CHECK:   [[B:%.*]] = alloca %struct.poly16x4x3_t, align 8
// CHECK:   [[__S1:%.*]] = alloca %struct.poly16x4x3_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.poly16x4x3_t, %struct.poly16x4x3_t* [[B]], i32 0, i32 0
// CHECK:   store [3 x <4 x i16>] [[B]].coerce, [3 x <4 x i16>]* [[COERCE_DIVE]], align 8
// CHECK:   [[TMP0:%.*]] = bitcast %struct.poly16x4x3_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.poly16x4x3_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[TMP0]], i8* align 8 [[TMP1]], i64 24, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast i16* %a to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.poly16x4x3_t, %struct.poly16x4x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <4 x i16>], [3 x <4 x i16>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <4 x i16>, <4 x i16>* [[ARRAYIDX]], align 8
// CHECK:   [[TMP4:%.*]] = bitcast <4 x i16> [[TMP3]] to <8 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.poly16x4x3_t, %struct.poly16x4x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <4 x i16>], [3 x <4 x i16>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <4 x i16>, <4 x i16>* [[ARRAYIDX2]], align 8
// CHECK:   [[TMP6:%.*]] = bitcast <4 x i16> [[TMP5]] to <8 x i8>
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds %struct.poly16x4x3_t, %struct.poly16x4x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <4 x i16>], [3 x <4 x i16>]* [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP7:%.*]] = load <4 x i16>, <4 x i16>* [[ARRAYIDX4]], align 8
// CHECK:   [[TMP8:%.*]] = bitcast <4 x i16> [[TMP7]] to <8 x i8>
// CHECK:   [[TMP9:%.*]] = bitcast <8 x i8> [[TMP4]] to <4 x i16>
// CHECK:   [[TMP10:%.*]] = bitcast <8 x i8> [[TMP6]] to <4 x i16>
// CHECK:   [[TMP11:%.*]] = bitcast <8 x i8> [[TMP8]] to <4 x i16>
// CHECK:   call void @llvm.aarch64.neon.st3lane.v4i16.p0i8(<4 x i16> [[TMP9]], <4 x i16> [[TMP10]], <4 x i16> [[TMP11]], i64 3, i8* [[TMP2]])
// CHECK:   ret void
void test_vst3_lane_p16(poly16_t  *a, poly16x4x3_t b) {
  vst3_lane_p16(a, b, 3);
}

// CHECK-LABEL: define{{.*}} void @test_vst3_lane_p64(i64* noundef %a, [3 x <1 x i64>] %b.coerce) #2 {
// CHECK:   [[B:%.*]] = alloca %struct.poly64x1x3_t, align 8
// CHECK:   [[__S1:%.*]] = alloca %struct.poly64x1x3_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.poly64x1x3_t, %struct.poly64x1x3_t* [[B]], i32 0, i32 0
// CHECK:   store [3 x <1 x i64>] [[B]].coerce, [3 x <1 x i64>]* [[COERCE_DIVE]], align 8
// CHECK:   [[TMP0:%.*]] = bitcast %struct.poly64x1x3_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.poly64x1x3_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[TMP0]], i8* align 8 [[TMP1]], i64 24, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast i64* %a to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.poly64x1x3_t, %struct.poly64x1x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <1 x i64>], [3 x <1 x i64>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <1 x i64>, <1 x i64>* [[ARRAYIDX]], align 8
// CHECK:   [[TMP4:%.*]] = bitcast <1 x i64> [[TMP3]] to <8 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.poly64x1x3_t, %struct.poly64x1x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <1 x i64>], [3 x <1 x i64>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <1 x i64>, <1 x i64>* [[ARRAYIDX2]], align 8
// CHECK:   [[TMP6:%.*]] = bitcast <1 x i64> [[TMP5]] to <8 x i8>
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds %struct.poly64x1x3_t, %struct.poly64x1x3_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <1 x i64>], [3 x <1 x i64>]* [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP7:%.*]] = load <1 x i64>, <1 x i64>* [[ARRAYIDX4]], align 8
// CHECK:   [[TMP8:%.*]] = bitcast <1 x i64> [[TMP7]] to <8 x i8>
// CHECK:   [[TMP9:%.*]] = bitcast <8 x i8> [[TMP4]] to <1 x i64>
// CHECK:   [[TMP10:%.*]] = bitcast <8 x i8> [[TMP6]] to <1 x i64>
// CHECK:   [[TMP11:%.*]] = bitcast <8 x i8> [[TMP8]] to <1 x i64>
// CHECK:   call void @llvm.aarch64.neon.st3lane.v1i64.p0i8(<1 x i64> [[TMP9]], <1 x i64> [[TMP10]], <1 x i64> [[TMP11]], i64 0, i8* [[TMP2]])
// CHECK:   ret void
void test_vst3_lane_p64(poly64_t  *a, poly64x1x3_t b) {
  vst3_lane_p64(a, b, 0);
}

// CHECK-LABEL: define{{.*}} void @test_vst4q_lane_u8(i8* noundef %a, [4 x <16 x i8>] %b.coerce) #2 {
// CHECK:   [[B:%.*]] = alloca %struct.uint8x16x4_t, align 16
// CHECK:   [[__S1:%.*]] = alloca %struct.uint8x16x4_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.uint8x16x4_t, %struct.uint8x16x4_t* [[B]], i32 0, i32 0
// CHECK:   store [4 x <16 x i8>] [[B]].coerce, [4 x <16 x i8>]* [[COERCE_DIVE]], align 16
// CHECK:   [[TMP0:%.*]] = bitcast %struct.uint8x16x4_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.uint8x16x4_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP0]], i8* align 16 [[TMP1]], i64 64, i1 false)
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.uint8x16x4_t, %struct.uint8x16x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <16 x i8>], [4 x <16 x i8>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP2:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX]], align 16
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.uint8x16x4_t, %struct.uint8x16x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <16 x i8>], [4 x <16 x i8>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP3:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX2]], align 16
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds %struct.uint8x16x4_t, %struct.uint8x16x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <16 x i8>], [4 x <16 x i8>]* [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP4:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX4]], align 16
// CHECK:   [[VAL5:%.*]] = getelementptr inbounds %struct.uint8x16x4_t, %struct.uint8x16x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <16 x i8>], [4 x <16 x i8>]* [[VAL5]], i64 0, i64 3
// CHECK:   [[TMP5:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX6]], align 16
// CHECK:   call void @llvm.aarch64.neon.st4lane.v16i8.p0i8(<16 x i8> [[TMP2]], <16 x i8> [[TMP3]], <16 x i8> [[TMP4]], <16 x i8> [[TMP5]], i64 15, i8* %a)
// CHECK:   ret void
void test_vst4q_lane_u8(uint8_t  *a, uint8x16x4_t b) {
  vst4q_lane_u8(a, b, 15);
}

// CHECK-LABEL: define{{.*}} void @test_vst4q_lane_u16(i16* noundef %a, [4 x <8 x i16>] %b.coerce) #2 {
// CHECK:   [[B:%.*]] = alloca %struct.uint16x8x4_t, align 16
// CHECK:   [[__S1:%.*]] = alloca %struct.uint16x8x4_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.uint16x8x4_t, %struct.uint16x8x4_t* [[B]], i32 0, i32 0
// CHECK:   store [4 x <8 x i16>] [[B]].coerce, [4 x <8 x i16>]* [[COERCE_DIVE]], align 16
// CHECK:   [[TMP0:%.*]] = bitcast %struct.uint16x8x4_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.uint16x8x4_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP0]], i8* align 16 [[TMP1]], i64 64, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast i16* %a to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.uint16x8x4_t, %struct.uint16x8x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <8 x i16>], [4 x <8 x i16>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <8 x i16>, <8 x i16>* [[ARRAYIDX]], align 16
// CHECK:   [[TMP4:%.*]] = bitcast <8 x i16> [[TMP3]] to <16 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.uint16x8x4_t, %struct.uint16x8x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <8 x i16>], [4 x <8 x i16>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <8 x i16>, <8 x i16>* [[ARRAYIDX2]], align 16
// CHECK:   [[TMP6:%.*]] = bitcast <8 x i16> [[TMP5]] to <16 x i8>
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds %struct.uint16x8x4_t, %struct.uint16x8x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <8 x i16>], [4 x <8 x i16>]* [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP7:%.*]] = load <8 x i16>, <8 x i16>* [[ARRAYIDX4]], align 16
// CHECK:   [[TMP8:%.*]] = bitcast <8 x i16> [[TMP7]] to <16 x i8>
// CHECK:   [[VAL5:%.*]] = getelementptr inbounds %struct.uint16x8x4_t, %struct.uint16x8x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <8 x i16>], [4 x <8 x i16>]* [[VAL5]], i64 0, i64 3
// CHECK:   [[TMP9:%.*]] = load <8 x i16>, <8 x i16>* [[ARRAYIDX6]], align 16
// CHECK:   [[TMP10:%.*]] = bitcast <8 x i16> [[TMP9]] to <16 x i8>
// CHECK:   [[TMP11:%.*]] = bitcast <16 x i8> [[TMP4]] to <8 x i16>
// CHECK:   [[TMP12:%.*]] = bitcast <16 x i8> [[TMP6]] to <8 x i16>
// CHECK:   [[TMP13:%.*]] = bitcast <16 x i8> [[TMP8]] to <8 x i16>
// CHECK:   [[TMP14:%.*]] = bitcast <16 x i8> [[TMP10]] to <8 x i16>
// CHECK:   call void @llvm.aarch64.neon.st4lane.v8i16.p0i8(<8 x i16> [[TMP11]], <8 x i16> [[TMP12]], <8 x i16> [[TMP13]], <8 x i16> [[TMP14]], i64 7, i8* [[TMP2]])
// CHECK:   ret void
void test_vst4q_lane_u16(uint16_t  *a, uint16x8x4_t b) {
  vst4q_lane_u16(a, b, 7);
}

// CHECK-LABEL: define{{.*}} void @test_vst4q_lane_u32(i32* noundef %a, [4 x <4 x i32>] %b.coerce) #2 {
// CHECK:   [[B:%.*]] = alloca %struct.uint32x4x4_t, align 16
// CHECK:   [[__S1:%.*]] = alloca %struct.uint32x4x4_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.uint32x4x4_t, %struct.uint32x4x4_t* [[B]], i32 0, i32 0
// CHECK:   store [4 x <4 x i32>] [[B]].coerce, [4 x <4 x i32>]* [[COERCE_DIVE]], align 16
// CHECK:   [[TMP0:%.*]] = bitcast %struct.uint32x4x4_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.uint32x4x4_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP0]], i8* align 16 [[TMP1]], i64 64, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast i32* %a to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.uint32x4x4_t, %struct.uint32x4x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <4 x i32>], [4 x <4 x i32>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <4 x i32>, <4 x i32>* [[ARRAYIDX]], align 16
// CHECK:   [[TMP4:%.*]] = bitcast <4 x i32> [[TMP3]] to <16 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.uint32x4x4_t, %struct.uint32x4x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <4 x i32>], [4 x <4 x i32>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <4 x i32>, <4 x i32>* [[ARRAYIDX2]], align 16
// CHECK:   [[TMP6:%.*]] = bitcast <4 x i32> [[TMP5]] to <16 x i8>
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds %struct.uint32x4x4_t, %struct.uint32x4x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <4 x i32>], [4 x <4 x i32>]* [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP7:%.*]] = load <4 x i32>, <4 x i32>* [[ARRAYIDX4]], align 16
// CHECK:   [[TMP8:%.*]] = bitcast <4 x i32> [[TMP7]] to <16 x i8>
// CHECK:   [[VAL5:%.*]] = getelementptr inbounds %struct.uint32x4x4_t, %struct.uint32x4x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <4 x i32>], [4 x <4 x i32>]* [[VAL5]], i64 0, i64 3
// CHECK:   [[TMP9:%.*]] = load <4 x i32>, <4 x i32>* [[ARRAYIDX6]], align 16
// CHECK:   [[TMP10:%.*]] = bitcast <4 x i32> [[TMP9]] to <16 x i8>
// CHECK:   [[TMP11:%.*]] = bitcast <16 x i8> [[TMP4]] to <4 x i32>
// CHECK:   [[TMP12:%.*]] = bitcast <16 x i8> [[TMP6]] to <4 x i32>
// CHECK:   [[TMP13:%.*]] = bitcast <16 x i8> [[TMP8]] to <4 x i32>
// CHECK:   [[TMP14:%.*]] = bitcast <16 x i8> [[TMP10]] to <4 x i32>
// CHECK:   call void @llvm.aarch64.neon.st4lane.v4i32.p0i8(<4 x i32> [[TMP11]], <4 x i32> [[TMP12]], <4 x i32> [[TMP13]], <4 x i32> [[TMP14]], i64 3, i8* [[TMP2]])
// CHECK:   ret void
void test_vst4q_lane_u32(uint32_t  *a, uint32x4x4_t b) {
  vst4q_lane_u32(a, b, 3);
}

// CHECK-LABEL: define{{.*}} void @test_vst4q_lane_u64(i64* noundef %a, [4 x <2 x i64>] %b.coerce) #2 {
// CHECK:   [[B:%.*]] = alloca %struct.uint64x2x4_t, align 16
// CHECK:   [[__S1:%.*]] = alloca %struct.uint64x2x4_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.uint64x2x4_t, %struct.uint64x2x4_t* [[B]], i32 0, i32 0
// CHECK:   store [4 x <2 x i64>] [[B]].coerce, [4 x <2 x i64>]* [[COERCE_DIVE]], align 16
// CHECK:   [[TMP0:%.*]] = bitcast %struct.uint64x2x4_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.uint64x2x4_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP0]], i8* align 16 [[TMP1]], i64 64, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast i64* %a to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.uint64x2x4_t, %struct.uint64x2x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <2 x i64>], [4 x <2 x i64>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <2 x i64>, <2 x i64>* [[ARRAYIDX]], align 16
// CHECK:   [[TMP4:%.*]] = bitcast <2 x i64> [[TMP3]] to <16 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.uint64x2x4_t, %struct.uint64x2x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <2 x i64>], [4 x <2 x i64>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <2 x i64>, <2 x i64>* [[ARRAYIDX2]], align 16
// CHECK:   [[TMP6:%.*]] = bitcast <2 x i64> [[TMP5]] to <16 x i8>
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds %struct.uint64x2x4_t, %struct.uint64x2x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <2 x i64>], [4 x <2 x i64>]* [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP7:%.*]] = load <2 x i64>, <2 x i64>* [[ARRAYIDX4]], align 16
// CHECK:   [[TMP8:%.*]] = bitcast <2 x i64> [[TMP7]] to <16 x i8>
// CHECK:   [[VAL5:%.*]] = getelementptr inbounds %struct.uint64x2x4_t, %struct.uint64x2x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <2 x i64>], [4 x <2 x i64>]* [[VAL5]], i64 0, i64 3
// CHECK:   [[TMP9:%.*]] = load <2 x i64>, <2 x i64>* [[ARRAYIDX6]], align 16
// CHECK:   [[TMP10:%.*]] = bitcast <2 x i64> [[TMP9]] to <16 x i8>
// CHECK:   [[TMP11:%.*]] = bitcast <16 x i8> [[TMP4]] to <2 x i64>
// CHECK:   [[TMP12:%.*]] = bitcast <16 x i8> [[TMP6]] to <2 x i64>
// CHECK:   [[TMP13:%.*]] = bitcast <16 x i8> [[TMP8]] to <2 x i64>
// CHECK:   [[TMP14:%.*]] = bitcast <16 x i8> [[TMP10]] to <2 x i64>
// CHECK:   call void @llvm.aarch64.neon.st4lane.v2i64.p0i8(<2 x i64> [[TMP11]], <2 x i64> [[TMP12]], <2 x i64> [[TMP13]], <2 x i64> [[TMP14]], i64 1, i8* [[TMP2]])
// CHECK:   ret void
void test_vst4q_lane_u64(uint64_t  *a, uint64x2x4_t b) {
  vst4q_lane_u64(a, b, 1);
}

// CHECK-LABEL: define{{.*}} void @test_vst4q_lane_s8(i8* noundef %a, [4 x <16 x i8>] %b.coerce) #2 {
// CHECK:   [[B:%.*]] = alloca %struct.int8x16x4_t, align 16
// CHECK:   [[__S1:%.*]] = alloca %struct.int8x16x4_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.int8x16x4_t, %struct.int8x16x4_t* [[B]], i32 0, i32 0
// CHECK:   store [4 x <16 x i8>] [[B]].coerce, [4 x <16 x i8>]* [[COERCE_DIVE]], align 16
// CHECK:   [[TMP0:%.*]] = bitcast %struct.int8x16x4_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.int8x16x4_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP0]], i8* align 16 [[TMP1]], i64 64, i1 false)
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.int8x16x4_t, %struct.int8x16x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <16 x i8>], [4 x <16 x i8>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP2:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX]], align 16
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.int8x16x4_t, %struct.int8x16x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <16 x i8>], [4 x <16 x i8>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP3:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX2]], align 16
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds %struct.int8x16x4_t, %struct.int8x16x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <16 x i8>], [4 x <16 x i8>]* [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP4:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX4]], align 16
// CHECK:   [[VAL5:%.*]] = getelementptr inbounds %struct.int8x16x4_t, %struct.int8x16x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <16 x i8>], [4 x <16 x i8>]* [[VAL5]], i64 0, i64 3
// CHECK:   [[TMP5:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX6]], align 16
// CHECK:   call void @llvm.aarch64.neon.st4lane.v16i8.p0i8(<16 x i8> [[TMP2]], <16 x i8> [[TMP3]], <16 x i8> [[TMP4]], <16 x i8> [[TMP5]], i64 15, i8* %a)
// CHECK:   ret void
void test_vst4q_lane_s8(int8_t  *a, int8x16x4_t b) {
  vst4q_lane_s8(a, b, 15);
}

// CHECK-LABEL: define{{.*}} void @test_vst4q_lane_s16(i16* noundef %a, [4 x <8 x i16>] %b.coerce) #2 {
// CHECK:   [[B:%.*]] = alloca %struct.int16x8x4_t, align 16
// CHECK:   [[__S1:%.*]] = alloca %struct.int16x8x4_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.int16x8x4_t, %struct.int16x8x4_t* [[B]], i32 0, i32 0
// CHECK:   store [4 x <8 x i16>] [[B]].coerce, [4 x <8 x i16>]* [[COERCE_DIVE]], align 16
// CHECK:   [[TMP0:%.*]] = bitcast %struct.int16x8x4_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.int16x8x4_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP0]], i8* align 16 [[TMP1]], i64 64, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast i16* %a to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.int16x8x4_t, %struct.int16x8x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <8 x i16>], [4 x <8 x i16>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <8 x i16>, <8 x i16>* [[ARRAYIDX]], align 16
// CHECK:   [[TMP4:%.*]] = bitcast <8 x i16> [[TMP3]] to <16 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.int16x8x4_t, %struct.int16x8x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <8 x i16>], [4 x <8 x i16>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <8 x i16>, <8 x i16>* [[ARRAYIDX2]], align 16
// CHECK:   [[TMP6:%.*]] = bitcast <8 x i16> [[TMP5]] to <16 x i8>
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds %struct.int16x8x4_t, %struct.int16x8x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <8 x i16>], [4 x <8 x i16>]* [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP7:%.*]] = load <8 x i16>, <8 x i16>* [[ARRAYIDX4]], align 16
// CHECK:   [[TMP8:%.*]] = bitcast <8 x i16> [[TMP7]] to <16 x i8>
// CHECK:   [[VAL5:%.*]] = getelementptr inbounds %struct.int16x8x4_t, %struct.int16x8x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <8 x i16>], [4 x <8 x i16>]* [[VAL5]], i64 0, i64 3
// CHECK:   [[TMP9:%.*]] = load <8 x i16>, <8 x i16>* [[ARRAYIDX6]], align 16
// CHECK:   [[TMP10:%.*]] = bitcast <8 x i16> [[TMP9]] to <16 x i8>
// CHECK:   [[TMP11:%.*]] = bitcast <16 x i8> [[TMP4]] to <8 x i16>
// CHECK:   [[TMP12:%.*]] = bitcast <16 x i8> [[TMP6]] to <8 x i16>
// CHECK:   [[TMP13:%.*]] = bitcast <16 x i8> [[TMP8]] to <8 x i16>
// CHECK:   [[TMP14:%.*]] = bitcast <16 x i8> [[TMP10]] to <8 x i16>
// CHECK:   call void @llvm.aarch64.neon.st4lane.v8i16.p0i8(<8 x i16> [[TMP11]], <8 x i16> [[TMP12]], <8 x i16> [[TMP13]], <8 x i16> [[TMP14]], i64 7, i8* [[TMP2]])
// CHECK:   ret void
void test_vst4q_lane_s16(int16_t  *a, int16x8x4_t b) {
  vst4q_lane_s16(a, b, 7);
}

// CHECK-LABEL: define{{.*}} void @test_vst4q_lane_s32(i32* noundef %a, [4 x <4 x i32>] %b.coerce) #2 {
// CHECK:   [[B:%.*]] = alloca %struct.int32x4x4_t, align 16
// CHECK:   [[__S1:%.*]] = alloca %struct.int32x4x4_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.int32x4x4_t, %struct.int32x4x4_t* [[B]], i32 0, i32 0
// CHECK:   store [4 x <4 x i32>] [[B]].coerce, [4 x <4 x i32>]* [[COERCE_DIVE]], align 16
// CHECK:   [[TMP0:%.*]] = bitcast %struct.int32x4x4_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.int32x4x4_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP0]], i8* align 16 [[TMP1]], i64 64, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast i32* %a to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.int32x4x4_t, %struct.int32x4x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <4 x i32>], [4 x <4 x i32>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <4 x i32>, <4 x i32>* [[ARRAYIDX]], align 16
// CHECK:   [[TMP4:%.*]] = bitcast <4 x i32> [[TMP3]] to <16 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.int32x4x4_t, %struct.int32x4x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <4 x i32>], [4 x <4 x i32>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <4 x i32>, <4 x i32>* [[ARRAYIDX2]], align 16
// CHECK:   [[TMP6:%.*]] = bitcast <4 x i32> [[TMP5]] to <16 x i8>
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds %struct.int32x4x4_t, %struct.int32x4x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <4 x i32>], [4 x <4 x i32>]* [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP7:%.*]] = load <4 x i32>, <4 x i32>* [[ARRAYIDX4]], align 16
// CHECK:   [[TMP8:%.*]] = bitcast <4 x i32> [[TMP7]] to <16 x i8>
// CHECK:   [[VAL5:%.*]] = getelementptr inbounds %struct.int32x4x4_t, %struct.int32x4x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <4 x i32>], [4 x <4 x i32>]* [[VAL5]], i64 0, i64 3
// CHECK:   [[TMP9:%.*]] = load <4 x i32>, <4 x i32>* [[ARRAYIDX6]], align 16
// CHECK:   [[TMP10:%.*]] = bitcast <4 x i32> [[TMP9]] to <16 x i8>
// CHECK:   [[TMP11:%.*]] = bitcast <16 x i8> [[TMP4]] to <4 x i32>
// CHECK:   [[TMP12:%.*]] = bitcast <16 x i8> [[TMP6]] to <4 x i32>
// CHECK:   [[TMP13:%.*]] = bitcast <16 x i8> [[TMP8]] to <4 x i32>
// CHECK:   [[TMP14:%.*]] = bitcast <16 x i8> [[TMP10]] to <4 x i32>
// CHECK:   call void @llvm.aarch64.neon.st4lane.v4i32.p0i8(<4 x i32> [[TMP11]], <4 x i32> [[TMP12]], <4 x i32> [[TMP13]], <4 x i32> [[TMP14]], i64 3, i8* [[TMP2]])
// CHECK:   ret void
void test_vst4q_lane_s32(int32_t  *a, int32x4x4_t b) {
  vst4q_lane_s32(a, b, 3);
}

// CHECK-LABEL: define{{.*}} void @test_vst4q_lane_s64(i64* noundef %a, [4 x <2 x i64>] %b.coerce) #2 {
// CHECK:   [[B:%.*]] = alloca %struct.int64x2x4_t, align 16
// CHECK:   [[__S1:%.*]] = alloca %struct.int64x2x4_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.int64x2x4_t, %struct.int64x2x4_t* [[B]], i32 0, i32 0
// CHECK:   store [4 x <2 x i64>] [[B]].coerce, [4 x <2 x i64>]* [[COERCE_DIVE]], align 16
// CHECK:   [[TMP0:%.*]] = bitcast %struct.int64x2x4_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.int64x2x4_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP0]], i8* align 16 [[TMP1]], i64 64, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast i64* %a to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.int64x2x4_t, %struct.int64x2x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <2 x i64>], [4 x <2 x i64>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <2 x i64>, <2 x i64>* [[ARRAYIDX]], align 16
// CHECK:   [[TMP4:%.*]] = bitcast <2 x i64> [[TMP3]] to <16 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.int64x2x4_t, %struct.int64x2x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <2 x i64>], [4 x <2 x i64>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <2 x i64>, <2 x i64>* [[ARRAYIDX2]], align 16
// CHECK:   [[TMP6:%.*]] = bitcast <2 x i64> [[TMP5]] to <16 x i8>
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds %struct.int64x2x4_t, %struct.int64x2x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <2 x i64>], [4 x <2 x i64>]* [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP7:%.*]] = load <2 x i64>, <2 x i64>* [[ARRAYIDX4]], align 16
// CHECK:   [[TMP8:%.*]] = bitcast <2 x i64> [[TMP7]] to <16 x i8>
// CHECK:   [[VAL5:%.*]] = getelementptr inbounds %struct.int64x2x4_t, %struct.int64x2x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <2 x i64>], [4 x <2 x i64>]* [[VAL5]], i64 0, i64 3
// CHECK:   [[TMP9:%.*]] = load <2 x i64>, <2 x i64>* [[ARRAYIDX6]], align 16
// CHECK:   [[TMP10:%.*]] = bitcast <2 x i64> [[TMP9]] to <16 x i8>
// CHECK:   [[TMP11:%.*]] = bitcast <16 x i8> [[TMP4]] to <2 x i64>
// CHECK:   [[TMP12:%.*]] = bitcast <16 x i8> [[TMP6]] to <2 x i64>
// CHECK:   [[TMP13:%.*]] = bitcast <16 x i8> [[TMP8]] to <2 x i64>
// CHECK:   [[TMP14:%.*]] = bitcast <16 x i8> [[TMP10]] to <2 x i64>
// CHECK:   call void @llvm.aarch64.neon.st4lane.v2i64.p0i8(<2 x i64> [[TMP11]], <2 x i64> [[TMP12]], <2 x i64> [[TMP13]], <2 x i64> [[TMP14]], i64 1, i8* [[TMP2]])
// CHECK:   ret void
void test_vst4q_lane_s64(int64_t  *a, int64x2x4_t b) {
  vst4q_lane_s64(a, b, 1);
}

// CHECK-LABEL: define{{.*}} void @test_vst4q_lane_f16(half* noundef %a, [4 x <8 x half>] %b.coerce) #2 {
// CHECK:   [[B:%.*]] = alloca %struct.float16x8x4_t, align 16
// CHECK:   [[__S1:%.*]] = alloca %struct.float16x8x4_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.float16x8x4_t, %struct.float16x8x4_t* [[B]], i32 0, i32 0
// CHECK:   store [4 x <8 x half>] [[B]].coerce, [4 x <8 x half>]* [[COERCE_DIVE]], align 16
// CHECK:   [[TMP0:%.*]] = bitcast %struct.float16x8x4_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.float16x8x4_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP0]], i8* align 16 [[TMP1]], i64 64, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast half* %a to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.float16x8x4_t, %struct.float16x8x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <8 x half>], [4 x <8 x half>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <8 x half>, <8 x half>* [[ARRAYIDX]], align 16
// CHECK:   [[TMP4:%.*]] = bitcast <8 x half> [[TMP3]] to <16 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.float16x8x4_t, %struct.float16x8x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <8 x half>], [4 x <8 x half>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <8 x half>, <8 x half>* [[ARRAYIDX2]], align 16
// CHECK:   [[TMP6:%.*]] = bitcast <8 x half> [[TMP5]] to <16 x i8>
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds %struct.float16x8x4_t, %struct.float16x8x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <8 x half>], [4 x <8 x half>]* [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP7:%.*]] = load <8 x half>, <8 x half>* [[ARRAYIDX4]], align 16
// CHECK:   [[TMP8:%.*]] = bitcast <8 x half> [[TMP7]] to <16 x i8>
// CHECK:   [[VAL5:%.*]] = getelementptr inbounds %struct.float16x8x4_t, %struct.float16x8x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <8 x half>], [4 x <8 x half>]* [[VAL5]], i64 0, i64 3
// CHECK:   [[TMP9:%.*]] = load <8 x half>, <8 x half>* [[ARRAYIDX6]], align 16
// CHECK:   [[TMP10:%.*]] = bitcast <8 x half> [[TMP9]] to <16 x i8>
// CHECK:   [[TMP11:%.*]] = bitcast <16 x i8> [[TMP4]] to <8 x half>
// CHECK:   [[TMP12:%.*]] = bitcast <16 x i8> [[TMP6]] to <8 x half>
// CHECK:   [[TMP13:%.*]] = bitcast <16 x i8> [[TMP8]] to <8 x half>
// CHECK:   [[TMP14:%.*]] = bitcast <16 x i8> [[TMP10]] to <8 x half>
// CHECK:   call void @llvm.aarch64.neon.st4lane.v8f16.p0i8(<8 x half> [[TMP11]], <8 x half> [[TMP12]], <8 x half> [[TMP13]], <8 x half> [[TMP14]], i64 7, i8* [[TMP2]])
// CHECK:   ret void
void test_vst4q_lane_f16(float16_t  *a, float16x8x4_t b) {
  vst4q_lane_f16(a, b, 7);
}

// CHECK-LABEL: define{{.*}} void @test_vst4q_lane_f32(float* noundef %a, [4 x <4 x float>] %b.coerce) #2 {
// CHECK:   [[B:%.*]] = alloca %struct.float32x4x4_t, align 16
// CHECK:   [[__S1:%.*]] = alloca %struct.float32x4x4_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.float32x4x4_t, %struct.float32x4x4_t* [[B]], i32 0, i32 0
// CHECK:   store [4 x <4 x float>] [[B]].coerce, [4 x <4 x float>]* [[COERCE_DIVE]], align 16
// CHECK:   [[TMP0:%.*]] = bitcast %struct.float32x4x4_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.float32x4x4_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP0]], i8* align 16 [[TMP1]], i64 64, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast float* %a to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.float32x4x4_t, %struct.float32x4x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <4 x float>], [4 x <4 x float>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <4 x float>, <4 x float>* [[ARRAYIDX]], align 16
// CHECK:   [[TMP4:%.*]] = bitcast <4 x float> [[TMP3]] to <16 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.float32x4x4_t, %struct.float32x4x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <4 x float>], [4 x <4 x float>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <4 x float>, <4 x float>* [[ARRAYIDX2]], align 16
// CHECK:   [[TMP6:%.*]] = bitcast <4 x float> [[TMP5]] to <16 x i8>
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds %struct.float32x4x4_t, %struct.float32x4x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <4 x float>], [4 x <4 x float>]* [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP7:%.*]] = load <4 x float>, <4 x float>* [[ARRAYIDX4]], align 16
// CHECK:   [[TMP8:%.*]] = bitcast <4 x float> [[TMP7]] to <16 x i8>
// CHECK:   [[VAL5:%.*]] = getelementptr inbounds %struct.float32x4x4_t, %struct.float32x4x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <4 x float>], [4 x <4 x float>]* [[VAL5]], i64 0, i64 3
// CHECK:   [[TMP9:%.*]] = load <4 x float>, <4 x float>* [[ARRAYIDX6]], align 16
// CHECK:   [[TMP10:%.*]] = bitcast <4 x float> [[TMP9]] to <16 x i8>
// CHECK:   [[TMP11:%.*]] = bitcast <16 x i8> [[TMP4]] to <4 x float>
// CHECK:   [[TMP12:%.*]] = bitcast <16 x i8> [[TMP6]] to <4 x float>
// CHECK:   [[TMP13:%.*]] = bitcast <16 x i8> [[TMP8]] to <4 x float>
// CHECK:   [[TMP14:%.*]] = bitcast <16 x i8> [[TMP10]] to <4 x float>
// CHECK:   call void @llvm.aarch64.neon.st4lane.v4f32.p0i8(<4 x float> [[TMP11]], <4 x float> [[TMP12]], <4 x float> [[TMP13]], <4 x float> [[TMP14]], i64 3, i8* [[TMP2]])
// CHECK:   ret void
void test_vst4q_lane_f32(float32_t  *a, float32x4x4_t b) {
  vst4q_lane_f32(a, b, 3);
}

// CHECK-LABEL: define{{.*}} void @test_vst4q_lane_f64(double* noundef %a, [4 x <2 x double>] %b.coerce) #2 {
// CHECK:   [[B:%.*]] = alloca %struct.float64x2x4_t, align 16
// CHECK:   [[__S1:%.*]] = alloca %struct.float64x2x4_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.float64x2x4_t, %struct.float64x2x4_t* [[B]], i32 0, i32 0
// CHECK:   store [4 x <2 x double>] [[B]].coerce, [4 x <2 x double>]* [[COERCE_DIVE]], align 16
// CHECK:   [[TMP0:%.*]] = bitcast %struct.float64x2x4_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.float64x2x4_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP0]], i8* align 16 [[TMP1]], i64 64, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast double* %a to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.float64x2x4_t, %struct.float64x2x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <2 x double>], [4 x <2 x double>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <2 x double>, <2 x double>* [[ARRAYIDX]], align 16
// CHECK:   [[TMP4:%.*]] = bitcast <2 x double> [[TMP3]] to <16 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.float64x2x4_t, %struct.float64x2x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <2 x double>], [4 x <2 x double>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <2 x double>, <2 x double>* [[ARRAYIDX2]], align 16
// CHECK:   [[TMP6:%.*]] = bitcast <2 x double> [[TMP5]] to <16 x i8>
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds %struct.float64x2x4_t, %struct.float64x2x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <2 x double>], [4 x <2 x double>]* [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP7:%.*]] = load <2 x double>, <2 x double>* [[ARRAYIDX4]], align 16
// CHECK:   [[TMP8:%.*]] = bitcast <2 x double> [[TMP7]] to <16 x i8>
// CHECK:   [[VAL5:%.*]] = getelementptr inbounds %struct.float64x2x4_t, %struct.float64x2x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <2 x double>], [4 x <2 x double>]* [[VAL5]], i64 0, i64 3
// CHECK:   [[TMP9:%.*]] = load <2 x double>, <2 x double>* [[ARRAYIDX6]], align 16
// CHECK:   [[TMP10:%.*]] = bitcast <2 x double> [[TMP9]] to <16 x i8>
// CHECK:   [[TMP11:%.*]] = bitcast <16 x i8> [[TMP4]] to <2 x double>
// CHECK:   [[TMP12:%.*]] = bitcast <16 x i8> [[TMP6]] to <2 x double>
// CHECK:   [[TMP13:%.*]] = bitcast <16 x i8> [[TMP8]] to <2 x double>
// CHECK:   [[TMP14:%.*]] = bitcast <16 x i8> [[TMP10]] to <2 x double>
// CHECK:   call void @llvm.aarch64.neon.st4lane.v2f64.p0i8(<2 x double> [[TMP11]], <2 x double> [[TMP12]], <2 x double> [[TMP13]], <2 x double> [[TMP14]], i64 1, i8* [[TMP2]])
// CHECK:   ret void
void test_vst4q_lane_f64(float64_t  *a, float64x2x4_t b) {
  vst4q_lane_f64(a, b, 1);
}

// CHECK-LABEL: define{{.*}} void @test_vst4q_lane_p8(i8* noundef %a, [4 x <16 x i8>] %b.coerce) #2 {
// CHECK:   [[B:%.*]] = alloca %struct.poly8x16x4_t, align 16
// CHECK:   [[__S1:%.*]] = alloca %struct.poly8x16x4_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.poly8x16x4_t, %struct.poly8x16x4_t* [[B]], i32 0, i32 0
// CHECK:   store [4 x <16 x i8>] [[B]].coerce, [4 x <16 x i8>]* [[COERCE_DIVE]], align 16
// CHECK:   [[TMP0:%.*]] = bitcast %struct.poly8x16x4_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.poly8x16x4_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP0]], i8* align 16 [[TMP1]], i64 64, i1 false)
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.poly8x16x4_t, %struct.poly8x16x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <16 x i8>], [4 x <16 x i8>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP2:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX]], align 16
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.poly8x16x4_t, %struct.poly8x16x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <16 x i8>], [4 x <16 x i8>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP3:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX2]], align 16
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds %struct.poly8x16x4_t, %struct.poly8x16x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <16 x i8>], [4 x <16 x i8>]* [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP4:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX4]], align 16
// CHECK:   [[VAL5:%.*]] = getelementptr inbounds %struct.poly8x16x4_t, %struct.poly8x16x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <16 x i8>], [4 x <16 x i8>]* [[VAL5]], i64 0, i64 3
// CHECK:   [[TMP5:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX6]], align 16
// CHECK:   call void @llvm.aarch64.neon.st4lane.v16i8.p0i8(<16 x i8> [[TMP2]], <16 x i8> [[TMP3]], <16 x i8> [[TMP4]], <16 x i8> [[TMP5]], i64 15, i8* %a)
// CHECK:   ret void
void test_vst4q_lane_p8(poly8_t  *a, poly8x16x4_t b) {
  vst4q_lane_p8(a, b, 15);
}

// CHECK-LABEL: define{{.*}} void @test_vst4q_lane_p16(i16* noundef %a, [4 x <8 x i16>] %b.coerce) #2 {
// CHECK:   [[B:%.*]] = alloca %struct.poly16x8x4_t, align 16
// CHECK:   [[__S1:%.*]] = alloca %struct.poly16x8x4_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.poly16x8x4_t, %struct.poly16x8x4_t* [[B]], i32 0, i32 0
// CHECK:   store [4 x <8 x i16>] [[B]].coerce, [4 x <8 x i16>]* [[COERCE_DIVE]], align 16
// CHECK:   [[TMP0:%.*]] = bitcast %struct.poly16x8x4_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.poly16x8x4_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP0]], i8* align 16 [[TMP1]], i64 64, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast i16* %a to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.poly16x8x4_t, %struct.poly16x8x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <8 x i16>], [4 x <8 x i16>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <8 x i16>, <8 x i16>* [[ARRAYIDX]], align 16
// CHECK:   [[TMP4:%.*]] = bitcast <8 x i16> [[TMP3]] to <16 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.poly16x8x4_t, %struct.poly16x8x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <8 x i16>], [4 x <8 x i16>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <8 x i16>, <8 x i16>* [[ARRAYIDX2]], align 16
// CHECK:   [[TMP6:%.*]] = bitcast <8 x i16> [[TMP5]] to <16 x i8>
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds %struct.poly16x8x4_t, %struct.poly16x8x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <8 x i16>], [4 x <8 x i16>]* [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP7:%.*]] = load <8 x i16>, <8 x i16>* [[ARRAYIDX4]], align 16
// CHECK:   [[TMP8:%.*]] = bitcast <8 x i16> [[TMP7]] to <16 x i8>
// CHECK:   [[VAL5:%.*]] = getelementptr inbounds %struct.poly16x8x4_t, %struct.poly16x8x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <8 x i16>], [4 x <8 x i16>]* [[VAL5]], i64 0, i64 3
// CHECK:   [[TMP9:%.*]] = load <8 x i16>, <8 x i16>* [[ARRAYIDX6]], align 16
// CHECK:   [[TMP10:%.*]] = bitcast <8 x i16> [[TMP9]] to <16 x i8>
// CHECK:   [[TMP11:%.*]] = bitcast <16 x i8> [[TMP4]] to <8 x i16>
// CHECK:   [[TMP12:%.*]] = bitcast <16 x i8> [[TMP6]] to <8 x i16>
// CHECK:   [[TMP13:%.*]] = bitcast <16 x i8> [[TMP8]] to <8 x i16>
// CHECK:   [[TMP14:%.*]] = bitcast <16 x i8> [[TMP10]] to <8 x i16>
// CHECK:   call void @llvm.aarch64.neon.st4lane.v8i16.p0i8(<8 x i16> [[TMP11]], <8 x i16> [[TMP12]], <8 x i16> [[TMP13]], <8 x i16> [[TMP14]], i64 7, i8* [[TMP2]])
// CHECK:   ret void
void test_vst4q_lane_p16(poly16_t  *a, poly16x8x4_t b) {
  vst4q_lane_p16(a, b, 7);
}

// CHECK-LABEL: define{{.*}} void @test_vst4q_lane_p64(i64* noundef %a, [4 x <2 x i64>] %b.coerce) #2 {
// CHECK:   [[B:%.*]] = alloca %struct.poly64x2x4_t, align 16
// CHECK:   [[__S1:%.*]] = alloca %struct.poly64x2x4_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.poly64x2x4_t, %struct.poly64x2x4_t* [[B]], i32 0, i32 0
// CHECK:   store [4 x <2 x i64>] [[B]].coerce, [4 x <2 x i64>]* [[COERCE_DIVE]], align 16
// CHECK:   [[TMP0:%.*]] = bitcast %struct.poly64x2x4_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.poly64x2x4_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 [[TMP0]], i8* align 16 [[TMP1]], i64 64, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast i64* %a to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.poly64x2x4_t, %struct.poly64x2x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <2 x i64>], [4 x <2 x i64>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <2 x i64>, <2 x i64>* [[ARRAYIDX]], align 16
// CHECK:   [[TMP4:%.*]] = bitcast <2 x i64> [[TMP3]] to <16 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.poly64x2x4_t, %struct.poly64x2x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <2 x i64>], [4 x <2 x i64>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <2 x i64>, <2 x i64>* [[ARRAYIDX2]], align 16
// CHECK:   [[TMP6:%.*]] = bitcast <2 x i64> [[TMP5]] to <16 x i8>
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds %struct.poly64x2x4_t, %struct.poly64x2x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <2 x i64>], [4 x <2 x i64>]* [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP7:%.*]] = load <2 x i64>, <2 x i64>* [[ARRAYIDX4]], align 16
// CHECK:   [[TMP8:%.*]] = bitcast <2 x i64> [[TMP7]] to <16 x i8>
// CHECK:   [[VAL5:%.*]] = getelementptr inbounds %struct.poly64x2x4_t, %struct.poly64x2x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <2 x i64>], [4 x <2 x i64>]* [[VAL5]], i64 0, i64 3
// CHECK:   [[TMP9:%.*]] = load <2 x i64>, <2 x i64>* [[ARRAYIDX6]], align 16
// CHECK:   [[TMP10:%.*]] = bitcast <2 x i64> [[TMP9]] to <16 x i8>
// CHECK:   [[TMP11:%.*]] = bitcast <16 x i8> [[TMP4]] to <2 x i64>
// CHECK:   [[TMP12:%.*]] = bitcast <16 x i8> [[TMP6]] to <2 x i64>
// CHECK:   [[TMP13:%.*]] = bitcast <16 x i8> [[TMP8]] to <2 x i64>
// CHECK:   [[TMP14:%.*]] = bitcast <16 x i8> [[TMP10]] to <2 x i64>
// CHECK:   call void @llvm.aarch64.neon.st4lane.v2i64.p0i8(<2 x i64> [[TMP11]], <2 x i64> [[TMP12]], <2 x i64> [[TMP13]], <2 x i64> [[TMP14]], i64 1, i8* [[TMP2]])
// CHECK:   ret void
void test_vst4q_lane_p64(poly64_t  *a, poly64x2x4_t b) {
  vst4q_lane_p64(a, b, 1);
}

// CHECK-LABEL: define{{.*}} void @test_vst4_lane_u8(i8* noundef %a, [4 x <8 x i8>] %b.coerce) #2 {
// CHECK:   [[B:%.*]] = alloca %struct.uint8x8x4_t, align 8
// CHECK:   [[__S1:%.*]] = alloca %struct.uint8x8x4_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.uint8x8x4_t, %struct.uint8x8x4_t* [[B]], i32 0, i32 0
// CHECK:   store [4 x <8 x i8>] [[B]].coerce, [4 x <8 x i8>]* [[COERCE_DIVE]], align 8
// CHECK:   [[TMP0:%.*]] = bitcast %struct.uint8x8x4_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.uint8x8x4_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[TMP0]], i8* align 8 [[TMP1]], i64 32, i1 false)
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.uint8x8x4_t, %struct.uint8x8x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <8 x i8>], [4 x <8 x i8>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP2:%.*]] = load <8 x i8>, <8 x i8>* [[ARRAYIDX]], align 8
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.uint8x8x4_t, %struct.uint8x8x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <8 x i8>], [4 x <8 x i8>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP3:%.*]] = load <8 x i8>, <8 x i8>* [[ARRAYIDX2]], align 8
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds %struct.uint8x8x4_t, %struct.uint8x8x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <8 x i8>], [4 x <8 x i8>]* [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP4:%.*]] = load <8 x i8>, <8 x i8>* [[ARRAYIDX4]], align 8
// CHECK:   [[VAL5:%.*]] = getelementptr inbounds %struct.uint8x8x4_t, %struct.uint8x8x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <8 x i8>], [4 x <8 x i8>]* [[VAL5]], i64 0, i64 3
// CHECK:   [[TMP5:%.*]] = load <8 x i8>, <8 x i8>* [[ARRAYIDX6]], align 8
// CHECK:   call void @llvm.aarch64.neon.st4lane.v8i8.p0i8(<8 x i8> [[TMP2]], <8 x i8> [[TMP3]], <8 x i8> [[TMP4]], <8 x i8> [[TMP5]], i64 7, i8* %a)
// CHECK:   ret void
void test_vst4_lane_u8(uint8_t  *a, uint8x8x4_t b) {
  vst4_lane_u8(a, b, 7);
}

// CHECK-LABEL: define{{.*}} void @test_vst4_lane_u16(i16* noundef %a, [4 x <4 x i16>] %b.coerce) #2 {
// CHECK:   [[B:%.*]] = alloca %struct.uint16x4x4_t, align 8
// CHECK:   [[__S1:%.*]] = alloca %struct.uint16x4x4_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.uint16x4x4_t, %struct.uint16x4x4_t* [[B]], i32 0, i32 0
// CHECK:   store [4 x <4 x i16>] [[B]].coerce, [4 x <4 x i16>]* [[COERCE_DIVE]], align 8
// CHECK:   [[TMP0:%.*]] = bitcast %struct.uint16x4x4_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.uint16x4x4_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[TMP0]], i8* align 8 [[TMP1]], i64 32, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast i16* %a to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.uint16x4x4_t, %struct.uint16x4x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <4 x i16>], [4 x <4 x i16>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <4 x i16>, <4 x i16>* [[ARRAYIDX]], align 8
// CHECK:   [[TMP4:%.*]] = bitcast <4 x i16> [[TMP3]] to <8 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.uint16x4x4_t, %struct.uint16x4x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <4 x i16>], [4 x <4 x i16>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <4 x i16>, <4 x i16>* [[ARRAYIDX2]], align 8
// CHECK:   [[TMP6:%.*]] = bitcast <4 x i16> [[TMP5]] to <8 x i8>
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds %struct.uint16x4x4_t, %struct.uint16x4x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <4 x i16>], [4 x <4 x i16>]* [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP7:%.*]] = load <4 x i16>, <4 x i16>* [[ARRAYIDX4]], align 8
// CHECK:   [[TMP8:%.*]] = bitcast <4 x i16> [[TMP7]] to <8 x i8>
// CHECK:   [[VAL5:%.*]] = getelementptr inbounds %struct.uint16x4x4_t, %struct.uint16x4x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <4 x i16>], [4 x <4 x i16>]* [[VAL5]], i64 0, i64 3
// CHECK:   [[TMP9:%.*]] = load <4 x i16>, <4 x i16>* [[ARRAYIDX6]], align 8
// CHECK:   [[TMP10:%.*]] = bitcast <4 x i16> [[TMP9]] to <8 x i8>
// CHECK:   [[TMP11:%.*]] = bitcast <8 x i8> [[TMP4]] to <4 x i16>
// CHECK:   [[TMP12:%.*]] = bitcast <8 x i8> [[TMP6]] to <4 x i16>
// CHECK:   [[TMP13:%.*]] = bitcast <8 x i8> [[TMP8]] to <4 x i16>
// CHECK:   [[TMP14:%.*]] = bitcast <8 x i8> [[TMP10]] to <4 x i16>
// CHECK:   call void @llvm.aarch64.neon.st4lane.v4i16.p0i8(<4 x i16> [[TMP11]], <4 x i16> [[TMP12]], <4 x i16> [[TMP13]], <4 x i16> [[TMP14]], i64 3, i8* [[TMP2]])
// CHECK:   ret void
void test_vst4_lane_u16(uint16_t  *a, uint16x4x4_t b) {
  vst4_lane_u16(a, b, 3);
}

// CHECK-LABEL: define{{.*}} void @test_vst4_lane_u32(i32* noundef %a, [4 x <2 x i32>] %b.coerce) #2 {
// CHECK:   [[B:%.*]] = alloca %struct.uint32x2x4_t, align 8
// CHECK:   [[__S1:%.*]] = alloca %struct.uint32x2x4_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.uint32x2x4_t, %struct.uint32x2x4_t* [[B]], i32 0, i32 0
// CHECK:   store [4 x <2 x i32>] [[B]].coerce, [4 x <2 x i32>]* [[COERCE_DIVE]], align 8
// CHECK:   [[TMP0:%.*]] = bitcast %struct.uint32x2x4_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.uint32x2x4_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[TMP0]], i8* align 8 [[TMP1]], i64 32, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast i32* %a to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.uint32x2x4_t, %struct.uint32x2x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <2 x i32>], [4 x <2 x i32>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <2 x i32>, <2 x i32>* [[ARRAYIDX]], align 8
// CHECK:   [[TMP4:%.*]] = bitcast <2 x i32> [[TMP3]] to <8 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.uint32x2x4_t, %struct.uint32x2x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <2 x i32>], [4 x <2 x i32>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <2 x i32>, <2 x i32>* [[ARRAYIDX2]], align 8
// CHECK:   [[TMP6:%.*]] = bitcast <2 x i32> [[TMP5]] to <8 x i8>
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds %struct.uint32x2x4_t, %struct.uint32x2x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <2 x i32>], [4 x <2 x i32>]* [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP7:%.*]] = load <2 x i32>, <2 x i32>* [[ARRAYIDX4]], align 8
// CHECK:   [[TMP8:%.*]] = bitcast <2 x i32> [[TMP7]] to <8 x i8>
// CHECK:   [[VAL5:%.*]] = getelementptr inbounds %struct.uint32x2x4_t, %struct.uint32x2x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <2 x i32>], [4 x <2 x i32>]* [[VAL5]], i64 0, i64 3
// CHECK:   [[TMP9:%.*]] = load <2 x i32>, <2 x i32>* [[ARRAYIDX6]], align 8
// CHECK:   [[TMP10:%.*]] = bitcast <2 x i32> [[TMP9]] to <8 x i8>
// CHECK:   [[TMP11:%.*]] = bitcast <8 x i8> [[TMP4]] to <2 x i32>
// CHECK:   [[TMP12:%.*]] = bitcast <8 x i8> [[TMP6]] to <2 x i32>
// CHECK:   [[TMP13:%.*]] = bitcast <8 x i8> [[TMP8]] to <2 x i32>
// CHECK:   [[TMP14:%.*]] = bitcast <8 x i8> [[TMP10]] to <2 x i32>
// CHECK:   call void @llvm.aarch64.neon.st4lane.v2i32.p0i8(<2 x i32> [[TMP11]], <2 x i32> [[TMP12]], <2 x i32> [[TMP13]], <2 x i32> [[TMP14]], i64 1, i8* [[TMP2]])
// CHECK:   ret void
void test_vst4_lane_u32(uint32_t  *a, uint32x2x4_t b) {
  vst4_lane_u32(a, b, 1);
}

// CHECK-LABEL: define{{.*}} void @test_vst4_lane_u64(i64* noundef %a, [4 x <1 x i64>] %b.coerce) #2 {
// CHECK:   [[B:%.*]] = alloca %struct.uint64x1x4_t, align 8
// CHECK:   [[__S1:%.*]] = alloca %struct.uint64x1x4_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.uint64x1x4_t, %struct.uint64x1x4_t* [[B]], i32 0, i32 0
// CHECK:   store [4 x <1 x i64>] [[B]].coerce, [4 x <1 x i64>]* [[COERCE_DIVE]], align 8
// CHECK:   [[TMP0:%.*]] = bitcast %struct.uint64x1x4_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.uint64x1x4_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[TMP0]], i8* align 8 [[TMP1]], i64 32, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast i64* %a to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.uint64x1x4_t, %struct.uint64x1x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <1 x i64>], [4 x <1 x i64>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <1 x i64>, <1 x i64>* [[ARRAYIDX]], align 8
// CHECK:   [[TMP4:%.*]] = bitcast <1 x i64> [[TMP3]] to <8 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.uint64x1x4_t, %struct.uint64x1x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <1 x i64>], [4 x <1 x i64>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <1 x i64>, <1 x i64>* [[ARRAYIDX2]], align 8
// CHECK:   [[TMP6:%.*]] = bitcast <1 x i64> [[TMP5]] to <8 x i8>
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds %struct.uint64x1x4_t, %struct.uint64x1x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <1 x i64>], [4 x <1 x i64>]* [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP7:%.*]] = load <1 x i64>, <1 x i64>* [[ARRAYIDX4]], align 8
// CHECK:   [[TMP8:%.*]] = bitcast <1 x i64> [[TMP7]] to <8 x i8>
// CHECK:   [[VAL5:%.*]] = getelementptr inbounds %struct.uint64x1x4_t, %struct.uint64x1x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <1 x i64>], [4 x <1 x i64>]* [[VAL5]], i64 0, i64 3
// CHECK:   [[TMP9:%.*]] = load <1 x i64>, <1 x i64>* [[ARRAYIDX6]], align 8
// CHECK:   [[TMP10:%.*]] = bitcast <1 x i64> [[TMP9]] to <8 x i8>
// CHECK:   [[TMP11:%.*]] = bitcast <8 x i8> [[TMP4]] to <1 x i64>
// CHECK:   [[TMP12:%.*]] = bitcast <8 x i8> [[TMP6]] to <1 x i64>
// CHECK:   [[TMP13:%.*]] = bitcast <8 x i8> [[TMP8]] to <1 x i64>
// CHECK:   [[TMP14:%.*]] = bitcast <8 x i8> [[TMP10]] to <1 x i64>
// CHECK:   call void @llvm.aarch64.neon.st4lane.v1i64.p0i8(<1 x i64> [[TMP11]], <1 x i64> [[TMP12]], <1 x i64> [[TMP13]], <1 x i64> [[TMP14]], i64 0, i8* [[TMP2]])
// CHECK:   ret void
void test_vst4_lane_u64(uint64_t  *a, uint64x1x4_t b) {
  vst4_lane_u64(a, b, 0);
}

// CHECK-LABEL: define{{.*}} void @test_vst4_lane_s8(i8* noundef %a, [4 x <8 x i8>] %b.coerce) #2 {
// CHECK:   [[B:%.*]] = alloca %struct.int8x8x4_t, align 8
// CHECK:   [[__S1:%.*]] = alloca %struct.int8x8x4_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.int8x8x4_t, %struct.int8x8x4_t* [[B]], i32 0, i32 0
// CHECK:   store [4 x <8 x i8>] [[B]].coerce, [4 x <8 x i8>]* [[COERCE_DIVE]], align 8
// CHECK:   [[TMP0:%.*]] = bitcast %struct.int8x8x4_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.int8x8x4_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[TMP0]], i8* align 8 [[TMP1]], i64 32, i1 false)
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.int8x8x4_t, %struct.int8x8x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <8 x i8>], [4 x <8 x i8>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP2:%.*]] = load <8 x i8>, <8 x i8>* [[ARRAYIDX]], align 8
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.int8x8x4_t, %struct.int8x8x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <8 x i8>], [4 x <8 x i8>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP3:%.*]] = load <8 x i8>, <8 x i8>* [[ARRAYIDX2]], align 8
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds %struct.int8x8x4_t, %struct.int8x8x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <8 x i8>], [4 x <8 x i8>]* [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP4:%.*]] = load <8 x i8>, <8 x i8>* [[ARRAYIDX4]], align 8
// CHECK:   [[VAL5:%.*]] = getelementptr inbounds %struct.int8x8x4_t, %struct.int8x8x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <8 x i8>], [4 x <8 x i8>]* [[VAL5]], i64 0, i64 3
// CHECK:   [[TMP5:%.*]] = load <8 x i8>, <8 x i8>* [[ARRAYIDX6]], align 8
// CHECK:   call void @llvm.aarch64.neon.st4lane.v8i8.p0i8(<8 x i8> [[TMP2]], <8 x i8> [[TMP3]], <8 x i8> [[TMP4]], <8 x i8> [[TMP5]], i64 7, i8* %a)
// CHECK:   ret void
void test_vst4_lane_s8(int8_t  *a, int8x8x4_t b) {
  vst4_lane_s8(a, b, 7);
}

// CHECK-LABEL: define{{.*}} void @test_vst4_lane_s16(i16* noundef %a, [4 x <4 x i16>] %b.coerce) #2 {
// CHECK:   [[B:%.*]] = alloca %struct.int16x4x4_t, align 8
// CHECK:   [[__S1:%.*]] = alloca %struct.int16x4x4_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.int16x4x4_t, %struct.int16x4x4_t* [[B]], i32 0, i32 0
// CHECK:   store [4 x <4 x i16>] [[B]].coerce, [4 x <4 x i16>]* [[COERCE_DIVE]], align 8
// CHECK:   [[TMP0:%.*]] = bitcast %struct.int16x4x4_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.int16x4x4_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[TMP0]], i8* align 8 [[TMP1]], i64 32, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast i16* %a to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.int16x4x4_t, %struct.int16x4x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <4 x i16>], [4 x <4 x i16>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <4 x i16>, <4 x i16>* [[ARRAYIDX]], align 8
// CHECK:   [[TMP4:%.*]] = bitcast <4 x i16> [[TMP3]] to <8 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.int16x4x4_t, %struct.int16x4x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <4 x i16>], [4 x <4 x i16>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <4 x i16>, <4 x i16>* [[ARRAYIDX2]], align 8
// CHECK:   [[TMP6:%.*]] = bitcast <4 x i16> [[TMP5]] to <8 x i8>
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds %struct.int16x4x4_t, %struct.int16x4x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <4 x i16>], [4 x <4 x i16>]* [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP7:%.*]] = load <4 x i16>, <4 x i16>* [[ARRAYIDX4]], align 8
// CHECK:   [[TMP8:%.*]] = bitcast <4 x i16> [[TMP7]] to <8 x i8>
// CHECK:   [[VAL5:%.*]] = getelementptr inbounds %struct.int16x4x4_t, %struct.int16x4x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <4 x i16>], [4 x <4 x i16>]* [[VAL5]], i64 0, i64 3
// CHECK:   [[TMP9:%.*]] = load <4 x i16>, <4 x i16>* [[ARRAYIDX6]], align 8
// CHECK:   [[TMP10:%.*]] = bitcast <4 x i16> [[TMP9]] to <8 x i8>
// CHECK:   [[TMP11:%.*]] = bitcast <8 x i8> [[TMP4]] to <4 x i16>
// CHECK:   [[TMP12:%.*]] = bitcast <8 x i8> [[TMP6]] to <4 x i16>
// CHECK:   [[TMP13:%.*]] = bitcast <8 x i8> [[TMP8]] to <4 x i16>
// CHECK:   [[TMP14:%.*]] = bitcast <8 x i8> [[TMP10]] to <4 x i16>
// CHECK:   call void @llvm.aarch64.neon.st4lane.v4i16.p0i8(<4 x i16> [[TMP11]], <4 x i16> [[TMP12]], <4 x i16> [[TMP13]], <4 x i16> [[TMP14]], i64 3, i8* [[TMP2]])
// CHECK:   ret void
void test_vst4_lane_s16(int16_t  *a, int16x4x4_t b) {
  vst4_lane_s16(a, b, 3);
}

// CHECK-LABEL: define{{.*}} void @test_vst4_lane_s32(i32* noundef %a, [4 x <2 x i32>] %b.coerce) #2 {
// CHECK:   [[B:%.*]] = alloca %struct.int32x2x4_t, align 8
// CHECK:   [[__S1:%.*]] = alloca %struct.int32x2x4_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.int32x2x4_t, %struct.int32x2x4_t* [[B]], i32 0, i32 0
// CHECK:   store [4 x <2 x i32>] [[B]].coerce, [4 x <2 x i32>]* [[COERCE_DIVE]], align 8
// CHECK:   [[TMP0:%.*]] = bitcast %struct.int32x2x4_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.int32x2x4_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[TMP0]], i8* align 8 [[TMP1]], i64 32, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast i32* %a to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.int32x2x4_t, %struct.int32x2x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <2 x i32>], [4 x <2 x i32>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <2 x i32>, <2 x i32>* [[ARRAYIDX]], align 8
// CHECK:   [[TMP4:%.*]] = bitcast <2 x i32> [[TMP3]] to <8 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.int32x2x4_t, %struct.int32x2x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <2 x i32>], [4 x <2 x i32>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <2 x i32>, <2 x i32>* [[ARRAYIDX2]], align 8
// CHECK:   [[TMP6:%.*]] = bitcast <2 x i32> [[TMP5]] to <8 x i8>
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds %struct.int32x2x4_t, %struct.int32x2x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <2 x i32>], [4 x <2 x i32>]* [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP7:%.*]] = load <2 x i32>, <2 x i32>* [[ARRAYIDX4]], align 8
// CHECK:   [[TMP8:%.*]] = bitcast <2 x i32> [[TMP7]] to <8 x i8>
// CHECK:   [[VAL5:%.*]] = getelementptr inbounds %struct.int32x2x4_t, %struct.int32x2x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <2 x i32>], [4 x <2 x i32>]* [[VAL5]], i64 0, i64 3
// CHECK:   [[TMP9:%.*]] = load <2 x i32>, <2 x i32>* [[ARRAYIDX6]], align 8
// CHECK:   [[TMP10:%.*]] = bitcast <2 x i32> [[TMP9]] to <8 x i8>
// CHECK:   [[TMP11:%.*]] = bitcast <8 x i8> [[TMP4]] to <2 x i32>
// CHECK:   [[TMP12:%.*]] = bitcast <8 x i8> [[TMP6]] to <2 x i32>
// CHECK:   [[TMP13:%.*]] = bitcast <8 x i8> [[TMP8]] to <2 x i32>
// CHECK:   [[TMP14:%.*]] = bitcast <8 x i8> [[TMP10]] to <2 x i32>
// CHECK:   call void @llvm.aarch64.neon.st4lane.v2i32.p0i8(<2 x i32> [[TMP11]], <2 x i32> [[TMP12]], <2 x i32> [[TMP13]], <2 x i32> [[TMP14]], i64 1, i8* [[TMP2]])
// CHECK:   ret void
void test_vst4_lane_s32(int32_t  *a, int32x2x4_t b) {
  vst4_lane_s32(a, b, 1);
}

// CHECK-LABEL: define{{.*}} void @test_vst4_lane_s64(i64* noundef %a, [4 x <1 x i64>] %b.coerce) #2 {
// CHECK:   [[B:%.*]] = alloca %struct.int64x1x4_t, align 8
// CHECK:   [[__S1:%.*]] = alloca %struct.int64x1x4_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.int64x1x4_t, %struct.int64x1x4_t* [[B]], i32 0, i32 0
// CHECK:   store [4 x <1 x i64>] [[B]].coerce, [4 x <1 x i64>]* [[COERCE_DIVE]], align 8
// CHECK:   [[TMP0:%.*]] = bitcast %struct.int64x1x4_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.int64x1x4_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[TMP0]], i8* align 8 [[TMP1]], i64 32, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast i64* %a to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.int64x1x4_t, %struct.int64x1x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <1 x i64>], [4 x <1 x i64>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <1 x i64>, <1 x i64>* [[ARRAYIDX]], align 8
// CHECK:   [[TMP4:%.*]] = bitcast <1 x i64> [[TMP3]] to <8 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.int64x1x4_t, %struct.int64x1x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <1 x i64>], [4 x <1 x i64>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <1 x i64>, <1 x i64>* [[ARRAYIDX2]], align 8
// CHECK:   [[TMP6:%.*]] = bitcast <1 x i64> [[TMP5]] to <8 x i8>
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds %struct.int64x1x4_t, %struct.int64x1x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <1 x i64>], [4 x <1 x i64>]* [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP7:%.*]] = load <1 x i64>, <1 x i64>* [[ARRAYIDX4]], align 8
// CHECK:   [[TMP8:%.*]] = bitcast <1 x i64> [[TMP7]] to <8 x i8>
// CHECK:   [[VAL5:%.*]] = getelementptr inbounds %struct.int64x1x4_t, %struct.int64x1x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <1 x i64>], [4 x <1 x i64>]* [[VAL5]], i64 0, i64 3
// CHECK:   [[TMP9:%.*]] = load <1 x i64>, <1 x i64>* [[ARRAYIDX6]], align 8
// CHECK:   [[TMP10:%.*]] = bitcast <1 x i64> [[TMP9]] to <8 x i8>
// CHECK:   [[TMP11:%.*]] = bitcast <8 x i8> [[TMP4]] to <1 x i64>
// CHECK:   [[TMP12:%.*]] = bitcast <8 x i8> [[TMP6]] to <1 x i64>
// CHECK:   [[TMP13:%.*]] = bitcast <8 x i8> [[TMP8]] to <1 x i64>
// CHECK:   [[TMP14:%.*]] = bitcast <8 x i8> [[TMP10]] to <1 x i64>
// CHECK:   call void @llvm.aarch64.neon.st4lane.v1i64.p0i8(<1 x i64> [[TMP11]], <1 x i64> [[TMP12]], <1 x i64> [[TMP13]], <1 x i64> [[TMP14]], i64 0, i8* [[TMP2]])
// CHECK:   ret void
void test_vst4_lane_s64(int64_t  *a, int64x1x4_t b) {
  vst4_lane_s64(a, b, 0);
}

// CHECK-LABEL: define{{.*}} void @test_vst4_lane_f16(half* noundef %a, [4 x <4 x half>] %b.coerce) #2 {
// CHECK:   [[B:%.*]] = alloca %struct.float16x4x4_t, align 8
// CHECK:   [[__S1:%.*]] = alloca %struct.float16x4x4_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.float16x4x4_t, %struct.float16x4x4_t* [[B]], i32 0, i32 0
// CHECK:   store [4 x <4 x half>] [[B]].coerce, [4 x <4 x half>]* [[COERCE_DIVE]], align 8
// CHECK:   [[TMP0:%.*]] = bitcast %struct.float16x4x4_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.float16x4x4_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[TMP0]], i8* align 8 [[TMP1]], i64 32, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast half* %a to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.float16x4x4_t, %struct.float16x4x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <4 x half>], [4 x <4 x half>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <4 x half>, <4 x half>* [[ARRAYIDX]], align 8
// CHECK:   [[TMP4:%.*]] = bitcast <4 x half> [[TMP3]] to <8 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.float16x4x4_t, %struct.float16x4x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <4 x half>], [4 x <4 x half>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <4 x half>, <4 x half>* [[ARRAYIDX2]], align 8
// CHECK:   [[TMP6:%.*]] = bitcast <4 x half> [[TMP5]] to <8 x i8>
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds %struct.float16x4x4_t, %struct.float16x4x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <4 x half>], [4 x <4 x half>]* [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP7:%.*]] = load <4 x half>, <4 x half>* [[ARRAYIDX4]], align 8
// CHECK:   [[TMP8:%.*]] = bitcast <4 x half> [[TMP7]] to <8 x i8>
// CHECK:   [[VAL5:%.*]] = getelementptr inbounds %struct.float16x4x4_t, %struct.float16x4x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <4 x half>], [4 x <4 x half>]* [[VAL5]], i64 0, i64 3
// CHECK:   [[TMP9:%.*]] = load <4 x half>, <4 x half>* [[ARRAYIDX6]], align 8
// CHECK:   [[TMP10:%.*]] = bitcast <4 x half> [[TMP9]] to <8 x i8>
// CHECK:   [[TMP11:%.*]] = bitcast <8 x i8> [[TMP4]] to <4 x half>
// CHECK:   [[TMP12:%.*]] = bitcast <8 x i8> [[TMP6]] to <4 x half>
// CHECK:   [[TMP13:%.*]] = bitcast <8 x i8> [[TMP8]] to <4 x half>
// CHECK:   [[TMP14:%.*]] = bitcast <8 x i8> [[TMP10]] to <4 x half>
// CHECK:   call void @llvm.aarch64.neon.st4lane.v4f16.p0i8(<4 x half> [[TMP11]], <4 x half> [[TMP12]], <4 x half> [[TMP13]], <4 x half> [[TMP14]], i64 3, i8* [[TMP2]])
// CHECK:   ret void
void test_vst4_lane_f16(float16_t  *a, float16x4x4_t b) {
  vst4_lane_f16(a, b, 3);
}

// CHECK-LABEL: define{{.*}} void @test_vst4_lane_f32(float* noundef %a, [4 x <2 x float>] %b.coerce) #2 {
// CHECK:   [[B:%.*]] = alloca %struct.float32x2x4_t, align 8
// CHECK:   [[__S1:%.*]] = alloca %struct.float32x2x4_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.float32x2x4_t, %struct.float32x2x4_t* [[B]], i32 0, i32 0
// CHECK:   store [4 x <2 x float>] [[B]].coerce, [4 x <2 x float>]* [[COERCE_DIVE]], align 8
// CHECK:   [[TMP0:%.*]] = bitcast %struct.float32x2x4_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.float32x2x4_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[TMP0]], i8* align 8 [[TMP1]], i64 32, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast float* %a to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.float32x2x4_t, %struct.float32x2x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <2 x float>], [4 x <2 x float>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <2 x float>, <2 x float>* [[ARRAYIDX]], align 8
// CHECK:   [[TMP4:%.*]] = bitcast <2 x float> [[TMP3]] to <8 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.float32x2x4_t, %struct.float32x2x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <2 x float>], [4 x <2 x float>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <2 x float>, <2 x float>* [[ARRAYIDX2]], align 8
// CHECK:   [[TMP6:%.*]] = bitcast <2 x float> [[TMP5]] to <8 x i8>
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds %struct.float32x2x4_t, %struct.float32x2x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <2 x float>], [4 x <2 x float>]* [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP7:%.*]] = load <2 x float>, <2 x float>* [[ARRAYIDX4]], align 8
// CHECK:   [[TMP8:%.*]] = bitcast <2 x float> [[TMP7]] to <8 x i8>
// CHECK:   [[VAL5:%.*]] = getelementptr inbounds %struct.float32x2x4_t, %struct.float32x2x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <2 x float>], [4 x <2 x float>]* [[VAL5]], i64 0, i64 3
// CHECK:   [[TMP9:%.*]] = load <2 x float>, <2 x float>* [[ARRAYIDX6]], align 8
// CHECK:   [[TMP10:%.*]] = bitcast <2 x float> [[TMP9]] to <8 x i8>
// CHECK:   [[TMP11:%.*]] = bitcast <8 x i8> [[TMP4]] to <2 x float>
// CHECK:   [[TMP12:%.*]] = bitcast <8 x i8> [[TMP6]] to <2 x float>
// CHECK:   [[TMP13:%.*]] = bitcast <8 x i8> [[TMP8]] to <2 x float>
// CHECK:   [[TMP14:%.*]] = bitcast <8 x i8> [[TMP10]] to <2 x float>
// CHECK:   call void @llvm.aarch64.neon.st4lane.v2f32.p0i8(<2 x float> [[TMP11]], <2 x float> [[TMP12]], <2 x float> [[TMP13]], <2 x float> [[TMP14]], i64 1, i8* [[TMP2]])
// CHECK:   ret void
void test_vst4_lane_f32(float32_t  *a, float32x2x4_t b) {
  vst4_lane_f32(a, b, 1);
}

// CHECK-LABEL: define{{.*}} void @test_vst4_lane_f64(double* noundef %a, [4 x <1 x double>] %b.coerce) #2 {
// CHECK:   [[B:%.*]] = alloca %struct.float64x1x4_t, align 8
// CHECK:   [[__S1:%.*]] = alloca %struct.float64x1x4_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.float64x1x4_t, %struct.float64x1x4_t* [[B]], i32 0, i32 0
// CHECK:   store [4 x <1 x double>] [[B]].coerce, [4 x <1 x double>]* [[COERCE_DIVE]], align 8
// CHECK:   [[TMP0:%.*]] = bitcast %struct.float64x1x4_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.float64x1x4_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[TMP0]], i8* align 8 [[TMP1]], i64 32, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast double* %a to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.float64x1x4_t, %struct.float64x1x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <1 x double>], [4 x <1 x double>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <1 x double>, <1 x double>* [[ARRAYIDX]], align 8
// CHECK:   [[TMP4:%.*]] = bitcast <1 x double> [[TMP3]] to <8 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.float64x1x4_t, %struct.float64x1x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <1 x double>], [4 x <1 x double>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <1 x double>, <1 x double>* [[ARRAYIDX2]], align 8
// CHECK:   [[TMP6:%.*]] = bitcast <1 x double> [[TMP5]] to <8 x i8>
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds %struct.float64x1x4_t, %struct.float64x1x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <1 x double>], [4 x <1 x double>]* [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP7:%.*]] = load <1 x double>, <1 x double>* [[ARRAYIDX4]], align 8
// CHECK:   [[TMP8:%.*]] = bitcast <1 x double> [[TMP7]] to <8 x i8>
// CHECK:   [[VAL5:%.*]] = getelementptr inbounds %struct.float64x1x4_t, %struct.float64x1x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <1 x double>], [4 x <1 x double>]* [[VAL5]], i64 0, i64 3
// CHECK:   [[TMP9:%.*]] = load <1 x double>, <1 x double>* [[ARRAYIDX6]], align 8
// CHECK:   [[TMP10:%.*]] = bitcast <1 x double> [[TMP9]] to <8 x i8>
// CHECK:   [[TMP11:%.*]] = bitcast <8 x i8> [[TMP4]] to <1 x double>
// CHECK:   [[TMP12:%.*]] = bitcast <8 x i8> [[TMP6]] to <1 x double>
// CHECK:   [[TMP13:%.*]] = bitcast <8 x i8> [[TMP8]] to <1 x double>
// CHECK:   [[TMP14:%.*]] = bitcast <8 x i8> [[TMP10]] to <1 x double>
// CHECK:   call void @llvm.aarch64.neon.st4lane.v1f64.p0i8(<1 x double> [[TMP11]], <1 x double> [[TMP12]], <1 x double> [[TMP13]], <1 x double> [[TMP14]], i64 0, i8* [[TMP2]])
// CHECK:   ret void
void test_vst4_lane_f64(float64_t  *a, float64x1x4_t b) {
  vst4_lane_f64(a, b, 0);
}

// CHECK-LABEL: define{{.*}} void @test_vst4_lane_p8(i8* noundef %a, [4 x <8 x i8>] %b.coerce) #2 {
// CHECK:   [[B:%.*]] = alloca %struct.poly8x8x4_t, align 8
// CHECK:   [[__S1:%.*]] = alloca %struct.poly8x8x4_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.poly8x8x4_t, %struct.poly8x8x4_t* [[B]], i32 0, i32 0
// CHECK:   store [4 x <8 x i8>] [[B]].coerce, [4 x <8 x i8>]* [[COERCE_DIVE]], align 8
// CHECK:   [[TMP0:%.*]] = bitcast %struct.poly8x8x4_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.poly8x8x4_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[TMP0]], i8* align 8 [[TMP1]], i64 32, i1 false)
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.poly8x8x4_t, %struct.poly8x8x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <8 x i8>], [4 x <8 x i8>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP2:%.*]] = load <8 x i8>, <8 x i8>* [[ARRAYIDX]], align 8
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.poly8x8x4_t, %struct.poly8x8x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <8 x i8>], [4 x <8 x i8>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP3:%.*]] = load <8 x i8>, <8 x i8>* [[ARRAYIDX2]], align 8
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds %struct.poly8x8x4_t, %struct.poly8x8x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <8 x i8>], [4 x <8 x i8>]* [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP4:%.*]] = load <8 x i8>, <8 x i8>* [[ARRAYIDX4]], align 8
// CHECK:   [[VAL5:%.*]] = getelementptr inbounds %struct.poly8x8x4_t, %struct.poly8x8x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <8 x i8>], [4 x <8 x i8>]* [[VAL5]], i64 0, i64 3
// CHECK:   [[TMP5:%.*]] = load <8 x i8>, <8 x i8>* [[ARRAYIDX6]], align 8
// CHECK:   call void @llvm.aarch64.neon.st4lane.v8i8.p0i8(<8 x i8> [[TMP2]], <8 x i8> [[TMP3]], <8 x i8> [[TMP4]], <8 x i8> [[TMP5]], i64 7, i8* %a)
// CHECK:   ret void
void test_vst4_lane_p8(poly8_t  *a, poly8x8x4_t b) {
  vst4_lane_p8(a, b, 7);
}

// CHECK-LABEL: define{{.*}} void @test_vst4_lane_p16(i16* noundef %a, [4 x <4 x i16>] %b.coerce) #2 {
// CHECK:   [[B:%.*]] = alloca %struct.poly16x4x4_t, align 8
// CHECK:   [[__S1:%.*]] = alloca %struct.poly16x4x4_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.poly16x4x4_t, %struct.poly16x4x4_t* [[B]], i32 0, i32 0
// CHECK:   store [4 x <4 x i16>] [[B]].coerce, [4 x <4 x i16>]* [[COERCE_DIVE]], align 8
// CHECK:   [[TMP0:%.*]] = bitcast %struct.poly16x4x4_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.poly16x4x4_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[TMP0]], i8* align 8 [[TMP1]], i64 32, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast i16* %a to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.poly16x4x4_t, %struct.poly16x4x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <4 x i16>], [4 x <4 x i16>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <4 x i16>, <4 x i16>* [[ARRAYIDX]], align 8
// CHECK:   [[TMP4:%.*]] = bitcast <4 x i16> [[TMP3]] to <8 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.poly16x4x4_t, %struct.poly16x4x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <4 x i16>], [4 x <4 x i16>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <4 x i16>, <4 x i16>* [[ARRAYIDX2]], align 8
// CHECK:   [[TMP6:%.*]] = bitcast <4 x i16> [[TMP5]] to <8 x i8>
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds %struct.poly16x4x4_t, %struct.poly16x4x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <4 x i16>], [4 x <4 x i16>]* [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP7:%.*]] = load <4 x i16>, <4 x i16>* [[ARRAYIDX4]], align 8
// CHECK:   [[TMP8:%.*]] = bitcast <4 x i16> [[TMP7]] to <8 x i8>
// CHECK:   [[VAL5:%.*]] = getelementptr inbounds %struct.poly16x4x4_t, %struct.poly16x4x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <4 x i16>], [4 x <4 x i16>]* [[VAL5]], i64 0, i64 3
// CHECK:   [[TMP9:%.*]] = load <4 x i16>, <4 x i16>* [[ARRAYIDX6]], align 8
// CHECK:   [[TMP10:%.*]] = bitcast <4 x i16> [[TMP9]] to <8 x i8>
// CHECK:   [[TMP11:%.*]] = bitcast <8 x i8> [[TMP4]] to <4 x i16>
// CHECK:   [[TMP12:%.*]] = bitcast <8 x i8> [[TMP6]] to <4 x i16>
// CHECK:   [[TMP13:%.*]] = bitcast <8 x i8> [[TMP8]] to <4 x i16>
// CHECK:   [[TMP14:%.*]] = bitcast <8 x i8> [[TMP10]] to <4 x i16>
// CHECK:   call void @llvm.aarch64.neon.st4lane.v4i16.p0i8(<4 x i16> [[TMP11]], <4 x i16> [[TMP12]], <4 x i16> [[TMP13]], <4 x i16> [[TMP14]], i64 3, i8* [[TMP2]])
// CHECK:   ret void
void test_vst4_lane_p16(poly16_t  *a, poly16x4x4_t b) {
  vst4_lane_p16(a, b, 3);
}

// CHECK-LABEL: define{{.*}} void @test_vst4_lane_p64(i64* noundef %a, [4 x <1 x i64>] %b.coerce) #2 {
// CHECK:   [[B:%.*]] = alloca %struct.poly64x1x4_t, align 8
// CHECK:   [[__S1:%.*]] = alloca %struct.poly64x1x4_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.poly64x1x4_t, %struct.poly64x1x4_t* [[B]], i32 0, i32 0
// CHECK:   store [4 x <1 x i64>] [[B]].coerce, [4 x <1 x i64>]* [[COERCE_DIVE]], align 8
// CHECK:   [[TMP0:%.*]] = bitcast %struct.poly64x1x4_t* [[__S1]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast %struct.poly64x1x4_t* [[B]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[TMP0]], i8* align 8 [[TMP1]], i64 32, i1 false)
// CHECK:   [[TMP2:%.*]] = bitcast i64* %a to i8*
// CHECK:   [[VAL:%.*]] = getelementptr inbounds %struct.poly64x1x4_t, %struct.poly64x1x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <1 x i64>], [4 x <1 x i64>]* [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <1 x i64>, <1 x i64>* [[ARRAYIDX]], align 8
// CHECK:   [[TMP4:%.*]] = bitcast <1 x i64> [[TMP3]] to <8 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.poly64x1x4_t, %struct.poly64x1x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <1 x i64>], [4 x <1 x i64>]* [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <1 x i64>, <1 x i64>* [[ARRAYIDX2]], align 8
// CHECK:   [[TMP6:%.*]] = bitcast <1 x i64> [[TMP5]] to <8 x i8>
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds %struct.poly64x1x4_t, %struct.poly64x1x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <1 x i64>], [4 x <1 x i64>]* [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP7:%.*]] = load <1 x i64>, <1 x i64>* [[ARRAYIDX4]], align 8
// CHECK:   [[TMP8:%.*]] = bitcast <1 x i64> [[TMP7]] to <8 x i8>
// CHECK:   [[VAL5:%.*]] = getelementptr inbounds %struct.poly64x1x4_t, %struct.poly64x1x4_t* [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <1 x i64>], [4 x <1 x i64>]* [[VAL5]], i64 0, i64 3
// CHECK:   [[TMP9:%.*]] = load <1 x i64>, <1 x i64>* [[ARRAYIDX6]], align 8
// CHECK:   [[TMP10:%.*]] = bitcast <1 x i64> [[TMP9]] to <8 x i8>
// CHECK:   [[TMP11:%.*]] = bitcast <8 x i8> [[TMP4]] to <1 x i64>
// CHECK:   [[TMP12:%.*]] = bitcast <8 x i8> [[TMP6]] to <1 x i64>
// CHECK:   [[TMP13:%.*]] = bitcast <8 x i8> [[TMP8]] to <1 x i64>
// CHECK:   [[TMP14:%.*]] = bitcast <8 x i8> [[TMP10]] to <1 x i64>
// CHECK:   call void @llvm.aarch64.neon.st4lane.v1i64.p0i8(<1 x i64> [[TMP11]], <1 x i64> [[TMP12]], <1 x i64> [[TMP13]], <1 x i64> [[TMP14]], i64 0, i8* [[TMP2]])
// CHECK:   ret void
void test_vst4_lane_p64(poly64_t  *a, poly64x1x4_t b) {
  vst4_lane_p64(a, b, 0);
}

// CHECK: attributes #0 ={{.*}}"min-legal-vector-width"="128"
// CHECK: attributes #1 ={{.*}}"min-legal-vector-width"="64"
// CHECK: attributes #2 ={{.*}}"min-legal-vector-width"="0"

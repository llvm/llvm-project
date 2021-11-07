// RUN: %clang_cc1 -triple arm64-none-linux-gnu -target-feature +neon \
// RUN:  -disable-O0-optnone -emit-llvm -o - %s | opt -S -mem2reg | FileCheck %s

// Test new aarch64 intrinsics and types

#include <arm_neon.h>

// CHECK-LABEL: define{{.*}} float @test_vcvtxd_f32_f64(double noundef %a) #0 {
// CHECK:   [[VCVTXD_F32_F64_I:%.*]] = call float @llvm.aarch64.sisd.fcvtxn(double %a) #2
// CHECK:   ret float [[VCVTXD_F32_F64_I]]
float32_t test_vcvtxd_f32_f64(float64_t a) {
  return (float32_t)vcvtxd_f32_f64(a);
}

// CHECK-LABEL: define{{.*}} i32 @test_vcvtas_s32_f32(float noundef %a) #0 {
// CHECK:   [[VCVTAS_S32_F32_I:%.*]] = call i32 @llvm.aarch64.neon.fcvtas.i32.f32(float %a) #2
// CHECK:   ret i32 [[VCVTAS_S32_F32_I]]
int32_t test_vcvtas_s32_f32(float32_t a) {
  return (int32_t)vcvtas_s32_f32(a);
}

// CHECK-LABEL: define{{.*}} i64 @test_test_vcvtad_s64_f64(double noundef %a) #0 {
// CHECK:   [[VCVTAD_S64_F64_I:%.*]] = call i64 @llvm.aarch64.neon.fcvtas.i64.f64(double %a) #2
// CHECK:   ret i64 [[VCVTAD_S64_F64_I]]
int64_t test_test_vcvtad_s64_f64(float64_t a) {
  return (int64_t)vcvtad_s64_f64(a);
}

// CHECK-LABEL: define{{.*}} i32 @test_vcvtas_u32_f32(float noundef %a) #0 {
// CHECK:   [[VCVTAS_U32_F32_I:%.*]] = call i32 @llvm.aarch64.neon.fcvtau.i32.f32(float %a) #2
// CHECK:   ret i32 [[VCVTAS_U32_F32_I]]
uint32_t test_vcvtas_u32_f32(float32_t a) {
  return (uint32_t)vcvtas_u32_f32(a);
}

// CHECK-LABEL: define{{.*}} i64 @test_vcvtad_u64_f64(double noundef %a) #0 {
// CHECK:   [[VCVTAD_U64_F64_I:%.*]] = call i64 @llvm.aarch64.neon.fcvtau.i64.f64(double %a) #2
// CHECK:   ret i64 [[VCVTAD_U64_F64_I]]
uint64_t test_vcvtad_u64_f64(float64_t a) {
  return (uint64_t)vcvtad_u64_f64(a);
}

// CHECK-LABEL: define{{.*}} i32 @test_vcvtms_s32_f32(float noundef %a) #0 {
// CHECK:   [[VCVTMS_S32_F32_I:%.*]] = call i32 @llvm.aarch64.neon.fcvtms.i32.f32(float %a) #2
// CHECK:   ret i32 [[VCVTMS_S32_F32_I]]
int32_t test_vcvtms_s32_f32(float32_t a) {
  return (int32_t)vcvtms_s32_f32(a);
}

// CHECK-LABEL: define{{.*}} i64 @test_vcvtmd_s64_f64(double noundef %a) #0 {
// CHECK:   [[VCVTMD_S64_F64_I:%.*]] = call i64 @llvm.aarch64.neon.fcvtms.i64.f64(double %a) #2
// CHECK:   ret i64 [[VCVTMD_S64_F64_I]]
int64_t test_vcvtmd_s64_f64(float64_t a) {
  return (int64_t)vcvtmd_s64_f64(a);
}

// CHECK-LABEL: define{{.*}} i32 @test_vcvtms_u32_f32(float noundef %a) #0 {
// CHECK:   [[VCVTMS_U32_F32_I:%.*]] = call i32 @llvm.aarch64.neon.fcvtmu.i32.f32(float %a) #2
// CHECK:   ret i32 [[VCVTMS_U32_F32_I]]
uint32_t test_vcvtms_u32_f32(float32_t a) {
  return (uint32_t)vcvtms_u32_f32(a);
}

// CHECK-LABEL: define{{.*}} i64 @test_vcvtmd_u64_f64(double noundef %a) #0 {
// CHECK:   [[VCVTMD_U64_F64_I:%.*]] = call i64 @llvm.aarch64.neon.fcvtmu.i64.f64(double %a) #2
// CHECK:   ret i64 [[VCVTMD_U64_F64_I]]
uint64_t test_vcvtmd_u64_f64(float64_t a) {
  return (uint64_t)vcvtmd_u64_f64(a);
}

// CHECK-LABEL: define{{.*}} i32 @test_vcvtns_s32_f32(float noundef %a) #0 {
// CHECK:   [[VCVTNS_S32_F32_I:%.*]] = call i32 @llvm.aarch64.neon.fcvtns.i32.f32(float %a) #2
// CHECK:   ret i32 [[VCVTNS_S32_F32_I]]
int32_t test_vcvtns_s32_f32(float32_t a) {
  return (int32_t)vcvtns_s32_f32(a);
}

// CHECK-LABEL: define{{.*}} i64 @test_vcvtnd_s64_f64(double noundef %a) #0 {
// CHECK:   [[VCVTND_S64_F64_I:%.*]] = call i64 @llvm.aarch64.neon.fcvtns.i64.f64(double %a) #2
// CHECK:   ret i64 [[VCVTND_S64_F64_I]]
int64_t test_vcvtnd_s64_f64(float64_t a) {
  return (int64_t)vcvtnd_s64_f64(a);
}

// CHECK-LABEL: define{{.*}} i32 @test_vcvtns_u32_f32(float noundef %a) #0 {
// CHECK:   [[VCVTNS_U32_F32_I:%.*]] = call i32 @llvm.aarch64.neon.fcvtnu.i32.f32(float %a) #2
// CHECK:   ret i32 [[VCVTNS_U32_F32_I]]
uint32_t test_vcvtns_u32_f32(float32_t a) {
  return (uint32_t)vcvtns_u32_f32(a);
}

// CHECK-LABEL: define{{.*}} i64 @test_vcvtnd_u64_f64(double noundef %a) #0 {
// CHECK:   [[VCVTND_U64_F64_I:%.*]] = call i64 @llvm.aarch64.neon.fcvtnu.i64.f64(double %a) #2
// CHECK:   ret i64 [[VCVTND_U64_F64_I]]
uint64_t test_vcvtnd_u64_f64(float64_t a) {
  return (uint64_t)vcvtnd_u64_f64(a);
}

// CHECK-LABEL: define{{.*}} i32 @test_vcvtps_s32_f32(float noundef %a) #0 {
// CHECK:   [[VCVTPS_S32_F32_I:%.*]] = call i32 @llvm.aarch64.neon.fcvtps.i32.f32(float %a) #2
// CHECK:   ret i32 [[VCVTPS_S32_F32_I]]
int32_t test_vcvtps_s32_f32(float32_t a) {
  return (int32_t)vcvtps_s32_f32(a);
}

// CHECK-LABEL: define{{.*}} i64 @test_vcvtpd_s64_f64(double noundef %a) #0 {
// CHECK:   [[VCVTPD_S64_F64_I:%.*]] = call i64 @llvm.aarch64.neon.fcvtps.i64.f64(double %a) #2
// CHECK:   ret i64 [[VCVTPD_S64_F64_I]]
int64_t test_vcvtpd_s64_f64(float64_t a) {
  return (int64_t)vcvtpd_s64_f64(a);
}

// CHECK-LABEL: define{{.*}} i32 @test_vcvtps_u32_f32(float noundef %a) #0 {
// CHECK:   [[VCVTPS_U32_F32_I:%.*]] = call i32 @llvm.aarch64.neon.fcvtpu.i32.f32(float %a) #2
// CHECK:   ret i32 [[VCVTPS_U32_F32_I]]
uint32_t test_vcvtps_u32_f32(float32_t a) {
  return (uint32_t)vcvtps_u32_f32(a);
}

// CHECK-LABEL: define{{.*}} i64 @test_vcvtpd_u64_f64(double noundef %a) #0 {
// CHECK:   [[VCVTPD_U64_F64_I:%.*]] = call i64 @llvm.aarch64.neon.fcvtpu.i64.f64(double %a) #2
// CHECK:   ret i64 [[VCVTPD_U64_F64_I]]
uint64_t test_vcvtpd_u64_f64(float64_t a) {
  return (uint64_t)vcvtpd_u64_f64(a);
}

// CHECK-LABEL: define{{.*}} i32 @test_vcvts_s32_f32(float noundef %a) #0 {
// CHECK:   [[TMP0:%.*]] = call i32 @llvm.aarch64.neon.fcvtzs.i32.f32(float %a)
// CHECK:   ret i32 [[TMP0]]
int32_t test_vcvts_s32_f32(float32_t a) {
  return (int32_t)vcvts_s32_f32(a);
}

// CHECK-LABEL: define{{.*}} i64 @test_vcvtd_s64_f64(double noundef %a) #0 {
// CHECK:   [[TMP0:%.*]] = call i64 @llvm.aarch64.neon.fcvtzs.i64.f64(double %a)
// CHECK:   ret i64 [[TMP0]]
int64_t test_vcvtd_s64_f64(float64_t a) {
  return (int64_t)vcvtd_s64_f64(a);
}

// CHECK-LABEL: define{{.*}} i32 @test_vcvts_u32_f32(float noundef %a) #0 {
// CHECK:   [[TMP0:%.*]] = call i32 @llvm.aarch64.neon.fcvtzu.i32.f32(float %a)
// CHECK:   ret i32 [[TMP0]]
uint32_t test_vcvts_u32_f32(float32_t a) {
  return (uint32_t)vcvts_u32_f32(a);
}

// CHECK-LABEL: define{{.*}} i64 @test_vcvtd_u64_f64(double noundef %a) #0 {
// CHECK:   [[TMP0:%.*]] = call i64 @llvm.aarch64.neon.fcvtzu.i64.f64(double %a)
// CHECK:   ret i64 [[TMP0]]
uint64_t test_vcvtd_u64_f64(float64_t a) {
  return (uint64_t)vcvtd_u64_f64(a);
}

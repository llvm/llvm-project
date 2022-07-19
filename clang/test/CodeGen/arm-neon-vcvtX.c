// RUN: %clang_cc1 -triple thumbv8-linux-gnueabihf -target-cpu cortex-a57 -ffreestanding -disable-O0-optnone -emit-llvm %s -o - | opt -S -mem2reg | FileCheck %s

// REQUIRES: aarch64-registered-target || arm-registered-target

#include <arm_neon.h>

// CHECK-LABEL: define{{.*}} <2 x i32> @test_vcvta_s32_f32(<2 x float> noundef %a) #0 {
// CHECK:   [[VCVTA_S32_V1_I:%.*]] = call <2 x i32> @llvm.arm.neon.vcvtas.v2i32.v2f32(<2 x float> %a) #3
// CHECK:   ret <2 x i32> [[VCVTA_S32_V1_I]]
int32x2_t test_vcvta_s32_f32(float32x2_t a) {
  return vcvta_s32_f32(a);
}

// CHECK-LABEL: define{{.*}} <2 x i32> @test_vcvta_u32_f32(<2 x float> noundef %a) #0 {
// CHECK:   [[VCVTA_U32_V1_I:%.*]] = call <2 x i32> @llvm.arm.neon.vcvtau.v2i32.v2f32(<2 x float> %a) #3
// CHECK:   ret <2 x i32> [[VCVTA_U32_V1_I]]
uint32x2_t test_vcvta_u32_f32(float32x2_t a) {
  return vcvta_u32_f32(a);
}

// CHECK-LABEL: define{{.*}} <4 x i32> @test_vcvtaq_s32_f32(<4 x float> noundef %a) #1 {
// CHECK:   [[VCVTAQ_S32_V1_I:%.*]] = call <4 x i32> @llvm.arm.neon.vcvtas.v4i32.v4f32(<4 x float> %a) #3
// CHECK:   ret <4 x i32> [[VCVTAQ_S32_V1_I]]
int32x4_t test_vcvtaq_s32_f32(float32x4_t a) {
  return vcvtaq_s32_f32(a);
}

// CHECK-LABEL: define{{.*}} <4 x i32> @test_vcvtaq_u32_f32(<4 x float> noundef %a) #1 {
// CHECK:   [[VCVTAQ_U32_V1_I:%.*]] = call <4 x i32> @llvm.arm.neon.vcvtau.v4i32.v4f32(<4 x float> %a) #3
// CHECK:   ret <4 x i32> [[VCVTAQ_U32_V1_I]]
uint32x4_t test_vcvtaq_u32_f32(float32x4_t a) {
  return vcvtaq_u32_f32(a);
}

// CHECK-LABEL: define{{.*}} <2 x i32> @test_vcvtn_s32_f32(<2 x float> noundef %a) #0 {
// CHECK:   [[VCVTN_S32_V1_I:%.*]] = call <2 x i32> @llvm.arm.neon.vcvtns.v2i32.v2f32(<2 x float> %a) #3
// CHECK:   ret <2 x i32> [[VCVTN_S32_V1_I]]
int32x2_t test_vcvtn_s32_f32(float32x2_t a) {
  return vcvtn_s32_f32(a);
}

// CHECK-LABEL: define{{.*}} <2 x i32> @test_vcvtn_u32_f32(<2 x float> noundef %a) #0 {
// CHECK:   [[VCVTN_U32_V1_I:%.*]] = call <2 x i32> @llvm.arm.neon.vcvtnu.v2i32.v2f32(<2 x float> %a) #3
// CHECK:   ret <2 x i32> [[VCVTN_U32_V1_I]]
uint32x2_t test_vcvtn_u32_f32(float32x2_t a) {
  return vcvtn_u32_f32(a);
}

// CHECK-LABEL: define{{.*}} <4 x i32> @test_vcvtnq_s32_f32(<4 x float> noundef %a) #1 {
// CHECK:   [[VCVTNQ_S32_V1_I:%.*]] = call <4 x i32> @llvm.arm.neon.vcvtns.v4i32.v4f32(<4 x float> %a) #3
// CHECK:   ret <4 x i32> [[VCVTNQ_S32_V1_I]]
int32x4_t test_vcvtnq_s32_f32(float32x4_t a) {
  return vcvtnq_s32_f32(a);
}

// CHECK-LABEL: define{{.*}} <4 x i32> @test_vcvtnq_u32_f32(<4 x float> noundef %a) #1 {
// CHECK:   [[VCVTNQ_U32_V1_I:%.*]] = call <4 x i32> @llvm.arm.neon.vcvtnu.v4i32.v4f32(<4 x float> %a) #3
// CHECK:   ret <4 x i32> [[VCVTNQ_U32_V1_I]]
uint32x4_t test_vcvtnq_u32_f32(float32x4_t a) {
  return vcvtnq_u32_f32(a);
}

// CHECK-LABEL: define{{.*}} <2 x i32> @test_vcvtp_s32_f32(<2 x float> noundef %a) #0 {
// CHECK:   [[VCVTP_S32_V1_I:%.*]] = call <2 x i32> @llvm.arm.neon.vcvtps.v2i32.v2f32(<2 x float> %a) #3
// CHECK:   ret <2 x i32> [[VCVTP_S32_V1_I]]
int32x2_t test_vcvtp_s32_f32(float32x2_t a) {
  return vcvtp_s32_f32(a);
}

// CHECK-LABEL: define{{.*}} <2 x i32> @test_vcvtp_u32_f32(<2 x float> noundef %a) #0 {
// CHECK:   [[VCVTP_U32_V1_I:%.*]] = call <2 x i32> @llvm.arm.neon.vcvtpu.v2i32.v2f32(<2 x float> %a) #3
// CHECK:   ret <2 x i32> [[VCVTP_U32_V1_I]]
uint32x2_t test_vcvtp_u32_f32(float32x2_t a) {
  return vcvtp_u32_f32(a);
}

// CHECK-LABEL: define{{.*}} <4 x i32> @test_vcvtpq_s32_f32(<4 x float> noundef %a) #1 {
// CHECK:   [[VCVTPQ_S32_V1_I:%.*]] = call <4 x i32> @llvm.arm.neon.vcvtps.v4i32.v4f32(<4 x float> %a) #3
// CHECK:   ret <4 x i32> [[VCVTPQ_S32_V1_I]]
int32x4_t test_vcvtpq_s32_f32(float32x4_t a) {
  return vcvtpq_s32_f32(a);
}

// CHECK-LABEL: define{{.*}} <4 x i32> @test_vcvtpq_u32_f32(<4 x float> noundef %a) #1 {
// CHECK:   [[VCVTPQ_U32_V1_I:%.*]] = call <4 x i32> @llvm.arm.neon.vcvtpu.v4i32.v4f32(<4 x float> %a) #3
// CHECK:   ret <4 x i32> [[VCVTPQ_U32_V1_I]]
uint32x4_t test_vcvtpq_u32_f32(float32x4_t a) {
  return vcvtpq_u32_f32(a);
}

// CHECK-LABEL: define{{.*}} <2 x i32> @test_vcvtm_s32_f32(<2 x float> noundef %a) #0 {
// CHECK:   [[VCVTM_S32_V1_I:%.*]] = call <2 x i32> @llvm.arm.neon.vcvtms.v2i32.v2f32(<2 x float> %a) #3
// CHECK:   ret <2 x i32> [[VCVTM_S32_V1_I]]
int32x2_t test_vcvtm_s32_f32(float32x2_t a) {
  return vcvtm_s32_f32(a);
}

// CHECK-LABEL: define{{.*}} <2 x i32> @test_vcvtm_u32_f32(<2 x float> noundef %a) #0 {
// CHECK:   [[VCVTM_U32_V1_I:%.*]] = call <2 x i32> @llvm.arm.neon.vcvtmu.v2i32.v2f32(<2 x float> %a) #3
// CHECK:   ret <2 x i32> [[VCVTM_U32_V1_I]]
uint32x2_t test_vcvtm_u32_f32(float32x2_t a) {
  return vcvtm_u32_f32(a);
}

// CHECK-LABEL: define{{.*}} <4 x i32> @test_vcvtmq_s32_f32(<4 x float> noundef %a) #1 {
// CHECK:   [[VCVTMQ_S32_V1_I:%.*]] = call <4 x i32> @llvm.arm.neon.vcvtms.v4i32.v4f32(<4 x float> %a) #3
// CHECK:   ret <4 x i32> [[VCVTMQ_S32_V1_I]]
int32x4_t test_vcvtmq_s32_f32(float32x4_t a) {
  return vcvtmq_s32_f32(a);
}

// CHECK-LABEL: define{{.*}} <4 x i32> @test_vcvtmq_u32_f32(<4 x float> noundef %a) #1 {
// CHECK:   [[VCVTMQ_U32_V1_I:%.*]] = call <4 x i32> @llvm.arm.neon.vcvtmu.v4i32.v4f32(<4 x float> %a) #3
// CHECK:   ret <4 x i32> [[VCVTMQ_U32_V1_I]]
uint32x4_t test_vcvtmq_u32_f32(float32x4_t a) {
  return vcvtmq_u32_f32(a);
}

// CHECK: attributes #0 ={{.*}}"min-legal-vector-width"="64"
// CHECK: attributes #1 ={{.*}}"min-legal-vector-width"="128"

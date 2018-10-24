// RUN: %clang_cc1 -triple arm64-none-linux-gnu -target-feature +neon \
// RUN:  -disable-O0-optnone -emit-llvm -o - %s | opt -S -mem2reg | FileCheck %s

// Test new aarch64 intrinsics and types

#include <arm_neon.h>

// CHECK-LABEL: define i16 @test_vaddlv_s8(<8 x i8> %a) #0 {
// CHECK:   [[VADDLV_I:%.*]] = call i32 @llvm.aarch64.neon.saddlv.i32.v8i8(<8 x i8> %a) #3
// CHECK:   [[TMP0:%.*]] = trunc i32 [[VADDLV_I]] to i16
// CHECK:   ret i16 [[TMP0]]
int16_t test_vaddlv_s8(int8x8_t a) {
  return vaddlv_s8(a);
}

// CHECK-LABEL: define i32 @test_vaddlv_s16(<4 x i16> %a) #0 {
// CHECK:   [[VADDLV_I:%.*]] = call i32 @llvm.aarch64.neon.saddlv.i32.v4i16(<4 x i16> %a) #3
// CHECK:   ret i32 [[VADDLV_I]]
int32_t test_vaddlv_s16(int16x4_t a) {
  return vaddlv_s16(a);
}

// CHECK-LABEL: define i16 @test_vaddlv_u8(<8 x i8> %a) #0 {
// CHECK:   [[VADDLV_I:%.*]] = call i32 @llvm.aarch64.neon.uaddlv.i32.v8i8(<8 x i8> %a) #3
// CHECK:   [[TMP0:%.*]] = trunc i32 [[VADDLV_I]] to i16
// CHECK:   ret i16 [[TMP0]]
uint16_t test_vaddlv_u8(uint8x8_t a) {
  return vaddlv_u8(a);
}

// CHECK-LABEL: define i32 @test_vaddlv_u16(<4 x i16> %a) #0 {
// CHECK:   [[VADDLV_I:%.*]] = call i32 @llvm.aarch64.neon.uaddlv.i32.v4i16(<4 x i16> %a) #3
// CHECK:   ret i32 [[VADDLV_I]]
uint32_t test_vaddlv_u16(uint16x4_t a) {
  return vaddlv_u16(a);
}

// CHECK-LABEL: define i16 @test_vaddlvq_s8(<16 x i8> %a) #1 {
// CHECK:   [[VADDLV_I:%.*]] = call i32 @llvm.aarch64.neon.saddlv.i32.v16i8(<16 x i8> %a) #3
// CHECK:   [[TMP0:%.*]] = trunc i32 [[VADDLV_I]] to i16
// CHECK:   ret i16 [[TMP0]]
int16_t test_vaddlvq_s8(int8x16_t a) {
  return vaddlvq_s8(a);
}

// CHECK-LABEL: define i32 @test_vaddlvq_s16(<8 x i16> %a) #1 {
// CHECK:   [[VADDLV_I:%.*]] = call i32 @llvm.aarch64.neon.saddlv.i32.v8i16(<8 x i16> %a) #3
// CHECK:   ret i32 [[VADDLV_I]]
int32_t test_vaddlvq_s16(int16x8_t a) {
  return vaddlvq_s16(a);
}

// CHECK-LABEL: define i64 @test_vaddlvq_s32(<4 x i32> %a) #1 {
// CHECK:   [[VADDLVQ_S32_I:%.*]] = call i64 @llvm.aarch64.neon.saddlv.i64.v4i32(<4 x i32> %a) #3
// CHECK:   ret i64 [[VADDLVQ_S32_I]]
int64_t test_vaddlvq_s32(int32x4_t a) {
  return vaddlvq_s32(a);
}

// CHECK-LABEL: define i16 @test_vaddlvq_u8(<16 x i8> %a) #1 {
// CHECK:   [[VADDLV_I:%.*]] = call i32 @llvm.aarch64.neon.uaddlv.i32.v16i8(<16 x i8> %a) #3
// CHECK:   [[TMP0:%.*]] = trunc i32 [[VADDLV_I]] to i16
// CHECK:   ret i16 [[TMP0]]
uint16_t test_vaddlvq_u8(uint8x16_t a) {
  return vaddlvq_u8(a);
}

// CHECK-LABEL: define i32 @test_vaddlvq_u16(<8 x i16> %a) #1 {
// CHECK:   [[VADDLV_I:%.*]] = call i32 @llvm.aarch64.neon.uaddlv.i32.v8i16(<8 x i16> %a) #3
// CHECK:   ret i32 [[VADDLV_I]]
uint32_t test_vaddlvq_u16(uint16x8_t a) {
  return vaddlvq_u16(a);
}

// CHECK-LABEL: define i64 @test_vaddlvq_u32(<4 x i32> %a) #1 {
// CHECK:   [[VADDLVQ_U32_I:%.*]] = call i64 @llvm.aarch64.neon.uaddlv.i64.v4i32(<4 x i32> %a) #3
// CHECK:   ret i64 [[VADDLVQ_U32_I]]
uint64_t test_vaddlvq_u32(uint32x4_t a) {
  return vaddlvq_u32(a);
}

// CHECK-LABEL: define i8 @test_vmaxv_s8(<8 x i8> %a) #0 {
// CHECK:   [[VMAXV_I:%.*]] = call i32 @llvm.aarch64.neon.smaxv.i32.v8i8(<8 x i8> %a) #3
// CHECK:   [[TMP0:%.*]] = trunc i32 [[VMAXV_I]] to i8
// CHECK:   ret i8 [[TMP0]]
int8_t test_vmaxv_s8(int8x8_t a) {
  return vmaxv_s8(a);
}

// CHECK-LABEL: define i16 @test_vmaxv_s16(<4 x i16> %a) #0 {
// CHECK:   [[VMAXV_I:%.*]] = call i32 @llvm.aarch64.neon.smaxv.i32.v4i16(<4 x i16> %a) #3
// CHECK:   [[TMP2:%.*]] = trunc i32 [[VMAXV_I]] to i16
// CHECK:   ret i16 [[TMP2]]
int16_t test_vmaxv_s16(int16x4_t a) {
  return vmaxv_s16(a);
}

// CHECK-LABEL: define i8 @test_vmaxv_u8(<8 x i8> %a) #0 {
// CHECK:   [[VMAXV_I:%.*]] = call i32 @llvm.aarch64.neon.umaxv.i32.v8i8(<8 x i8> %a) #3
// CHECK:   [[TMP0:%.*]] = trunc i32 [[VMAXV_I]] to i8
// CHECK:   ret i8 [[TMP0]]
uint8_t test_vmaxv_u8(uint8x8_t a) {
  return vmaxv_u8(a);
}

// CHECK-LABEL: define i16 @test_vmaxv_u16(<4 x i16> %a) #0 {
// CHECK:   [[VMAXV_I:%.*]] = call i32 @llvm.aarch64.neon.umaxv.i32.v4i16(<4 x i16> %a) #3
// CHECK:   [[TMP2:%.*]] = trunc i32 [[VMAXV_I]] to i16
// CHECK:   ret i16 [[TMP2]]
uint16_t test_vmaxv_u16(uint16x4_t a) {
  return vmaxv_u16(a);
}

// CHECK-LABEL: define i8 @test_vmaxvq_s8(<16 x i8> %a) #1 {
// CHECK:   [[VMAXV_I:%.*]] = call i32 @llvm.aarch64.neon.smaxv.i32.v16i8(<16 x i8> %a) #3
// CHECK:   [[TMP0:%.*]] = trunc i32 [[VMAXV_I]] to i8
// CHECK:   ret i8 [[TMP0]]
int8_t test_vmaxvq_s8(int8x16_t a) {
  return vmaxvq_s8(a);
}

// CHECK-LABEL: define i16 @test_vmaxvq_s16(<8 x i16> %a) #1 {
// CHECK:   [[VMAXV_I:%.*]] = call i32 @llvm.aarch64.neon.smaxv.i32.v8i16(<8 x i16> %a) #3
// CHECK:   [[TMP2:%.*]] = trunc i32 [[VMAXV_I]] to i16
// CHECK:   ret i16 [[TMP2]]
int16_t test_vmaxvq_s16(int16x8_t a) {
  return vmaxvq_s16(a);
}

// CHECK-LABEL: define i32 @test_vmaxvq_s32(<4 x i32> %a) #1 {
// CHECK:   [[VMAXVQ_S32_I:%.*]] = call i32 @llvm.aarch64.neon.smaxv.i32.v4i32(<4 x i32> %a) #3
// CHECK:   ret i32 [[VMAXVQ_S32_I]]
int32_t test_vmaxvq_s32(int32x4_t a) {
  return vmaxvq_s32(a);
}

// CHECK-LABEL: define i8 @test_vmaxvq_u8(<16 x i8> %a) #1 {
// CHECK:   [[VMAXV_I:%.*]] = call i32 @llvm.aarch64.neon.umaxv.i32.v16i8(<16 x i8> %a) #3
// CHECK:   [[TMP0:%.*]] = trunc i32 [[VMAXV_I]] to i8
// CHECK:   ret i8 [[TMP0]]
uint8_t test_vmaxvq_u8(uint8x16_t a) {
  return vmaxvq_u8(a);
}

// CHECK-LABEL: define i16 @test_vmaxvq_u16(<8 x i16> %a) #1 {
// CHECK:   [[VMAXV_I:%.*]] = call i32 @llvm.aarch64.neon.umaxv.i32.v8i16(<8 x i16> %a) #3
// CHECK:   [[TMP2:%.*]] = trunc i32 [[VMAXV_I]] to i16
// CHECK:   ret i16 [[TMP2]]
uint16_t test_vmaxvq_u16(uint16x8_t a) {
  return vmaxvq_u16(a);
}

// CHECK-LABEL: define i32 @test_vmaxvq_u32(<4 x i32> %a) #1 {
// CHECK:   [[VMAXVQ_U32_I:%.*]] = call i32 @llvm.aarch64.neon.umaxv.i32.v4i32(<4 x i32> %a) #3
// CHECK:   ret i32 [[VMAXVQ_U32_I]]
uint32_t test_vmaxvq_u32(uint32x4_t a) {
  return vmaxvq_u32(a);
}

// CHECK-LABEL: define i8 @test_vminv_s8(<8 x i8> %a) #0 {
// CHECK:   [[VMINV_I:%.*]] = call i32 @llvm.aarch64.neon.sminv.i32.v8i8(<8 x i8> %a) #3
// CHECK:   [[TMP0:%.*]] = trunc i32 [[VMINV_I]] to i8
// CHECK:   ret i8 [[TMP0]]
int8_t test_vminv_s8(int8x8_t a) {
  return vminv_s8(a);
}

// CHECK-LABEL: define i16 @test_vminv_s16(<4 x i16> %a) #0 {
// CHECK:   [[VMINV_I:%.*]] = call i32 @llvm.aarch64.neon.sminv.i32.v4i16(<4 x i16> %a) #3
// CHECK:   [[TMP2:%.*]] = trunc i32 [[VMINV_I]] to i16
// CHECK:   ret i16 [[TMP2]]
int16_t test_vminv_s16(int16x4_t a) {
  return vminv_s16(a);
}

// CHECK-LABEL: define i8 @test_vminv_u8(<8 x i8> %a) #0 {
// CHECK:   [[VMINV_I:%.*]] = call i32 @llvm.aarch64.neon.uminv.i32.v8i8(<8 x i8> %a) #3
// CHECK:   [[TMP0:%.*]] = trunc i32 [[VMINV_I]] to i8
// CHECK:   ret i8 [[TMP0]]
uint8_t test_vminv_u8(uint8x8_t a) {
  return vminv_u8(a);
}

// CHECK-LABEL: define i16 @test_vminv_u16(<4 x i16> %a) #0 {
// CHECK:   [[VMINV_I:%.*]] = call i32 @llvm.aarch64.neon.uminv.i32.v4i16(<4 x i16> %a) #3
// CHECK:   [[TMP2:%.*]] = trunc i32 [[VMINV_I]] to i16
// CHECK:   ret i16 [[TMP2]]
uint16_t test_vminv_u16(uint16x4_t a) {
  return vminv_u16(a);
}

// CHECK-LABEL: define i8 @test_vminvq_s8(<16 x i8> %a) #1 {
// CHECK:   [[VMINV_I:%.*]] = call i32 @llvm.aarch64.neon.sminv.i32.v16i8(<16 x i8> %a) #3
// CHECK:   [[TMP0:%.*]] = trunc i32 [[VMINV_I]] to i8
// CHECK:   ret i8 [[TMP0]]
int8_t test_vminvq_s8(int8x16_t a) {
  return vminvq_s8(a);
}

// CHECK-LABEL: define i16 @test_vminvq_s16(<8 x i16> %a) #1 {
// CHECK:   [[VMINV_I:%.*]] = call i32 @llvm.aarch64.neon.sminv.i32.v8i16(<8 x i16> %a) #3
// CHECK:   [[TMP2:%.*]] = trunc i32 [[VMINV_I]] to i16
// CHECK:   ret i16 [[TMP2]]
int16_t test_vminvq_s16(int16x8_t a) {
  return vminvq_s16(a);
}

// CHECK-LABEL: define i32 @test_vminvq_s32(<4 x i32> %a) #1 {
// CHECK:   [[VMINVQ_S32_I:%.*]] = call i32 @llvm.aarch64.neon.sminv.i32.v4i32(<4 x i32> %a) #3
// CHECK:   ret i32 [[VMINVQ_S32_I]]
int32_t test_vminvq_s32(int32x4_t a) {
  return vminvq_s32(a);
}

// CHECK-LABEL: define i8 @test_vminvq_u8(<16 x i8> %a) #1 {
// CHECK:   [[VMINV_I:%.*]] = call i32 @llvm.aarch64.neon.uminv.i32.v16i8(<16 x i8> %a) #3
// CHECK:   [[TMP0:%.*]] = trunc i32 [[VMINV_I]] to i8
// CHECK:   ret i8 [[TMP0]]
uint8_t test_vminvq_u8(uint8x16_t a) {
  return vminvq_u8(a);
}

// CHECK-LABEL: define i16 @test_vminvq_u16(<8 x i16> %a) #1 {
// CHECK:   [[VMINV_I:%.*]] = call i32 @llvm.aarch64.neon.uminv.i32.v8i16(<8 x i16> %a) #3
// CHECK:   [[TMP2:%.*]] = trunc i32 [[VMINV_I]] to i16
// CHECK:   ret i16 [[TMP2]]
uint16_t test_vminvq_u16(uint16x8_t a) {
  return vminvq_u16(a);
}

// CHECK-LABEL: define i32 @test_vminvq_u32(<4 x i32> %a) #1 {
// CHECK:   [[VMINVQ_U32_I:%.*]] = call i32 @llvm.aarch64.neon.uminv.i32.v4i32(<4 x i32> %a) #3
// CHECK:   ret i32 [[VMINVQ_U32_I]]
uint32_t test_vminvq_u32(uint32x4_t a) {
  return vminvq_u32(a);
}

// CHECK-LABEL: define i8 @test_vaddv_s8(<8 x i8> %a) #0 {
// CHECK:   [[VADDV_I:%.*]] = call i32 @llvm.aarch64.neon.saddv.i32.v8i8(<8 x i8> %a) #3
// CHECK:   [[TMP0:%.*]] = trunc i32 [[VADDV_I]] to i8
// CHECK:   ret i8 [[TMP0]]
int8_t test_vaddv_s8(int8x8_t a) {
  return vaddv_s8(a);
}

// CHECK-LABEL: define i16 @test_vaddv_s16(<4 x i16> %a) #0 {
// CHECK:   [[VADDV_I:%.*]] = call i32 @llvm.aarch64.neon.saddv.i32.v4i16(<4 x i16> %a) #3
// CHECK:   [[TMP2:%.*]] = trunc i32 [[VADDV_I]] to i16
// CHECK:   ret i16 [[TMP2]]
int16_t test_vaddv_s16(int16x4_t a) {
  return vaddv_s16(a);
}

// CHECK-LABEL: define i8 @test_vaddv_u8(<8 x i8> %a) #0 {
// CHECK:   [[VADDV_I:%.*]] = call i32 @llvm.aarch64.neon.uaddv.i32.v8i8(<8 x i8> %a) #3
// CHECK:   [[TMP0:%.*]] = trunc i32 [[VADDV_I]] to i8
// CHECK:   ret i8 [[TMP0]]
uint8_t test_vaddv_u8(uint8x8_t a) {
  return vaddv_u8(a);
}

// CHECK-LABEL: define i16 @test_vaddv_u16(<4 x i16> %a) #0 {
// CHECK:   [[VADDV_I:%.*]] = call i32 @llvm.aarch64.neon.uaddv.i32.v4i16(<4 x i16> %a) #3
// CHECK:   [[TMP2:%.*]] = trunc i32 [[VADDV_I]] to i16
// CHECK:   ret i16 [[TMP2]]
uint16_t test_vaddv_u16(uint16x4_t a) {
  return vaddv_u16(a);
}

// CHECK-LABEL: define i8 @test_vaddvq_s8(<16 x i8> %a) #1 {
// CHECK:   [[VADDV_I:%.*]] = call i32 @llvm.aarch64.neon.saddv.i32.v16i8(<16 x i8> %a) #3
// CHECK:   [[TMP0:%.*]] = trunc i32 [[VADDV_I]] to i8
// CHECK:   ret i8 [[TMP0]]
int8_t test_vaddvq_s8(int8x16_t a) {
  return vaddvq_s8(a);
}

// CHECK-LABEL: define i16 @test_vaddvq_s16(<8 x i16> %a) #1 {
// CHECK:   [[VADDV_I:%.*]] = call i32 @llvm.aarch64.neon.saddv.i32.v8i16(<8 x i16> %a) #3
// CHECK:   [[TMP2:%.*]] = trunc i32 [[VADDV_I]] to i16
// CHECK:   ret i16 [[TMP2]]
int16_t test_vaddvq_s16(int16x8_t a) {
  return vaddvq_s16(a);
}

// CHECK-LABEL: define i32 @test_vaddvq_s32(<4 x i32> %a) #1 {
// CHECK:   [[VADDVQ_S32_I:%.*]] = call i32 @llvm.aarch64.neon.saddv.i32.v4i32(<4 x i32> %a) #3
// CHECK:   ret i32 [[VADDVQ_S32_I]]
int32_t test_vaddvq_s32(int32x4_t a) {
  return vaddvq_s32(a);
}

// CHECK-LABEL: define i8 @test_vaddvq_u8(<16 x i8> %a) #1 {
// CHECK:   [[VADDV_I:%.*]] = call i32 @llvm.aarch64.neon.uaddv.i32.v16i8(<16 x i8> %a) #3
// CHECK:   [[TMP0:%.*]] = trunc i32 [[VADDV_I]] to i8
// CHECK:   ret i8 [[TMP0]]
uint8_t test_vaddvq_u8(uint8x16_t a) {
  return vaddvq_u8(a);
}

// CHECK-LABEL: define i16 @test_vaddvq_u16(<8 x i16> %a) #1 {
// CHECK:   [[VADDV_I:%.*]] = call i32 @llvm.aarch64.neon.uaddv.i32.v8i16(<8 x i16> %a) #3
// CHECK:   [[TMP2:%.*]] = trunc i32 [[VADDV_I]] to i16
// CHECK:   ret i16 [[TMP2]]
uint16_t test_vaddvq_u16(uint16x8_t a) {
  return vaddvq_u16(a);
}

// CHECK-LABEL: define i32 @test_vaddvq_u32(<4 x i32> %a) #1 {
// CHECK:   [[VADDVQ_U32_I:%.*]] = call i32 @llvm.aarch64.neon.uaddv.i32.v4i32(<4 x i32> %a) #3
// CHECK:   ret i32 [[VADDVQ_U32_I]]
uint32_t test_vaddvq_u32(uint32x4_t a) {
  return vaddvq_u32(a);
}

// CHECK-LABEL: define float @test_vmaxvq_f32(<4 x float> %a) #1 {
// CHECK:   [[VMAXVQ_F32_I:%.*]] = call float @llvm.aarch64.neon.fmaxv.f32.v4f32(<4 x float> %a) #3
// CHECK:   ret float [[VMAXVQ_F32_I]]
float32_t test_vmaxvq_f32(float32x4_t a) {
  return vmaxvq_f32(a);
}

// CHECK-LABEL: define float @test_vminvq_f32(<4 x float> %a) #1 {
// CHECK:   [[VMINVQ_F32_I:%.*]] = call float @llvm.aarch64.neon.fminv.f32.v4f32(<4 x float> %a) #3
// CHECK:   ret float [[VMINVQ_F32_I]]
float32_t test_vminvq_f32(float32x4_t a) {
  return vminvq_f32(a);
}

// CHECK-LABEL: define float @test_vmaxnmvq_f32(<4 x float> %a) #1 {
// CHECK:   [[VMAXNMVQ_F32_I:%.*]] = call float @llvm.aarch64.neon.fmaxnmv.f32.v4f32(<4 x float> %a) #3
// CHECK:   ret float [[VMAXNMVQ_F32_I]]
float32_t test_vmaxnmvq_f32(float32x4_t a) {
  return vmaxnmvq_f32(a);
}

// CHECK-LABEL: define float @test_vminnmvq_f32(<4 x float> %a) #1 {
// CHECK:   [[VMINNMVQ_F32_I:%.*]] = call float @llvm.aarch64.neon.fminnmv.f32.v4f32(<4 x float> %a) #3
// CHECK:   ret float [[VMINNMVQ_F32_I]]
float32_t test_vminnmvq_f32(float32x4_t a) {
  return vminnmvq_f32(a);
}

// CHECK: attributes #0 ={{.*}}"min-legal-vector-width"="64"
// CHECK: attributes #1 ={{.*}}"min-legal-vector-width"="128"

// REQUIRES: arm-registered-target
// RUN: %clang_cc1 -triple armv8.3a-arm-none-eabi -target-cpu generic \
// RUN: -target-feature +fullfp16 -mfloat-abi soft -S -emit-llvm -o - %s | \
// RUN: opt -S -sroa -o - | FileCheck %s

#include <arm_neon.h>

void foo16x4_rot90(float16x4_t a, float16x4_t b)
{
// CHECK: call <4 x half> @llvm.arm.neon.vcadd.rot90.v4f16
  float16x4_t result = vcadd_rot90_f16(a, b);
}

void foo32x2_rot90(float32x2_t a, float32x2_t b)
{
// CHECK: call <2 x float> @llvm.arm.neon.vcadd.rot90.v2f32
  float32x2_t result = vcadd_rot90_f32(a, b);
}

void foo16x8_rot90(float16x8_t a, float16x8_t b)
{
// CHECK: call <8 x half> @llvm.arm.neon.vcadd.rot90.v8f16
  float16x8_t result = vcaddq_rot90_f16(a, b);
}

void foo32x4_rot90(float32x4_t a, float32x4_t b)
{
// CHECK: call <4 x float> @llvm.arm.neon.vcadd.rot90.v4f32
  float32x4_t result = vcaddq_rot90_f32(a, b);
}

void foo16x4_rot270(float16x4_t a, float16x4_t b)
{
// CHECK: call <4 x half> @llvm.arm.neon.vcadd.rot270.v4f16
  float16x4_t result = vcadd_rot270_f16(a, b);
}

void foo32x2_rot270(float32x2_t a, float32x2_t b)
{
// CHECK: call <2 x float> @llvm.arm.neon.vcadd.rot270.v2f32
  float32x2_t result = vcadd_rot270_f32(a, b);
}

void foo16x8_rot270(float16x8_t a, float16x8_t b)
{
// CHECK: call <8 x half> @llvm.arm.neon.vcadd.rot270.v8f16
  float16x8_t result = vcaddq_rot270_f16(a, b);
}

void foo32x4_rot270(float32x4_t a, float32x4_t b)
{
// CHECK: call <4 x float> @llvm.arm.neon.vcadd.rot270.v4f32
  float32x4_t result = vcaddq_rot270_f32(a, b);
}

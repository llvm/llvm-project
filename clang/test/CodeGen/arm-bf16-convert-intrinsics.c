// RUN: %clang_cc1 \
// RUN:   -triple aarch64-arm-none-eabi -target-feature +neon -target-feature +bf16 \
// RUN:   -disable-O0-optnone -emit-llvm -o - %s \
// RUN:   | opt -S -mem2reg -instcombine \
// RUN:   | FileCheck --check-prefixes=CHECK,CHECK-A64 %s
// RUN: %clang_cc1 \
// RUN:   -triple armv8.6a-arm-none-eabi -target-feature +neon \
// RUN:   -target-feature +bf16 -mfloat-abi hard \
// RUN:   -disable-O0-optnone -emit-llvm -o - %s \
// RUN:   | opt -S -mem2reg -instcombine \
// RUN:   | FileCheck --check-prefixes=CHECK,CHECK-A32-HARDFP %s
// RUN: %clang_cc1 \
// RUN:   -triple armv8.6a-arm-none-eabi -target-feature +neon \
// RUN:   -target-feature +bf16 -mfloat-abi softfp \
// RUN:   -disable-O0-optnone -emit-llvm -o - %s \
// RUN:   | opt -S -mem2reg -instcombine \
// RUN:   | FileCheck --check-prefixes=CHECK,CHECK-A32-SOFTFP %s

#include <arm_neon.h>

// CHECK-LABEL: test_vcvt_f32_bf16
// CHECK: %[[EXT:.*]] = zext <4 x i16> %{{.*}} to <4 x i32>
// CHECK: shl nuw <4 x i32> %[[EXT]], <i32 16, i32 16, i32 16, i32 16>
float32x4_t test_vcvt_f32_bf16(bfloat16x4_t a) {
  return vcvt_f32_bf16(a);
}

// CHECK-LABEL: test_vcvtq_low_f32_bf16
// CHECK: shufflevector <8 x bfloat> %{{.*}}, <8 x bfloat> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
// CHECK: %[[EXT:.*]] = zext <4 x i16> %{{.*}} to <4 x i32>
// CHECK: shl nuw <4 x i32> %[[EXT]], <i32 16, i32 16, i32 16, i32 16>
float32x4_t test_vcvtq_low_f32_bf16(bfloat16x8_t a) {
  return vcvtq_low_f32_bf16(a);
}

// CHECK-LABEL: test_vcvtq_high_f32_bf16
// CHECK: shufflevector <8 x bfloat> %{{.*}}, <8 x bfloat> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK: %[[EXT:.*]] = zext <4 x i16> %{{.*}} to <4 x i32>
// CHECK: shl nuw <4 x i32> %[[EXT]], <i32 16, i32 16, i32 16, i32 16>
float32x4_t test_vcvtq_high_f32_bf16(bfloat16x8_t a) {
  return vcvtq_high_f32_bf16(a);
}

// CHECK-LABEL: test_vcvt_bf16_f32
// CHECK-A64: %[[CVT:.*]] = call <8 x bfloat> @llvm.aarch64.neon.bfcvtn(<4 x float> %a)
// CHECK-A64: shufflevector <8 x bfloat> %[[CVT]], <8 x bfloat> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
// CHECK-A32-HARDFP: call <4 x bfloat> @llvm.arm.neon.vcvtfp2bf.v4bf16(<4 x float> %a)
// CHECK-A32-SOFTFP: call <4 x i16> @llvm.arm.neon.vcvtfp2bf.v4i16(<4 x float> %a)
bfloat16x4_t test_vcvt_bf16_f32(float32x4_t a) {
  return vcvt_bf16_f32(a);
}

// CHECK-LABEL: test_vcvtq_low_bf16_f32
// CHECK-A64: call <8 x bfloat> @llvm.aarch64.neon.bfcvtn(<4 x float> %a)
// CHECK-A32-HARDFP: %[[CVT:.*]] = call <4 x bfloat> @llvm.arm.neon.vcvtfp2bf.v4bf16
// CHECK-A32-HARDFP: shufflevector <4 x bfloat> zeroinitializer, <4 x bfloat> %[[CVT]], <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
// CHECK-A32-SOFTFP: call <4 x i16> @llvm.arm.neon.vcvtfp2bf.v4i16
// CHECK-A32-SOFTFP: shufflevector <4 x bfloat> zeroinitializer, <4 x bfloat> %{{.*}}, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
bfloat16x8_t test_vcvtq_low_bf16_f32(float32x4_t a) {
  return vcvtq_low_bf16_f32(a);
}

// CHECK-LABEL: test_vcvtq_high_bf16_f32
// CHECK-A64: call <8 x bfloat> @llvm.aarch64.neon.bfcvtn2(<8 x bfloat> %inactive, <4 x float> %a)
// CHECK-A32-HARDFP: %[[CVT:.*]] = call <4 x bfloat> @llvm.arm.neon.vcvtfp2bf.v4bf16(<4 x float> %a)
// CHECK-A32-HARDFP: %[[INACT:.*]] = shufflevector <8 x bfloat> %inactive, <8 x bfloat> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
// CHECK-A32-HARDFP: shufflevector <4 x bfloat> %[[CVT]], <4 x bfloat> %[[INACT]], <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
// CHECK-A32-SOFTFP: call <4 x i16> @llvm.arm.neon.vcvtfp2bf.v4i16(<4 x float> %a)
// CHECK-A32-SOFTFP: shufflevector <8 x bfloat> %{{.*}}, <8 x bfloat> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
// CHECK-A32-SOFTFP: shufflevector <4 x bfloat> %{{.*}}, <4 x bfloat> %{{.*}}, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
bfloat16x8_t test_vcvtq_high_bf16_f32(bfloat16x8_t inactive, float32x4_t a) {
  return vcvtq_high_bf16_f32(inactive, a);
}

// CHECK-LABEL: test_vcvth_bf16_f32
// CHECK-A64: call bfloat @llvm.aarch64.neon.bfcvt(float %a)
// CHECK-A32-HARDFP: call bfloat @llvm.arm.neon.vcvtbfp2bf(float %a)
// CHECK-A32-SOFTFP: call bfloat @llvm.arm.neon.vcvtbfp2bf(float %a)
bfloat16_t test_vcvth_bf16_f32(float32_t a) {
  return vcvth_bf16_f32(a);
}

// CHECK-LABEL: test_vcvtah_f32_bf16
// CHECK: shl i32 %{{.*}}, 16
float32_t test_vcvtah_f32_bf16(bfloat16_t a) {
  return vcvtah_f32_bf16(a);
}


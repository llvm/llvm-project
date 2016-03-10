// RUN: %clang_cc1 -triple thumbv8-linux-gnueabihf -target-cpu cortex-a57 -ffreestanding -emit-llvm %s -o - | opt -S -mem2reg | FileCheck %s

#include <arm_neon.h>

// CHECK-LABEL: define <2 x float> @test_vmaxnm_f32(<2 x float> %a, <2 x float> %b) #0 {
// CHECK:   [[TMP0:%.*]] = bitcast <2 x float> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x float> %b to <8 x i8>
// CHECK:   [[VMAXNM_V_I:%.*]] = bitcast <8 x i8> [[TMP0]] to <2 x float>
// CHECK:   [[VMAXNM_V1_I:%.*]] = bitcast <8 x i8> [[TMP1]] to <2 x float>
// CHECK:   [[VMAXNM_V2_I:%.*]] = call <2 x float> @llvm.arm.neon.vmaxnm.v2f32(<2 x float> [[VMAXNM_V_I]], <2 x float> [[VMAXNM_V1_I]]) #2
// CHECK:   [[VMAXNM_V3_I:%.*]] = bitcast <2 x float> [[VMAXNM_V2_I]] to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <8 x i8> [[VMAXNM_V3_I]] to <2 x float>
// CHECK:   ret <2 x float> [[TMP2]]
float32x2_t test_vmaxnm_f32(float32x2_t a, float32x2_t b) {
  return vmaxnm_f32(a, b);
}

// CHECK-LABEL: define <4 x float> @test_vmaxnmq_f32(<4 x float> %a, <4 x float> %b) #0 {
// CHECK:   [[TMP0:%.*]] = bitcast <4 x float> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x float> %b to <16 x i8>
// CHECK:   [[VMAXNMQ_V_I:%.*]] = bitcast <16 x i8> [[TMP0]] to <4 x float>
// CHECK:   [[VMAXNMQ_V1_I:%.*]] = bitcast <16 x i8> [[TMP1]] to <4 x float>
// CHECK:   [[VMAXNMQ_V2_I:%.*]] = call <4 x float> @llvm.arm.neon.vmaxnm.v4f32(<4 x float> [[VMAXNMQ_V_I]], <4 x float> [[VMAXNMQ_V1_I]]) #2
// CHECK:   [[VMAXNMQ_V3_I:%.*]] = bitcast <4 x float> [[VMAXNMQ_V2_I]] to <16 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <16 x i8> [[VMAXNMQ_V3_I]] to <4 x float>
// CHECK:   ret <4 x float> [[TMP2]]
float32x4_t test_vmaxnmq_f32(float32x4_t a, float32x4_t b) {
  return vmaxnmq_f32(a, b);
}

// CHECK-LABEL: define <2 x float> @test_vminnm_f32(<2 x float> %a, <2 x float> %b) #0 {
// CHECK:   [[TMP0:%.*]] = bitcast <2 x float> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x float> %b to <8 x i8>
// CHECK:   [[VMINNM_V_I:%.*]] = bitcast <8 x i8> [[TMP0]] to <2 x float>
// CHECK:   [[VMINNM_V1_I:%.*]] = bitcast <8 x i8> [[TMP1]] to <2 x float>
// CHECK:   [[VMINNM_V2_I:%.*]] = call <2 x float> @llvm.arm.neon.vminnm.v2f32(<2 x float> [[VMINNM_V_I]], <2 x float> [[VMINNM_V1_I]]) #2
// CHECK:   [[VMINNM_V3_I:%.*]] = bitcast <2 x float> [[VMINNM_V2_I]] to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <8 x i8> [[VMINNM_V3_I]] to <2 x float>
// CHECK:   ret <2 x float> [[TMP2]]
float32x2_t test_vminnm_f32(float32x2_t a, float32x2_t b) {
  return vminnm_f32(a, b);
}

// CHECK-LABEL: define <4 x float> @test_vminnmq_f32(<4 x float> %a, <4 x float> %b) #0 {
// CHECK:   [[TMP0:%.*]] = bitcast <4 x float> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x float> %b to <16 x i8>
// CHECK:   [[VMINNMQ_V_I:%.*]] = bitcast <16 x i8> [[TMP0]] to <4 x float>
// CHECK:   [[VMINNMQ_V1_I:%.*]] = bitcast <16 x i8> [[TMP1]] to <4 x float>
// CHECK:   [[VMINNMQ_V2_I:%.*]] = call <4 x float> @llvm.arm.neon.vminnm.v4f32(<4 x float> [[VMINNMQ_V_I]], <4 x float> [[VMINNMQ_V1_I]]) #2
// CHECK:   [[VMINNMQ_V3_I:%.*]] = bitcast <4 x float> [[VMINNMQ_V2_I]] to <16 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <16 x i8> [[VMINNMQ_V3_I]] to <4 x float>
// CHECK:   ret <4 x float> [[TMP2]]
float32x4_t test_vminnmq_f32(float32x4_t a, float32x4_t b) {
  return vminnmq_f32(a, b);
}

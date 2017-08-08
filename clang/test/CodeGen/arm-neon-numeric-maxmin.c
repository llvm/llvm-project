// RUN: %clang_cc1 -triple thumbv8-linux-gnueabihf -target-cpu cortex-a57 -ffreestanding -disable-O0-optnone -emit-llvm %s -o - | opt -S -mem2reg | FileCheck %s

#include <arm_neon.h>

// CHECK-LABEL: define <2 x float> @test_vmaxnm_f32(<2 x float> %a, <2 x float> %b) #0 {
// CHECK:   [[VMAXNM_V2_I:%.*]] = call <2 x float> @llvm.arm.neon.vmaxnm.v2f32(<2 x float> %a, <2 x float> %b) #2
// CHECK:   ret <2 x float> [[VMAXNM_V2_I]]
float32x2_t test_vmaxnm_f32(float32x2_t a, float32x2_t b) {
  return vmaxnm_f32(a, b);
}

// CHECK-LABEL: define <4 x float> @test_vmaxnmq_f32(<4 x float> %a, <4 x float> %b) #0 {
// CHECK:   [[VMAXNMQ_V2_I:%.*]] = call <4 x float> @llvm.arm.neon.vmaxnm.v4f32(<4 x float> %a, <4 x float> %b) #2
// CHECK:   ret <4 x float> [[VMAXNMQ_V2_I]]
float32x4_t test_vmaxnmq_f32(float32x4_t a, float32x4_t b) {
  return vmaxnmq_f32(a, b);
}

// CHECK-LABEL: define <2 x float> @test_vminnm_f32(<2 x float> %a, <2 x float> %b) #0 {
// CHECK:   [[VMINNM_V2_I:%.*]] = call <2 x float> @llvm.arm.neon.vminnm.v2f32(<2 x float> %a, <2 x float> %b) #2
// CHECK:   ret <2 x float> [[VMINNM_V2_I]]
float32x2_t test_vminnm_f32(float32x2_t a, float32x2_t b) {
  return vminnm_f32(a, b);
}

// CHECK-LABEL: define <4 x float> @test_vminnmq_f32(<4 x float> %a, <4 x float> %b) #0 {
// CHECK:   [[VMINNMQ_V2_I:%.*]] = call <4 x float> @llvm.arm.neon.vminnm.v4f32(<4 x float> %a, <4 x float> %b) #2
// CHECK:   ret <4 x float> [[VMINNMQ_V2_I]]
float32x4_t test_vminnmq_f32(float32x4_t a, float32x4_t b) {
  return vminnmq_f32(a, b);
}

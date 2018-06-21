// RUN: %clang_cc1 -triple thumbv8-linux-gnueabihf -target-cpu cortex-a57 -ffreestanding -disable-O0-optnone -emit-llvm %s -o - | opt -S -mem2reg | FileCheck %s

#include <arm_neon.h>

// CHECK-LABEL: define <2 x float> @test_vrnda_f32(<2 x float> %a) #0 {
// CHECK:   [[VRNDA_V1_I:%.*]] = call <2 x float> @llvm.arm.neon.vrinta.v2f32(<2 x float> %a) #2
// CHECK:   ret <2 x float> [[VRNDA_V1_I]]
float32x2_t test_vrnda_f32(float32x2_t a) {
  return vrnda_f32(a);
}

// CHECK-LABEL: define <4 x float> @test_vrndaq_f32(<4 x float> %a) #0 {
// CHECK:   [[VRNDAQ_V1_I:%.*]] = call <4 x float> @llvm.arm.neon.vrinta.v4f32(<4 x float> %a) #2
// CHECK:   ret <4 x float> [[VRNDAQ_V1_I]]
float32x4_t test_vrndaq_f32(float32x4_t a) {
  return vrndaq_f32(a);
}

// CHECK-LABEL: define <2 x float> @test_vrndm_f32(<2 x float> %a) #0 {
// CHECK:   [[VRNDM_V1_I:%.*]] = call <2 x float> @llvm.arm.neon.vrintm.v2f32(<2 x float> %a) #2
// CHECK:   ret <2 x float> [[VRNDM_V1_I]]
float32x2_t test_vrndm_f32(float32x2_t a) {
  return vrndm_f32(a);
}

// CHECK-LABEL: define <4 x float> @test_vrndmq_f32(<4 x float> %a) #0 {
// CHECK:   [[VRNDMQ_V1_I:%.*]] = call <4 x float> @llvm.arm.neon.vrintm.v4f32(<4 x float> %a) #2
// CHECK:   ret <4 x float> [[VRNDMQ_V1_I]]
float32x4_t test_vrndmq_f32(float32x4_t a) {
  return vrndmq_f32(a);
}

// CHECK-LABEL: define <2 x float> @test_vrndn_f32(<2 x float> %a) #0 {
// CHECK:   [[VRNDN_V1_I:%.*]] = call <2 x float> @llvm.arm.neon.vrintn.v2f32(<2 x float> %a) #2
// CHECK:   ret <2 x float> [[VRNDN_V1_I]]
float32x2_t test_vrndn_f32(float32x2_t a) {
  return vrndn_f32(a);
}

// CHECK-LABEL: define <4 x float> @test_vrndnq_f32(<4 x float> %a) #0 {
// CHECK:   [[VRNDNQ_V1_I:%.*]] = call <4 x float> @llvm.arm.neon.vrintn.v4f32(<4 x float> %a) #2
// CHECK:   ret <4 x float> [[VRNDNQ_V1_I]]
float32x4_t test_vrndnq_f32(float32x4_t a) {
  return vrndnq_f32(a);
}

// CHECK-LABEL: define <2 x float> @test_vrndp_f32(<2 x float> %a) #0 {
// CHECK:   [[VRNDP_V1_I:%.*]] = call <2 x float> @llvm.arm.neon.vrintp.v2f32(<2 x float> %a) #2
// CHECK:   ret <2 x float> [[VRNDP_V1_I]]
float32x2_t test_vrndp_f32(float32x2_t a) {
  return vrndp_f32(a);
}

// CHECK-LABEL: define <4 x float> @test_vrndpq_f32(<4 x float> %a) #0 {
// CHECK:   [[VRNDPQ_V1_I:%.*]] = call <4 x float> @llvm.arm.neon.vrintp.v4f32(<4 x float> %a) #2
// CHECK:   ret <4 x float> [[VRNDPQ_V1_I]]
float32x4_t test_vrndpq_f32(float32x4_t a) {
  return vrndpq_f32(a);
}

// CHECK-LABEL: define <2 x float> @test_vrndx_f32(<2 x float> %a) #0 {
// CHECK:   [[VRNDX_V1_I:%.*]] = call <2 x float> @llvm.arm.neon.vrintx.v2f32(<2 x float> %a) #2
// CHECK:   ret <2 x float> [[VRNDX_V1_I]]
float32x2_t test_vrndx_f32(float32x2_t a) {
  return vrndx_f32(a);
}

// CHECK-LABEL: define <4 x float> @test_vrndxq_f32(<4 x float> %a) #0 {
// CHECK:   [[VRNDXQ_V1_I:%.*]] = call <4 x float> @llvm.arm.neon.vrintx.v4f32(<4 x float> %a) #2
// CHECK:   ret <4 x float> [[VRNDXQ_V1_I]]
float32x4_t test_vrndxq_f32(float32x4_t a) {
  return vrndxq_f32(a);
}

// CHECK-LABEL: define <2 x float> @test_vrnd_f32(<2 x float> %a) #0 {
// CHECK:   [[VRND_V1_I:%.*]] = call <2 x float> @llvm.arm.neon.vrintz.v2f32(<2 x float> %a) #2
// CHECK:   ret <2 x float> [[VRND_V1_I]]
float32x2_t test_vrnd_f32(float32x2_t a) {
  return vrnd_f32(a);
}

// CHECK-LABEL: define <4 x float> @test_vrndq_f32(<4 x float> %a) #0 {
// CHECK:   [[VRNDQ_V1_I:%.*]] = call <4 x float> @llvm.arm.neon.vrintz.v4f32(<4 x float> %a) #2
// CHECK:   ret <4 x float> [[VRNDQ_V1_I]]
float32x4_t test_vrndq_f32(float32x4_t a) {
  return vrndq_f32(a);
}

// RUN: %clang_cc1 -triple thumbv8-linux-gnueabihf -target-cpu cortex-a57 -ffreestanding -emit-llvm %s -o - | opt -S -mem2reg | FileCheck %s

#include <arm_neon.h>

// CHECK-LABEL: define <2 x float> @test_vrnda_f32(<2 x float> %a) #0 {
// CHECK:   [[TMP0:%.*]] = bitcast <2 x float> %a to <8 x i8>
// CHECK:   [[VRNDA_V_I:%.*]] = bitcast <8 x i8> [[TMP0]] to <2 x float>
// CHECK:   [[VRNDA_V1_I:%.*]] = call <2 x float> @llvm.arm.neon.vrinta.v2f32(<2 x float> [[VRNDA_V_I]]) #2
// CHECK:   [[VRNDA_V2_I:%.*]] = bitcast <2 x float> [[VRNDA_V1_I]] to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i8> [[VRNDA_V2_I]] to <2 x float>
// CHECK:   ret <2 x float> [[TMP1]]
float32x2_t test_vrnda_f32(float32x2_t a) {
  return vrnda_f32(a);
}

// CHECK-LABEL: define <4 x float> @test_vrndaq_f32(<4 x float> %a) #0 {
// CHECK:   [[TMP0:%.*]] = bitcast <4 x float> %a to <16 x i8>
// CHECK:   [[VRNDAQ_V_I:%.*]] = bitcast <16 x i8> [[TMP0]] to <4 x float>
// CHECK:   [[VRNDAQ_V1_I:%.*]] = call <4 x float> @llvm.arm.neon.vrinta.v4f32(<4 x float> [[VRNDAQ_V_I]]) #2
// CHECK:   [[VRNDAQ_V2_I:%.*]] = bitcast <4 x float> [[VRNDAQ_V1_I]] to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <16 x i8> [[VRNDAQ_V2_I]] to <4 x float>
// CHECK:   ret <4 x float> [[TMP1]]
float32x4_t test_vrndaq_f32(float32x4_t a) {
  return vrndaq_f32(a);
}

// CHECK-LABEL: define <2 x float> @test_vrndm_f32(<2 x float> %a) #0 {
// CHECK:   [[TMP0:%.*]] = bitcast <2 x float> %a to <8 x i8>
// CHECK:   [[VRNDM_V_I:%.*]] = bitcast <8 x i8> [[TMP0]] to <2 x float>
// CHECK:   [[VRNDM_V1_I:%.*]] = call <2 x float> @llvm.arm.neon.vrintm.v2f32(<2 x float> [[VRNDM_V_I]]) #2
// CHECK:   [[VRNDM_V2_I:%.*]] = bitcast <2 x float> [[VRNDM_V1_I]] to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i8> [[VRNDM_V2_I]] to <2 x float>
// CHECK:   ret <2 x float> [[TMP1]]
float32x2_t test_vrndm_f32(float32x2_t a) {
  return vrndm_f32(a);
}

// CHECK-LABEL: define <4 x float> @test_vrndmq_f32(<4 x float> %a) #0 {
// CHECK:   [[TMP0:%.*]] = bitcast <4 x float> %a to <16 x i8>
// CHECK:   [[VRNDMQ_V_I:%.*]] = bitcast <16 x i8> [[TMP0]] to <4 x float>
// CHECK:   [[VRNDMQ_V1_I:%.*]] = call <4 x float> @llvm.arm.neon.vrintm.v4f32(<4 x float> [[VRNDMQ_V_I]]) #2
// CHECK:   [[VRNDMQ_V2_I:%.*]] = bitcast <4 x float> [[VRNDMQ_V1_I]] to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <16 x i8> [[VRNDMQ_V2_I]] to <4 x float>
// CHECK:   ret <4 x float> [[TMP1]]
float32x4_t test_vrndmq_f32(float32x4_t a) {
  return vrndmq_f32(a);
}

// CHECK-LABEL: define <2 x float> @test_vrndn_f32(<2 x float> %a) #0 {
// CHECK:   [[TMP0:%.*]] = bitcast <2 x float> %a to <8 x i8>
// CHECK:   [[VRNDN_V_I:%.*]] = bitcast <8 x i8> [[TMP0]] to <2 x float>
// CHECK:   [[VRNDN_V1_I:%.*]] = call <2 x float> @llvm.arm.neon.vrintn.v2f32(<2 x float> [[VRNDN_V_I]]) #2
// CHECK:   [[VRNDN_V2_I:%.*]] = bitcast <2 x float> [[VRNDN_V1_I]] to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i8> [[VRNDN_V2_I]] to <2 x float>
// CHECK:   ret <2 x float> [[TMP1]]
float32x2_t test_vrndn_f32(float32x2_t a) {
  return vrndn_f32(a);
}

// CHECK-LABEL: define <4 x float> @test_vrndnq_f32(<4 x float> %a) #0 {
// CHECK:   [[TMP0:%.*]] = bitcast <4 x float> %a to <16 x i8>
// CHECK:   [[VRNDNQ_V_I:%.*]] = bitcast <16 x i8> [[TMP0]] to <4 x float>
// CHECK:   [[VRNDNQ_V1_I:%.*]] = call <4 x float> @llvm.arm.neon.vrintn.v4f32(<4 x float> [[VRNDNQ_V_I]]) #2
// CHECK:   [[VRNDNQ_V2_I:%.*]] = bitcast <4 x float> [[VRNDNQ_V1_I]] to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <16 x i8> [[VRNDNQ_V2_I]] to <4 x float>
// CHECK:   ret <4 x float> [[TMP1]]
float32x4_t test_vrndnq_f32(float32x4_t a) {
  return vrndnq_f32(a);
}

// CHECK-LABEL: define <2 x float> @test_vrndp_f32(<2 x float> %a) #0 {
// CHECK:   [[TMP0:%.*]] = bitcast <2 x float> %a to <8 x i8>
// CHECK:   [[VRNDP_V_I:%.*]] = bitcast <8 x i8> [[TMP0]] to <2 x float>
// CHECK:   [[VRNDP_V1_I:%.*]] = call <2 x float> @llvm.arm.neon.vrintp.v2f32(<2 x float> [[VRNDP_V_I]]) #2
// CHECK:   [[VRNDP_V2_I:%.*]] = bitcast <2 x float> [[VRNDP_V1_I]] to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i8> [[VRNDP_V2_I]] to <2 x float>
// CHECK:   ret <2 x float> [[TMP1]]
float32x2_t test_vrndp_f32(float32x2_t a) {
  return vrndp_f32(a);
}

// CHECK-LABEL: define <4 x float> @test_vrndpq_f32(<4 x float> %a) #0 {
// CHECK:   [[TMP0:%.*]] = bitcast <4 x float> %a to <16 x i8>
// CHECK:   [[VRNDPQ_V_I:%.*]] = bitcast <16 x i8> [[TMP0]] to <4 x float>
// CHECK:   [[VRNDPQ_V1_I:%.*]] = call <4 x float> @llvm.arm.neon.vrintp.v4f32(<4 x float> [[VRNDPQ_V_I]]) #2
// CHECK:   [[VRNDPQ_V2_I:%.*]] = bitcast <4 x float> [[VRNDPQ_V1_I]] to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <16 x i8> [[VRNDPQ_V2_I]] to <4 x float>
// CHECK:   ret <4 x float> [[TMP1]]
float32x4_t test_vrndpq_f32(float32x4_t a) {
  return vrndpq_f32(a);
}

// CHECK-LABEL: define <2 x float> @test_vrndx_f32(<2 x float> %a) #0 {
// CHECK:   [[TMP0:%.*]] = bitcast <2 x float> %a to <8 x i8>
// CHECK:   [[VRNDX_V_I:%.*]] = bitcast <8 x i8> [[TMP0]] to <2 x float>
// CHECK:   [[VRNDX_V1_I:%.*]] = call <2 x float> @llvm.arm.neon.vrintx.v2f32(<2 x float> [[VRNDX_V_I]]) #2
// CHECK:   [[VRNDX_V2_I:%.*]] = bitcast <2 x float> [[VRNDX_V1_I]] to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i8> [[VRNDX_V2_I]] to <2 x float>
// CHECK:   ret <2 x float> [[TMP1]]
float32x2_t test_vrndx_f32(float32x2_t a) {
  return vrndx_f32(a);
}

// CHECK-LABEL: define <4 x float> @test_vrndxq_f32(<4 x float> %a) #0 {
// CHECK:   [[TMP0:%.*]] = bitcast <4 x float> %a to <16 x i8>
// CHECK:   [[VRNDXQ_V_I:%.*]] = bitcast <16 x i8> [[TMP0]] to <4 x float>
// CHECK:   [[VRNDXQ_V1_I:%.*]] = call <4 x float> @llvm.arm.neon.vrintx.v4f32(<4 x float> [[VRNDXQ_V_I]]) #2
// CHECK:   [[VRNDXQ_V2_I:%.*]] = bitcast <4 x float> [[VRNDXQ_V1_I]] to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <16 x i8> [[VRNDXQ_V2_I]] to <4 x float>
// CHECK:   ret <4 x float> [[TMP1]]
float32x4_t test_vrndxq_f32(float32x4_t a) {
  return vrndxq_f32(a);
}

// CHECK-LABEL: define <2 x float> @test_vrnd_f32(<2 x float> %a) #0 {
// CHECK:   [[TMP0:%.*]] = bitcast <2 x float> %a to <8 x i8>
// CHECK:   [[VRND_V_I:%.*]] = bitcast <8 x i8> [[TMP0]] to <2 x float>
// CHECK:   [[VRND_V1_I:%.*]] = call <2 x float> @llvm.arm.neon.vrintz.v2f32(<2 x float> [[VRND_V_I]]) #2
// CHECK:   [[VRND_V2_I:%.*]] = bitcast <2 x float> [[VRND_V1_I]] to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i8> [[VRND_V2_I]] to <2 x float>
// CHECK:   ret <2 x float> [[TMP1]]
float32x2_t test_vrnd_f32(float32x2_t a) {
  return vrnd_f32(a);
}

// CHECK-LABEL: define <4 x float> @test_vrndq_f32(<4 x float> %a) #0 {
// CHECK:   [[TMP0:%.*]] = bitcast <4 x float> %a to <16 x i8>
// CHECK:   [[VRNDQ_V_I:%.*]] = bitcast <16 x i8> [[TMP0]] to <4 x float>
// CHECK:   [[VRNDQ_V1_I:%.*]] = call <4 x float> @llvm.arm.neon.vrintz.v4f32(<4 x float> [[VRNDQ_V_I]]) #2
// CHECK:   [[VRNDQ_V2_I:%.*]] = bitcast <4 x float> [[VRNDQ_V1_I]] to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <16 x i8> [[VRNDQ_V2_I]] to <4 x float>
// CHECK:   ret <4 x float> [[TMP1]]
float32x4_t test_vrndq_f32(float32x4_t a) {
  return vrndq_f32(a);
}

// RUN: %clang_cc1 -triple thumbv8-linux-gnueabihf -target-cpu cortex-a57 \
// RUN:     -ffreestanding -disable-O0-optnone -emit-llvm %s -o - | \
// RUN:     opt -S -mem2reg | FileCheck -check-prefixes=CHECK,CHECK-A32 %s
// RUN: %clang_cc1 -triple arm64-linux-gnueabihf -target-feature +neon \
// RUN:     -ffreestanding -disable-O0-optnone -emit-llvm %s -o - | \
// RUN:     opt -S -mem2reg | FileCheck -check-prefixes=CHECK,CHECK-A64 %s

#include <arm_neon.h>

// CHECK-LABEL: define{{.*}} <2 x float> @test_vrnda_f32(<2 x float> noundef %a)
// CHECK-A32: [[VRNDA_V1_I:%.*]] = call <2 x float> @llvm.arm.neon.vrinta.v2f32(<2 x float> %a)
// CHECK-A64: [[VRNDA_V1_I:%.*]] = call <2 x float> @llvm.round.v2f32(<2 x float> %a)
// CHECK: ret <2 x float> [[VRNDA_V1_I]]
float32x2_t test_vrnda_f32(float32x2_t a) {
  return vrnda_f32(a);
}

// CHECK-LABEL: define{{.*}} <4 x float> @test_vrndaq_f32(<4 x float> noundef %a)
// CHECK-A32: [[VRNDAQ_V1_I:%.*]] = call <4 x float> @llvm.arm.neon.vrinta.v4f32(<4 x float> %a)
// CHECK-A64: [[VRNDAQ_V1_I:%.*]] = call <4 x float> @llvm.round.v4f32(<4 x float> %a)
// CHECK: ret <4 x float> [[VRNDAQ_V1_I]]
float32x4_t test_vrndaq_f32(float32x4_t a) {
  return vrndaq_f32(a);
}

// CHECK-LABEL: define{{.*}} <2 x float> @test_vrndm_f32(<2 x float> noundef %a)
// CHECK-A32: [[VRNDM_V1_I:%.*]] = call <2 x float> @llvm.arm.neon.vrintm.v2f32(<2 x float> %a)
// CHECK-A64: [[VRNDM_V1_I:%.*]] = call <2 x float> @llvm.floor.v2f32(<2 x float> %a)
// CHECK: ret <2 x float> [[VRNDM_V1_I]]
float32x2_t test_vrndm_f32(float32x2_t a) {
  return vrndm_f32(a);
}

// CHECK-LABEL: define{{.*}} <4 x float> @test_vrndmq_f32(<4 x float> noundef %a)
// CHECK-A32: [[VRNDMQ_V1_I:%.*]] = call <4 x float> @llvm.arm.neon.vrintm.v4f32(<4 x float> %a)
// CHECK-A64: [[VRNDMQ_V1_I:%.*]] = call <4 x float> @llvm.floor.v4f32(<4 x float> %a)
// CHECK: ret <4 x float> [[VRNDMQ_V1_I]]
float32x4_t test_vrndmq_f32(float32x4_t a) {
  return vrndmq_f32(a);
}

// CHECK-LABEL: define{{.*}} <2 x float> @test_vrndn_f32(<2 x float> noundef %a)
// CHECK-A32: [[VRNDN_V1_I:%.*]] = call <2 x float> @llvm.arm.neon.vrintn.v2f32(<2 x float> %a)
// CHECK-A64: [[VRNDN_V1_I:%.*]] = call <2 x float> @llvm.roundeven.v2f32(<2 x float> %a)
// CHECK: ret <2 x float> [[VRNDN_V1_I]]
float32x2_t test_vrndn_f32(float32x2_t a) {
  return vrndn_f32(a);
}

// CHECK-LABEL: define{{.*}} <4 x float> @test_vrndnq_f32(<4 x float> noundef %a)
// CHECK-A32: [[VRNDNQ_V1_I:%.*]] = call <4 x float> @llvm.arm.neon.vrintn.v4f32(<4 x float> %a)
// CHECK-A64: [[VRNDNQ_V1_I:%.*]] = call <4 x float> @llvm.roundeven.v4f32(<4 x float> %a)
// CHECK: ret <4 x float> [[VRNDNQ_V1_I]]
float32x4_t test_vrndnq_f32(float32x4_t a) {
  return vrndnq_f32(a);
}

// CHECK-LABEL: define{{.*}} <2 x float> @test_vrndp_f32(<2 x float> noundef %a)
// CHECK-A32: [[VRNDP_V1_I:%.*]] = call <2 x float> @llvm.arm.neon.vrintp.v2f32(<2 x float> %a)
// CHECK-A64: [[VRNDP_V1_I:%.*]] = call <2 x float> @llvm.ceil.v2f32(<2 x float> %a)
// CHECK: ret <2 x float> [[VRNDP_V1_I]]
float32x2_t test_vrndp_f32(float32x2_t a) {
  return vrndp_f32(a);
}

// CHECK-LABEL: define{{.*}} <4 x float> @test_vrndpq_f32(<4 x float> noundef %a)
// CHECK-A32: [[VRNDPQ_V1_I:%.*]] = call <4 x float> @llvm.arm.neon.vrintp.v4f32(<4 x float> %a)
// CHECK-A64: [[VRNDPQ_V1_I:%.*]] = call <4 x float> @llvm.ceil.v4f32(<4 x float> %a)
// CHECK: ret <4 x float> [[VRNDPQ_V1_I]]
float32x4_t test_vrndpq_f32(float32x4_t a) {
  return vrndpq_f32(a);
}

// CHECK-LABEL: define{{.*}} <2 x float> @test_vrndx_f32(<2 x float> noundef %a)
// CHECK-A32: [[VRNDX_V1_I:%.*]] = call <2 x float> @llvm.arm.neon.vrintx.v2f32(<2 x float> %a)
// CHECK-A64: [[VRNDX_V1_I:%.*]] = call <2 x float> @llvm.rint.v2f32(<2 x float> %a)
// CHECK: ret <2 x float> [[VRNDX_V1_I]]
float32x2_t test_vrndx_f32(float32x2_t a) {
  return vrndx_f32(a);
}

// CHECK-LABEL: define{{.*}} <4 x float> @test_vrndxq_f32(<4 x float> noundef %a)
// CHECK-A32: [[VRNDXQ_V1_I:%.*]] = call <4 x float> @llvm.arm.neon.vrintx.v4f32(<4 x float> %a)
// CHECK-A64: [[VRNDXQ_V1_I:%.*]] = call <4 x float> @llvm.rint.v4f32(<4 x float> %a)
// CHECK: ret <4 x float> [[VRNDXQ_V1_I]]
float32x4_t test_vrndxq_f32(float32x4_t a) {
  return vrndxq_f32(a);
}

// CHECK-LABEL: define{{.*}} <2 x float> @test_vrnd_f32(<2 x float> noundef %a)
// CHECK-A32: [[VRND_V1_I:%.*]] = call <2 x float> @llvm.arm.neon.vrintz.v2f32(<2 x float> %a)
// CHECK-A64: [[VRND_V1_I:%.*]] = call <2 x float> @llvm.trunc.v2f32(<2 x float> %a)
// CHECK: ret <2 x float> [[VRND_V1_I]]
float32x2_t test_vrnd_f32(float32x2_t a) {
  return vrnd_f32(a);
}

// CHECK-LABEL: define{{.*}} <4 x float> @test_vrndq_f32(<4 x float> noundef %a)
// CHECK-A32: [[VRNDQ_V1_I:%.*]] = call <4 x float> @llvm.arm.neon.vrintz.v4f32(<4 x float> %a)
// CHECK-A64: [[VRNDQ_V1_I:%.*]] = call <4 x float> @llvm.trunc.v4f32(<4 x float> %a)
// CHECK: ret <4 x float> [[VRNDQ_V1_I]]
float32x4_t test_vrndq_f32(float32x4_t a) {
  return vrndq_f32(a);
}

// CHECK-LABEL: define{{.*}} float @test_vrndns_f32(float noundef %a)
// CHECK-A32: [[VRNDN_I:%.*]] = call float @llvm.arm.neon.vrintn.f32(float %a)
// CHECK-A64: [[VRNDN_I:%.*]] = call float @llvm.roundeven.f32(float %a)
// CHECK: ret float [[VRNDN_I]]
float32_t test_vrndns_f32(float32_t a) {
  return vrndns_f32(a);
}

// CHECK-LABEL: define{{.*}} <2 x float> @test_vrndi_f32(<2 x float> noundef %a)
// CHECK: [[TMP0:%.*]] = bitcast <2 x float> %a to <8 x i8>
// CHECK: [[VRNDI1_I:%.*]] = call <2 x float> @llvm.nearbyint.v2f32(<2 x float> %a)
// CHECK: ret <2 x float> [[VRNDI1_I]]
float32x2_t test_vrndi_f32(float32x2_t a) {
  return vrndi_f32(a);
}

// CHECK-LABEL: define{{.*}} <4 x float> @test_vrndiq_f32(<4 x float> noundef %a)
// CHECK: [[TMP0:%.*]] = bitcast <4 x float> %a to <16 x i8>
// CHECK: [[VRNDI1_I:%.*]] = call <4 x float> @llvm.nearbyint.v4f32(<4 x float> %a)
// CHECK: ret <4 x float> [[VRNDI1_I]]
float32x4_t test_vrndiq_f32(float32x4_t a) {
  return vrndiq_f32(a);
}

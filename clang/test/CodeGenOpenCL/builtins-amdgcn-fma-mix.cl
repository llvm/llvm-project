// RUN: %clang_cc1 -cl-std=CL2.0 -triple amdgcn-amd-amdhsa -target-cpu gfx1250 -emit-llvm -o - %s | FileCheck %s

// REQUIRES: amdgpu-registered-target

typedef unsigned int uint;

// CHECK-LABEL: define{{.*}} float @test_fma_mix_f32(
// CHECK: call float @llvm.amdgcn.fma.mix.f32(i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 2, i32 2, i32 2)
float test_fma_mix_f32(uint src0, uint src1, uint src2) {
  return __builtin_amdgcn_fma_mix_f32(src0, src1, src2, 2, 2, 2);
}

// CHECK-LABEL: define{{.*}} float @test_fma_mix_f32_bf16(
// CHECK: call float @llvm.amdgcn.fma.mix.f32.bf16(i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 3, i32 2, i32 2)
float test_fma_mix_f32_bf16(uint src0, uint src1, uint src2) {
  return __builtin_amdgcn_fma_mix_f32_bf16(src0, src1, src2, 3, 2, 2);
}

// CHECK-LABEL: define{{.*}} i32 @test_fma_mixlo_f16(
// CHECK: call i32 @llvm.amdgcn.fma.mixlo.f16(i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 2, i32 2, i32 2)
uint test_fma_mixlo_f16(uint src0, uint src1, uint src2, uint dst) {
  return __builtin_amdgcn_fma_mixlo_f16(src0, src1, src2, dst, 2, 2, 2);
}

// CHECK-LABEL: define{{.*}} i32 @test_fma_mixhi_f16(
// CHECK: call i32 @llvm.amdgcn.fma.mixhi.f16(i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 2, i32 2, i32 2)
uint test_fma_mixhi_f16(uint src0, uint src1, uint src2, uint dst) {
  return __builtin_amdgcn_fma_mixhi_f16(src0, src1, src2, dst, 2, 2, 2);
}

// CHECK-LABEL: define{{.*}} i32 @test_fma_mixlo_bf16(
// CHECK: call i32 @llvm.amdgcn.fma.mixlo.bf16(i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 2, i32 2, i32 2)
uint test_fma_mixlo_bf16(uint src0, uint src1, uint src2, uint dst) {
  return __builtin_amdgcn_fma_mixlo_bf16(src0, src1, src2, dst, 2, 2, 2);
}

// CHECK-LABEL: define{{.*}} i32 @test_fma_mixhi_bf16(
// CHECK: call i32 @llvm.amdgcn.fma.mixhi.bf16(i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 2, i32 2, i32 2)
uint test_fma_mixhi_bf16(uint src0, uint src1, uint src2, uint dst) {
  return __builtin_amdgcn_fma_mixhi_bf16(src0, src1, src2, dst, 2, 2, 2);
}

// RUN: %clang_cc1 -triple arm64-none-linux-gnu -target-feature +fullfp16 -target-feature +v8.5a \
// RUN:    -fclangir -disable-O0-optnone \
// RUN:  -flax-vector-conversions=none -emit-cir -o %t.cir %s
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

// RUN: %clang_cc1 -triple arm64-none-linux-gnu -target-feature +fullfp16 -target-feature +v8.5a \
// RUN:    -fclangir -disable-O0-optnone \
// RUN:  -flax-vector-conversions=none -emit-llvm -fno-clangir-call-conv-lowering -o - %s \
// RUN: | opt -S -passes=mem2reg,simplifycfg -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// REQUIRES: aarch64-registered-target || arm-registered-target

// This test mimics clang/test/CodeGen/AArch64/v8.2a-neon-frint3264-intrinsics.c, which eventually
// CIR shall be able to support fully. Since this is going to take some time to converge,
// the unsupported/NYI code is commented out, so that we can incrementally improve this.
// The NYI filecheck used contains the LLVM output from OG codegen that should guide the
// correct result when implementing this into the CIR pipeline.

#include <arm_neon.h>

float32x2_t test_vrnd32x_f32(float32x2_t a) {
  return vrnd32x_f32(a);

  // CIR-LABEL: vrnd32x_f32
  // CIR: [[TMP0:%.*]] = cir.llvm.intrinsic "aarch64.neon.frint32x" {{.*}} : (!cir.vector<!cir.float x 2>) -> !cir.vector<!cir.float x 2>

  // LLVM-LABEL: @test_vrnd32x_f32
  // LLVM:  [[RND:%.*]] =  call <2 x float> @llvm.aarch64.neon.frint32x.v2f32(<2 x float> %0)
  // LLVM:  ret <2 x float> [[RND]]
}


float32x4_t test_vrnd32xq_f32(float32x4_t a) {
  return vrnd32xq_f32(a);

  // CIR-LABEL: vrnd32xq_f32
  // CIR: [[TMP0:%.*]] = cir.llvm.intrinsic "aarch64.neon.frint32x" {{.*}} : (!cir.vector<!cir.float x 4>) -> !cir.vector<!cir.float x 4>

  // LLVM-LABEL: @test_vrnd32xq_f32
  // LLVM:  [[RND:%.*]] =  call <4 x float> @llvm.aarch64.neon.frint32x.v4f32(<4 x float> %0)
  // LLVM:  ret <4 x float> [[RND]]
}

float32x2_t test_vrnd32z_f32(float32x2_t a) {
  return vrnd32z_f32(a);

  // CIR-LABEL: vrnd32z_f32
  // CIR: [[TMP0:%.*]] = cir.llvm.intrinsic "aarch64.neon.frint32z" {{.*}} : (!cir.vector<!cir.float x 2>) -> !cir.vector<!cir.float x 2>

  // LLVM-LABEL: @test_vrnd32z_f32
  // LLVM:  [[RND:%.*]] =  call <2 x float> @llvm.aarch64.neon.frint32z.v2f32(<2 x float> %0)
  // LLVM:  ret <2 x float> [[RND]]
}

float32x4_t test_vrnd32zq_f32(float32x4_t a) {
  return vrnd32zq_f32(a);

  // CIR-LABEL: vrnd32zq_f32
  // CIR: [[TMP0:%.*]] = cir.llvm.intrinsic "aarch64.neon.frint32z" {{.*}} : (!cir.vector<!cir.float x 4>) -> !cir.vector<!cir.float x 4>

  // LLVM-LABEL: @test_vrnd32zq_f32
  // LLVM:  [[RND:%.*]] =  call <4 x float> @llvm.aarch64.neon.frint32z.v4f32(<4 x float> %0)
  // LLVM:  ret <4 x float> [[RND]]
}

// CHECK-LABEL: test_vrnd64x_f32
// CHECK:  [[RND:%.*]] =  call <2 x float> @llvm.aarch64.neon.frint64x.v2f32(<2 x float> %a)
// CHECK:  ret <2 x float> [[RND]]
// float32x2_t test_vrnd64x_f32(float32x2_t a) {
//   return vrnd64x_f32(a);
// }

// CHECK-LABEL: test_vrnd64xq_f32
// CHECK:  [[RND:%.*]] =  call <4 x float> @llvm.aarch64.neon.frint64x.v4f32(<4 x float> %a)
// CHECK:  ret <4 x float> [[RND]]
// float32x4_t test_vrnd64xq_f32(float32x4_t a) {
//   return vrnd64xq_f32(a);
// }

// CHECK-LABEL: test_vrnd64z_f32
// CHECK:  [[RND:%.*]] =  call <2 x float> @llvm.aarch64.neon.frint64z.v2f32(<2 x float> %a)
// CHECK:  ret <2 x float> [[RND]]
// float32x2_t test_vrnd64z_f32(float32x2_t a) {
//   return vrnd64z_f32(a);
// }

// CHECK-LABEL: test_vrnd64zq_f32
// CHECK:  [[RND:%.*]] =  call <4 x float> @llvm.aarch64.neon.frint64z.v4f32(<4 x float> %a)
// CHECK:  ret <4 x float> [[RND]]
// float32x4_t test_vrnd64zq_f32(float32x4_t a) {
//   return vrnd64zq_f32(a);
// }

float64x1_t test_vrnd32x_f64(float64x1_t a) {
  return vrnd32x_f64(a);

  // CIR-LABEL: vrnd32x_f64
  // CIR: [[TMP0:%.*]] = cir.llvm.intrinsic "aarch64.neon.frint32x" {{.*}} : (!cir.vector<!cir.double x 1>) -> !cir.vector<!cir.double x 1>

  // LLVM-LABEL: @test_vrnd32x_f64
  // LLVM:  [[RND:%.*]] =  call <1 x double> @llvm.aarch64.neon.frint32x.v1f64(<1 x double> %0)
  // LLVM:  ret <1 x double> [[RND]]
}


float64x2_t test_vrnd32xq_f64(float64x2_t a) {
  return vrnd32xq_f64(a);

  // CIR-LABEL: vrnd32xq_f64
  // CIR: [[TMP0:%.*]] = cir.llvm.intrinsic "aarch64.neon.frint32x" {{.*}} : (!cir.vector<!cir.double x 2>) -> !cir.vector<!cir.double x 2>

  // LLVM-LABEL: @test_vrnd32xq_f64
  // LLVM:  [[RND:%.*]] =  call <2 x double> @llvm.aarch64.neon.frint32x.v2f64(<2 x double> %0)
  // LLVM:  ret <2 x double> [[RND]]
}

float64x1_t test_vrnd32z_f64(float64x1_t a) {
  return vrnd32z_f64(a);

  // CIR-LABEL: vrnd32z_f64
  // CIR: [[TMP0:%.*]] = cir.llvm.intrinsic "aarch64.neon.frint32z" {{.*}} : (!cir.vector<!cir.double x 1>) -> !cir.vector<!cir.double x 1>

  // LLVM-LABEL: @test_vrnd32z_f64
  // LLVM:  [[RND:%.*]] =  call <1 x double> @llvm.aarch64.neon.frint32z.v1f64(<1 x double> %0)
  // LLVM:  ret <1 x double> [[RND]]
}

float64x2_t test_vrnd32zq_f64(float64x2_t a) {
  return vrnd32zq_f64(a);

  // CIR-LABEL: vrnd32zq_f64
  // CIR: [[TMP0:%.*]] = cir.llvm.intrinsic "aarch64.neon.frint32z" {{.*}} : (!cir.vector<!cir.double x 2>) -> !cir.vector<!cir.double x 2>

  // LLVM-LABEL: @test_vrnd32zq_f64
  // LLVM:  [[RND:%.*]] =  call <2 x double> @llvm.aarch64.neon.frint32z.v2f64(<2 x double> %0)
  // LLVM:  ret <2 x double> [[RND]]
}

// CHECK-LABEL: test_vrnd64x_f64
// CHECK:  [[RND:%.*]] =  call <1 x double> @llvm.aarch64.neon.frint64x.v1f64(<1 x double> %a)
// CHECK:  ret <1 x double> [[RND]]
// float64x1_t test_vrnd64x_f64(float64x1_t a) {
//   return vrnd64x_f64(a);
// }

// CHECK-LABEL: test_vrnd64xq_f64
// CHECK:  [[RND:%.*]] =  call <2 x double> @llvm.aarch64.neon.frint64x.v2f64(<2 x double> %a)
// CHECK:  ret <2 x double> [[RND]]
// float64x2_t test_vrnd64xq_f64(float64x2_t a) {
//   return vrnd64xq_f64(a);
// }

// CHECK-LABEL: test_vrnd64z_f64
// CHECK:  [[RND:%.*]] =  call <1 x double> @llvm.aarch64.neon.frint64z.v1f64(<1 x double> %a)
// CHECK:  ret <1 x double> [[RND]]
// float64x1_t test_vrnd64z_f64(float64x1_t a) {
//   return vrnd64z_f64(a);
// }

// CHECK-LABEL: test_vrnd64zq_f64
// CHECK:  [[RND:%.*]] =  call <2 x double> @llvm.aarch64.neon.frint64z.v2f64(<2 x double> %a)
// CHECK:  ret <2 x double> [[RND]]
// float64x2_t test_vrnd64zq_f64(float64x2_t a) {
//   return vrnd64zq_f64(a);
// }

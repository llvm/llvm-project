// RUN: %clang_cc1 -triple arm64-none-linux-gnu -target-feature +neon -target-feature +v8.5a\
// RUN: -flax-vector-conversions=none -S -disable-O0-optnone -emit-llvm -o - %s \
// RUN: | opt -S -passes=mem2reg \
// RUN: | FileCheck %s

// REQUIRES: aarch64-registered-target

#include <arm_neon.h>

// CHECK-LABEL: test_vrnd32x_f32
// CHECK:  [[RND:%.*]] =  call <2 x float> @llvm.aarch64.neon.frint32x.v2f32(<2 x float> %a)
// CHECK:  ret <2 x float> [[RND]]
float32x2_t test_vrnd32x_f32(float32x2_t a) {
  return vrnd32x_f32(a);
}

// CHECK-LABEL: test_vrnd32xq_f32
// CHECK:  [[RND:%.*]] =  call <4 x float> @llvm.aarch64.neon.frint32x.v4f32(<4 x float> %a)
// CHECK:  ret <4 x float> [[RND]]
float32x4_t test_vrnd32xq_f32(float32x4_t a) {
  return vrnd32xq_f32(a);
}

// CHECK-LABEL: test_vrnd32z_f32
// CHECK:  [[RND:%.*]] =  call <2 x float> @llvm.aarch64.neon.frint32z.v2f32(<2 x float> %a)
// CHECK:  ret <2 x float> [[RND]]
float32x2_t test_vrnd32z_f32(float32x2_t a) {
  return vrnd32z_f32(a);
}

// CHECK-LABEL: test_vrnd32zq_f32
// CHECK:  [[RND:%.*]] =  call <4 x float> @llvm.aarch64.neon.frint32z.v4f32(<4 x float> %a)
// CHECK:  ret <4 x float> [[RND]]
float32x4_t test_vrnd32zq_f32(float32x4_t a) {
  return vrnd32zq_f32(a);
}

// CHECK-LABEL: test_vrnd64x_f32
// CHECK:  [[RND:%.*]] =  call <2 x float> @llvm.aarch64.neon.frint64x.v2f32(<2 x float> %a)
// CHECK:  ret <2 x float> [[RND]]
float32x2_t test_vrnd64x_f32(float32x2_t a) {
  return vrnd64x_f32(a);
}

// CHECK-LABEL: test_vrnd64xq_f32
// CHECK:  [[RND:%.*]] =  call <4 x float> @llvm.aarch64.neon.frint64x.v4f32(<4 x float> %a)
// CHECK:  ret <4 x float> [[RND]]
float32x4_t test_vrnd64xq_f32(float32x4_t a) {
  return vrnd64xq_f32(a);
}

// CHECK-LABEL: test_vrnd64z_f32
// CHECK:  [[RND:%.*]] =  call <2 x float> @llvm.aarch64.neon.frint64z.v2f32(<2 x float> %a)
// CHECK:  ret <2 x float> [[RND]]
float32x2_t test_vrnd64z_f32(float32x2_t a) {
  return vrnd64z_f32(a);
}

// CHECK-LABEL: test_vrnd64zq_f32
// CHECK:  [[RND:%.*]] =  call <4 x float> @llvm.aarch64.neon.frint64z.v4f32(<4 x float> %a)
// CHECK:  ret <4 x float> [[RND]]
float32x4_t test_vrnd64zq_f32(float32x4_t a) {
  return vrnd64zq_f32(a);
}

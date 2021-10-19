// RUN: %clang_cc1 -triple arm64-none-linux-gnu -target-feature +v8.5a\
// RUN: -flax-vector-conversions=none -S -disable-O0-optnone -emit-llvm -o - %s \
// RUN: | opt -S -mem2reg \
// RUN: | FileCheck %s

// REQUIRES: aarch64-registered-target

#include <arm_acle.h>

// CHECK-LABEL: test_frint32zf
// CHECK:  [[RND:%.*]] =  call float @llvm.aarch64.frint32z.f32(float %a)
// CHECK:  ret float [[RND]]
float test_frint32zf(float a) {
  return __frint32zf(a);
}

// CHECK-LABEL: test_frint32z
// CHECK:  [[RND:%.*]] =  call double @llvm.aarch64.frint32z.f64(double %a)
// CHECK:  ret double [[RND]]
double test_frint32z(double a) {
  return __frint32z(a);
}

// CHECK-LABEL: test_frint64zf
// CHECK:  [[RND:%.*]] =  call float @llvm.aarch64.frint64z.f32(float %a)
// CHECK:  ret float [[RND]]
float test_frint64zf(float a) {
  return __frint64zf(a);
}

// CHECK-LABEL: test_frint64z
// CHECK:  [[RND:%.*]] =  call double @llvm.aarch64.frint64z.f64(double %a)
// CHECK:  ret double [[RND]]
double test_frint64z(double a) {
  return __frint64z(a);
}

// CHECK-LABEL: test_frint32xf
// CHECK:  [[RND:%.*]] =  call float @llvm.aarch64.frint32x.f32(float %a)
// CHECK:  ret float [[RND]]
float test_frint32xf(float a) {
  return __frint32xf(a);
}

// CHECK-LABEL: test_frint32x
// CHECK:  [[RND:%.*]] =  call double @llvm.aarch64.frint32x.f64(double %a)
// CHECK:  ret double [[RND]]
double test_frint32x(double a) {
  return __frint32x(a);
}

// CHECK-LABEL: test_frint64xf
// CHECK:  [[RND:%.*]] =  call float @llvm.aarch64.frint64x.f32(float %a)
// CHECK:  ret float [[RND]]
float test_frint64xf(float a) {
  return __frint64xf(a);
}

// CHECK-LABEL: test_frint64x
// CHECK:  [[RND:%.*]] =  call double @llvm.aarch64.frint64x.f64(double %a)
// CHECK:  ret double [[RND]]
double test_frint64x(double a) {
  return __frint64x(a);
}

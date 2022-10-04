// RUN: %clang_cc1 -triple arm64-none-linux-gnu -target-feature +v8.5a\
// RUN: -flax-vector-conversions=none -S -disable-O0-optnone -emit-llvm -o - %s \
// RUN: | opt -S -mem2reg \
// RUN: | FileCheck %s

// REQUIRES: aarch64-registered-target

#include <arm_acle.h>

// CHECK-LABEL: test_rint32zf
// CHECK:  [[RND:%.*]] =  call float @llvm.aarch64.frint32z.f32(float %a)
// CHECK:  ret float [[RND]]
float test_rint32zf(float a) {
  return __rint32zf(a);
}

// CHECK-LABEL: test_rint32z
// CHECK:  [[RND:%.*]] =  call double @llvm.aarch64.frint32z.f64(double %a)
// CHECK:  ret double [[RND]]
double test_rint32z(double a) {
  return __rint32z(a);
}

// CHECK-LABEL: test_rint64zf
// CHECK:  [[RND:%.*]] =  call float @llvm.aarch64.frint64z.f32(float %a)
// CHECK:  ret float [[RND]]
float test_rint64zf(float a) {
  return __rint64zf(a);
}

// CHECK-LABEL: test_rint64z
// CHECK:  [[RND:%.*]] =  call double @llvm.aarch64.frint64z.f64(double %a)
// CHECK:  ret double [[RND]]
double test_rint64z(double a) {
  return __rint64z(a);
}

// CHECK-LABEL: test_rint32xf
// CHECK:  [[RND:%.*]] =  call float @llvm.aarch64.frint32x.f32(float %a)
// CHECK:  ret float [[RND]]
float test_rint32xf(float a) {
  return __rint32xf(a);
}

// CHECK-LABEL: test_rint32x
// CHECK:  [[RND:%.*]] =  call double @llvm.aarch64.frint32x.f64(double %a)
// CHECK:  ret double [[RND]]
double test_rint32x(double a) {
  return __rint32x(a);
}

// CHECK-LABEL: test_rint64xf
// CHECK:  [[RND:%.*]] =  call float @llvm.aarch64.frint64x.f32(float %a)
// CHECK:  ret float [[RND]]
float test_rint64xf(float a) {
  return __rint64xf(a);
}

// CHECK-LABEL: test_rint64x
// CHECK:  [[RND:%.*]] =  call double @llvm.aarch64.frint64x.f64(double %a)
// CHECK:  ret double [[RND]]
double test_rint64x(double a) {
  return __rint64x(a);
}

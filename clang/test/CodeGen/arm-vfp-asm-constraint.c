// REQUIRES: arm-registered-target
// RUN: %clang_cc1 -triple armv7-unknown-unknown -mfpmath vfp -emit-llvm -o - %s | FileCheck %s

// CHECK-NOT: error:

double fabs(double x) { // CHECK-LABEL: @fabs(
  // CHECK: call double asm "vabs.f64 ${0:P}, ${1:P}", "=w,w"(double
  __asm__("vabs.f64 %P0, %P1"
          : "=w"(x)
          : "w"(x));
  return x;
}

float fabsf(float x) { // CHECK-LABEL: @fabsf(
  // CHECK: call float asm "vabs.f32 $0, $1", "=t,t"(float
  __asm__("vabs.f32 %0, %1"
          : "=t"(x)
          : "t"(x));
  return x;
}

double sqrt(double x) { // CHECK-LABEL: @sqrt(
  // CHECK: call double asm "vsqrt.f64 ${0:P}, ${1:P}", "=w,w"(double
  __asm__("vsqrt.f64 %P0, %P1"
          : "=w"(x)
          : "w"(x));
  return x;
}

float sqrtf(float x) { // CHECK-LABEL: @sqrtf(
  // CHECK: call float asm "vsqrt.f32 $0, $1", "=t,t"(float
  __asm__("vsqrt.f32 %0, %1"
          : "=t"(x)
          : "t"(x));
  return x;
}

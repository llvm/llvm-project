// RUN: %clang_cc1 %s -cl-single-precision-constant -emit-llvm -o - | FileCheck %s

float fn(float f) {
  // CHECK: tail call float @llvm.fmuladd.f32(float %f, float 2.000000e+00, float 1.000000e+00)
  return f*2. + 1.;
}

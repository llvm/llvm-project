// RUN: %clang_cc1 -triple mips64-linux-gnu -frounding-math -ffp-exception-behavior=strict -O2 -verify=rounding,exception -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple mips64-linux-gnu -ffp-exception-behavior=strict -O2 -verify=exception -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple mips64-linux-gnu -frounding-math -O2 -verify=rounding -emit-llvm -o - %s | FileCheck %s
//
// Verify that constrained intrinsics are not used.
// As more targets gain support for constrained intrinsics the triple
// in this test will need to change.

// rounding-warning@* {{overriding currently unsupported rounding mode on this target}}
// exception-warning@* {{overriding currently unsupported use of floating point exceptions on this target}}
float fp_precise_1(float a, float b, float c) {
// CHECK: define float @_Z12fp_precise_1fff
// CHECK: %[[M:.+]] = fmul float{{.*}}
// CHECK: fadd float %[[M]], %c
  return a * b + c;
}

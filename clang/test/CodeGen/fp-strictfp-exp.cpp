// RUN: %clang_cc1 -triple mips64-linux-gnu -fexperimental-strict-floating-point -frounding-math -ffp-exception-behavior=strict -O2 -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple mips64-linux-gnu -fexperimental-strict-floating-point -ffp-exception-behavior=strict -O2 -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple mips64-linux-gnu -fexperimental-strict-floating-point -frounding-math -O2 -emit-llvm -o - %s | FileCheck %s
//
// Verify that constrained intrinsics are used due to the experimental flag.
// As more targets gain support for constrained intrinsics the triple
// in this test will need to change.

float fp_precise_1(float a, float b, float c) {
// CHECK-LABEL: define float @_Z12fp_precise_1fff
// CHECK: %[[M:.+]] = tail call float @llvm.experimental.constrained.fmul.f32(float {{.*}}, float {{.*}}, metadata {{.*}})
// CHECK: tail call float @llvm.experimental.constrained.fadd.f32(float %[[M]], float %c, metadata {{.*}})
  return a * b + c;
}

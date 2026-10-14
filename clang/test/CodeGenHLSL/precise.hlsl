// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.6-library -ffp-contract=on -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s --check-prefixes=CHECK,PRECISE
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.6-library -ffp-contract=on -emit-llvm -disable-llvm-passes -o - %s -DNO_PRECISE | FileCheck %s --check-prefixes=CHECK,DEFAULT
// RUN: %clang_cc1 -triple spirv-unknown-vulkan1.3-library -ffp-contract=on -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s --check-prefixes=CHECK,PRECISE

// A simplified version of precise is implemented: any use of the keyword in a
// translation unit puts the whole translation unit in IEEE-strict mode, which
// disables floating-point contraction. This includes code emitted before the
// keyword is seen.

// CHECK-LABEL: define {{.*}}float @_Z6Beforeff(
// PRECISE: %[[MUL:.*]] = fmul {{.*}}float
// PRECISE: fadd {{.*}}float %[[MUL]],
// PRECISE-NOT: fmuladd
// DEFAULT: call {{.*}}float @llvm.fmuladd.f32(
export float Before(float A, float B) { return A * B + A; }

float Two() { return 2.0; }
// The dynamic initializer of a global is also emitted in strict mode.
// CHECK-LABEL: define internal {{.*}}void @__cxx_global_var_init(
// PRECISE: %[[MUL:.*]] = fmul {{.*}}float
// PRECISE: fadd {{.*}}float %[[MUL]],
// PRECISE-NOT: fmuladd
// DEFAULT: call {{.*}}float @llvm.fmuladd.f32(
static float G = Two() * Two() + Two();

// CHECK-LABEL: define {{.*}}float @_Z5Afterf(
// PRECISE: %[[MUL:.*]] = fmul {{.*}}float
// PRECISE: fadd {{.*}}float %[[MUL]],
// PRECISE-NOT: fmuladd
// DEFAULT: call {{.*}}float @llvm.fmuladd.f32(
export float After(float A) {
#ifdef NO_PRECISE
  float X = A * A + A;
#else
  precise float X = A * A + A;
#endif
  return X;
}

// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.6-library -ffp-contract=on -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s --check-prefixes=CHECK,PRECISE --implicit-check-not=nofpclass --implicit-check-not='"no-signed-zeros-fp-math"="true"'
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.6-library -ffp-contract=on -emit-llvm -disable-llvm-passes -o - %s -DNO_PRECISE | FileCheck %s --check-prefixes=CHECK,DEFAULT
// RUN: %clang_cc1 -triple spirv-unknown-vulkan1.3-library -ffp-contract=on -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s --check-prefixes=CHECK,PRECISE --implicit-check-not=nofpclass --implicit-check-not='"no-signed-zeros-fp-math"="true"'
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.6-library -ffast-math -ffp-contract=fast -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s --check-prefixes=CHECK,PRECISE --implicit-check-not=nofpclass --implicit-check-not='"no-signed-zeros-fp-math"="true"'
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.6-library -O2 -emit-llvm -o - %s | FileCheck %s --check-prefix=OPT-PRECISE
// RUN: %clang_cc1 -triple spirv-unknown-vulkan1.3-library -O2 -emit-llvm -o - %s | FileCheck %s --check-prefix=OPT-PRECISE
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.6-library -O2 -emit-llvm -o - %s -DNO_PRECISE | FileCheck %s --check-prefix=OPT-DEFAULT
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.6-library -fno-fast-math -ffp-contract=off -DFP_PRAGMAS -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s --check-prefixes=CHECK,PRECISE
// RUN: %clang_cc1 -triple spirv-unknown-vulkan1.3-library -fno-fast-math -ffp-contract=off -DFP_PRAGMAS -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s --check-prefixes=CHECK,PRECISE
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.6-library -fno-fast-math -ffp-contract=off -DFP_PRAGMAS -DNO_PRECISE -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s --check-prefixes=CHECK,DEFAULT
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.6-library -fno-fast-math -DFP_PRAGMAS -O2 -emit-llvm -o - %s | FileCheck %s --check-prefix=OPT-PRECISE

// A simplified version of precise is implemented: any use of the keyword in a
// translation unit puts the whole translation unit in IEEE-strict mode, which
// disables floating-point contraction and unsafe optimizations. This includes
// code emitted before the keyword is seen.

#ifdef FP_PRAGMAS
#pragma float_control(precise, off)
#pragma clang fp contract(on)
#endif

// CHECK-LABEL: define {{.*}}float @_Z6Beforeff(
// PRECISE: %[[MUL:.*]] = fmul float
// PRECISE: fadd float %[[MUL]],
// PRECISE-NOT: fmuladd
// DEFAULT: call {{.*}}float @llvm.fmuladd.f32(
export float Before(float A, float B) { return A * B + A; }

// OPT-PRECISE-LABEL: define {{.*}}float @_Z11Reassociatef(
// OPT-PRECISE: %[[ADD:.*]] = fadd float %{{.*}}, 1.000000e+00
// OPT-PRECISE: %[[SUB:.*]] = fsub float %[[ADD]], %{{.*}}
// OPT-PRECISE: ret float %[[SUB]]
// OPT-DEFAULT-LABEL: define {{.*}}float @_Z11Reassociatef(
// OPT-DEFAULT: ret float 1.000000e+00
export float Reassociate(float A) { return (A + 1.0) - A; }

float Two() { return 2.0; }
// The dynamic initializer of a global is also emitted in strict mode.
// CHECK-LABEL: define internal {{.*}}void @__cxx_global_var_init(
// PRECISE: %[[MUL:.*]] = fmul float
// PRECISE: fadd float %[[MUL]],
// PRECISE-NOT: fmuladd
// DEFAULT: call {{.*}}float @llvm.fmuladd.f32(
static float G = Two() * Two() + Two();

// CHECK-LABEL: define {{.*}}float @_Z5Afterf(
// PRECISE: %[[MUL:.*]] = fmul float
// PRECISE: fadd float %[[MUL]],
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

// `precise` must match explicit strict flags plus -fno-fast-math, because HLSL
// enables fast math by default and precise disables it.
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -ffp-contract=on -disable-llvm-passes -emit-llvm -verify=precise -o %t.precise.ll %s
// RUN: FileCheck %s --check-prefix=PRECISE --implicit-check-not=llvm.fmuladd --implicit-check-not=llvm.experimental.constrained --implicit-check-not=no-signed-zeros-fp-math < %t.precise.ll
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -ffp-contract=off -fmath-errno -frounding-math -ffp-exception-behavior=strict -complex-range=full -fno-fast-math -disable-llvm-passes -emit-llvm -verify=precise -o %t.strict.ll %s
// RUN: diff %t.precise.ll %t.strict.ll

// Targets with strict-FP support: `precise` must also make every function use
// constrained intrinsics and strictfp, matching explicit strict flags.
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -fexperimental-strict-floating-point -ffp-contract=on -disable-llvm-passes -emit-llvm -verify=xstrict -o %t.xprecise.ll %s
// RUN: FileCheck %s --check-prefix=XPRECISE < %t.xprecise.ll
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -fexperimental-strict-floating-point -ffp-contract=off -fmath-errno -frounding-math -ffp-exception-behavior=strict -complex-range=full -fno-fast-math -disable-llvm-passes -emit-llvm -verify=xstrict -o %t.xstrict.ll %s
// RUN: diff %t.xprecise.ll %t.xstrict.ll

// xstrict-no-diagnostics

// precise-warning@* {{overriding currently unsupported rounding mode on this target}}
// precise-warning@* {{overriding currently unsupported use of floating point exceptions on this target}}

// PRECISE-LABEL: define {{.*}} @_Z5firstfff(
// PRECISE: fmul float
// PRECISE: fadd float
// XPRECISE: define {{.*}} @_Z5firstfff({{.*}}) #[[ATTR:[0-9]+]]
// XPRECISE: call float @llvm.experimental.constrained.fmul.f32
// XPRECISE: call float @llvm.experimental.constrained.fadd.f32
float first(float a, float b, float c) {
  return a * b + c;
}

// PRECISE-LABEL: define {{.*}} @_Z4lastfff(
// PRECISE: fmul float
// PRECISE: fadd float
float last(float a, float b, float c) {
  return a * b + c;
}

// Inline member functions are handed to codegen separately, via
// HandleInlineFunctionDefinition, as soon as the class is complete.
struct S {
  float m(float a, float b, float c) { return a * b + c; }
};

export float callMember(S s, float a, float b, float c) {
  return s.m(a, b, c);
}

// PRECISE-LABEL: define {{.*}} @_ZN1S1mEfff(
// PRECISE: fmul float
// PRECISE: fadd float
// XPRECISE: define {{.*}} @_ZN1S1mEfff({{.*}}) #[[ATTR]]
// XPRECISE: call float @llvm.experimental.constrained.fmul.f32
// XPRECISE: call float @llvm.experimental.constrained.fadd.f32

void marker() {
  precise float unused;
}

// XPRECISE: attributes #[[ATTR]] = {{{.*}}strictfp

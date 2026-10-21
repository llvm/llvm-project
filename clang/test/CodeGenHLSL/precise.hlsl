// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -ffp-contract=on -disable-llvm-passes -emit-llvm -verify=precise -o %t.precise.ll %s
// RUN: FileCheck %s --check-prefix=PRECISE --implicit-check-not=llvm.fmuladd --implicit-check-not=llvm.experimental.constrained < %t.precise.ll
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -ffp-contract=off -fmath-errno -frounding-math -ffp-exception-behavior=strict -complex-range=full -disable-llvm-passes -emit-llvm -verify=precise -o %t.strict.ll %s
// RUN: diff %t.precise.ll %t.strict.ll

// precise-warning@* {{overriding currently unsupported rounding mode on this target}}
// precise-warning@* {{overriding currently unsupported use of floating point exceptions on this target}}

// PRECISE-LABEL: define {{.*}} @_Z5firstfff(
// PRECISE: fmul reassoc nnan ninf nsz arcp afn float
// PRECISE: fadd reassoc nnan ninf nsz arcp afn float
float first(float a, float b, float c) {
  return a * b + c;
}

// PRECISE-LABEL: define {{.*}} @_Z4lastfff(
// PRECISE: fmul reassoc nnan ninf nsz arcp afn float
// PRECISE: fadd reassoc nnan ninf nsz arcp afn float
float last(float a, float b, float c) {
  return a * b + c;
}

void marker() {
  precise float unused;
}

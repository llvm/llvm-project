// `precise` marks the whole translation unit as precise even when the
// declaration it appears on is never used or never instantiated. Each case is
// compiled separately so one cannot mask the other.
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -DTEMPLATE -disable-llvm-passes -emit-llvm -verify -o - %s | FileCheck %s --implicit-check-not=llvm.fmuladd --implicit-check-not=no-signed-zeros-fp-math
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -DTYPEDEF -disable-llvm-passes -emit-llvm -verify -o - %s | FileCheck %s --implicit-check-not=llvm.fmuladd --implicit-check-not=no-signed-zeros-fp-math

// expected-warning@* {{overriding currently unsupported rounding mode on this target}}
// expected-warning@* {{overriding currently unsupported use of floating point exceptions on this target}}

#ifdef TEMPLATE
template <typename T> T neverInstantiated(T a) {
  precise T x = a;
  return x;
}
#endif

#ifdef TYPEDEF
typedef precise float unused_precise_float;
#endif

// CHECK-LABEL: define {{.*}} @_Z1ffff(
// CHECK: fmul float
// CHECK: fadd float
export float f(float a, float b, float c) {
  return a * b + c;
}

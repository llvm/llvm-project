// RUN: %clang_cc1 -triple arm64ec-w64-windows-gnu -emit-llvm -o - %s | FileCheck %s

// The elements of a `long double` vector are real `x86_fp80` values.  The IR
// keeps the vector operations intact; the AArch64 backend splits the vector
// and softens every lane to a compiler-rt helper.
typedef long double V __attribute__((vector_size(32)));
typedef double D2 __attribute__((vector_size(16)));

// Whole vectors move around as they are.
V copy(V a) { return a; }
// CHECK-LABEL: define {{.*}} @copy(
// CHECK-NOT: call
// CHECK: ret void

V add(V a, V b) { return a + b; }
// CHECK-LABEL: define {{.*}} @add(
// CHECK: fadd <2 x x86_fp80>
// CHECK-NOT: __arm64ec_

V div(V a, V b) { return a / b; }
// CHECK-LABEL: define {{.*}} @div(
// CHECK: fdiv <2 x x86_fp80>
// CHECK-NOT: __arm64ec_

// Negation stays a floating-point operation on the vector.
V neg(V a) { return -a; }
// CHECK-LABEL: define {{.*}} @neg(
// CHECK: fneg <2 x x86_fp80>
// CHECK-NOT: __arm64ec_

// A compound assignment takes the same path.
V compound(V a, V b) {
  a += b;
  return a;
}
// CHECK-LABEL: define {{.*}} @compound(
// CHECK: fadd <2 x x86_fp80>
// CHECK-NOT: __arm64ec_

// A scalar operand is splatted at the IR level.
V splat(V a) { return a + 1.0L; }
// CHECK-LABEL: define {{.*}} @splat(
// CHECK: fadd <2 x x86_fp80> {{.*}}, splat (x86_fp80 1.000000e+00)
// CHECK-NOT: __arm64ec_

// Comparison stays a vector compare.
int lt(V a, V b) { return (a < b)[0]; }
// CHECK-LABEL: define {{.*}} @lt(
// CHECK: fcmp olt <2 x x86_fp80>
// CHECK-NOT: __arm64ec_

// Conversions stay vector conversions.
D2 convert(V a) { return __builtin_convertvector(a, D2); }
// CHECK-LABEL: define {{.*}} @convert(
// CHECK: fptrunc <2 x x86_fp80> {{.*}} to <2 x double>
// CHECK-NOT: __arm64ec_

// The reverse conversion goes the other way.
V back(D2 a) { return __builtin_convertvector(a, V); }
// CHECK-LABEL: define {{.*}} @back(
// CHECK: fpext <2 x double> {{.*}} to <2 x x86_fp80>
// CHECK-NOT: __arm64ec_


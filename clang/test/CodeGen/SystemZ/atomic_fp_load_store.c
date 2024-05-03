// RUN: %clang_cc1 -triple s390x-linux-gnu -O1 -emit-llvm %s -o - | FileCheck %s
//
// Test that floating point atomic stores and loads do not get casted to/from
// integer.

#include <stdatomic.h>

_Atomic float Af;
_Atomic double Ad;
_Atomic long double Ald;

//// Atomic stores of floating point values.
void fun0(float Arg) {
// CHECK-LABEL: @fun0
// CHECK:       store atomic float %Arg, ptr @Af seq_cst, align 4
  Af = Arg;
}

void fun1(double Arg) {
// CHECK-LABEL: @fun1
// CHECK:       store atomic double %Arg, ptr @Ad seq_cst, align 8
  Ad = Arg;
}

void fun2(long double Arg) {
// CHECK-LABEL: @fun2
// CHECK:       store atomic fp128 %Arg, ptr @Ald seq_cst, align 16
  Ald = Arg;
}

void fun3(_Atomic float *Dst, float Arg) {
// CHECK-LABEL: @fun
// CHECK:       store atomic float %Arg, ptr %Dst seq_cst, align 4
  *Dst = Arg;
}

void fun4(_Atomic double *Dst, double Arg) {
// CHECK-LABEL: @fun4
// CHECK:       store atomic double %Arg, ptr %Dst seq_cst, align 8
  *Dst = Arg;
}

void fun5(_Atomic long double *Dst, long double Arg) {
// CHECK-LABEL: @fun5
// CHECK:       store atomic fp128 %Arg, ptr %Dst seq_cst, align 16
  *Dst = Arg;
}

//// Atomic loads of floating point values.
float fun6() {
// CHECK-LABEL: @fun6
// CHECK:       %atomic-load = load atomic float, ptr @Af seq_cst, align 4
  return Af;
}

float fun7() {
// CHECK-LABEL: @fun7
// CHECK:       %atomic-load = load atomic double, ptr @Ad seq_cst, align 8
  return Ad;
}

float fun8() {
// CHECK-LABEL: @fun8
// CHECK:       %atomic-load = load atomic fp128, ptr @Ald seq_cst, align 16
  return Ald;
}

float fun9(_Atomic float *Src) {
// CHECK-LABEL: @fun9
// CHECK:       %atomic-load = load atomic float, ptr %Src seq_cst, align 4
  return *Src;
}

double fun10(_Atomic double *Src) {
// CHECK-LABEL: @fun10
// CHECK:       %atomic-load = load atomic double, ptr %Src seq_cst, align 8
  return *Src;
}

long double fun11(_Atomic long double *Src) {
// CHECK-LABEL: @fun11
// CHECK:       %atomic-load = load atomic fp128, ptr %Src seq_cst, align 16
  return *Src;
}

//// Same, but with 'volatile' as well:

_Atomic volatile float Af_vol;
_Atomic volatile double Ad_vol;
_Atomic volatile long double Ald_vol;

//// Atomic volatile stores of floating point values.
void fun0_vol(float Arg) {
// CHECK-LABEL: @fun0_vol
// CHECK:       store atomic volatile float %Arg, ptr @Af_vol seq_cst, align 4
  Af_vol = Arg;
}

void fun1_vol(double Arg) {
// CHECK-LABEL: @fun1_vol
// CHECK:       store atomic volatile double %Arg, ptr @Ad_vol seq_cst, align 8
  Ad_vol = Arg;
}

void fun2_vol(long double Arg) {
// CHECK-LABEL: @fun2_vol
// CHECK:       store atomic volatile fp128 %Arg, ptr @Ald_vol seq_cst, align 16
  Ald_vol = Arg;
}

void fun3_vol(_Atomic volatile float *Dst, float Arg) {
// CHECK-LABEL: @fun3_vol
// CHECK:       store atomic volatile float %Arg, ptr %Dst seq_cst, align 4
  *Dst = Arg;
}

void fun4_vol(_Atomic volatile double *Dst, double Arg) {
// CHECK-LABEL: @fun4_vol
// CHECK:       store atomic volatile double %Arg, ptr %Dst seq_cst, align 8
  *Dst = Arg;
}

void fun5_vol(_Atomic volatile long double *Dst, long double Arg) {
// CHECK-LABEL: @fun5_vol
// CHECK:       store atomic volatile fp128 %Arg, ptr %Dst seq_cst, align 16
  *Dst = Arg;
}

//// Atomic volatile loads of floating point values.
float fun6_vol() {
// CHECK-LABEL: @fun6_vol
// CHECK:       %atomic-load = load atomic volatile float, ptr @Af_vol seq_cst, align 4
  return Af_vol;
}

float fun7_vol() {
// CHECK-LABEL: @fun7_vol
// CHECK:       %atomic-load = load atomic volatile double, ptr @Ad_vol seq_cst, align 8
  return Ad_vol;
}

float fun8_vol() {
// CHECK-LABEL: @fun8_vol
// CHECK:       %atomic-load = load atomic volatile fp128, ptr @Ald_vol seq_cst, align 16
  return Ald_vol;
}

float fun9_vol(_Atomic volatile float *Src) {
// CHECK-LABEL: @fun9_vol
// CHECK:       %atomic-load = load atomic volatile float, ptr %Src seq_cst, align 4
  return *Src;
}

double fun10_vol(_Atomic volatile double *Src) {
// CHECK-LABEL: @fun10_vol
// CHECK:       %atomic-load = load atomic volatile double, ptr %Src seq_cst, align 8
  return *Src;
}

long double fun11_vol(_Atomic volatile long double *Src) {
// CHECK-LABEL: @fun11_vol
// CHECK:       %atomic-load = load atomic volatile fp128, ptr %Src seq_cst, align 16
  return *Src;
}

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -isystem %S/Inputs/include \
// RUN:   -Wsystem-headers -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s

#include <tgmath.h>

// C11 7.25p5: conj() of a real argument calls the complex function, so the
// result is x - 0i, not x + 0i.

float complex test_conj_float(float x) { return conj(x); }
double complex test_conj_double(double x) { return conj(x); }
long double complex test_conj_long_double(long double x) { return conj(x); }

// CHECK-LABEL: define internal <2 x float> @_ZL9__tg_conjf(
// CHECK:         store float -0.000000e+00, ptr {{%.*}}

// CHECK-LABEL: define internal { double, double } @_ZL9__tg_conjd(
// CHECK:         store double -0.000000e+00, ptr {{%.*}}

// CHECK-LABEL: define internal { x86_fp80, x86_fp80 } @_ZL9__tg_conje(
// CHECK:         store x86_fp80 -0.000000e+00, ptr {{%.*}}

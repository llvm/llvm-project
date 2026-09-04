// RUN: %clang_cc1 -triple x86_64-linux-gnu -ast-print %s | FileCheck %s

// Floating literals print as the shortest decimal form that round-trips.

double d1 = 3.14;
// CHECK: double d1 = 3.14;
double d2 = 4.0;
// CHECK: double d2 = 4.;
double d3 = 0.5;
// CHECK: double d3 = 0.5;
double d4 = 1e10;
// CHECK: double d4 = 1.0E+10;
double d5 = 1e-300;
// CHECK: double d5 = 1.0E-300;
double d6 = 0.1;
// CHECK: double d6 = 0.1;
// The smallest double denormal: the shortest round-trip form is 5E-324.
double d7 = 4.9406564584124654e-324;
// CHECK: double d7 = 5.0E-324;
// DBL_MAX needs all 17 significant digits; nothing shorter round-trips.
double d8 = 1.7976931348623157e+308;
// CHECK: double d8 = 1.7976931348623157E+308;
double d9 = -3.14;
// CHECK: double d9 = -3.14;

float f1 = 3.14f;
// CHECK: float f1 = 3.14F;
float f2 = 0.1f;
// CHECK: float f2 = 0.1F;
// A hex float keeps its value, printed in decimal.
float f3 = 0x1.8p3f;
// CHECK: float f3 = 12.F;

// x86 80-bit extended precision.
long double ld1 = 0.1L;
// CHECK: long double ld1 = 0.1L;
long double ld2 = 3.14L;
// CHECK: long double ld2 = 3.14L;

__float128 q1 = 0.1q;
// CHECK: __float128 q1 = 0.1Q;

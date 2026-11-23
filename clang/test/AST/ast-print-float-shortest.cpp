// RUN: %clang_cc1 -triple x86_64-linux-gnu -ast-print %s | FileCheck %s --check-prefixes=CHECK,F16,X87,Q128
// RUN: %clang_cc1 -triple aarch64-linux-gnu -ast-print %s | FileCheck %s --check-prefixes=CHECK,F16,QUAD
// RUN: %clang_cc1 -triple powerpc64-linux-gnu -ast-print %s | FileCheck %s --check-prefixes=CHECK,PPC

// Floating literals print as the shortest decimal form that round-trips to
// the same value in the literal's type.

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

// A hex float keeps its value, printed in decimal.
double h1 = 0x10.1p0;
// CHECK: double h1 = 16.0625;
double h2 = 0x1p-1074;
// CHECK: double h2 = 5.0E-324;

// Integers keep their digits; the shortest form would only switch them to
// scientific notation. More than three trailing zeros still do, as before.
double i1 = 10.0;
// CHECK: double i1 = 10.;
double i2 = 100.0;
// CHECK: double i2 = 100.;
double i3 = 1200.0;
// CHECK: double i3 = 1200.;
double i4 = 120000.0;
// CHECK: double i4 = 1.2E+5;
// 2^53 + 1 rounds to 2^53.
double i5 = 9007199254740993.0;
// CHECK: double i5 = 9007199254740992.;
// 1e23 is not exactly representable; 1E23 is still the shortest form.
double i6 = 1e23;
// CHECK: double i6 = 1.0E+23;

// The smallest normal and the largest denormal double.
double b1 = 2.2250738585072014e-308;
// CHECK: double b1 = 2.2250738585072014E-308;
double b2 = 2.2250738585072009e-308;
// CHECK: double b2 = 2.225073858507201E-308;
// 0.1 + 0.2 needs all 17 digits.
double b3 = 0.30000000000000004;
// CHECK: double b3 = 0.30000000000000004;
double b4 = 0.0001;
// CHECK: double b4 = 1.0E-4;
// A significand that looks like a power of five (Ryu's LooksLikePow5).
double b5 = 5.764607523034235e39;
// CHECK: double b5 = 5.764607523034235E+39;

float f1 = 3.14f;
// CHECK: float f1 = 3.14F;
float f2 = 0.1f;
// CHECK: float f2 = 0.1F;
float f3 = 0x1.8p3f;
// CHECK: float f3 = 12.F;
float f4 = 0x10.1p0f;
// CHECK: float f4 = 16.0625F;
float f5 = 100.0f;
// CHECK: float f5 = 100.F;
// FLT_MAX and the smallest float denormal.
float f6 = 3.4028235e38f;
// CHECK: float f6 = 3.4028235E+38F;
float f7 = 1.4e-45f;
// CHECK: float f7 = 1.0E-45F;
// 2^24 + 1 rounds to 2^24.
float f8 = 16777217.0f;
// CHECK: float f8 = 16777216.F;
// The upper bound of the rounding interval is a round-trip form when the
// significand is even (Ryu's BoundaryRoundEven).
float f9 = 3.4366717e10f;
// CHECK: float f9 = 3.436672E+10F;

#ifdef __FLT16_MANT_DIG__
_Float16 h16_1 = 0.1f16;
// F16: _Float16 h16_1 = 0.1F16;
_Float16 h16_2 = 65504.0f16;
// F16: _Float16 h16_2 = 65504.F16;
// 16.0625 is exact in _Float16, but 16.06 already round-trips.
_Float16 h16_3 = 0x10.1p0f16;
// F16: _Float16 h16_3 = 16.06F16;
// The smallest _Float16 denormal.
_Float16 h16_4 = 5.9604644775390625e-8f16;
// F16: _Float16 h16_4 = 6.0E-8F16;
#endif

// long double: x87 80-bit on x86_64, IEEE quad on AArch64, IBM double-double
// on PowerPC.
long double ld1 = 0.1L;
// CHECK: long double ld1 = 0.1L;
long double ld2 = 3.14L;
// CHECK: long double ld2 = 3.14L;
long double ld3 = 0x10.1p0L;
// CHECK: long double ld3 = 16.0625L;
long double ld4 = 100.0L;
// CHECK: long double ld4 = 100.L;
// 1 + 2^-52 is exact in every long double format; the digit count follows
// the format's precision.
long double ld5 = 1.0000000000000002220446049250313L;
// X87: long double ld5 = 1.000000000000000222L;
// QUAD: long double ld5 = 1.0000000000000002220446049250313L;
// PPC: long double ld5 = 1.0000000000000002220446049250313L;
// The smallest double denormal is a denormal only in double-double.
long double ld6 = 4.9406564584124654e-324L;
// X87: long double ld6 = 4.9406564584124654E-324L;
// QUAD: long double ld6 = 4.9406564584124654E-324L;
// PPC: long double ld6 = 5.0E-324L;
// 1 + 1e-37 is below every format's precision.
long double ld7 = 1.0000000000000000000000000000000000001L;
// CHECK: long double ld7 = 1.L;

#ifdef __SIZEOF_FLOAT128__
__float128 q1 = 0.1q;
// Q128: __float128 q1 = 0.1Q;
__float128 q2 = 0x10.1p0q;
// Q128: __float128 q2 = 16.0625Q;
__float128 q3 = 100.0q;
// Q128: __float128 q3 = 100.Q;
#endif

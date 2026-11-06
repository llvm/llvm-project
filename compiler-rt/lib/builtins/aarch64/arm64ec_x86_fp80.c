//===-- arm64ec_x86_fp80.c - ARM64EC x87 `long double` helpers ----*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if __LDBL_MANT_DIG__ == 64

#define SRC_80
#define DST_BFLOAT
#include "../fp_trunc_impl.inc"

long double __arm64ec_addxf3(long double a, long double b) { return a + b; }

long double __arm64ec_subxf3(long double a, long double b) { return a - b; }

long double __arm64ec_mulxf3(long double a, long double b) { return a * b; }

long double __arm64ec_divxf3(long double a, long double b) { return a / b; }

long double __powixf2(long double, int);

long double __arm64ec_powixf2(long double base, int exponent) {
  return __powixf2(base, exponent);
}

int __arm64ec_lexf2(long double a, long double b) { return __leXf2__(a, b); }

COMPILER_RT_ALIAS(__arm64ec_lexf2, __arm64ec_eqxf2)

COMPILER_RT_ALIAS(__arm64ec_lexf2, __arm64ec_ltxf2)

COMPILER_RT_ALIAS(__arm64ec_lexf2, __arm64ec_nexf2)

int __arm64ec_gexf2(long double a, long double b) { return __geXf2__(a, b); }

COMPILER_RT_ALIAS(__arm64ec_gexf2, __arm64ec_gtxf2)

int __arm64ec_unordxf2(long double a, long double b) {
  return __unordXf2__(a, b);
}

long double __arm64ec_extendhfxf2(_Float16 value) { return value; }

long double __arm64ec_extendsfxf2(float value) { return value; }

long double __arm64ec_extenddfxf2(double value) { return value; }

__float128 __arm64ec_extendxftf2(long double value) { return value; }

_Float16 __arm64ec_truncxfhf2(long double value) { return value; }

__bf16 __arm64ec_truncxfbf2(long double value) { return __truncXfYf2__(value); }

float __arm64ec_truncxfsf2(long double value) { return value; }

double __arm64ec_truncxfdf2(long double value) { return value; }

long double __arm64ec_trunctfxf2(__float128 value) { return value; }

int __arm64ec_fixxfsi(long double value) { return value; }

long long __arm64ec_fixxfdi(long double value) { return value; }

__int128 __arm64ec_fixxfti(long double value) { return value; }

unsigned int __arm64ec_fixunsxfsi(long double value) { return value; }

unsigned long long __arm64ec_fixunsxfdi(long double value) { return value; }

unsigned __int128 __arm64ec_fixunsxfti(long double value) { return value; }

long double __arm64ec_floatsixf(int value) { return value; }

long double __arm64ec_floatdixf(long long value) { return value; }

long double __arm64ec_floattixf(__int128 value) { return value; }

long double __arm64ec_floatunsixf(unsigned int value) { return value; }

long double __arm64ec_floatundixf(unsigned long long value) { return value; }

long double __arm64ec_floatuntixf(unsigned __int128 value) { return value; }

#endif // __LDBL_MANT_DIG__ == 64

// RUN: %clang_builtins %s %librt -o %t && %run %t
//===-- floatuntisf.c - Test __floatuntisf --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file tests __floatuntisf for the compiler_rt library.
//
//===----------------------------------------------------------------------===//

#include "int_lib.h"
#include <float.h>
#include <stdio.h>

#if defined(CRT_HAS_128BIT) && HAS_80_BIT_LONG_DOUBLE

// Returns: convert a to a float, rounding toward even.

// Assumption: float is a IEEE 32 bit floating point type 
//             tu_int is a 128 bit integral type

// seee eeee emmm mmmm mmmm mmmm mmmm mmmm

COMPILER_RT_ABI float __floatuntisf(tu_int a);

int test__floatuntisf(tu_int a, float expected)
{
    float x = __floatuntisf(a);
    if (x != expected)
    {
        utwords at;
        at.all = a;
        printf("error in __floatuntisf(0x%.16llX%.16llX) = %a, expected %a\n",
               at.s.high, at.s.low, x, expected);
    }
    return x != expected;
}

COMPILE_TIME_ASSERT(sizeof(tu_int) == 2*sizeof(du_int));
COMPILE_TIME_ASSERT(sizeof(tu_int)*CHAR_BIT == 128);
COMPILE_TIME_ASSERT(sizeof(float)*CHAR_BIT == 32);

#endif

int main()
{
#if defined(CRT_HAS_128BIT) && HAS_80_BIT_LONG_DOUBLE
    if (test__floatuntisf(0, 0.0F))
        return 1;

    if (test__floatuntisf(1, 1.0F))
        return 1;
    if (test__floatuntisf(2, 2.0F))
        return 1;
    if (test__floatuntisf(20, 20.0F))
        return 1;

    if (test__floatuntisf(0x7FFFFF8000000000LL, 0x1.FFFFFEp+62F))
        return 1;
    if (test__floatuntisf(0x7FFFFF0000000000LL, 0x1.FFFFFCp+62F))
        return 1;

    if (test__floatuntisf(make_ti(0x8000008000000000LL, 0), 0x1.000001p+127F))
        return 1;
    if (test__floatuntisf(make_ti(0x8000000000000800LL, 0), 0x1.0p+127F))
        return 1;
    if (test__floatuntisf(make_ti(0x8000010000000000LL, 0), 0x1.000002p+127F))
        return 1;

    if (test__floatuntisf(make_ti(0x8000000000000000LL, 0), 0x1.000000p+127F))
        return 1;

    if (test__floatuntisf(0x0007FB72E8000000LL, 0x1.FEDCBAp+50F))
        return 1;

    if (test__floatuntisf(0x0007FB72EA000000LL, 0x1.FEDCBA8p+50F))
        return 1;
    if (test__floatuntisf(0x0007FB72EB000000LL, 0x1.FEDCBACp+50F))
        return 1;

    if (test__floatuntisf(0x0007FB72EC000000LL, 0x1.FEDCBBp+50F))
        return 1;

    if (test__floatuntisf(0x0007FB72E6000000LL, 0x1.FEDCB98p+50F))
        return 1;
    if (test__floatuntisf(0x0007FB72E7000000LL, 0x1.FEDCB9Cp+50F))
        return 1;
    if (test__floatuntisf(0x0007FB72E4000000LL, 0x1.FEDCB9p+50F))
        return 1;

    if (test__floatuntisf(0xFFFFFFFFFFFFFFFELL, 0x1p+64F))
        return 1;
    if (test__floatuntisf(0xFFFFFFFFFFFFFFFFLL, 0x1p+64F))
        return 1;

    if (test__floatuntisf(0x0007FB72E8000000LL, 0x1.FEDCBAp+50F))
        return 1;

    if (test__floatuntisf(0x0007FB72EA000000LL, 0x1.FEDCBAp+50F))
        return 1;
    if (test__floatuntisf(0x0007FB72EB000000LL, 0x1.FEDCBAp+50F))
        return 1;
    if (test__floatuntisf(0x0007FB72EBFFFFFFLL, 0x1.FEDCBAp+50F))
        return 1;
    if (test__floatuntisf(0x0007FB72EC000000LL, 0x1.FEDCBCp+50F))
        return 1;
    if (test__floatuntisf(0x0007FB72E8000001LL, 0x1.FEDCBAp+50F))
        return 1;

    if (test__floatuntisf(0x0007FB72E6000000LL, 0x1.FEDCBAp+50F))
        return 1;
    if (test__floatuntisf(0x0007FB72E7000000LL, 0x1.FEDCBAp+50F))
        return 1;
    if (test__floatuntisf(0x0007FB72E7FFFFFFLL, 0x1.FEDCBAp+50F))
        return 1;
    if (test__floatuntisf(0x0007FB72E4000001LL, 0x1.FEDCBAp+50F))
        return 1;
    if (test__floatuntisf(0x0007FB72E4000000LL, 0x1.FEDCB8p+50F))
        return 1;

    if (test__floatuntisf(make_ti(0x0000000000001FEDLL, 0xCB90000000000001LL),
                          0x1.FEDCBAp+76F))
        return 1;
    if (test__floatuntisf(make_ti(0x0000000000001FEDLL, 0xCBA0000000000000LL),
                          0x1.FEDCBAp+76F))
        return 1;
    if (test__floatuntisf(make_ti(0x0000000000001FEDLL, 0xCBAFFFFFFFFFFFFFLL),
                          0x1.FEDCBAp+76F))
        return 1;
    if (test__floatuntisf(make_ti(0x0000000000001FEDLL, 0xCBB0000000000000LL),
                          0x1.FEDCBCp+76F))
        return 1;
    if (test__floatuntisf(make_ti(0x0000000000001FEDLL, 0xCBB0000000000001LL),
                          0x1.FEDCBCp+76F))
        return 1;
    if (test__floatuntisf(make_ti(0x0000000000001FEDLL, 0xCBBFFFFFFFFFFFFFLL),
                          0x1.FEDCBCp+76F))
        return 1;
    if (test__floatuntisf(make_ti(0x0000000000001FEDLL, 0xCBC0000000000000LL),
                          0x1.FEDCBCp+76F))
        return 1;
    if (test__floatuntisf(make_ti(0x0000000000001FEDLL, 0xCBC0000000000001LL),
                          0x1.FEDCBCp+76F))
        return 1;
    if (test__floatuntisf(make_ti(0x0000000000001FEDLL, 0xCBD0000000000000LL),
                          0x1.FEDCBCp+76F))
        return 1;
    if (test__floatuntisf(make_ti(0x0000000000001FEDLL, 0xCBD0000000000001LL),
                          0x1.FEDCBEp+76F))
        return 1;
    if (test__floatuntisf(make_ti(0x0000000000001FEDLL, 0xCBDFFFFFFFFFFFFFLL),
                          0x1.FEDCBEp+76F))
        return 1;
    if (test__floatuntisf(make_ti(0x0000000000001FEDLL, 0xCBE0000000000000LL),
                          0x1.FEDCBEp+76F))
        return 1;

#else
    printf("skipped\n");
#endif
   return 0;
}

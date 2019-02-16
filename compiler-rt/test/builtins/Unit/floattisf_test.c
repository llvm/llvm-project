// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: int128
//===-- floattisf_test.c - Test __floattisf -------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file tests __floattisf for the compiler_rt library.
//
//===----------------------------------------------------------------------===//

#include "int_lib.h"
#include <float.h>
#include <stdio.h>

#ifdef CRT_HAS_128BIT

// Returns: convert a to a float, rounding toward even.

// Assumption: float is a IEEE 32 bit floating point type 
//             ti_int is a 64 bit integral type

// seee eeee emmm mmmm mmmm mmmm mmmm mmmm

COMPILER_RT_ABI float __floattisf(ti_int a);

int test__floattisf(ti_int a, float expected)
{
    float x = __floattisf(a);
    if (x != expected)
    {
        twords at;
        at.all = a;
        printf("error in __floattisf(0x%.16llX%.16llX) = %a, expected %a\n",
               at.s.high, at.s.low, x, expected);
    }
    return x != expected;
}

char assumption_1[sizeof(ti_int) == 2*sizeof(di_int)] = {0};
char assumption_2[sizeof(ti_int)*CHAR_BIT == 128] = {0};
char assumption_3[sizeof(float)*CHAR_BIT == 32] = {0};

#endif

int main()
{
#ifdef CRT_HAS_128BIT
    if (test__floattisf(0, 0.0F))
        return 1;

    if (test__floattisf(1, 1.0F))
        return 1;
    if (test__floattisf(2, 2.0F))
        return 1;
    if (test__floattisf(-1, -1.0F))
        return 1;
    if (test__floattisf(-2, -2.0F))
        return 1;

    if (test__floattisf(0x7FFFFF8000000000LL, 0x1.FFFFFEp+62F))
        return 1;
    if (test__floattisf(0x7FFFFF0000000000LL, 0x1.FFFFFCp+62F))
        return 1;

    if (test__floattisf(make_ti(0xFFFFFFFFFFFFFFFFLL, 0x8000008000000000LL),
                        -0x1.FFFFFEp+62F))
        return 1;
    if (test__floattisf(make_ti(0xFFFFFFFFFFFFFFFFLL, 0x8000010000000000LL),
                        -0x1.FFFFFCp+62F))
        return 1;

    if (test__floattisf(make_ti(0xFFFFFFFFFFFFFFFFLL, 0x8000000000000000LL),
                        -0x1.000000p+63F))
        return 1;
    if (test__floattisf(make_ti(0xFFFFFFFFFFFFFFFFLL, 0x8000000000000001LL),
                        -0x1.000000p+63F))
        return 1;

    if (test__floattisf(0x0007FB72E8000000LL, 0x1.FEDCBAp+50F))
        return 1;

    if (test__floattisf(0x0007FB72EA000000LL, 0x1.FEDCBAp+50F))
        return 1;
    if (test__floattisf(0x0007FB72EB000000LL, 0x1.FEDCBAp+50F))
        return 1;
    if (test__floattisf(0x0007FB72EBFFFFFFLL, 0x1.FEDCBAp+50F))
        return 1;
    if (test__floattisf(0x0007FB72EC000000LL, 0x1.FEDCBCp+50F))
        return 1;
    if (test__floattisf(0x0007FB72E8000001LL, 0x1.FEDCBAp+50F))
        return 1;

    if (test__floattisf(0x0007FB72E6000000LL, 0x1.FEDCBAp+50F))
        return 1;
    if (test__floattisf(0x0007FB72E7000000LL, 0x1.FEDCBAp+50F))
        return 1;
    if (test__floattisf(0x0007FB72E7FFFFFFLL, 0x1.FEDCBAp+50F))
        return 1;
    if (test__floattisf(0x0007FB72E4000001LL, 0x1.FEDCBAp+50F))
        return 1;
    if (test__floattisf(0x0007FB72E4000000LL, 0x1.FEDCB8p+50F))
        return 1;

    if (test__floattisf(make_ti(0x0007FB72E8000000LL, 0), 0x1.FEDCBAp+114F))
        return 1;

    if (test__floattisf(make_ti(0x0007FB72EA000000LL, 0), 0x1.FEDCBAp+114F))
        return 1;
    if (test__floattisf(make_ti(0x0007FB72EB000000LL, 0), 0x1.FEDCBAp+114F))
        return 1;
    if (test__floattisf(make_ti(0x0007FB72EBFFFFFFLL, 0), 0x1.FEDCBAp+114F))
        return 1;
    if (test__floattisf(make_ti(0x0007FB72EC000000LL, 0), 0x1.FEDCBCp+114F))
        return 1;
    if (test__floattisf(make_ti(0x0007FB72E8000001LL, 0), 0x1.FEDCBAp+114F))
        return 1;

    if (test__floattisf(make_ti(0x0007FB72E6000000LL, 0), 0x1.FEDCBAp+114F))
        return 1;
    if (test__floattisf(make_ti(0x0007FB72E7000000LL, 0), 0x1.FEDCBAp+114F))
        return 1;
    if (test__floattisf(make_ti(0x0007FB72E7FFFFFFLL, 0), 0x1.FEDCBAp+114F))
        return 1;
    if (test__floattisf(make_ti(0x0007FB72E4000001LL, 0), 0x1.FEDCBAp+114F))
        return 1;
    if (test__floattisf(make_ti(0x0007FB72E4000000LL, 0), 0x1.FEDCB8p+114F))
        return 1;

#else
    printf("skipped\n");
#endif
   return 0;
}

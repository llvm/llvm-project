// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_fixdfti
// REQUIRES: int128
//===-- fixdfti_test.c - Test __fixdfti -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file tests __fixdfti for the compiler_rt library.
//
//===----------------------------------------------------------------------===//

#include "int_lib.h"
#include <stdio.h>

#ifdef CRT_HAS_128BIT

// Returns: convert a to a signed long long, rounding toward zero.

// Assumption: double is a IEEE 64 bit floating point type 
//             su_int is a 32 bit integral type
//             value in double is representable in ti_int (no range checking performed)

// seee eeee eeee mmmm mmmm mmmm mmmm mmmm | mmmm mmmm mmmm mmmm mmmm mmmm mmmm mmmm

COMPILER_RT_ABI ti_int __fixdfti(double a);

int test__fixdfti(double a, ti_int expected)
{
    ti_int x = __fixdfti(a);
    if (x != expected)
    {
        twords xt;
        xt.all = x;
        twords expectedt;
        expectedt.all = expected;
        printf("error in __fixdfti(%A) = 0x%.16llX%.16llX, expected 0x%.16llX%.16llX\n",
        a, xt.s.high, xt.s.low, expectedt.s.high, expectedt.s.low);
    }
    return x != expected;
}

char assumption_1[sizeof(ti_int) == 2*sizeof(di_int)] = {0};
char assumption_2[sizeof(su_int)*CHAR_BIT == 32] = {0};
char assumption_3[sizeof(double)*CHAR_BIT == 64] = {0};

#endif

int main()
{
#ifdef CRT_HAS_128BIT
    if (test__fixdfti(0.0, 0))
        return 1;

    if (test__fixdfti(0.5, 0))
        return 1;
    if (test__fixdfti(0.99, 0))
        return 1;
    if (test__fixdfti(1.0, 1))
        return 1;
    if (test__fixdfti(1.5, 1))
        return 1;
    if (test__fixdfti(1.99, 1))
        return 1;
    if (test__fixdfti(2.0, 2))
        return 1;
    if (test__fixdfti(2.01, 2))
        return 1;
    if (test__fixdfti(-0.5, 0))
        return 1;
    if (test__fixdfti(-0.99, 0))
        return 1;
    if (test__fixdfti(-1.0, -1))
        return 1;
    if (test__fixdfti(-1.5, -1))
        return 1;
    if (test__fixdfti(-1.99, -1))
        return 1;
    if (test__fixdfti(-2.0, -2))
        return 1;
    if (test__fixdfti(-2.01, -2))
        return 1;

    if (test__fixdfti(0x1.FFFFFEp+62, 0x7FFFFF8000000000LL))
        return 1;
    if (test__fixdfti(0x1.FFFFFCp+62, 0x7FFFFF0000000000LL))
        return 1;

    if (test__fixdfti(-0x1.FFFFFEp+62, make_ti(0xFFFFFFFFFFFFFFFFLL,
                                               0x8000008000000000LL)))
        return 1;
    if (test__fixdfti(-0x1.FFFFFCp+62, make_ti(0xFFFFFFFFFFFFFFFFLL,
                                               0x8000010000000000LL)))
        return 1;

    if (test__fixdfti(0x1.FFFFFFFFFFFFFp+62, 0x7FFFFFFFFFFFFC00LL))
        return 1;
    if (test__fixdfti(0x1.FFFFFFFFFFFFEp+62, 0x7FFFFFFFFFFFF800LL))
        return 1;

    if (test__fixdfti(-0x1.FFFFFFFFFFFFFp+62, make_ti(0xFFFFFFFFFFFFFFFFLL,
                                                      0x8000000000000400LL)))
        return 1;
    if (test__fixdfti(-0x1.FFFFFFFFFFFFEp+62, make_ti(0xFFFFFFFFFFFFFFFFLL,
                                                      0x8000000000000800LL)))
        return 1;

    if (test__fixdfti(0x1.FFFFFFFFFFFFFp+126, make_ti(0x7FFFFFFFFFFFFC00LL, 0)))
        return 1;
    if (test__fixdfti(0x1.FFFFFFFFFFFFEp+126, make_ti(0x7FFFFFFFFFFFF800LL, 0)))
        return 1;

    if (test__fixdfti(-0x1.FFFFFFFFFFFFFp+126, make_ti(0x8000000000000400LL, 0)))
        return 1;
    if (test__fixdfti(-0x1.FFFFFFFFFFFFEp+126, make_ti(0x8000000000000800LL, 0)))
        return 1;

#else
    printf("skipped\n");
#endif
   return 0;
}

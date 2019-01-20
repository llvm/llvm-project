// RUN: %clang_builtins %s %librt -o %t && %run %t
//===--------------- fixtfti_test.c - Test __fixtfti ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file tests __fixtfti for the compiler_rt library.
//
//===----------------------------------------------------------------------===//

#include "int_lib.h"
#include <stdio.h>

#if __LDBL_MANT_DIG__ == 113

#include "fp_test.h"

ti_int __fixtfti(long double a);

int test__fixtfti(long double a, ti_int expected)
{
    ti_int x = __fixtfti(a);
    int ret = (x != expected);

    if (ret)
    {
        twords xt;
        xt.all = x;

        twords expectedt;
        expectedt.all = expected;

        printf("error in test__fixtfti(%.20Lf) = 0x%.16llX%.16llX, "
               "expected 0x%.16llX%.16llX\n",
               a, xt.s.high, xt.s.low, expectedt.s.high, expectedt.s.low);
    }
    return ret;
}

char assumption_1[sizeof(long double) * CHAR_BIT == 128] = {0};

#endif

int main()
{
#if __LDBL_MANT_DIG__ == 113
    if (test__fixtfti(makeInf128(), make_ti(0x7fffffffffffffffLL,
                                            0xffffffffffffffffLL)))
        return 1;
    if (test__fixtfti(0, make_ti(0x0LL, 0x0LL)))
        return 1;
    if (test__fixtfti(0x1.23456789abcdefp+5L, make_ti(0x0LL, 0x24LL)))
        return 1;
    if (test__fixtfti(0x1.23456789abcdefp-3L, make_ti(0x0LL, 0x0LL)))
        return 1;
    if (test__fixtfti(0x1.23456789abcdef12345678p+20L,
                      make_ti(0x0LL, 0x123456LL)))
        return 1;
    if (test__fixtfti(0x1.23456789abcdef123456789abcdep+112L,
                      make_ti(0x123456789abcdLL, 0xef123456789abcdeLL)))
        return 1;
    if (test__fixtfti(-0x1.23456789abcdef123456789abcdep+112L,
                      make_ti(0xFFFEDCBA98765432LL, 0x10EDCBA987654322LL)))
        return 1;
    if (test__fixtfti(0x1.23456789abcdefp+256L, make_ti(0x7fffffffffffffffLL,
                                                        0xffffffffffffffffLL)))
        return 1;
    if (test__fixtfti(-0x1.23456789abcdefp+20L, make_ti(0xffffffffffffffffLL,
                                                        0xffffffffffedcbaaLL)))
        return 1;
    if (test__fixtfti(-0x1.23456789abcdefp+256L, make_ti(0x8000000000000000LL,
                                                         0x0)))
        return 1;

#else
    printf("skipped\n");

#endif
    return 0;
}

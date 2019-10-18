// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_fixtfdi
//===--------------- fixtfdi_test.c - Test __fixtfdi ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file tests __fixtfdi for the compiler_rt library.
//
//===----------------------------------------------------------------------===//

#include "int_lib.h"
#include <stdio.h>

#if __LDBL_MANT_DIG__ == 113

#include "fp_test.h"

di_int __fixtfdi(long double a);

int test__fixtfdi(long double a, di_int expected)
{
    di_int x = __fixtfdi(a);
    int ret = (x != expected);

    if (ret)
    {
        printf("error in test__fixtfdi(%.20Lf) = %llX, "
               "expected %llX\n", a, x, expected);
    }
    return ret;
}

char assumption_1[sizeof(long double) * CHAR_BIT == 128] = {0};

#endif

int main()
{
#if __LDBL_MANT_DIG__ == 113
    if (test__fixtfdi(makeInf128(), 0x7fffffffffffffffLL))
        return 1;
    if (test__fixtfdi(0, 0x0))
        return 1;
    if (test__fixtfdi(0x1.23456789abcdefp+5L, 0x24LL))
        return 1;
    if (test__fixtfdi(0x1.23456789abcdefp-3L, 0x0LL))
        return 1;
    if (test__fixtfdi(0x1.23456789abcdef12345678p+20L, 0x123456LL))
        return 1;
    if (test__fixtfdi(0x1.23456789abcdef12345678p+40L, 0x123456789abLL))
        return 1;
    if (test__fixtfdi(0x1.23456789abcdef12345678p+60L, 0x123456789abcdef1LL))
        return 1;
    if (test__fixtfdi(0x1.23456789abcdefp+256L, 0x7fffffffffffffffLL))
        return 1;
    if (test__fixtfdi(-0x1.23456789abcdefp+20L, 0xffffffffffedcbaaLL))
        return 1;
    if (test__fixtfdi(-0x1.23456789abcdefp+40L, 0xfffffedcba987655LL))
        return 1;
    if (test__fixtfdi(-0x1.23456789abcdefp+256L, 0x8000000000000000LL))
        return 1;

#else
    printf("skipped\n");

#endif
    return 0;
}

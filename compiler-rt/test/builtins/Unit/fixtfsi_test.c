//===--------------- fixtfsi_test.c - Test __fixtfsi ----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file tests __fixtfsi for the compiler_rt library.
//
//===----------------------------------------------------------------------===//

#include <stdio.h>

#if __LDBL_MANT_DIG__ == 113

#include "fp_test.h"

int __fixtfsi(long double a);

int test__fixtfsi(long double a, int expected)
{
    int x = __fixtfsi(a);
    int ret = (x != expected);

    if (ret){
        printf("error in test__fixtfsi(%.20Lf) = %d, "
               "expected %d\n", a, x, expected);
    }
    return ret;
}

char assumption_1[sizeof(long double) * CHAR_BIT == 128] = {0};

#endif

int main()
{
#if __LDBL_MANT_DIG__ == 113
    if (test__fixtfsi(makeInf128(), 0x7fffffff))
        return 1;
    if (test__fixtfsi(0, 0x0))
        return 1;
    if (test__fixtfsi(0x1.23456789abcdefp+5, 0x24))
        return 1;
    if (test__fixtfsi(0x1.23456789abcdefp-3, 0x0))
        return 1;
    if (test__fixtfsi(0x1.23456789abcdefp+20, 0x123456))
        return 1;
    if (test__fixtfsi(0x1.23456789abcdefp+40, 0x7fffffff))
        return 1;
    if (test__fixtfsi(0x1.23456789abcdefp+256, 0x7fffffff))
        return 1;
    if (test__fixtfsi(-0x1.23456789abcdefp+20, 0xffedcbaa))
        return 1;
    if (test__fixtfsi(-0x1.23456789abcdefp+40, 0x80000000))
        return 1;

#else
    printf("skipped\n");

#endif
    return 0;
}

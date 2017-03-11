// RUN: %clang_builtins %s %librt -o %t && %run %t
//===------------ gttf2_test.c - Test __gttf2------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file tests __gttf2 for the compiler_rt library.
//
//===----------------------------------------------------------------------===//

#include <stdio.h>

#if __LP64__ && __LDBL_MANT_DIG__ == 113

#include "fp_test.h"

int __gttf2(long double a, long double b);

int test__gttf2(long double a, long double b, enum EXPECTED_RESULT expected)
{
    int x = __gttf2(a, b);
    int ret = compareResultCMP(x, expected);

    if (ret){
        printf("error in test__gttf2(%.20Lf, %.20Lf) = %d, "
               "expected %s\n", a, b, x, expectedStr(expected));
    }
    return ret;
}

char assumption_1[sizeof(long double) * CHAR_BIT == 128] = {0};

#endif

int main()
{
#if __LP64__ && __LDBL_MANT_DIG__ == 113
    // NaN
    if (test__gttf2(makeQNaN128(),
                    0x1.234567890abcdef1234567890abcp+3L,
                    LESS_EQUAL_0))
        return 1;
    // <
    // exp
    if (test__gttf2(0x1.234567890abcdef1234567890abcp-3L,
                    0x1.234567890abcdef1234567890abcp+3L,
                    LESS_EQUAL_0))
        return 1;
    // mantissa
    if (test__gttf2(0x1.234567890abcdef1234567890abcp+3L,
                    0x1.334567890abcdef1234567890abcp+3L,
                    LESS_EQUAL_0))
        return 1;
    // sign
    if (test__gttf2(-0x1.234567890abcdef1234567890abcp+3L,
                    0x1.234567890abcdef1234567890abcp+3L,
                    LESS_EQUAL_0))
        return 1;
    // ==
    if (test__gttf2(0x1.234567890abcdef1234567890abcp+3L,
                    0x1.234567890abcdef1234567890abcp+3L,
                    LESS_EQUAL_0))
        return 1;
    // >
    // exp
    if (test__gttf2(0x1.234567890abcdef1234567890abcp+3L,
                    0x1.234567890abcdef1234567890abcp-3L,
                    GREATER_0))
        return 1;
    // mantissa
    if (test__gttf2(0x1.334567890abcdef1234567890abcp+3L,
                    0x1.234567890abcdef1234567890abcp+3L,
                    GREATER_0))
        return 1;
    // sign
    if (test__gttf2(0x1.234567890abcdef1234567890abcp+3L,
                    -0x1.234567890abcdef1234567890abcp+3L,
                    GREATER_0))
        return 1;

#else
    printf("skipped\n");

#endif
    return 0;
}

// RUN: %clang_builtins %s %librt -o %t && %run %t

//===--------------- truncsfhf2_test.c - Test __truncsfhf2 ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file tests __truncsfhf2 for the compiler_rt library.
//
//===----------------------------------------------------------------------===//

#include <stdio.h>

#include "fp_test.h"

uint16_t __truncsfhf2(float a);

int test__truncsfhf2(float a, uint16_t expected)
{
    uint16_t x = __truncsfhf2(a);
    int ret = compareResultH(x, expected);

    if (ret){
        printf("error in test__truncsfhf2(%f) = %#.4x, "
               "expected %#.4x\n", a, x, fromRep16(expected));
    }
    return ret;
}

char assumption_1[sizeof(__fp16) * CHAR_BIT == 16] = {0};

int main()
{
    // qNaN
    if (test__truncsfhf2(makeQNaN32(),
                         UINT16_C(0x7e00)))
        return 1;
    // NaN
    if (test__truncsfhf2(makeNaN32(UINT32_C(0x8000)),
                         UINT16_C(0x7e00)))
        return 1;
    // inf
    if (test__truncsfhf2(makeInf32(),
                         UINT16_C(0x7c00)))
        return 1;
    if (test__truncsfhf2(-makeInf32(),
                         UINT16_C(0xfc00)))
        return 1;
    // zero
    if (test__truncsfhf2(0.0f, UINT16_C(0x0)))
        return 1;
    if (test__truncsfhf2(-0.0f, UINT16_C(0x8000)))
        return 1;

    if (test__truncsfhf2(3.1415926535f,
                         UINT16_C(0x4248)))
        return 1;
    if (test__truncsfhf2(-3.1415926535f,
                         UINT16_C(0xc248)))
        return 1;
    if (test__truncsfhf2(0x1.987124876876324p+100f,
                         UINT16_C(0x7c00)))
        return 1;
    if (test__truncsfhf2(0x1.987124876876324p+12f,
                         UINT16_C(0x6e62)))
        return 1;
    if (test__truncsfhf2(0x1.0p+0f,
                         UINT16_C(0x3c00)))
        return 1;
    if (test__truncsfhf2(0x1.0p-14f,
                         UINT16_C(0x0400)))
        return 1;
    // denormal
    if (test__truncsfhf2(0x1.0p-20f,
                         UINT16_C(0x0010)))
        return 1;
    if (test__truncsfhf2(0x1.0p-24f,
                         UINT16_C(0x0001)))
        return 1;
    if (test__truncsfhf2(-0x1.0p-24f,
                         UINT16_C(0x8001)))
        return 1;
    if (test__truncsfhf2(0x1.5p-25f,
                         UINT16_C(0x0001)))
        return 1;
    // and back to zero
    if (test__truncsfhf2(0x1.0p-25f,
                         UINT16_C(0x0000)))
        return 1;
    if (test__truncsfhf2(-0x1.0p-25f,
                         UINT16_C(0x8000)))
        return 1;
    // max (precise)
    if (test__truncsfhf2(65504.0f,
                         UINT16_C(0x7bff)))
        return 1;
    // max (rounded)
    if (test__truncsfhf2(65519.0f,
                         UINT16_C(0x7bff)))
        return 1;
    // max (to +inf)
    if (test__truncsfhf2(65520.0f,
                         UINT16_C(0x7c00)))
        return 1;
    if (test__truncsfhf2(65536.0f,
                         UINT16_C(0x7c00)))
        return 1;
    if (test__truncsfhf2(-65520.0f,
                         UINT16_C(0xfc00)))
        return 1;
    return 0;
}

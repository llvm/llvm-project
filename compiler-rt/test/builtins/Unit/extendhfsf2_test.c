//===--------------- extendhfsf2_test.c - Test __extendhfsf2 --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file tests __extendhfsf2 for the compiler_rt library.
//
//===----------------------------------------------------------------------===//

#include <stdio.h>

#include "fp_test.h"

float __extendhfsf2(uint16_t a);

int test__extendhfsf2(uint16_t a, float expected)
{
    float x = __extendhfsf2(a);
    int ret = compareResultH(x, expected);

    if (ret){
        printf("error in test__extendhfsf2(%#.4x) = %f, "
               "expected %f\n", a, x, expected);
    }
    return ret;
}

char assumption_1[sizeof(__fp16) * CHAR_BIT == 16] = {0};

int main()
{
    // qNaN
    if (test__extendhfsf2(UINT16_C(0x7e00),
                          makeQNaN32()))
        return 1;
    // NaN
    if (test__extendhfsf2(UINT16_C(0x7e00),
                          makeNaN32(UINT32_C(0x8000))))
        return 1;
    // inf
    if (test__extendhfsf2(UINT16_C(0x7c00),
                          makeInf32()))
        return 1;
    if (test__extendhfsf2(UINT16_C(0xfc00),
                          -makeInf32()))
        return 1;
    // zero
    if (test__extendhfsf2(UINT16_C(0x0),
                          0.0f))
        return 1;
    if (test__extendhfsf2(UINT16_C(0x8000),
                          -0.0f))
        return 1;

    if (test__extendhfsf2(UINT16_C(0x4248),
                          3.1415926535f))
        return 1;
    if (test__extendhfsf2(UINT16_C(0xc248),
                          -3.1415926535f))
        return 1;
    if (test__extendhfsf2(UINT16_C(0x7c00),
                          0x1.987124876876324p+100f))
        return 1;
    if (test__extendhfsf2(UINT16_C(0x6e62),
                          0x1.988p+12f))
        return 1;
    if (test__extendhfsf2(UINT16_C(0x3c00),
                          0x1.0p+0f))
        return 1;
    if (test__extendhfsf2(UINT16_C(0x0400),
                          0x1.0p-14f))
        return 1;
    // denormal
    if (test__extendhfsf2(UINT16_C(0x0010),
                          0x1.0p-20f))
        return 1;
    if (test__extendhfsf2(UINT16_C(0x0001),
                          0x1.0p-24f))
        return 1;
    if (test__extendhfsf2(UINT16_C(0x8001),
                          -0x1.0p-24f))
        return 1;
    if (test__extendhfsf2(UINT16_C(0x0001),
                          0x1.5p-25f))
        return 1;
    // and back to zero
    if (test__extendhfsf2(UINT16_C(0x0000),
                          0x1.0p-25f))
        return 1;
    if (test__extendhfsf2(UINT16_C(0x8000),
                          -0x1.0p-25f))
        return 1;
    // max (precise)
    if (test__extendhfsf2(UINT16_C(0x7bff),
                          65504.0f))
        return 1;
    // max (rounded)
    if (test__extendhfsf2(UINT16_C(0x7bff),
                          65504.0f))
        return 1;
    // max (to +inf)
    if (test__extendhfsf2(UINT16_C(0x7c00),
                          makeInf32()))
        return 1;
    if (test__extendhfsf2(UINT16_C(0xfc00),
                          -makeInf32()))
        return 1;
    return 0;
}

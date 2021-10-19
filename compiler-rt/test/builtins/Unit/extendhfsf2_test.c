// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_extendhfsf2

#include <stdio.h>

#include "fp_test.h"

float __extendhfsf2(TYPE_FP16 a);

int test__extendhfsf2(TYPE_FP16 a, uint32_t expected)
{
    float x = __extendhfsf2(a);
    int ret = compareResultF(x, expected);

    if (ret){
        printf("error in test__extendhfsf2(%#.4x) = %f, "
               "expected %f\n", toRep16(a), x, fromRep32(expected));
    }
    return ret;
}

char assumption_1[sizeof(TYPE_FP16) * CHAR_BIT == 16] = {0};

int main()
{
    // qNaN
    if (test__extendhfsf2(fromRep16(0x7e00),
                          UINT32_C(0x7fc00000)))
        return 1;
    // NaN
    if (test__extendhfsf2(fromRep16(0x7f80),
                          UINT32_C(0x7ff00000)))
        return 1;
    // inf
    if (test__extendhfsf2(fromRep16(0x7c00),
                          UINT32_C(0x7f800000)))
        return 1;
    // -inf
    if (test__extendhfsf2(fromRep16(0xfc00),
                          UINT32_C(0xff800000)))
        return 1;
    // zero
    if (test__extendhfsf2(fromRep16(0x0),
                          UINT32_C(0x00000000)))
        return 1;
    // -zero
    if (test__extendhfsf2(fromRep16(0x8000),
                          UINT32_C(0x80000000)))
        return 1;
    if (test__extendhfsf2(fromRep16(0x4248),
                          UINT32_C(0x40490000)))
        return 1;
    if (test__extendhfsf2(fromRep16(0xc248),
                          UINT32_C(0xc0490000)))
        return 1;
    if (test__extendhfsf2(fromRep16(0x6e62),
                          UINT32_C(0x45cc4000)))
        return 1;
    if (test__extendhfsf2(fromRep16(0x3c00),
                          UINT32_C(0x3f800000)))
        return 1;
    if (test__extendhfsf2(fromRep16(0x0400),
                          UINT32_C(0x38800000)))
        return 1;
    // denormal
    if (test__extendhfsf2(fromRep16(0x0010),
                          UINT32_C(0x35800000)))
        return 1;
    if (test__extendhfsf2(fromRep16(0x0001),
                          UINT32_C(0x33800000)))
        return 1;
    if (test__extendhfsf2(fromRep16(0x8001),
                          UINT32_C(0xb3800000)))
        return 1;
    if (test__extendhfsf2(fromRep16(0x0001),
                          UINT32_C(0x33800000)))
        return 1;
    // max (precise)
    if (test__extendhfsf2(fromRep16(0x7bff),
                          UINT32_C(0x477fe000)))
        return 1;
    // max (rounded)
    if (test__extendhfsf2(fromRep16(0x7bff),
                          UINT32_C(0x477fe000)))
        return 1;
    return 0;
}

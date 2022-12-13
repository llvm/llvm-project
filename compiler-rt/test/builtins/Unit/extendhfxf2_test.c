// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_extendhfxf2

#include <stdio.h>

#include "fp_test.h"

long double __extendhfxf2(TYPE_FP16 a);

int test__extendhfxf2(TYPE_FP16 a, uint64_t expectedHi, uint64_t expectedLo)
{
    long double x = __extendhfxf2(a);
    int ret = compareResultLD(x, expectedHi, expectedLo);

    if (ret){
        printf("error in test__extendhfxf2(%#.4x) = %Lf, "
               "expected %Lf\n", toRep16(a), x, fromRep80(expectedHi, expectedLo));

    }
    return ret;
}

char assumption_1[sizeof(TYPE_FP16) * CHAR_BIT == 16] = {0};

int main()
{
    // qNaN
    if (test__extendhfxf2(fromRep16(0x7e00),
                          UINT64_C(0x7fff),
                          UINT64_C(0xc000000000000000)))
        return 1;
    // NaN
    if (test__extendhfxf2(fromRep16(0x7f80),
                          UINT64_C(0x7fff),
                          UINT64_C(0xf000000000000000)))
        return 1;
    // inf
    if (test__extendhfxf2(fromRep16(0x7c00),
                          UINT64_C(0x7fff),
                          UINT64_C(0x8000000000000000)))
        return 1;
    // -inf
    if (test__extendhfxf2(fromRep16(0xfc00),
                          UINT64_C(0xffff),
                          UINT64_C(0x8000000000000000)))
        return 1;
    // zero
    if (test__extendhfxf2(fromRep16(0x0),
                          UINT64_C(0x0000),
                          UINT64_C(0x0000000000000000)))
        return 1;
    // -zero
    if (test__extendhfxf2(fromRep16(0x8000),
                          UINT64_C(0x8000),
                          UINT64_C(0x0000000000000000)))
        return 1;
    if (test__extendhfxf2(fromRep16(0x4248),
                          UINT64_C(0x4000),
                          UINT64_C(0xc900000000000000)))
        return 1;
    if (test__extendhfxf2(fromRep16(0xc248),
                          UINT64_C(0xc000),
                          UINT64_C(0xc900000000000000)))
        return 1;
    if (test__extendhfxf2(fromRep16(0x6e62),
                          UINT64_C(0x400b),
                          UINT64_C(0xcc40000000000000)))
        return 1;
    if (test__extendhfxf2(fromRep16(0x3c00),
                          UINT64_C(0x3fff),
                          UINT64_C(0x8000000000000000)))
        return 1;
    if (test__extendhfxf2(fromRep16(0x0400),
                          UINT64_C(0x3ff1),
                          UINT64_C(0x8000000000000000)))
        return 1;
    // denormal
    if (test__extendhfxf2(fromRep16(0x0010),
                          UINT64_C(0x3feb),
                          UINT64_C(0x8000000000000000)))
        return 1;
    if (test__extendhfxf2(fromRep16(0x0001),
                          UINT64_C(0x3fe7),
                          UINT64_C(0x8000000000000000)))
        return 1;
    if (test__extendhfxf2(fromRep16(0x8001),
                          UINT64_C(0xbfe7),
                          UINT64_C(0x8000000000000000)))
        return 1;
    // max (precise)
    if (test__extendhfxf2(fromRep16(0x7bff),
                          UINT64_C(0x400e),
                          UINT64_C(0xffe0000000000000)))
        return 1;
    return 0;
}

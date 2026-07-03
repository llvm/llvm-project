// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_extendhfdf2

#include <stdio.h>

#include "fp_test.h"

double __extendhfdf2(TYPE_FP16 a);

int test__extendhfdf2(TYPE_FP16 a, uint64_t expected)
{
    double x = __extendhfdf2(a);
    int ret = compareResultD(x, expected);

    if (ret){
        printf("error in test__extendhfdf2(%#.4x) = %f, "
               "expected %f\n", toRep16(a), x, fromRep64(expected));
    }
    return ret;
}

char assumption_1[sizeof(TYPE_FP16) * CHAR_BIT == 16] = {0};

int main()
{
    // qNaN
    if (test__extendhfdf2(makeQNaN16(),
                          UINT64_C(0x7ff8000000000000)))
        return 1;
    // NaN
    if (test__extendhfdf2(fromRep16(0x7f80),
                          UINT64_C(0x7ffe000000000000)))
        return 1;
    // inf
    if (test__extendhfdf2(makeInf16(),
                          UINT64_C(0x7ff0000000000000)))
        return 1;
    // -inf
    if (test__extendhfdf2(makeNegativeInf16(),
                          UINT64_C(0xfff0000000000000)))
        return 1;
    // zero
    if (test__extendhfdf2(fromRep16(0x0),
                          UINT64_C(0x0)))
        return 1;
    // -zero
    if (test__extendhfdf2(fromRep16(0x8000),
                          UINT64_C(0x8000000000000000)))
        return 1;
    if (test__extendhfdf2(fromRep16(0x4248),
                          UINT64_C(0x4009200000000000)))
        return 1;
    if (test__extendhfdf2(fromRep16(0xc248),
                          UINT64_C(0xc009200000000000)))
        return 1;
    if (test__extendhfdf2(fromRep16(0x6e62),
                          UINT64_C(0x40b9880000000000)))
        return 1;
    if (test__extendhfdf2(fromRep16(0x3c00),
                          UINT64_C(0x3ff0000000000000)))
        return 1;
    if (test__extendhfdf2(fromRep16(0x0400),
                          UINT64_C(0x3f10000000000000)))
        return 1;
    // denormal
    if (test__extendhfdf2(fromRep16(0x0010),
                          UINT64_C(0x3eb0000000000000)))
        return 1;
    if (test__extendhfdf2(fromRep16(0x0001),
                          UINT64_C(0x3e70000000000000)))
        return 1;
    if (test__extendhfdf2(fromRep16(0x8001),
                          UINT64_C(0xbe70000000000000)))
        return 1;
    if (test__extendhfdf2(fromRep16(0x0001),
                          UINT64_C(0x3e70000000000000)))
        return 1;
    // max (precise)
    if (test__extendhfdf2(fromRep16(0x7bff),
                          UINT64_C(0x40effc0000000000)))
        return 1;
    // max (rounded)
    if (test__extendhfdf2(fromRep16(0x7bff),
                          UINT64_C(0x40effc0000000000)))
        return 1;
    return 0;
}

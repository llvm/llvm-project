// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_truncdfhf2

#include <stdio.h>

#include "fp_test.h"

TYPE_FP16 __truncdfhf2(double a);

int test__truncdfhf2(double a, uint16_t expected)
{
    TYPE_FP16 x = __truncdfhf2(a);
    int ret = compareResultH(x, expected);

    if (ret){
        printf("error in test__truncdfhf2(%lf) = %#.4x, "
               "expected %#.4x\n", a, toRep16(x), expected);
    }
    return ret;
}

char assumption_1[sizeof(TYPE_FP16) * CHAR_BIT == 16] = {0};

int main()
{
    // qNaN
    if (test__truncdfhf2(makeQNaN64(),
                         UINT16_C(0x7e00)))
        return 1;
    // NaN
    if (test__truncdfhf2(makeNaN64(UINT64_C(0x8000)),
                         UINT16_C(0x7e00)))
        return 1;
    // inf
    if (test__truncdfhf2(makeInf64(),
                         UINT16_C(0x7c00)))
        return 1;
    if (test__truncdfhf2(-makeInf64(),
                         UINT16_C(0xfc00)))
        return 1;
    // zero
    if (test__truncdfhf2(0.0, UINT16_C(0x0)))
        return 1;
    if (test__truncdfhf2(-0.0, UINT16_C(0x8000)))
        return 1;

    if (test__truncdfhf2(3.1415926535,
                         UINT16_C(0x4248)))
        return 1;
    if (test__truncdfhf2(-3.1415926535,
                         UINT16_C(0xc248)))
        return 1;
    if (test__truncdfhf2(0x1.987124876876324p+1000,
                         UINT16_C(0x7c00)))
        return 1;
    if (test__truncdfhf2(0x1.987124876876324p+12,
                         UINT16_C(0x6e62)))
        return 1;
    if (test__truncdfhf2(0x1.0p+0,
                         UINT16_C(0x3c00)))
        return 1;
    if (test__truncdfhf2(0x1.0p-14,
                         UINT16_C(0x0400)))
        return 1;
    // denormal
    if (test__truncdfhf2(0x1.0p-20,
                         UINT16_C(0x0010)))
        return 1;
    if (test__truncdfhf2(0x1.0p-24,
                         UINT16_C(0x0001)))
        return 1;
    if (test__truncdfhf2(-0x1.0p-24,
                         UINT16_C(0x8001)))
        return 1;
    if (test__truncdfhf2(0x1.5p-25,
                         UINT16_C(0x0001)))
        return 1;
    // and back to zero
    if (test__truncdfhf2(0x1.0p-25,
                         UINT16_C(0x0000)))
        return 1;
    if (test__truncdfhf2(-0x1.0p-25,
                         UINT16_C(0x8000)))
        return 1;
    // max (precise)
    if (test__truncdfhf2(65504.0,
                         UINT16_C(0x7bff)))
        return 1;
    // max (rounded)
    if (test__truncdfhf2(65519.0,
                         UINT16_C(0x7bff)))
        return 1;
    // max (to +inf)
    if (test__truncdfhf2(65520.0,
                         UINT16_C(0x7c00)))
        return 1;
    if (test__truncdfhf2(-65520.0,
                         UINT16_C(0xfc00)))
        return 1;
    if (test__truncdfhf2(65536.0,
                         UINT16_C(0x7c00)))
        return 1;
    return 0;
}

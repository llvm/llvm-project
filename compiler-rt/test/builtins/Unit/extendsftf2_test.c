// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_extendsftf2

#include "int_lib.h"
#include <stdio.h>

#if __LDBL_MANT_DIG__ == 113

#include "fp_test.h"

COMPILER_RT_ABI long double __extendsftf2(float a);

int test__extendsftf2(float a, uint64_t expectedHi, uint64_t expectedLo)
{
    long double x = __extendsftf2(a);
    int ret = compareResultLD(x, expectedHi, expectedLo);

    if (ret)
    {
        printf("error in test__extendsftf2(%f) = %.20Lf, "
               "expected %.20Lf\n", a, x, fromRep128(expectedHi, expectedLo));
    }
    return ret;
}

char assumption_1[sizeof(long double) * CHAR_BIT == 128] = {0};

#endif

int main()
{
#if __LDBL_MANT_DIG__ == 113
    // qNaN
    if (test__extendsftf2(makeQNaN32(),
                          UINT64_C(0x7fff800000000000),
                          UINT64_C(0x0)))
        return 1;
    // NaN
    if (test__extendsftf2(makeNaN32(UINT32_C(0x410000)),
                          UINT64_C(0x7fff820000000000),
                          UINT64_C(0x0)))
        return 1;
    // inf
    if (test__extendsftf2(makeInf32(),
                          UINT64_C(0x7fff000000000000),
                          UINT64_C(0x0)))
        return 1;
    // zero
    if (test__extendsftf2(0.0f, UINT64_C(0x0), UINT64_C(0x0)))
        return 1;

    if (test__extendsftf2(0x1.23456p+5f,
                          UINT64_C(0x4004234560000000),
                          UINT64_C(0x0)))
        return 1;
    if (test__extendsftf2(0x1.edcbap-9f,
                          UINT64_C(0x3ff6edcba0000000),
                          UINT64_C(0x0)))
        return 1;
    if (test__extendsftf2(0x1.23456p+45f,
                          UINT64_C(0x402c234560000000),
                          UINT64_C(0x0)))
        return 1;
    if (test__extendsftf2(0x1.edcbap-45f,
                          UINT64_C(0x3fd2edcba0000000),
                          UINT64_C(0x0)))
        return 1;

#else
    printf("skipped\n");

#endif
    return 0;
}

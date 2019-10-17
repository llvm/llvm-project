// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_extenddftf2
//===--------------- extenddftf2_test.c - Test __extenddftf2 --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file tests __extenddftf2 for the compiler_rt library.
//
//===----------------------------------------------------------------------===//

#include "int_lib.h"
#include <stdio.h>

#if __LDBL_MANT_DIG__ == 113

#include "fp_test.h"

COMPILER_RT_ABI long double __extenddftf2(double a);

int test__extenddftf2(double a, uint64_t expectedHi, uint64_t expectedLo)
{
    long double x = __extenddftf2(a);
    int ret = compareResultLD(x, expectedHi, expectedLo);

    if (ret){
        printf("error in test__extenddftf2(%f) = %.20Lf, "
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
    if (test__extenddftf2(makeQNaN64(),
                          UINT64_C(0x7fff800000000000),
                          UINT64_C(0x0)))
        return 1;
    // NaN
    if (test__extenddftf2(makeNaN64(UINT64_C(0x7100000000000)),
                          UINT64_C(0x7fff710000000000),
                          UINT64_C(0x0)))
        return 1;
    // inf
    if (test__extenddftf2(makeInf64(),
                          UINT64_C(0x7fff000000000000),
                          UINT64_C(0x0)))
        return 1;
    // zero
    if (test__extenddftf2(0.0, UINT64_C(0x0), UINT64_C(0x0)))
        return 1;

    if (test__extenddftf2(0x1.23456789abcdefp+5,
                          UINT64_C(0x400423456789abcd),
                          UINT64_C(0xf000000000000000)))
        return 1;
    if (test__extenddftf2(0x1.edcba987654321fp-9,
                          UINT64_C(0x3ff6edcba9876543),
                          UINT64_C(0x2000000000000000)))
        return 1;
    if (test__extenddftf2(0x1.23456789abcdefp+45,
                          UINT64_C(0x402c23456789abcd),
                          UINT64_C(0xf000000000000000)))
        return 1;
    if (test__extenddftf2(0x1.edcba987654321fp-45,
                          UINT64_C(0x3fd2edcba9876543),
                          UINT64_C(0x2000000000000000)))
        return 1;

#else
    printf("skipped\n");

#endif
    return 0;
}

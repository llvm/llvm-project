// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_fixunstfsi
//===--------------- fixunstfsi_test.c - Test __fixunstfsi ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file tests __fixunstfsi for the compiler_rt library.
//
//===----------------------------------------------------------------------===//

#include <stdio.h>

#if __LDBL_MANT_DIG__ == 113

#include "fp_test.h"

unsigned int __fixunstfsi(long double a);

int test__fixunstfsi(long double a, unsigned int expected)
{
    unsigned int x = __fixunstfsi(a);
    int ret = (x != expected);

    if (ret)
    {
        printf("error in test__fixunstfsi(%.20Lf) = %u, "
               "expected %u\n", a, x, expected);
    }
    return ret;
}

char assumption_1[sizeof(long double) * CHAR_BIT == 128] = {0};

#endif

int main()
{
#if __LDBL_MANT_DIG__ == 113
    if (test__fixunstfsi(makeInf128(), UINT32_C(0xffffffff)))
        return 1;
    if (test__fixunstfsi(0, UINT32_C(0x0)))
        return 1;
    if (test__fixunstfsi(0x1.23456789abcdefp+5, UINT32_C(0x24)))
        return 1;
    if (test__fixunstfsi(0x1.23456789abcdefp-3, UINT32_C(0x0)))
        return 1;
    if (test__fixunstfsi(0x1.23456789abcdefp+20, UINT32_C(0x123456)))
        return 1;
    if (test__fixunstfsi(0x1.23456789abcdefp+40, UINT32_C(0xffffffff)))
        return 1;
    if (test__fixunstfsi(0x1.23456789abcdefp+256, UINT32_C(0xffffffff)))
        return 1;
    if (test__fixunstfsi(-0x1.23456789abcdefp+3, UINT32_C(0x0)))
        return 1;

    if (test__fixunstfsi(0x1.p+32, 0xFFFFFFFFLL))
        return 1;

#else
    printf("skipped\n");

#endif
    return 0;
}

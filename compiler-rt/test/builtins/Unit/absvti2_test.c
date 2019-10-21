// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_absvti2
// REQUIRES: int128
//===-- absvti2_test.c - Test __absvti2 -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file tests __absvti2 for the compiler_rt library.
//
//===----------------------------------------------------------------------===//

#include "int_lib.h"
#include <stdio.h>
#include <stdlib.h>

#ifdef CRT_HAS_128BIT

// Returns: absolute value

// Effects: aborts if abs(x) < 0

COMPILER_RT_ABI ti_int __absvti2(ti_int a);

int test__absvti2(ti_int a)
{
    ti_int x = __absvti2(a);
    ti_int expected = a;
    if (expected < 0)
        expected = -expected;
    if (x != expected || expected < 0)
    {
        twords at;
        at.all = a;
        twords xt;
        xt.all = x;
        twords expectedt;
        expectedt.all = expected;
        printf("error in __absvti2(0x%llX%.16llX) = "
               "0x%llX%.16llX, expected positive 0x%llX%.16llX\n",
               at.s.high, at.s.low, xt.s.high, xt.s.low,
               expectedt.s.high, expectedt.s.low);
    }
    return x != expected;
}

#endif

int main()
{
#ifdef CRT_HAS_128BIT

//     if (test__absvti2(make_ti(0x8000000000000000LL, 0)))  // should abort
//         return 1;
    if (test__absvti2(0x0000000000000000LL))
        return 1;
    if (test__absvti2(0x0000000000000001LL))
        return 1;
    if (test__absvti2(0x0000000000000002LL))
        return 1;
    if (test__absvti2(make_ti(0x7FFFFFFFFFFFFFFFLL, 0xFFFFFFFFFFFFFFFELL)))
        return 1;
    if (test__absvti2(make_ti(0x7FFFFFFFFFFFFFFFLL, 0xFFFFFFFFFFFFFFFFLL)))
        return 1;
    if (test__absvti2(make_ti(0x8000000000000000LL, 0x0000000000000001LL)))
        return 1;
    if (test__absvti2(make_ti(0x8000000000000000LL, 0x0000000000000002LL)))
        return 1;
    if (test__absvti2(make_ti(0xFFFFFFFFFFFFFFFFLL, 0xFFFFFFFFFFFFFFFELL)))
        return 1;
    if (test__absvti2(make_ti(0xFFFFFFFFFFFFFFFFLL, 0xFFFFFFFFFFFFFFFFLL)))
        return 1;

    int i;
    for (i = 0; i < 10000; ++i)
        if (test__absvti2(make_ti(((ti_int)rand() << 32) | rand(),
                                  ((ti_int)rand() << 32) | rand())))
            return 1;
#else
    printf("skipped\n");
#endif
    return 0;
}

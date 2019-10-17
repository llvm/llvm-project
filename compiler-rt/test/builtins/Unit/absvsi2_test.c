// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_absvsi2
//===-- absvsi2_test.c - Test __absvsi2 -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file tests __absvsi2 for the compiler_rt library.
//
//===----------------------------------------------------------------------===//

#include "int_lib.h"
#include <stdio.h>
#include <stdlib.h>

// Returns: absolute value

// Effects: aborts if abs(x) < 0

COMPILER_RT_ABI si_int __absvsi2(si_int a);

int test__absvsi2(si_int a)
{
    si_int x = __absvsi2(a);
    si_int expected = a;
    if (expected < 0)
        expected = -expected;
    if (x != expected || expected < 0)
        printf("error in __absvsi2(0x%X) = %d, expected positive %d\n",
               a, x, expected);
    return x != expected;
}

int main()
{
//     if (test__absvsi2(0x80000000))  // should abort
//         return 1;
    if (test__absvsi2(0x00000000))
        return 1;
    if (test__absvsi2(0x00000001))
        return 1;
    if (test__absvsi2(0x00000002))
        return 1;
    if (test__absvsi2(0x7FFFFFFE))
        return 1;
    if (test__absvsi2(0x7FFFFFFF))
        return 1;
    if (test__absvsi2(0x80000001))
        return 1;
    if (test__absvsi2(0x80000002))
        return 1;
    if (test__absvsi2(0xFFFFFFFE))
        return 1;
    if (test__absvsi2(0xFFFFFFFF))
        return 1;

    int i;
    for (i = 0; i < 10000; ++i)
        if (test__absvsi2(rand()))
            return 1;

    return 0;
}

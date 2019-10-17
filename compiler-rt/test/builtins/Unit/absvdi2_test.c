// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_absvdi2
//===-- absvdi2_test.c - Test __absvdi2 -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file tests __absvdi2 for the compiler_rt library.
//
//===----------------------------------------------------------------------===//

#include "int_lib.h"
#include <stdio.h>
#include <stdlib.h>

// Returns: absolute value

// Effects: aborts if abs(x) < 0

COMPILER_RT_ABI di_int __absvdi2(di_int a);

int test__absvdi2(di_int a)
{
    di_int x = __absvdi2(a);
    di_int expected = a;
    if (expected < 0)
        expected = -expected;
    if (x != expected || expected < 0)
        printf("error in __absvdi2(0x%llX) = %lld, expected positive %lld\n",
               a, x, expected);
    return x != expected;
}

int main()
{
//     if (test__absvdi2(0x8000000000000000LL))  // should abort
//         return 1;
    if (test__absvdi2(0x0000000000000000LL))
        return 1;
    if (test__absvdi2(0x0000000000000001LL))
        return 1;
    if (test__absvdi2(0x0000000000000002LL))
        return 1;
    if (test__absvdi2(0x7FFFFFFFFFFFFFFELL))
        return 1;
    if (test__absvdi2(0x7FFFFFFFFFFFFFFFLL))
        return 1;
    if (test__absvdi2(0x8000000000000001LL))
        return 1;
    if (test__absvdi2(0x8000000000000002LL))
        return 1;
    if (test__absvdi2(0xFFFFFFFFFFFFFFFELL))
        return 1;
    if (test__absvdi2(0xFFFFFFFFFFFFFFFFLL))
        return 1;

    int i;
    for (i = 0; i < 10000; ++i)
        if (test__absvdi2(((di_int)rand() << 32) | rand()))
            return 1;

    return 0;
}

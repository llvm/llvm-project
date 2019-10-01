// RUN: %clang_builtins %s %librt -o %t && %run %t
//===-- divdi3_test.c - Test __divdi3 -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file tests __divdi3 for the compiler_rt library.
//
//===----------------------------------------------------------------------===//

#include "int_lib.h"
#include <stdio.h>

// Returns: a / b

COMPILER_RT_ABI di_int __divdi3(di_int a, di_int b);

int test__divdi3(di_int a, di_int b, di_int expected)
{
    di_int x = __divdi3(a, b);
    if (x != expected)
        printf("error in __divdi3: %lld / %lld = %lld, expected %lld\n",
               a, b, x, expected);
    return x != expected;
}

char assumption_1[sizeof(di_int) == 2*sizeof(si_int)] = {0};

int main()
{
    if (test__divdi3(0, 1, 0))
        return 1;
    if (test__divdi3(0, -1, 0))
        return 1;

    if (test__divdi3(2, 1, 2))
        return 1;
    if (test__divdi3(2, -1, -2))
        return 1;
    if (test__divdi3(-2, 1, -2))
        return 1;
    if (test__divdi3(-2, -1, 2))
        return 1;

    if (test__divdi3(0x8000000000000000LL, 1, 0x8000000000000000LL))
        return 1;
    if (test__divdi3(0x8000000000000000LL, -1, 0x8000000000000000LL))
        return 1;
    if (test__divdi3(0x8000000000000000LL, -2, 0x4000000000000000LL))
        return 1;
    if (test__divdi3(0x8000000000000000LL, 2, 0xC000000000000000LL))
        return 1;

    return 0;
}

// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: int128
//===-- ctzti2_test.c - Test __ctzti2 -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file tests __ctzti2 for the compiler_rt library.
//
//===----------------------------------------------------------------------===//

#include "int_lib.h"
#include <stdio.h>

#ifdef CRT_HAS_128BIT

// Returns: the number of trailing 0-bits

// Precondition: a != 0

COMPILER_RT_ABI si_int __ctzti2(ti_int a);

int test__ctzti2(ti_int a, si_int expected)
{
    si_int x = __ctzti2(a);
    if (x != expected)
    {
        twords at;
        at.all = a;
        printf("error in __ctzti2(0x%llX%.16llX) = %d, expected %d\n",
               at.s.high, at.s.low, x, expected);
    }
    return x != expected;
}

char assumption_1[sizeof(ti_int) == 2*sizeof(di_int)] = {0};

#endif

int main()
{
#ifdef CRT_HAS_128BIT
    if (test__ctzti2(0x00000001, 0))
        return 1;
    if (test__ctzti2(0x00000002, 1))
        return 1;
    if (test__ctzti2(0x00000003, 0))
        return 1;
    if (test__ctzti2(0x00000004, 2))
        return 1;
    if (test__ctzti2(0x00000005, 0))
        return 1;
    if (test__ctzti2(0x0000000A, 1))
        return 1;
    if (test__ctzti2(0x10000000, 28))
        return 1;
    if (test__ctzti2(0x20000000, 29))
        return 1;
    if (test__ctzti2(0x60000000, 29))
        return 1;
    if (test__ctzti2(0x80000000uLL, 31))
        return 1;
    if (test__ctzti2(0x0000050000000000uLL, 40))
        return 1;
    if (test__ctzti2(0x0200080000000000uLL, 43))
        return 1;
    if (test__ctzti2(0x7200000000000000uLL, 57))
        return 1;
    if (test__ctzti2(0x8000000000000000uLL, 63))
        return 1;
    if (test__ctzti2(make_ti(0x00000000A0000000LL, 0x0000000000000000LL), 93))
        return 1;
    if (test__ctzti2(make_ti(0xF000000000000000LL, 0x0000000000000000LL), 124))
        return 1;
    if (test__ctzti2(make_ti(0x8000000000000000LL, 0x0000000000000000LL), 127))
        return 1;
#else
    printf("skipped\n");
#endif

   return 0;
}

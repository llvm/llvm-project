// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_parityti2
// REQUIRES: int128
//===-- parityti2_test.c - Test __parityti2 -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file tests __parityti2 for the compiler_rt library.
//
//===----------------------------------------------------------------------===//

#include "int_lib.h"
#include <stdio.h>
#include <stdlib.h>

#ifdef CRT_HAS_128BIT

// Returns: 1 if number of bits is odd else returns 0

COMPILER_RT_ABI si_int __parityti2(ti_int a);

int naive_parity(ti_int a)
{
    int r = 0;
    for (; a; a = a & (a - 1))
        r = ~r;
    return r & 1;
}

int test__parityti2(ti_int a)
{
    si_int x = __parityti2(a);
    si_int expected = naive_parity(a);
    if (x != expected)
    {
        twords at;
        at.all = a;
        printf("error in __parityti2(0x%.16llX%.16llX) = %d, expected %d\n",
               at.s.high, at.s.low, x, expected);
    }
    return x != expected;
}

char assumption_1[sizeof(ti_int) == 2*sizeof(di_int)] = {0};
char assumption_2[sizeof(di_int)*CHAR_BIT == 64] = {0};

#endif

int main()
{
#ifdef CRT_HAS_128BIT
    int i;
    for (i = 0; i < 10000; ++i)
        if (test__parityti2(((ti_int)rand() << 96) + ((ti_int)rand() << 64) +
                            ((ti_int)rand() << 32) + rand()))
            return 1;

#else
    printf("skipped\n");
#endif
   return 0;
}

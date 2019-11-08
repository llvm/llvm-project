// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_negvti2
// REQUIRES: int128
//===-- negvti2_test.c - Test __negvti2 -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file tests __negvti2 for the compiler_rt library.
//
//===----------------------------------------------------------------------===//

#include "int_lib.h"
#include <stdio.h>

#ifdef CRT_HAS_128BIT

// Returns: -a

// Effects: aborts if -a overflows

COMPILER_RT_ABI ti_int __negvti2(ti_int a);
COMPILER_RT_ABI ti_int __negti2(ti_int a);

int test__negvti2(ti_int a)
{
    ti_int x = __negvti2(a);
    ti_int expected = __negti2(a);
    if (x != expected)
    {
        twords at;
        at.all = a;
        twords xt;
        xt.all = x;
        twords expectedt;
        expectedt.all = expected;
        printf("error in __negvti2(0x%.16llX%.16llX) = 0x%.16llX%.16llX, "
               "expected 0x%.16llX%.16llX\n",
               at.s.high, at.s.low, xt.s.high, xt.s.low, expectedt.s.high, expectedt.s.low);
    }
    return x != expected;
}

#endif

int main()
{
#ifdef CRT_HAS_128BIT
    if (test__negvti2(0))
        return 1;
    if (test__negvti2(1))
        return 1;
    if (test__negvti2(-1))
        return 1;
    if (test__negvti2(2))
        return 1;
    if (test__negvti2(-2))
        return 1;
    if (test__negvti2(3))
        return 1;
    if (test__negvti2(-3))
        return 1;
    if (test__negvti2(make_ti(0x0000000000000000LL, 0x00000000FFFFFFFELL)))
        return 1;
    if (test__negvti2(make_ti(0xFFFFFFFFFFFFFFFFLL, 0xFFFFFFFF00000002LL)))
        return 1;
    if (test__negvti2(make_ti(0x0000000000000000LL, 0x00000000FFFFFFFFLL)))
        return 1;
    if (test__negvti2(make_ti(0xFFFFFFFFFFFFFFFFLL, 0xFFFFFFFF00000001LL)))
        return 1;
    if (test__negvti2(make_ti(0x0000000000000000LL, 0x0000000100000000LL)))
        return 1;
    if (test__negvti2(make_ti(0xFFFFFFFFFFFFFFFFLL, 0xFFFFFFFF00000000LL)))
        return 1;
    if (test__negvti2(make_ti(0x0000000000000000LL, 0x0000000200000000LL)))
        return 1;
    if (test__negvti2(make_ti(0xFFFFFFFFFFFFFFFFLL, 0xFFFFFFFE00000000LL)))
        return 1;
    if (test__negvti2(make_ti(0x0000000000000000LL, 0x0000000300000000LL)))
        return 1;
    if (test__negvti2(make_ti(0xFFFFFFFFFFFFFFFFLL, 0xFFFFFFFD00000000LL)))
        return 1;
    if (test__negvti2(make_ti(0x0000000000000000LL, 0x7FFFFFFFFFFFFFFFLL)))
        return 1;
    if (test__negvti2(make_ti(0xFFFFFFFFFFFFFFFFLL, 0x8000000000000001LL)))
        return 1;
    if (test__negvti2(make_ti(0x0000000000000000LL, 0x7FFFFFFFFFFFFFFFLL)))
        return 1;
    if (test__negvti2(make_ti(0xFFFFFFFFFFFFFFFFLL, 0xFFFFFFFE00000000LL)))
        return 1;
    if (test__negvti2(make_ti(0x0000000000000000LL, 0x0000000200000000LL)))
        return 1;
    if (test__negvti2(make_ti(0xFFFFFFFFFFFFFFFFLL, 0xFFFFFFFF00000000LL)))
        return 1;
    if (test__negvti2(make_ti(0x0000000000000000LL, 0x0000000100000000LL)))
        return 1;
//     if (test__negvti2(make_ti(0x8000000000000000LL, 0x0000000000000000LL))) // abort
//         return 1;
    if (test__negvti2(make_ti(0x8000000000000000LL, 0x0000000000000001LL)))
        return 1;
    if (test__negvti2(make_ti(0x7FFFFFFFFFFFFFFFLL, 0xFFFFFFFFFFFFFFFFLL)))
        return 1;

#else
    printf("skipped\n");
#endif
   return 0;
}

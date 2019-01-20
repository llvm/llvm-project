// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: int128
//===-- udivti3_test.c - Test __udivti3 -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file tests __udivti3 for the compiler_rt library.
//
//===----------------------------------------------------------------------===//

#include "int_lib.h"
#include <stdio.h>

#ifdef CRT_HAS_128BIT

// Returns: a / b

COMPILER_RT_ABI tu_int __udivti3(tu_int a, tu_int b);

int test__udivti3(tu_int a, tu_int b, tu_int expected_q)
{
    tu_int q = __udivti3(a, b);
    if (q != expected_q)
    {
        utwords at;
        at.all = a;
        utwords bt;
        bt.all = b;
        utwords qt;
        qt.all = q;
        utwords expected_qt;
        expected_qt.all = expected_q;
        printf("error in __udivti3: 0x%llX%.16llX / 0x%llX%.16llX = "
               "0x%llX%.16llX, expected 0x%llX%.16llX\n",
               at.s.high, at.s.low, bt.s.high, bt.s.low, qt.s.high, qt.s.low,
               expected_qt.s.high, expected_qt.s.low);
    }
    return q != expected_q;
}

#endif

int main()
{
#ifdef CRT_HAS_128BIT
    if (test__udivti3(0, 1, 0))
        return 1;
    if (test__udivti3(2, 1, 2))
        return 1;
    if (test__udivti3(make_tu(0x8000000000000000uLL, 0), 1,
                      make_tu(0x8000000000000000uLL, 0)))
        return 1;
    if (test__udivti3(make_tu(0x8000000000000000uLL, 0), 2,
                      make_tu(0x4000000000000000uLL, 0)))
        return 1;
    if (test__udivti3(make_tu(0xFFFFFFFFFFFFFFFFuLL, 0xFFFFFFFFFFFFFFFFuLL), 2,
                      make_tu(0x7FFFFFFFFFFFFFFFuLL, 0xFFFFFFFFFFFFFFFFuLL)))
        return 1;

#else
    printf("skipped\n");
#endif
    return 0;
}

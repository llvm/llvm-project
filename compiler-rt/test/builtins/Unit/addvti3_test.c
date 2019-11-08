// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_addvti3
// REQUIRES: int128
//===-- addvti3_test.c - Test __addvti3 -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file tests __addvti3 for the compiler_rt library.
//
//===----------------------------------------------------------------------===//

#include "int_lib.h"
#include <stdio.h>

#ifdef CRT_HAS_128BIT

// Returns: a + b

// Effects: aborts if a + b overflows

COMPILER_RT_ABI ti_int __addvti3(ti_int a, ti_int b);

int test__addvti3(ti_int a, ti_int b)
{
    ti_int x = __addvti3(a, b);
    ti_int expected = a + b;
    if (x != expected)
    {
        twords at;
        at.all = a;
        twords bt;
        bt.all = b;
        twords xt;
        xt.all = x;
        twords expectedt;
        expectedt.all = expected;
        printf("error in test__addvti3(0x%llX%.16llX, 0x%llX%.16llX) = "
               "0x%llX%.16llX, expected 0x%llX%.16llX\n",
                at.s.high, at.s.low, bt.s.high, bt.s.low, xt.s.high, xt.s.low,
                expectedt.s.high, expectedt.s.low);
    }
    return x != expected;
}

#endif

int main()
{
#ifdef CRT_HAS_128BIT
// should abort
//     test__addvti3(make_ti(0x8000000000000000LL, 0x0000000000000000LL),
//                   make_ti(0xFFFFFFFFFFFFFFFFLL, 0xFFFFFFFFFFFFFFFFLL));
// should abort
//     test__addvti3(make_ti(0xFFFFFFFFFFFFFFFFLL, 0xFFFFFFFFFFFFFFFFLL),
//                   make_ti(0x8000000000000000LL, 0x0000000000000000LL));
// should abort
//     test__addvti3(make_ti(0x0000000000000000LL, 0x0000000000000001LL),
//                   make_ti(0x7FFFFFFFFFFFFFFFLL, 0xFFFFFFFFFFFFFFFFLL));
// should abort
//     test__addvti3(make_ti(0x7FFFFFFFFFFFFFFFLL, 0xFFFFFFFFFFFFFFFFLL),
//                   make_ti(0x0000000000000000LL, 0x0000000000000001LL));

    if (test__addvti3(make_ti(0x8000000000000000LL, 0x0000000000000000LL),
                      make_ti(0x0000000000000000LL, 0x0000000000000001LL)))
        return 1;
    if (test__addvti3(make_ti(0x0000000000000000LL, 0x0000000000000001LL),
                      make_ti(0x8000000000000000LL, 0x0000000000000000LL)))
        return 1;
    if (test__addvti3(make_ti(0x8000000000000000LL, 0x0000000000000000LL),
                      make_ti(0x0000000000000000LL, 0x0000000000000000LL)))
        return 1;
    if (test__addvti3(make_ti(0x0000000000000000LL, 0x0000000000000000LL),
                      make_ti(0x8000000000000000LL, 0x0000000000000000LL)))
        return 1;
    if (test__addvti3(make_ti(0x7FFFFFFFFFFFFFFFLL, 0xFFFFFFFFFFFFFFFFLL),
                      make_ti(0xFFFFFFFFFFFFFFFFLL, 0xFFFFFFFFFFFFFFFFLL)))
        return 1;
    if (test__addvti3(make_ti(0xFFFFFFFFFFFFFFFFLL, 0xFFFFFFFFFFFFFFFFLL),
                      make_ti(0x7FFFFFFFFFFFFFFFLL, 0xFFFFFFFFFFFFFFFFLL)))
        return 1;
    if (test__addvti3(make_ti(0x7FFFFFFFFFFFFFFFLL, 0xFFFFFFFFFFFFFFFFLL),
                      make_ti(0x0000000000000000LL, 0x0000000000000000LL)))
        return 1;
    if (test__addvti3(make_ti(0x0000000000000000LL, 0x0000000000000000LL),
                      make_ti(0x7FFFFFFFFFFFFFFFLL, 0xFFFFFFFFFFFFFFFFLL)))
        return 1;

#else
    printf("skipped\n");
#endif
    return 0;
}

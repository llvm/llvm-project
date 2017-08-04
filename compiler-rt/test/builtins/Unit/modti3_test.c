// RUN: %clang_builtins %s %librt -o %t && %run %t
//===-- modti3_test.c - Test __modti3 -------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file tests __modti3 for the compiler_rt library.
//
//===----------------------------------------------------------------------===//

#include "int_lib.h"
#include <stdio.h>

#ifdef CRT_HAS_128BIT

// Returns: a % b

COMPILER_RT_ABI ti_int __modti3(ti_int a, ti_int b);

int test__modti3(ti_int a, ti_int b, ti_int expected)
{
    ti_int x = __modti3(a, b);
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
        printf("error in __modti3: 0x%.16llX%.16llX %% 0x%.16llX%.16llX = "
               "0x%.16llX%.16llX, expected 0x%.16llX%.16llX\n",
               at.s.high, at.s.low, bt.s.high, bt.s.low, xt.s.high, xt.s.low,
               expectedt.s.high, expectedt.s.low);
    }
    return x != expected;
}

char assumption_1[sizeof(ti_int) == 2*sizeof(di_int)] = {0};

#endif

int main()
{
#ifdef CRT_HAS_128BIT
    if (test__modti3(0, 1, 0))
        return 1;
    if (test__modti3(0, -1, 0))
        return 1;

    if (test__modti3(5, 3, 2))
        return 1;
    if (test__modti3(5, -3, 2))
        return 1;
    if (test__modti3(-5, 3, -2))
        return 1;
    if (test__modti3(-5, -3, -2))
        return 1;

    if (test__modti3(0x8000000000000000LL, 1, 0x0LL))
        return 1;
    if (test__modti3(0x8000000000000000LL, -1, 0x0LL))
        return 1;
    if (test__modti3(0x8000000000000000LL, 2, 0x0LL))
        return 1;
    if (test__modti3(0x8000000000000000LL, -2, 0x0LL))
        return 1;
    if (test__modti3(0x8000000000000000LL, 3, 2))
        return 1;
    if (test__modti3(0x8000000000000000LL, -3, 2))
        return 1;

    if (test__modti3(make_ti(0x8000000000000000LL, 0), 1, 0x0LL))
        return 1;
    if (test__modti3(make_ti(0x8000000000000000LL, 0), -1, 0x0LL))
        return 1;
    if (test__modti3(make_ti(0x8000000000000000LL, 0), 2, 0x0LL))
        return 1;
    if (test__modti3(make_ti(0x8000000000000000LL, 0), -2, 0x0LL))
        return 1;
    if (test__modti3(make_ti(0x8000000000000000LL, 0), 3, -2))
        return 1;
    if (test__modti3(make_ti(0x8000000000000000LL, 0), -3, -2))
        return 1;

#else
    printf("skipped\n");
#endif
    return 0;
}

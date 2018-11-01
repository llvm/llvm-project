// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: int128
//===-- cmpti2_test.c - Test __cmpti2 -------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file tests __cmpti2 for the compiler_rt library.
//
//===----------------------------------------------------------------------===//

#include "int_lib.h"
#include <stdio.h>

#ifdef CRT_HAS_128BIT

// Returns:  if (a <  b) returns 0
//           if (a == b) returns 1
//           if (a >  b) returns 2

COMPILER_RT_ABI si_int __cmpti2(ti_int a, ti_int b);

int test__cmpti2(ti_int a, ti_int b, si_int expected)
{
    si_int x = __cmpti2(a, b);
    if (x != expected)
    {
        twords at;
        at.all = a;
        twords bt;
        bt.all = b;
        printf("error in __cmpti2(0x%llX%.16llX, 0x%llX%.16llX) = %d, expected %d\n",
               at.s.high, at.s.low, bt.s.high, bt.s.low, x, expected);
    }
    return x != expected;
}

char assumption_1[sizeof(ti_int) == 2*sizeof(di_int)] = {0};

#endif

int main()
{
#ifdef CRT_HAS_128BIT
    if (test__cmpti2(0, 0, 1))
        return 1;
    if (test__cmpti2(1, 1, 1))
        return 1;
    if (test__cmpti2(2, 2, 1))
        return 1;
    if (test__cmpti2(0x7FFFFFFF, 0x7FFFFFFF, 1))
        return 1;
    if (test__cmpti2(0x80000000, 0x80000000, 1))
        return 1;
    if (test__cmpti2(0x80000001, 0x80000001, 1))
        return 1;
    if (test__cmpti2(0xFFFFFFFF, 0xFFFFFFFF, 1))
        return 1;
    if (test__cmpti2(0x000000010000000LL, 0x000000010000000LL, 1))
        return 1;
    if (test__cmpti2(0xFFFFFFFFFFFFFFFFLL, 0xFFFFFFFFFFFFFFFFLL, 1))
        return 1;

    if (test__cmpti2(0x0000000200000002LL, 0x0000000300000001LL, 0))
        return 1;
    if (test__cmpti2(0x0000000200000002LL, 0x0000000300000002LL, 0))
        return 1;
    if (test__cmpti2(0x0000000200000002LL, 0x0000000300000003LL, 0))
        return 1;

    if (test__cmpti2(0x0000000200000002LL, 0x0000000100000001LL, 2))
        return 1;
    if (test__cmpti2(0x0000000200000002LL, 0x0000000100000002LL, 2))
        return 1;
    if (test__cmpti2(0x0000000200000002LL, 0x0000000100000003LL, 2))
        return 1;

    if (test__cmpti2(0x0000000200000002LL, 0x0000000200000001LL, 2))
        return 1;
    if (test__cmpti2(0x0000000200000002LL, 0x0000000200000002LL, 1))
        return 1;
    if (test__cmpti2(0x0000000200000002LL, 0x0000000200000003LL, 0))
        return 1;

    if (test__cmpti2(make_ti(2, 2), make_ti(3, 1), 0))
        return 1;
    if (test__cmpti2(make_ti(2, 2), make_ti(3, 2), 0))
        return 1;
    if (test__cmpti2(make_ti(2, 2), make_ti(3, 3), 0))
        return 1;

    if (test__cmpti2(make_ti(2, 2), make_ti(1, 1), 2))
        return 1;
    if (test__cmpti2(make_ti(2, 2), make_ti(1, 2), 2))
        return 1;
    if (test__cmpti2(make_ti(2, 2), make_ti(1, 3), 2))
        return 1;

    if (test__cmpti2(make_ti(2, 2), make_ti(2, 1), 2))
        return 1;
    if (test__cmpti2(make_ti(2, 2), make_ti(2, 2), 1))
        return 1;
    if (test__cmpti2(make_ti(2, 2), make_ti(2, 3), 0))
        return 1;

#else
    printf("skipped\n");
#endif
   return 0;
}

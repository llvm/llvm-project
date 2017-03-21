// RUN: %clang_builtins %s %librt -o %t && %run %t
//===-- clzti2_test.c - Test __clzti2 -------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file tests __clzti2 for the compiler_rt library.
//
//===----------------------------------------------------------------------===//

#include "int_lib.h"
#include <stdio.h>

#ifdef CRT_HAS_128BIT

// Returns: the number of leading 0-bits

// Precondition: a != 0

COMPILER_RT_ABI si_int __clzti2(ti_int a);

int test__clzti2(ti_int a, si_int expected)
{
    si_int x = __clzti2(a);
    if (x != expected)
    {
        twords at;
        at.all = a;
        printf("error in __clzti2(0x%llX%.16llX) = %d, expected %d\n",
               at.s.high, at.s.low, x, expected);
    }
    return x != expected;
}

char assumption_1[sizeof(ti_int) == 2*sizeof(di_int)] = {0};

#endif

int main()
{
#ifdef CRT_HAS_128BIT
    const int N = (int)(sizeof(ti_int) * CHAR_BIT);

    if (test__clzti2(0x00000001, N-1))
        return 1;
    if (test__clzti2(0x00000002, N-2))
        return 1;
    if (test__clzti2(0x00000003, N-2))
        return 1;
    if (test__clzti2(0x00000004, N-3))
        return 1;
    if (test__clzti2(0x00000005, N-3))
        return 1;
    if (test__clzti2(0x0000000A, N-4))
        return 1;
    if (test__clzti2(0x1000000A, N*3/4+3))
        return 1;
    if (test__clzti2(0x2000000A, N*3/4+2))
        return 1;
    if (test__clzti2(0x6000000A, N*3/4+1))
        return 1;
    if (test__clzti2(0x8000000AuLL, N*3/4))
        return 1;
    if (test__clzti2(0x000005008000000AuLL, 85))
        return 1;
    if (test__clzti2(0x020005008000000AuLL, 70))
        return 1;
    if (test__clzti2(0x720005008000000AuLL, 65))
        return 1;
    if (test__clzti2(0x820005008000000AuLL, 64))
        return 1;

    if (test__clzti2(make_ti(0x0000000080000000LL, 0x8000000800000000LL), 32))
        return 1;
    if (test__clzti2(make_ti(0x0000000100000000LL, 0x8000000800000000LL), 31))
        return 1;
    if (test__clzti2(make_ti(0x1000000100000000LL, 0x8000000800000000LL), 3))
        return 1;
    if (test__clzti2(make_ti(0x7000000100000000LL, 0x8000000800000000LL), 1))
        return 1;
    if (test__clzti2(make_ti(0x8000000100000000LL, 0x8000000800000000LL), 0))
        return 1;
#else
    printf("skipped\n");
#endif
   return 0;
}

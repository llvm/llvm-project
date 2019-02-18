// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: int128
//===-- negti2_test.c - Test __negti2 -------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file tests __negti2 for the compiler_rt library.
//
//===----------------------------------------------------------------------===//

#include "int_lib.h"
#include <stdio.h>

#ifdef CRT_HAS_128BIT

// Returns: -a

COMPILER_RT_ABI ti_int __negti2(ti_int a);

int test__negti2(ti_int a, ti_int expected)
{
    ti_int x = __negti2(a);
    if (x != expected)
    {
        twords at;
        at.all = a;
        twords xt;
        xt.all = x;
        twords expectedt;
        expectedt.all = expected;
        printf("error in __negti2: -0x%.16llX%.16llX = 0x%.16llX%.16llX, "
               "expected 0x%.16llX%.16llX\n",
               at.s.high, at.s.low, xt.s.high, xt.s.low, expectedt.s.high, expectedt.s.low);
    }
    return x != expected;
}

char assumption_1[sizeof(ti_int) == 2*sizeof(di_int)] = {0};

#endif

int main()
{
#ifdef CRT_HAS_128BIT
    if (test__negti2(0, 0))
        return 1;
    if (test__negti2(1, -1))
        return 1;
    if (test__negti2(-1, 1))
        return 1;
    if (test__negti2(2, -2))
        return 1;
    if (test__negti2(-2, 2))
        return 1;
    if (test__negti2(3, -3))
        return 1;
    if (test__negti2(-3, 3))
        return 1;
    if (test__negti2(make_ti(0x0000000000000000LL, 0x00000000FFFFFFFELL),
                     make_ti(0xFFFFFFFFFFFFFFFFLL, 0xFFFFFFFF00000002LL)))
        return 1;
    if (test__negti2(make_ti(0xFFFFFFFFFFFFFFFFLL, 0xFFFFFFFF00000002LL),
                     make_ti(0x0000000000000000LL, 0x00000000FFFFFFFELL)))
        return 1;
    if (test__negti2(make_ti(0x0000000000000000LL, 0x00000000FFFFFFFFLL),
                     make_ti(0xFFFFFFFFFFFFFFFFLL, 0xFFFFFFFF00000001LL)))
        return 1;
    if (test__negti2(make_ti(0xFFFFFFFFFFFFFFFFLL, 0xFFFFFFFF00000001LL),
                     make_ti(0x0000000000000000LL, 0x00000000FFFFFFFFLL)))
        return 1;
    if (test__negti2(make_ti(0x0000000000000000LL, 0x0000000100000000LL),
                     make_ti(0xFFFFFFFFFFFFFFFFLL, 0xFFFFFFFF00000000LL)))
        return 1;
    if (test__negti2(make_ti(0xFFFFFFFFFFFFFFFFLL, 0xFFFFFFFF00000000LL),
                     make_ti(0x0000000000000000LL, 0x0000000100000000LL)))
        return 1;
    if (test__negti2(make_ti(0x0000000000000000LL, 0x0000000200000000LL),
                     make_ti(0xFFFFFFFFFFFFFFFFLL, 0xFFFFFFFE00000000LL)))
        return 1;
    if (test__negti2(make_ti(0xFFFFFFFFFFFFFFFFLL, 0xFFFFFFFE00000000LL),
                     make_ti(0x0000000000000000LL, 0x0000000200000000LL)))
        return 1;
    if (test__negti2(make_ti(0x0000000000000000LL, 0x0000000300000000LL),
                     make_ti(0xFFFFFFFFFFFFFFFFLL, 0xFFFFFFFD00000000LL)))
        return 1;
    if (test__negti2(make_ti(0xFFFFFFFFFFFFFFFFLL, 0xFFFFFFFD00000000LL),
                     make_ti(0x0000000000000000LL, 0x0000000300000000LL)))
        return 1;
    if (test__negti2(make_ti(0x0000000000000000LL, 0x7FFFFFFFFFFFFFFFLL),
                     make_ti(0xFFFFFFFFFFFFFFFFLL, 0x8000000000000001LL)))
        return 1;
    if (test__negti2(make_ti(0xFFFFFFFFFFFFFFFFLL, 0x8000000000000001LL),
                     make_ti(0x0000000000000000LL, 0x7FFFFFFFFFFFFFFFLL)))
        return 1;
    if (test__negti2(make_ti(0x0000000000000000LL, 0x7FFFFFFFFFFFFFFFLL),
                     make_ti(0xFFFFFFFFFFFFFFFFLL, 0x8000000000000001LL)))
        return 1;
    if (test__negti2(make_ti(0xFFFFFFFFFFFFFFFFLL, 0xFFFFFFFE00000000LL),
                     make_ti(0x0000000000000000LL, 0x0000000200000000LL)))
        return 1;
    if (test__negti2(make_ti(0x0000000000000000LL, 0x0000000200000000LL),
                     make_ti(0xFFFFFFFFFFFFFFFFLL, 0xFFFFFFFE00000000LL)))
        return 1;
    if (test__negti2(make_ti(0xFFFFFFFFFFFFFFFFLL, 0xFFFFFFFF00000000LL),
                     make_ti(0x0000000000000000LL, 0x0000000100000000LL)))
        return 1;
    if (test__negti2(make_ti(0x0000000000000000LL, 0x0000000100000000LL),
                     make_ti(0xFFFFFFFFFFFFFFFFLL, 0xFFFFFFFF00000000LL)))
        return 1;
    if (test__negti2(make_ti(0x8000000000000000LL, 0x0000000000000000LL),
                     make_ti(0x8000000000000000LL, 0x0000000000000000LL)))
        return 1;
    if (test__negti2(make_ti(0x8000000000000000LL, 0x0000000000000001LL),
                     make_ti(0x7FFFFFFFFFFFFFFFLL, 0xFFFFFFFFFFFFFFFFLL)))
        return 1;
    if (test__negti2(make_ti(0x7FFFFFFFFFFFFFFFLL, 0xFFFFFFFFFFFFFFFFLL),
                     make_ti(0x8000000000000000LL, 0x0000000000000001LL)))
        return 1;

#else
    printf("skipped\n");
#endif
   return 0;
}

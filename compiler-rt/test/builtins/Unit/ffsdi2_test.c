//===-- ffsdi2_test.c - Test __ffsdi2 -------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file tests __ffsdi2 for the compiler_rt library.
//
//===----------------------------------------------------------------------===//

#include "int_lib.h"
#include <stdio.h>

// Returns: the index of the least significant 1-bit in a, or
// the value zero if a is zero. The least significant bit is index one.

COMPILER_RT_ABI si_int __ffsdi2(di_int a);

int test__ffsdi2(di_int a, si_int expected)
{
    si_int x = __ffsdi2(a);
    if (x != expected)
        printf("error in __ffsdi2(0x%llX) = %d, expected %d\n", a, x, expected);
    return x != expected;
}

char assumption_1[sizeof(di_int) == 2*sizeof(si_int)] = {0};

int main()
{
    if (test__ffsdi2(0x00000000, 0))
        return 1;
    if (test__ffsdi2(0x00000001, 1))
        return 1;
    if (test__ffsdi2(0x00000002, 2))
        return 1;
    if (test__ffsdi2(0x00000003, 1))
        return 1;
    if (test__ffsdi2(0x00000004, 3))
        return 1;
    if (test__ffsdi2(0x00000005, 1))
        return 1;
    if (test__ffsdi2(0x0000000A, 2))
        return 1;
    if (test__ffsdi2(0x10000000, 29))
        return 1;
    if (test__ffsdi2(0x20000000, 30))
        return 1;
    if (test__ffsdi2(0x60000000, 30))
        return 1;
    if (test__ffsdi2(0x80000000uLL, 32))
        return 1;
    if (test__ffsdi2(0x0000050000000000uLL, 41))
        return 1;
    if (test__ffsdi2(0x0200080000000000uLL, 44))
        return 1;
    if (test__ffsdi2(0x7200000000000000uLL, 58))
        return 1;
    if (test__ffsdi2(0x8000000000000000uLL, 64))
        return 1;

   return 0;
}

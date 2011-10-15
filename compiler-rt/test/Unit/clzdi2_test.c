//===-- clzdi2_test.c - Test __clzdi2 -------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file tests __clzdi2 for the compiler_rt library.
//
//===----------------------------------------------------------------------===//

#include "int_lib.h"
#include <stdio.h>

// Returns: the number of leading 0-bits

// Precondition: a != 0

si_int __clzdi2(di_int a);

int test__clzdi2(di_int a, si_int expected)
{
    si_int x = __clzdi2(a);
    if (x != expected)
        printf("error in __clzdi2(0x%llX) = %d, expected %d\n", a, x, expected);
    return x != expected;
}

char assumption_1[sizeof(di_int) == 2*sizeof(si_int)] = {0};

int main()
{
    const int N = (int)(sizeof(di_int) * CHAR_BIT);
//    if (test__clzdi2(0x00000000, N))  // undefined
//        return 1;
    if (test__clzdi2(0x00000001, N-1))
        return 1;
    if (test__clzdi2(0x00000002, N-2))
        return 1;
    if (test__clzdi2(0x00000003, N-2))
        return 1;
    if (test__clzdi2(0x00000004, N-3))
        return 1;
    if (test__clzdi2(0x00000005, N-3))
        return 1;
    if (test__clzdi2(0x0000000A, N-4))
        return 1;
    if (test__clzdi2(0x1000000A, N/2+3))
        return 1;
    if (test__clzdi2(0x2000000A, N/2+2))
        return 1;
    if (test__clzdi2(0x6000000A, N/2+1))
        return 1;
    if (test__clzdi2(0x8000000AuLL, N/2))
        return 1;
    if (test__clzdi2(0x000005008000000AuLL, 21))
        return 1;
    if (test__clzdi2(0x020005008000000AuLL, 6))
        return 1;
    if (test__clzdi2(0x720005008000000AuLL, 1))
        return 1;
    if (test__clzdi2(0x820005008000000AuLL, 0))
        return 1;

   return 0;
}

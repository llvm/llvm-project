//===-- addvdi3_test.c - Test __addvdi3 -----------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file tests __addvdi3 for the compiler_rt library.
//
//===----------------------------------------------------------------------===//

#include "int_lib.h"
#include <stdio.h>

// Returns: a + b

// Effects: aborts if a + b overflows

COMPILER_RT_ABI di_int __addvdi3(di_int a, di_int b);

int test__addvdi3(di_int a, di_int b)
{
    di_int x = __addvdi3(a, b);
    di_int expected = a + b;
    if (x != expected)
        printf("error in test__addvdi3(0x%llX, 0x%llX) = %lld, expected %lld\n",
                a, b, x, expected);
    return x != expected;
}

int main()
{
//     test__addvdi3(0x8000000000000000LL, -1);  // should abort
//     test__addvdi3(-1, 0x8000000000000000LL);  // should abort
//     test__addvdi3(1, 0x7FFFFFFFFFFFFFFFLL);  // should abort
//     test__addvdi3(0x7FFFFFFFFFFFFFFFLL, 1);  // should abort

    if (test__addvdi3(0x8000000000000000LL, 1))
        return 1;
    if (test__addvdi3(1, 0x8000000000000000LL))
        return 1;
    if (test__addvdi3(0x8000000000000000LL, 0))
        return 1;
    if (test__addvdi3(0, 0x8000000000000000LL))
        return 1;
    if (test__addvdi3(0x7FFFFFFFFFFFFFFLL, -1))
        return 1;
    if (test__addvdi3(-1, 0x7FFFFFFFFFFFFFFLL))
        return 1;
    if (test__addvdi3(0x7FFFFFFFFFFFFFFFLL, 0))
        return 1;
    if (test__addvdi3(0, 0x7FFFFFFFFFFFFFFFLL))
        return 1;

    return 0;
}

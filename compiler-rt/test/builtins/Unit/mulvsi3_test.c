//===-- mulvsi3_test.c - Test __mulvsi3 -----------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file tests __mulvsi3 for the compiler_rt library.
//
//===----------------------------------------------------------------------===//

#include "int_lib.h"
#include <stdio.h>

// Returns: a * b

// Effects: aborts if a * b overflows

COMPILER_RT_ABI si_int __mulvsi3(si_int a, si_int b);

int test__mulvsi3(si_int a, si_int b, si_int expected)
{
    si_int x = __mulvsi3(a, b);
    if (x != expected)
        printf("error in __mulvsi3: %d * %d = %d, expected %d\n",
               a, b, x, expected);
    return x != expected;
}

int main()
{
    if (test__mulvsi3(0, 0, 0))
        return 1;
    if (test__mulvsi3(0, 1, 0))
        return 1;
    if (test__mulvsi3(1, 0, 0))
        return 1;
    if (test__mulvsi3(0, 10, 0))
        return 1;
    if (test__mulvsi3(10, 0, 0))
        return 1;
    if (test__mulvsi3(0, 0x1234567, 0))
        return 1;
    if (test__mulvsi3(0x1234567, 0, 0))
        return 1;

    if (test__mulvsi3(0, -1, 0))
        return 1;
    if (test__mulvsi3(-1, 0, 0))
        return 1;
    if (test__mulvsi3(0, -10, 0))
        return 1;
    if (test__mulvsi3(-10, 0, 0))
        return 1;
    if (test__mulvsi3(0, -0x1234567, 0))
        return 1;
    if (test__mulvsi3(-0x1234567, 0, 0))
        return 1;

    if (test__mulvsi3(1, 1, 1))
        return 1;
    if (test__mulvsi3(1, 10, 10))
        return 1;
    if (test__mulvsi3(10, 1, 10))
        return 1;
    if (test__mulvsi3(1, 0x1234567, 0x1234567))
        return 1;
    if (test__mulvsi3(0x1234567, 1, 0x1234567))
        return 1;

    if (test__mulvsi3(1, -1, -1))
        return 1;
    if (test__mulvsi3(1, -10, -10))
        return 1;
    if (test__mulvsi3(-10, 1, -10))
        return 1;
    if (test__mulvsi3(1, -0x1234567, -0x1234567))
        return 1;
    if (test__mulvsi3(-0x1234567, 1, -0x1234567))
        return 1;

//     if (test__mulvsi3(0x7FFFFFFF, -2, 0x80000001))  // abort
//         return 1;
//     if (test__mulvsi3(-2, 0x7FFFFFFF, 0x80000001))  // abort
//         return 1;
    if (test__mulvsi3(0x7FFFFFFF, -1, 0x80000001))
        return 1;
    if (test__mulvsi3(-1, 0x7FFFFFFF, 0x80000001))
        return 1;
    if (test__mulvsi3(0x7FFFFFFF, 0, 0))
        return 1;
    if (test__mulvsi3(0, 0x7FFFFFFF, 0))
        return 1;
    if (test__mulvsi3(0x7FFFFFFF, 1, 0x7FFFFFFF))
        return 1;
    if (test__mulvsi3(1, 0x7FFFFFFF, 0x7FFFFFFF))
        return 1;
//     if (test__mulvsi3(0x7FFFFFFF, 2, 0x80000001))  // abort
//         return 1;
//     if (test__mulvsi3(2, 0x7FFFFFFF, 0x80000001))  // abort
//         return 1;

//     if (test__mulvsi3(0x80000000, -2, 0x80000000))  // abort
//         return 1;
//     if (test__mulvsi3(-2, 0x80000000, 0x80000000))  // abort
//         return 1;
//     if (test__mulvsi3(0x80000000, -1, 0x80000000))  // abort
//         return 1;
//     if (test__mulvsi3(-1, 0x80000000, 0x80000000))  // abort
//         return 1;
    if (test__mulvsi3(0x80000000, 0, 0))
        return 1;
    if (test__mulvsi3(0, 0x80000000, 0))
        return 1;
    if (test__mulvsi3(0x80000000, 1, 0x80000000))
        return 1;
    if (test__mulvsi3(1, 0x80000000, 0x80000000))
        return 1;
//     if (test__mulvsi3(0x80000000, 2, 0x80000000))  // abort
//         return 1;
//     if (test__mulvsi3(2, 0x80000000, 0x80000000))  // abort
//         return 1;

//     if (test__mulvsi3(0x80000001, -2, 0x80000001))  // abort
//         return 1;
//     if (test__mulvsi3(-2, 0x80000001, 0x80000001))  // abort
//         return 1;
    if (test__mulvsi3(0x80000001, -1, 0x7FFFFFFF))
        return 1;
    if (test__mulvsi3(-1, 0x80000001, 0x7FFFFFFF))
        return 1;
    if (test__mulvsi3(0x80000001, 0, 0))
        return 1;
    if (test__mulvsi3(0, 0x80000001, 0))
        return 1;
    if (test__mulvsi3(0x80000001, 1, 0x80000001))
        return 1;
    if (test__mulvsi3(1, 0x80000001, 0x80000001))
        return 1;
//     if (test__mulvsi3(0x80000001, 2, 0x80000000))  // abort
//         return 1;
//     if (test__mulvsi3(2, 0x80000001, 0x80000000))  // abort
//         return 1;

    return 0;
}

//===-- negvsi2_test.c - Test __negvsi2 -----------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file tests __negvsi2 for the compiler_rt library.
//
//===----------------------------------------------------------------------===//

#include "int_lib.h"
#include <stdio.h>

// Returns: -a

// Effects: aborts if -a overflows

COMPILER_RT_ABI si_int __negvsi2(si_int a);

int test__negvsi2(si_int a)
{
    si_int x = __negvsi2(a);
    si_int expected = -a;
    if (x != expected)
        printf("error in __negvsi2(0x%X) = %d, expected %d\n", a, x, expected);
    return x != expected;
}

int main()
{
//     if (test__negvsi2(0x80000000))  // should abort
//         return 1;
    if (test__negvsi2(0x00000000))
        return 1;
    if (test__negvsi2(0x00000001))
        return 1;
    if (test__negvsi2(0x00000002))
        return 1;
    if (test__negvsi2(0x7FFFFFFE))
        return 1;
    if (test__negvsi2(0x7FFFFFFF))
        return 1;
    if (test__negvsi2(0x80000001))
        return 1;
    if (test__negvsi2(0x80000002))
        return 1;
    if (test__negvsi2(0xFFFFFFFE))
        return 1;
    if (test__negvsi2(0xFFFFFFFF))
        return 1;

    return 0;
}

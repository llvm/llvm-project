// RUN: %clang_builtins %s %librt -o %t && %run %t
//===-- negvdi2_test.c - Test __negvdi2 -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file tests __negvdi2 for the compiler_rt library.
//
//===----------------------------------------------------------------------===//

#include "int_lib.h"
#include <stdio.h>

// Returns: -a

// Effects: aborts if -a overflows

COMPILER_RT_ABI di_int __negvdi2(di_int a);

int test__negvdi2(di_int a)
{
    di_int x = __negvdi2(a);
    di_int expected = -a;
    if (x != expected)
        printf("error in __negvdi2(0x%llX) = %lld, expected %lld\n",
               a, x, expected);
    return x != expected;
}

int main()
{
//     if (test__negvdi2(0x8000000000000000LL))  // should abort
//         return 1;
    if (test__negvdi2(0x0000000000000000LL))
        return 1;
    if (test__negvdi2(0x0000000000000001LL))
        return 1;
    if (test__negvdi2(0x0000000000000002LL))
        return 1;
    if (test__negvdi2(0x7FFFFFFFFFFFFFFELL))
        return 1;
    if (test__negvdi2(0x7FFFFFFFFFFFFFFFLL))
        return 1;
    if (test__negvdi2(0x8000000000000001LL))
        return 1;
    if (test__negvdi2(0x8000000000000002LL))
        return 1;
    if (test__negvdi2(0xFFFFFFFFFFFFFFFELL))
        return 1;
    if (test__negvdi2(0xFFFFFFFFFFFFFFFFLL))
        return 1;

    return 0;
}

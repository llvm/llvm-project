// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_cmpdi2
//===-- cmpdi2_test.c - Test __cmpdi2 -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file tests __cmpdi2 for the compiler_rt library.
//
//===----------------------------------------------------------------------===//

#include "int_lib.h"
#include <stdio.h>

// Returns:  if (a <  b) returns 0
//           if (a == b) returns 1
//           if (a >  b) returns 2

COMPILER_RT_ABI si_int __cmpdi2(di_int a, di_int b);

int test__cmpdi2(di_int a, di_int b, si_int expected)
{
    si_int x = __cmpdi2(a, b);
    if (x != expected)
        printf("error in __cmpdi2(0x%llX, 0x%llX) = %d, expected %d\n",
               a, b, x, expected);
    return x != expected;
}

char assumption_1[sizeof(di_int) == 2*sizeof(si_int)] = {0};

int main()
{
    if (test__cmpdi2(0, 0, 1))
        return 1;
    if (test__cmpdi2(1, 1, 1))
        return 1;
    if (test__cmpdi2(2, 2, 1))
        return 1;
    if (test__cmpdi2(0x7FFFFFFF, 0x7FFFFFFF, 1))
        return 1;
    if (test__cmpdi2(0x80000000, 0x80000000, 1))
        return 1;
    if (test__cmpdi2(0x80000001, 0x80000001, 1))
        return 1;
    if (test__cmpdi2(0xFFFFFFFF, 0xFFFFFFFF, 1))
        return 1;
    if (test__cmpdi2(0x000000010000000LL, 0x000000010000000LL, 1))
        return 1;
    if (test__cmpdi2(0xFFFFFFFFFFFFFFFFLL, 0xFFFFFFFFFFFFFFFFLL, 1))
        return 1;

    if (test__cmpdi2(0x0000000200000002LL, 0x0000000300000001LL, 0))
        return 1;
    if (test__cmpdi2(0x0000000200000002LL, 0x0000000300000002LL, 0))
        return 1;
    if (test__cmpdi2(0x0000000200000002LL, 0x0000000300000003LL, 0))
        return 1;

    if (test__cmpdi2(0x0000000200000002LL, 0x0000000100000001LL, 2))
        return 1;
    if (test__cmpdi2(0x0000000200000002LL, 0x0000000100000002LL, 2))
        return 1;
    if (test__cmpdi2(0x0000000200000002LL, 0x0000000100000003LL, 2))
        return 1;

    if (test__cmpdi2(0x0000000200000002LL, 0x0000000200000001LL, 2))
        return 1;
    if (test__cmpdi2(0x0000000200000002LL, 0x0000000200000002LL, 1))
        return 1;
    if (test__cmpdi2(0x0000000200000002LL, 0x0000000200000003LL, 0))
        return 1;

   return 0;
}

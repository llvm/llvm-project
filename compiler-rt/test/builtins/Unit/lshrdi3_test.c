// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_lshrdi3
//===-- lshrdi3_test.c - Test __lshrdi3 -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file tests __lshrdi3 for the compiler_rt library.
//
//===----------------------------------------------------------------------===//

#include "int_lib.h"
#include <stdio.h>

// Returns: logical a >> b

// Precondition:  0 <= b < bits_in_dword

COMPILER_RT_ABI di_int __lshrdi3(di_int a, si_int b);

int test__lshrdi3(di_int a, si_int b, di_int expected)
{
    di_int x = __lshrdi3(a, b);
    if (x != expected)
        printf("error in __lshrdi3: %llX >> %d = %llX, expected %llX\n",
               a, b, __lshrdi3(a, b), expected);
    return x != expected;
}

char assumption_1[sizeof(di_int) == 2*sizeof(si_int)] = {0};

int main()
{
    if (test__lshrdi3(0x0123456789ABCDEFLL, 0, 0x123456789ABCDEFLL))
        return 1;
    if (test__lshrdi3(0x0123456789ABCDEFLL, 1, 0x91A2B3C4D5E6F7LL))
        return 1;
    if (test__lshrdi3(0x0123456789ABCDEFLL, 2, 0x48D159E26AF37BLL))
        return 1;
    if (test__lshrdi3(0x0123456789ABCDEFLL, 3, 0x2468ACF13579BDLL))
        return 1;
    if (test__lshrdi3(0x0123456789ABCDEFLL, 4, 0x123456789ABCDELL))
        return 1;

    if (test__lshrdi3(0x0123456789ABCDEFLL, 28, 0x12345678LL))
        return 1;
    if (test__lshrdi3(0x0123456789ABCDEFLL, 29, 0x91A2B3CLL))
        return 1;
    if (test__lshrdi3(0x0123456789ABCDEFLL, 30, 0x48D159ELL))
        return 1;
    if (test__lshrdi3(0x0123456789ABCDEFLL, 31, 0x2468ACFLL))
        return 1;

    if (test__lshrdi3(0x0123456789ABCDEFLL, 32, 0x1234567LL))
        return 1;

    if (test__lshrdi3(0x0123456789ABCDEFLL, 33, 0x91A2B3LL))
        return 1;
    if (test__lshrdi3(0x0123456789ABCDEFLL, 34, 0x48D159LL))
        return 1;
    if (test__lshrdi3(0x0123456789ABCDEFLL, 35, 0x2468ACLL))
        return 1;
    if (test__lshrdi3(0x0123456789ABCDEFLL, 36, 0x123456LL))
        return 1;

    if (test__lshrdi3(0x0123456789ABCDEFLL, 60, 0))
        return 1;
    if (test__lshrdi3(0x0123456789ABCDEFLL, 61, 0))
        return 1;
    if (test__lshrdi3(0x0123456789ABCDEFLL, 62, 0))
        return 1;
    if (test__lshrdi3(0x0123456789ABCDEFLL, 63, 0))
        return 1;

    if (test__lshrdi3(0xFEDCBA9876543210LL, 0, 0xFEDCBA9876543210LL))
        return 1;
    if (test__lshrdi3(0xFEDCBA9876543210LL, 1, 0x7F6E5D4C3B2A1908LL))
        return 1;
    if (test__lshrdi3(0xFEDCBA9876543210LL, 2, 0x3FB72EA61D950C84LL))
        return 1;
    if (test__lshrdi3(0xFEDCBA9876543210LL, 3, 0x1FDB97530ECA8642LL))
        return 1;
    if (test__lshrdi3(0xFEDCBA9876543210LL, 4, 0xFEDCBA987654321LL))
        return 1;

    if (test__lshrdi3(0xFEDCBA9876543210LL, 28, 0xFEDCBA987LL))
        return 1;
    if (test__lshrdi3(0xFEDCBA9876543210LL, 29, 0x7F6E5D4C3LL))
        return 1;
    if (test__lshrdi3(0xFEDCBA9876543210LL, 30, 0x3FB72EA61LL))
        return 1;
    if (test__lshrdi3(0xFEDCBA9876543210LL, 31, 0x1FDB97530LL))
        return 1;

    if (test__lshrdi3(0xFEDCBA9876543210LL, 32, 0xFEDCBA98LL))
        return 1;

    if (test__lshrdi3(0xFEDCBA9876543210LL, 33, 0x7F6E5D4CLL))
        return 1;
    if (test__lshrdi3(0xFEDCBA9876543210LL, 34, 0x3FB72EA6LL))
        return 1;
    if (test__lshrdi3(0xFEDCBA9876543210LL, 35, 0x1FDB9753LL))
        return 1;
    if (test__lshrdi3(0xFEDCBA9876543210LL, 36, 0xFEDCBA9LL))
        return 1;

    if (test__lshrdi3(0xAEDCBA9876543210LL, 60, 0xALL))
        return 1;
    if (test__lshrdi3(0xAEDCBA9876543210LL, 61, 0x5LL))
        return 1;
    if (test__lshrdi3(0xAEDCBA9876543210LL, 62, 0x2LL))
        return 1;
    if (test__lshrdi3(0xAEDCBA9876543210LL, 63, 0x1LL))
        return 1;
    return 0;
}

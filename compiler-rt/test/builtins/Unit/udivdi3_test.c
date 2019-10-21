// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_udivdi3
//===-- udivdi3_test.c - Test __udivdi3 -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file tests __udivdi3 for the compiler_rt library.
//
//===----------------------------------------------------------------------===//

#include "int_lib.h"
#include <stdio.h>

// Returns: a / b

COMPILER_RT_ABI du_int __udivdi3(du_int a, du_int b);

int test__udivdi3(du_int a, du_int b, du_int expected_q)
{
    du_int q = __udivdi3(a, b);
    if (q != expected_q)
        printf("error in __udivdi3: %lld / %lld = %lld, expected %lld\n",
               a, b, q, expected_q);
    return q != expected_q;
}

int main()
{
    if (test__udivdi3(0, 1, 0))
        return 1;
    if (test__udivdi3(2, 1, 2))
        return 1;
    if (test__udivdi3(0x8000000000000000uLL, 1, 0x8000000000000000uLL))
        return 1;
    if (test__udivdi3(0x8000000000000000uLL, 2, 0x4000000000000000uLL))
        return 1;
    if (test__udivdi3(0xFFFFFFFFFFFFFFFFuLL, 2, 0x7FFFFFFFFFFFFFFFuLL))
        return 1;

    return 0;
}

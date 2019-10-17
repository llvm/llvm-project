// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_mulvdi3
//===-- mulvdi3_test.c - Test __mulvdi3 -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file tests __mulvdi3 for the compiler_rt library.
//
//===----------------------------------------------------------------------===//

#include "int_lib.h"
#include <stdio.h>

// Returns: a * b

// Effects: aborts if a * b overflows

COMPILER_RT_ABI di_int __mulvdi3(di_int a, di_int b);

int test__mulvdi3(di_int a, di_int b, di_int expected)
{
    di_int x = __mulvdi3(a, b);
    if (x != expected)
        printf("error in __mulvdi3: %lld * %lld = %lld, expected %lld\n",
               a, b, x, expected);
    return x != expected;
}

int main()
{
    if (test__mulvdi3(0, 0, 0))
        return 1;
    if (test__mulvdi3(0, 1, 0))
        return 1;
    if (test__mulvdi3(1, 0, 0))
        return 1;
    if (test__mulvdi3(0, 10, 0))
        return 1;
    if (test__mulvdi3(10, 0, 0))
        return 1;
    if (test__mulvdi3(0, 81985529216486895LL, 0))
        return 1;
    if (test__mulvdi3(81985529216486895LL, 0, 0))
        return 1;

    if (test__mulvdi3(0, -1, 0))
        return 1;
    if (test__mulvdi3(-1, 0, 0))
        return 1;
    if (test__mulvdi3(0, -10, 0))
        return 1;
    if (test__mulvdi3(-10, 0, 0))
        return 1;
    if (test__mulvdi3(0, -81985529216486895LL, 0))
        return 1;
    if (test__mulvdi3(-81985529216486895LL, 0, 0))
        return 1;

    if (test__mulvdi3(1, 1, 1))
        return 1;
    if (test__mulvdi3(1, 10, 10))
        return 1;
    if (test__mulvdi3(10, 1, 10))
        return 1;
    if (test__mulvdi3(1, 81985529216486895LL, 81985529216486895LL))
        return 1;
    if (test__mulvdi3(81985529216486895LL, 1, 81985529216486895LL))
        return 1;

    if (test__mulvdi3(1, -1, -1))
        return 1;
    if (test__mulvdi3(1, -10, -10))
        return 1;
    if (test__mulvdi3(-10, 1, -10))
        return 1;
    if (test__mulvdi3(1, -81985529216486895LL, -81985529216486895LL))
        return 1;
    if (test__mulvdi3(-81985529216486895LL, 1, -81985529216486895LL))
        return 1;

    if (test__mulvdi3(3037000499LL, 3037000499LL, 9223372030926249001LL))
        return 1;
    if (test__mulvdi3(-3037000499LL, 3037000499LL, -9223372030926249001LL))
        return 1;
    if (test__mulvdi3(3037000499LL, -3037000499LL, -9223372030926249001LL))
        return 1;
    if (test__mulvdi3(-3037000499LL, -3037000499LL, 9223372030926249001LL))
        return 1;

    if (test__mulvdi3(4398046511103LL, 2097152LL, 9223372036852678656LL))
        return 1;
    if (test__mulvdi3(-4398046511103LL, 2097152LL, -9223372036852678656LL))
        return 1;
    if (test__mulvdi3(4398046511103LL, -2097152LL, -9223372036852678656LL))
        return 1;
    if (test__mulvdi3(-4398046511103LL, -2097152LL, 9223372036852678656LL))
        return 1;

    if (test__mulvdi3(2097152LL, 4398046511103LL, 9223372036852678656LL))
        return 1;
    if (test__mulvdi3(-2097152LL, 4398046511103LL, -9223372036852678656LL))
        return 1;
    if (test__mulvdi3(2097152LL, -4398046511103LL, -9223372036852678656LL))
        return 1;
    if (test__mulvdi3(-2097152LL, -4398046511103LL, 9223372036852678656LL))
        return 1;

//     if (test__mulvdi3(0x7FFFFFFFFFFFFFFFLL, -2, 0x8000000000000001LL))  // abort
//         return 1;
//     if (test__mulvdi3(-2, 0x7FFFFFFFFFFFFFFFLL, 0x8000000000000001LL))  // abort
//         return 1;
    if (test__mulvdi3(0x7FFFFFFFFFFFFFFFLL, -1, 0x8000000000000001LL))
        return 1;
    if (test__mulvdi3(-1, 0x7FFFFFFFFFFFFFFFLL, 0x8000000000000001LL))
        return 1;
    if (test__mulvdi3(0x7FFFFFFFFFFFFFFFLL, 0, 0))
        return 1;
    if (test__mulvdi3(0, 0x7FFFFFFFFFFFFFFFLL, 0))
        return 1;
    if (test__mulvdi3(0x7FFFFFFFFFFFFFFFLL, 1, 0x7FFFFFFFFFFFFFFFLL))
        return 1;
    if (test__mulvdi3(1, 0x7FFFFFFFFFFFFFFFLL, 0x7FFFFFFFFFFFFFFFLL))
        return 1;
//     if (test__mulvdi3(0x7FFFFFFFFFFFFFFFLL, 2, 0x8000000000000001LL))  // abort
//         return 1;
//     if (test__mulvdi3(2, 0x7FFFFFFFFFFFFFFFLL, 0x8000000000000001LL))  // abort
//         return 1;

//     if (test__mulvdi3(0x8000000000000000LL, -2, 0x8000000000000000LL))  // abort
//         return 1;
//     if (test__mulvdi3(-2, 0x8000000000000000LL, 0x8000000000000000LL))  // abort
//         return 1;
//     if (test__mulvdi3(0x8000000000000000LL, -1, 0x8000000000000000LL))  // abort
//         return 1;
//     if (test__mulvdi3(-1, 0x8000000000000000LL, 0x8000000000000000LL))  // abort
//         return 1;
    if (test__mulvdi3(0x8000000000000000LL, 0, 0))
        return 1;
    if (test__mulvdi3(0, 0x8000000000000000LL, 0))
        return 1;
    if (test__mulvdi3(0x8000000000000000LL, 1, 0x8000000000000000LL))
        return 1;
    if (test__mulvdi3(1, 0x8000000000000000LL, 0x8000000000000000LL))
        return 1;
//     if (test__mulvdi3(0x8000000000000000LL, 2, 0x8000000000000000LL))  // abort
//         return 1;
//     if (test__mulvdi3(2, 0x8000000000000000LL, 0x8000000000000000LL))  // abort
//         return 1;

//     if (test__mulvdi3(0x8000000000000001LL, -2, 0x8000000000000001LL))  // abort
//         return 1;
//     if (test__mulvdi3(-2, 0x8000000000000001LL, 0x8000000000000001LL))  // abort
//         return 1;
    if (test__mulvdi3(0x8000000000000001LL, -1, 0x7FFFFFFFFFFFFFFFLL))
        return 1;
    if (test__mulvdi3(-1, 0x8000000000000001LL, 0x7FFFFFFFFFFFFFFFLL))
        return 1;
    if (test__mulvdi3(0x8000000000000001LL, 0, 0))
        return 1;
    if (test__mulvdi3(0, 0x8000000000000001LL, 0))
        return 1;
    if (test__mulvdi3(0x8000000000000001LL, 1, 0x8000000000000001LL))
        return 1;
    if (test__mulvdi3(1, 0x8000000000000001LL, 0x8000000000000001LL))
        return 1;
//     if (test__mulvdi3(0x8000000000000001LL, 2, 0x8000000000000000LL))  // abort
//         return 1;
//     if (test__mulvdi3(2, 0x8000000000000001LL, 0x8000000000000000LL))  // abort
//         return 1;

    return 0;
}

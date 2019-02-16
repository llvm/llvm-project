// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: int128
//===-- multi3_test.c - Test __multi3 -------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file tests __multi3 for the compiler_rt library.
//
//===----------------------------------------------------------------------===//

#include "int_lib.h"
#include <stdio.h>

#ifdef CRT_HAS_128BIT

COMPILER_RT_ABI ti_int __multi3(ti_int a, ti_int b);

int test__multi3(ti_int a, ti_int b, ti_int expected)
{
    ti_int x = __multi3(a, b);
    if (x != expected)
    {
        twords at;
        at.all = a;
        twords bt;
        bt.all = b;
        twords xt;
        xt.all = x;
        twords expectedt;
        expectedt.all = expected;
        printf("error in __multi3: 0x%.16llX%.16llX * 0x%.16llX%.16llX = "
               "0x%.16llX%.16llX, expected 0x%.16llX%.16llX\n",
               at.s.high, at.s.low, bt.s.high, bt.s.low, xt.s.high, xt.s.low,
               expectedt.s.high, expectedt.s.low);
    }
    return x != expected;
}

char assumption_1[sizeof(ti_int) == 2*sizeof(di_int)] = {0};

#endif

int main()
{
#ifdef CRT_HAS_128BIT
    if (test__multi3(0, 0, 0))
        return 1;
    if (test__multi3(0, 1, 0))
        return 1;
    if (test__multi3(1, 0, 0))
        return 1;
    if (test__multi3(0, 10, 0))
        return 1;
    if (test__multi3(10, 0, 0))
        return 1;
    if (test__multi3(0, 81985529216486895LL, 0))
        return 1;
    if (test__multi3(81985529216486895LL, 0, 0))
        return 1;

    if (test__multi3(0, -1, 0))
        return 1;
    if (test__multi3(-1, 0, 0))
        return 1;
    if (test__multi3(0, -10, 0))
        return 1;
    if (test__multi3(-10, 0, 0))
        return 1;
    if (test__multi3(0, -81985529216486895LL, 0))
        return 1;
    if (test__multi3(-81985529216486895LL, 0, 0))
        return 1;

    if (test__multi3(1, 1, 1))
        return 1;
    if (test__multi3(1, 10, 10))
        return 1;
    if (test__multi3(10, 1, 10))
        return 1;
    if (test__multi3(1, 81985529216486895LL, 81985529216486895LL))
        return 1;
    if (test__multi3(81985529216486895LL, 1, 81985529216486895LL))
        return 1;

    if (test__multi3(1, -1, -1))
        return 1;
    if (test__multi3(1, -10, -10))
        return 1;
    if (test__multi3(-10, 1, -10))
        return 1;
    if (test__multi3(1, -81985529216486895LL, -81985529216486895LL))
        return 1;
    if (test__multi3(-81985529216486895LL, 1, -81985529216486895LL))
        return 1;

    if (test__multi3(3037000499LL, 3037000499LL, 9223372030926249001LL))
        return 1;
    if (test__multi3(-3037000499LL, 3037000499LL, -9223372030926249001LL))
        return 1;
    if (test__multi3(3037000499LL, -3037000499LL, -9223372030926249001LL))
        return 1;
    if (test__multi3(-3037000499LL, -3037000499LL, 9223372030926249001LL))
        return 1;

    if (test__multi3(4398046511103LL, 2097152LL, 9223372036852678656LL))
        return 1;
    if (test__multi3(-4398046511103LL, 2097152LL, -9223372036852678656LL))
        return 1;
    if (test__multi3(4398046511103LL, -2097152LL, -9223372036852678656LL))
        return 1;
    if (test__multi3(-4398046511103LL, -2097152LL, 9223372036852678656LL))
        return 1;

    if (test__multi3(2097152LL, 4398046511103LL, 9223372036852678656LL))
        return 1;
    if (test__multi3(-2097152LL, 4398046511103LL, -9223372036852678656LL))
        return 1;
    if (test__multi3(2097152LL, -4398046511103LL, -9223372036852678656LL))
        return 1;
    if (test__multi3(-2097152LL, -4398046511103LL, 9223372036852678656LL))
        return 1;

    if (test__multi3(make_ti(0x00000000000000B5LL, 0x04F333F9DE5BE000LL),
                     make_ti(0x0000000000000000LL, 0x00B504F333F9DE5BLL),
                     make_ti(0x7FFFFFFFFFFFF328LL, 0xDF915DA296E8A000LL)))
        return 1;
#else
    printf("skipped\n");
#endif
    return 0;
}

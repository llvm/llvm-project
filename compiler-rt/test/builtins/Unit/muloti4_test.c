// RUN: %clang_builtins %s %librt -o %t && %run %t
//===-- muloti4_test.c - Test __muloti4 -----------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file tests __muloti3 for the compiler_rt library.
//
//===----------------------------------------------------------------------===//

#include "int_lib.h"
#include <stdio.h>

#ifdef CRT_HAS_128BIT

// Returns: a * b

// Effects: sets overflow if a * b overflows

COMPILER_RT_ABI ti_int __muloti4(ti_int a, ti_int b, int *overflow);

int test__muloti4(ti_int a, ti_int b, ti_int expected, int expected_overflow)
{
    int ov;
    ti_int x = __muloti4(a, b, &ov);
    if (ov != expected_overflow) {
      twords at;
      at.all = a;
      twords bt;
      bt.all = b;
      twords xt;
      xt.all = x;
      twords expectedt;
      expectedt.all = expected;

      printf("error in __muloti4: overflow=%d expected=%d\n",
	     ov, expected_overflow);
      printf("error in __muloti4: 0x%.16llX%.16llX * 0x%.16llX%.16llX = "
	     "0x%.16llX%.16llX, expected 0x%.16llX%.16llX\n",
	     at.s.high, at.s.low, bt.s.high, bt.s.low, xt.s.high, xt.s.low,
	     expectedt.s.high, expectedt.s.low);
      return 1;
    }
    else if (!expected_overflow && x != expected)
    {
        twords at;
        at.all = a;
        twords bt;
        bt.all = b;
        twords xt;
        xt.all = x;
        twords expectedt;
        expectedt.all = expected;
        printf("error in __muloti4: 0x%.16llX%.16llX * 0x%.16llX%.16llX = "
               "0x%.16llX%.16llX, expected 0x%.16llX%.16llX\n",
               at.s.high, at.s.low, bt.s.high, bt.s.low, xt.s.high, xt.s.low,
               expectedt.s.high, expectedt.s.low);
	return 1;
    }
    return 0;
}

#endif

int main()
{
#ifdef CRT_HAS_128BIT
    if (test__muloti4(0, 0, 0, 0))
        return 1;
    if (test__muloti4(0, 1, 0, 0))
        return 1;
    if (test__muloti4(1, 0, 0, 0))
        return 1;
    if (test__muloti4(0, 10, 0, 0))
        return 1;
    if (test__muloti4(10, 0, 0, 0))
        return 1;
    if (test__muloti4(0, 81985529216486895LL, 0, 0))
        return 1;
    if (test__muloti4(81985529216486895LL, 0, 0, 0))
        return 1;

    if (test__muloti4(0, -1, 0, 0))
        return 1;
    if (test__muloti4(-1, 0, 0, 0))
        return 1;
    if (test__muloti4(0, -10, 0, 0))
        return 1;
    if (test__muloti4(-10, 0, 0, 0))
        return 1;
    if (test__muloti4(0, -81985529216486895LL, 0, 0))
        return 1;
    if (test__muloti4(-81985529216486895LL, 0, 0, 0))
        return 1;

    if (test__muloti4(1, 1, 1, 0))
        return 1;
    if (test__muloti4(1, 10, 10, 0))
        return 1;
    if (test__muloti4(10, 1, 10, 0))
        return 1;
    if (test__muloti4(1, 81985529216486895LL, 81985529216486895LL, 0))
        return 1;
    if (test__muloti4(81985529216486895LL, 1, 81985529216486895LL, 0))
        return 1;

    if (test__muloti4(1, -1, -1, 0))
        return 1;
    if (test__muloti4(1, -10, -10, 0))
        return 1;
    if (test__muloti4(-10, 1, -10, 0))
        return 1;
    if (test__muloti4(1, -81985529216486895LL, -81985529216486895LL, 0))
        return 1;
    if (test__muloti4(-81985529216486895LL, 1, -81985529216486895LL, 0))
        return 1;

    if (test__muloti4(3037000499LL, 3037000499LL, 9223372030926249001LL, 0))
        return 1;
    if (test__muloti4(-3037000499LL, 3037000499LL, -9223372030926249001LL, 0))
        return 1;
    if (test__muloti4(3037000499LL, -3037000499LL, -9223372030926249001LL, 0))
        return 1;
    if (test__muloti4(-3037000499LL, -3037000499LL, 9223372030926249001LL, 0))
        return 1;

    if (test__muloti4(4398046511103LL, 2097152LL, 9223372036852678656LL, 0))
        return 1;
    if (test__muloti4(-4398046511103LL, 2097152LL, -9223372036852678656LL, 0))
        return 1;
    if (test__muloti4(4398046511103LL, -2097152LL, -9223372036852678656LL, 0))
        return 1;
    if (test__muloti4(-4398046511103LL, -2097152LL, 9223372036852678656LL, 0))
        return 1;

    if (test__muloti4(2097152LL, 4398046511103LL, 9223372036852678656LL, 0))
        return 1;
    if (test__muloti4(-2097152LL, 4398046511103LL, -9223372036852678656LL, 0))
        return 1;
    if (test__muloti4(2097152LL, -4398046511103LL, -9223372036852678656LL, 0))
        return 1;
    if (test__muloti4(-2097152LL, -4398046511103LL, 9223372036852678656LL, 0))
        return 1;

    if (test__muloti4(make_ti(0x00000000000000B5LL, 0x04F333F9DE5BE000LL),
                      make_ti(0x0000000000000000LL, 0x00B504F333F9DE5BLL),
                      make_ti(0x7FFFFFFFFFFFF328LL, 0xDF915DA296E8A000LL), 0))
        return 1;

     if (test__muloti4(make_ti(0x7FFFFFFFFFFFFFFFLL, 0xFFFFFFFFFFFFFFFFLL),
                       -2,
                       make_ti(0x8000000000000000LL, 0x0000000000000001LL), 1))
       return 1;
     if (test__muloti4(-2,
                       make_ti(0x7FFFFFFFFFFFFFFFLL, 0xFFFFFFFFFFFFFFFFLL),
                       make_ti(0x8000000000000000LL, 0x0000000000000001LL), 1))
         return 1;
    if (test__muloti4(make_ti(0x7FFFFFFFFFFFFFFFLL, 0xFFFFFFFFFFFFFFFFLL),
                      -1,
                      make_ti(0x8000000000000000LL, 0x0000000000000001LL), 0))
        return 1;
    if (test__muloti4(-1,
                      make_ti(0x7FFFFFFFFFFFFFFFLL, 0xFFFFFFFFFFFFFFFFLL),
                      make_ti(0x8000000000000000LL, 0x0000000000000001LL), 0))
        return 1;
    if (test__muloti4(make_ti(0x7FFFFFFFFFFFFFFFLL, 0xFFFFFFFFFFFFFFFFLL),
                      0,
                      0, 0))
        return 1;
    if (test__muloti4(0,
                      make_ti(0x7FFFFFFFFFFFFFFFLL, 0xFFFFFFFFFFFFFFFFLL),
                      0, 0))
        return 1;
    if (test__muloti4(make_ti(0x7FFFFFFFFFFFFFFFLL, 0xFFFFFFFFFFFFFFFFLL),
                      1,
                      make_ti(0x7FFFFFFFFFFFFFFFLL, 0xFFFFFFFFFFFFFFFFLL), 0))
        return 1;
    if (test__muloti4(1,
                      make_ti(0x7FFFFFFFFFFFFFFFLL, 0xFFFFFFFFFFFFFFFFLL),
                      make_ti(0x7FFFFFFFFFFFFFFFLL, 0xFFFFFFFFFFFFFFFFLL), 0))
        return 1;
     if (test__muloti4(make_ti(0x7FFFFFFFFFFFFFFFLL, 0xFFFFFFFFFFFFFFFFLL),
                       2,
                       make_ti(0x8000000000000000LL, 0x0000000000000001LL), 1))
         return 1;
     if (test__muloti4(2,
                       make_ti(0x7FFFFFFFFFFFFFFFLL, 0xFFFFFFFFFFFFFFFFLL),
                       make_ti(0x8000000000000000LL, 0x0000000000000001LL), 1))
         return 1;

     if (test__muloti4(make_ti(0x8000000000000000LL, 0x0000000000000000LL),
                       -2,
                       make_ti(0x8000000000000000LL, 0x0000000000000000LL), 1))
         return 1;
     if (test__muloti4(-2,
                       make_ti(0x8000000000000000LL, 0x0000000000000000LL),
                       make_ti(0x8000000000000000LL, 0x0000000000000000LL), 1))
         return 1;
     if (test__muloti4(make_ti(0x8000000000000000LL, 0x0000000000000000LL),
                       -1,
                       make_ti(0x8000000000000000LL, 0x0000000000000000LL), 1))
         return 1;
     if (test__muloti4(-1,
                       make_ti(0x8000000000000000LL, 0x0000000000000000LL),
                       make_ti(0x8000000000000000LL, 0x0000000000000000LL), 1))
         return 1;
    if (test__muloti4(make_ti(0x8000000000000000LL, 0x0000000000000000LL),
                      0,
                      0, 0))
        return 1;
    if (test__muloti4(0,
                      make_ti(0x8000000000000000LL, 0x0000000000000000LL),
                      0, 0))
        return 1;
    if (test__muloti4(make_ti(0x8000000000000000LL, 0x0000000000000000LL),
                      1,
                      make_ti(0x8000000000000000LL, 0x0000000000000000LL), 0))
        return 1;
    if (test__muloti4(1,
                      make_ti(0x8000000000000000LL, 0x0000000000000000LL),
                      make_ti(0x8000000000000000LL, 0x0000000000000000LL), 0))
        return 1;
     if (test__muloti4(make_ti(0x8000000000000000LL, 0x0000000000000000LL),
                       2,
                       make_ti(0x8000000000000000LL, 0x0000000000000000LL), 1))
         return 1;
     if (test__muloti4(2,
                       make_ti(0x8000000000000000LL, 0x0000000000000000LL),
                       make_ti(0x8000000000000000LL, 0x0000000000000000LL), 1))
         return 1;

     if (test__muloti4(make_ti(0x8000000000000000LL, 0x0000000000000001LL),
                       -2,
                       make_ti(0x8000000000000000LL, 0x0000000000000001LL), 1))
         return 1;
     if (test__muloti4(-2,
                       make_ti(0x8000000000000000LL, 0x0000000000000001LL),
                       make_ti(0x8000000000000000LL, 0x0000000000000001LL), 1))
         return 1;
    if (test__muloti4(make_ti(0x8000000000000000LL, 0x0000000000000001LL),
                      -1,
                      make_ti(0x7FFFFFFFFFFFFFFFLL, 0xFFFFFFFFFFFFFFFFLL), 0))
        return 1;
    if (test__muloti4(-1,
                      make_ti(0x8000000000000000LL, 0x0000000000000001LL),
                      make_ti(0x7FFFFFFFFFFFFFFFLL, 0xFFFFFFFFFFFFFFFFLL), 0))
        return 1;
    if (test__muloti4(make_ti(0x8000000000000000LL, 0x0000000000000001LL),
                      0,
                      0, 0))
        return 1;
    if (test__muloti4(0,
                      make_ti(0x8000000000000000LL, 0x0000000000000001LL),
                      0, 0))
        return 1;
    if (test__muloti4(make_ti(0x8000000000000000LL, 0x0000000000000001LL),
                      1,
                      make_ti(0x8000000000000000LL, 0x0000000000000001LL), 0))
        return 1;
    if (test__muloti4(1,
                      make_ti(0x8000000000000000LL, 0x0000000000000001LL),
                      make_ti(0x8000000000000000LL, 0x0000000000000001LL), 0))
        return 1;
     if (test__muloti4(make_ti(0x8000000000000000LL, 0x0000000000000001LL),
                       2,
                       make_ti(0x8000000000000000LL, 0x0000000000000000LL), 1))
         return 1;
     if (test__muloti4(2,
                       make_ti(0x8000000000000000LL, 0x0000000000000001LL),
                       make_ti(0x8000000000000000LL, 0x0000000000000000LL), 1))
         return 1;

#else
    printf("skipped\n");
#endif
    return 0;
}

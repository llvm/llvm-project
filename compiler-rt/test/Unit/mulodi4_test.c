//===-- mulodi4_test.c - Test __mulodi4 -----------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file tests __mulodi4 for the compiler_rt library.
//
//===----------------------------------------------------------------------===//

#include "int_lib.h"
#include <stdio.h>

extern di_int __mulodi4(di_int a, di_int b, int* overflow);

int test__mulodi4(di_int a, di_int b, di_int expected, int expected_overflow)
{
    int ov;
    di_int x = __mulodi4(a, b, &ov);
    if (ov != expected_overflow)
      printf("error in __mulodi4: overflow=%d expected=%d\n",
	     ov, expected_overflow);
    else if (!expected_overflow && x != expected) {
        printf("error in __mulodi4: 0x%llX * 0x%llX = 0x%llX (overflow=%d), "
			   "expected 0x%llX (overflow=%d)\n",
               a, b, x, ov, expected, expected_overflow);
		return 1;
    }
    return 0;
}

int main()
{
    if (test__mulodi4(0, 0, 0, 0))
        return 1;
    if (test__mulodi4(0, 1, 0, 0))
        return 1;
    if (test__mulodi4(1, 0, 0, 0))
        return 1;
    if (test__mulodi4(0, 10, 0, 0))
        return 1;
    if (test__mulodi4(10, 0, 0, 0))
        return 1;
    if (test__mulodi4(0, 81985529216486895LL, 0, 0))
        return 1;
    if (test__mulodi4(81985529216486895LL, 0, 0, 0))
        return 1;

    if (test__mulodi4(0, -1, 0, 0))
        return 1;
    if (test__mulodi4(-1, 0, 0, 0))
        return 1;
    if (test__mulodi4(0, -10, 0, 0))
        return 1;
    if (test__mulodi4(-10, 0, 0, 0))
        return 1;
    if (test__mulodi4(0, -81985529216486895LL, 0, 0))
        return 1;
    if (test__mulodi4(-81985529216486895LL, 0, 0, 0))
        return 1;

    if (test__mulodi4(1, 1, 1, 0))
        return 1;
    if (test__mulodi4(1, 10, 10, 0))
        return 1;
    if (test__mulodi4(10, 1, 10, 0))
        return 1;
    if (test__mulodi4(1, 81985529216486895LL, 81985529216486895LL, 0))
        return 1;
    if (test__mulodi4(81985529216486895LL, 1, 81985529216486895LL, 0))
        return 1;

    if (test__mulodi4(1, -1, -1, 0))
        return 1;
    if (test__mulodi4(1, -10, -10, 0))
        return 1;
    if (test__mulodi4(-10, 1, -10, 0))
        return 1;
    if (test__mulodi4(1, -81985529216486895LL, -81985529216486895LL, 0))
        return 1;
    if (test__mulodi4(-81985529216486895LL, 1, -81985529216486895LL, 0))
        return 1;

    if (test__mulodi4(3037000499LL, 3037000499LL, 9223372030926249001LL, 0))
        return 1;
    if (test__mulodi4(-3037000499LL, 3037000499LL, -9223372030926249001LL, 0))
        return 1;
    if (test__mulodi4(3037000499LL, -3037000499LL, -9223372030926249001LL, 0))
        return 1;
    if (test__mulodi4(-3037000499LL, -3037000499LL, 9223372030926249001LL, 0))
        return 1;

    if (test__mulodi4(4398046511103LL, 2097152LL, 9223372036852678656LL, 0))
        return 1;
    if (test__mulodi4(-4398046511103LL, 2097152LL, -9223372036852678656LL, 0))
        return 1;
    if (test__mulodi4(4398046511103LL, -2097152LL, -9223372036852678656LL, 0))
        return 1;
    if (test__mulodi4(-4398046511103LL, -2097152LL, 9223372036852678656LL, 0))
        return 1;

    if (test__mulodi4(2097152LL, 4398046511103LL, 9223372036852678656LL, 0))
        return 1;
    if (test__mulodi4(-2097152LL, 4398046511103LL, -9223372036852678656LL, 0))
        return 1;
    if (test__mulodi4(2097152LL, -4398046511103LL, -9223372036852678656LL, 0))
        return 1;
    if (test__mulodi4(-2097152LL, -4398046511103LL, 9223372036852678656LL, 0))
        return 1;

     if (test__mulodi4(0x7FFFFFFFFFFFFFFFLL, -2, 2, 1))
         return 1;
     if (test__mulodi4(-2, 0x7FFFFFFFFFFFFFFFLL, 2, 1))
         return 1;
    if (test__mulodi4(0x7FFFFFFFFFFFFFFFLL, -1, 0x8000000000000001LL, 0))
        return 1;
    if (test__mulodi4(-1, 0x7FFFFFFFFFFFFFFFLL, 0x8000000000000001LL, 0))
        return 1;
    if (test__mulodi4(0x7FFFFFFFFFFFFFFFLL, 0, 0, 0))
        return 1;
    if (test__mulodi4(0, 0x7FFFFFFFFFFFFFFFLL, 0, 0))
        return 1;
    if (test__mulodi4(0x7FFFFFFFFFFFFFFFLL, 1, 0x7FFFFFFFFFFFFFFFLL, 0))
        return 1;
    if (test__mulodi4(1, 0x7FFFFFFFFFFFFFFFLL, 0x7FFFFFFFFFFFFFFFLL, 0))
        return 1;
     if (test__mulodi4(0x7FFFFFFFFFFFFFFFLL, 2, 0x8000000000000001LL, 1))
         return 1;
     if (test__mulodi4(2, 0x7FFFFFFFFFFFFFFFLL, 0x8000000000000001LL, 1))
         return 1;

     if (test__mulodi4(0x8000000000000000LL, -2, 0x8000000000000000LL, 1))
         return 1;
     if (test__mulodi4(-2, 0x8000000000000000LL, 0x8000000000000000LL, 1))
         return 1;
     if (test__mulodi4(0x8000000000000000LL, -1, 0x8000000000000000LL, 1))
         return 1;
     if (test__mulodi4(-1, 0x8000000000000000LL, 0x8000000000000000LL, 1))
         return 1;
    if (test__mulodi4(0x8000000000000000LL, 0, 0, 0))
        return 1;
    if (test__mulodi4(0, 0x8000000000000000LL, 0, 0))
        return 1;
    if (test__mulodi4(0x8000000000000000LL, 1, 0x8000000000000000LL, 0))
        return 1;
    if (test__mulodi4(1, 0x8000000000000000LL, 0x8000000000000000LL, 0))
        return 1;
     if (test__mulodi4(0x8000000000000000LL, 2, 0x8000000000000000LL, 1))
         return 1;
     if (test__mulodi4(2, 0x8000000000000000LL, 0x8000000000000000LL, 1))
         return 1;

     if (test__mulodi4(0x8000000000000001LL, -2, 0x8000000000000001LL, 1))
         return 1;
     if (test__mulodi4(-2, 0x8000000000000001LL, 0x8000000000000001LL, 1))
         return 1;
    if (test__mulodi4(0x8000000000000001LL, -1, 0x7FFFFFFFFFFFFFFFLL, 0))
        return 1;
    if (test__mulodi4(-1, 0x8000000000000001LL, 0x7FFFFFFFFFFFFFFFLL, 0))
        return 1;
    if (test__mulodi4(0x8000000000000001LL, 0, 0, 0))
        return 1;
    if (test__mulodi4(0, 0x8000000000000001LL, 0, 0))
        return 1;
    if (test__mulodi4(0x8000000000000001LL, 1, 0x8000000000000001LL, 0))
        return 1;
    if (test__mulodi4(1, 0x8000000000000001LL, 0x8000000000000001LL, 0))
        return 1;
     if (test__mulodi4(0x8000000000000001LL, 2, 0x8000000000000000LL, 1))
         return 1;
     if (test__mulodi4(2, 0x8000000000000001LL, 0x8000000000000000LL, 1))
         return 1;

    return 0;
}

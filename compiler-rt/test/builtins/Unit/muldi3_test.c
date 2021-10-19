// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_muldi3

#include "int_lib.h"
#include <stdio.h>

COMPILER_RT_ABI di_int __muldi3(di_int a, di_int b);

int test__muldi3(di_int a, di_int b, di_int expected)
{
    di_int x = __muldi3(a, b);
    if (x != expected)
        printf("error in __muldi3: %lld * %lld = %lld, expected %lld\n",
               a, b, x, expected);
    return x != expected;
}

char assumption_1[sizeof(di_int) == 2*sizeof(si_int)] = {0};

int main()
{
    if (test__muldi3(0, 0, 0))
        return 1;
    if (test__muldi3(0, 1, 0))
        return 1;
    if (test__muldi3(1, 0, 0))
        return 1;
    if (test__muldi3(0, 10, 0))
        return 1;
    if (test__muldi3(10, 0, 0))
        return 1;
    if (test__muldi3(0, 81985529216486895LL, 0))
        return 1;
    if (test__muldi3(81985529216486895LL, 0, 0))
        return 1;

    if (test__muldi3(0, -1, 0))
        return 1;
    if (test__muldi3(-1, 0, 0))
        return 1;
    if (test__muldi3(0, -10, 0))
        return 1;
    if (test__muldi3(-10, 0, 0))
        return 1;
    if (test__muldi3(0, -81985529216486895LL, 0))
        return 1;
    if (test__muldi3(-81985529216486895LL, 0, 0))
        return 1;

    if (test__muldi3(1, 1, 1))
        return 1;
    if (test__muldi3(1, 10, 10))
        return 1;
    if (test__muldi3(10, 1, 10))
        return 1;
    if (test__muldi3(1, 81985529216486895LL, 81985529216486895LL))
        return 1;
    if (test__muldi3(81985529216486895LL, 1, 81985529216486895LL))
        return 1;

    if (test__muldi3(1, -1, -1))
        return 1;
    if (test__muldi3(1, -10, -10))
        return 1;
    if (test__muldi3(-10, 1, -10))
        return 1;
    if (test__muldi3(1, -81985529216486895LL, -81985529216486895LL))
        return 1;
    if (test__muldi3(-81985529216486895LL, 1, -81985529216486895LL))
        return 1;

    if (test__muldi3(3037000499LL, 3037000499LL, 9223372030926249001LL))
        return 1;
    if (test__muldi3(-3037000499LL, 3037000499LL, -9223372030926249001LL))
        return 1;
    if (test__muldi3(3037000499LL, -3037000499LL, -9223372030926249001LL))
        return 1;
    if (test__muldi3(-3037000499LL, -3037000499LL, 9223372030926249001LL))
        return 1;

    if (test__muldi3(4398046511103LL, 2097152LL, 9223372036852678656LL))
        return 1;
    if (test__muldi3(-4398046511103LL, 2097152LL, -9223372036852678656LL))
        return 1;
    if (test__muldi3(4398046511103LL, -2097152LL, -9223372036852678656LL))
        return 1;
    if (test__muldi3(-4398046511103LL, -2097152LL, 9223372036852678656LL))
        return 1;

    if (test__muldi3(2097152LL, 4398046511103LL, 9223372036852678656LL))
        return 1;
    if (test__muldi3(-2097152LL, 4398046511103LL, -9223372036852678656LL))
        return 1;
    if (test__muldi3(2097152LL, -4398046511103LL, -9223372036852678656LL))
        return 1;
    if (test__muldi3(-2097152LL, -4398046511103LL, 9223372036852678656LL))
        return 1;

    return 0;
}

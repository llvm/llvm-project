// REQUIRES-ANY: riscv32-target-arch
// RUN: %clang_builtins %s %librt -o %t && %run %t
//===-- mulsi3_test.c - Test __mulsi3 -------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file tests __mulsi3 for the compiler_rt library.
//
//===----------------------------------------------------------------------===//

#include "int_lib.h"
#include <stdio.h>
#include <limits.h>

#if !defined(__riscv_mul) && __riscv_xlen == 32
// Based on mulsi3_test.c

COMPILER_RT_ABI si_int __mulsi3(si_int a, si_int b);

int test__mulsi3(si_int a, si_int b, si_int expected)
{
    si_int x = __mulsi3(a, b);
    if (x != expected)
        printf("error in __mulsi3: %d * %d = %d, expected %d\n",
               a, b, __mulsi3(a, b), expected);
    return x != expected;
}
#endif

int main()
{
#if !defined(__riscv_mul) && __riscv_xlen == 32
    if (test__mulsi3(0, 0, 0))
        return 1;
    if (test__mulsi3(0, 1, 0))
        return 1;
    if (test__mulsi3(1, 0, 0))
        return 1;
    if (test__mulsi3(0, 10, 0))
        return 1;
    if (test__mulsi3(10, 0, 0))
        return 1;
    if (test__mulsi3(0, INT_MAX, 0))
        return 1;
    if (test__mulsi3(INT_MAX, 0, 0))
        return 1;

    if (test__mulsi3(0, -1, 0))
        return 1;
    if (test__mulsi3(-1, 0, 0))
        return 1;
    if (test__mulsi3(0, -10, 0))
        return 1;
    if (test__mulsi3(-10, 0, 0))
        return 1;
    if (test__mulsi3(0, INT_MIN, 0))
        return 1;
    if (test__mulsi3(INT_MIN, 0, 0))
        return 1;

    if (test__mulsi3(1, 1, 1))
        return 1;
    if (test__mulsi3(1, 10, 10))
        return 1;
    if (test__mulsi3(10, 1, 10))
        return 1;
    if (test__mulsi3(1, INT_MAX, INT_MAX))
        return 1;
    if (test__mulsi3(INT_MAX, 1, INT_MAX))
        return 1;

    if (test__mulsi3(1, -1, -1))
        return 1;
    if (test__mulsi3(1, -10, -10))
        return 1;
    if (test__mulsi3(-10, 1, -10))
        return 1;
    if (test__mulsi3(1, INT_MIN, INT_MIN))
        return 1;
    if (test__mulsi3(INT_MIN, 1, INT_MIN))
        return 1;

    if (test__mulsi3(46340, 46340, 2147395600))
        return 1;
    if (test__mulsi3(-46340, 46340, -2147395600))
        return 1;
    if (test__mulsi3(46340, -46340, -2147395600))
        return 1;
    if (test__mulsi3(-46340, -46340, 2147395600))
        return 1;

    if (test__mulsi3(4194303, 8192, 34359730176))
        return 1;
    if (test__mulsi3(-4194303, 8192, -34359730176))
        return 1;
    if (test__mulsi3(4194303, -8192, -34359730176))
        return 1;
    if (test__mulsi3(-4194303, -8192, 34359730176))
        return 1;

    if (test__mulsi3(8192, 4194303, 34359730176))
        return 1;
    if (test__mulsi3(-8192, 4194303, -34359730176))
        return 1;
    if (test__mulsi3(8192, -4194303, -34359730176))
        return 1;
    if (test__mulsi3(-8192, -4194303, 34359730176))
        return 1;
#else
    printf("skipped\n");
#endif

    return 0;
}

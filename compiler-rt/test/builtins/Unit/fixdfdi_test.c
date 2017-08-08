// RUN: %clang_builtins %s %librt -o %t && %run %t
//===-- fixdfdi_test.c - Test __fixdfdi -----------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file tests __fixdfdi for the compiler_rt library.
//
//===----------------------------------------------------------------------===//

#include "int_lib.h"
#include <stdio.h>

// Returns: convert a to a signed long long, rounding toward zero.

// Assumption: double is a IEEE 64 bit floating point type 
//             su_int is a 32 bit integral type
//             value in double is representable in di_int (no range checking performed)

// seee eeee eeee mmmm mmmm mmmm mmmm mmmm | mmmm mmmm mmmm mmmm mmmm mmmm mmmm mmmm

COMPILER_RT_ABI di_int __fixdfdi(double a);

int test__fixdfdi(double a, di_int expected)
{
    di_int x = __fixdfdi(a);
    if (x != expected)
        printf("error in __fixdfdi(%A) = %llX, expected %llX\n", a, x, expected);
    return x != expected;
}

char assumption_1[sizeof(di_int) == 2*sizeof(si_int)] = {0};
char assumption_2[sizeof(su_int)*CHAR_BIT == 32] = {0};
char assumption_3[sizeof(double)*CHAR_BIT == 64] = {0};

int main()
{
    if (test__fixdfdi(0.0, 0))
        return 1;

    if (test__fixdfdi(0.5, 0))
        return 1;
    if (test__fixdfdi(0.99, 0))
        return 1;
    if (test__fixdfdi(1.0, 1))
        return 1;
    if (test__fixdfdi(1.5, 1))
        return 1;
    if (test__fixdfdi(1.99, 1))
        return 1;
    if (test__fixdfdi(2.0, 2))
        return 1;
    if (test__fixdfdi(2.01, 2))
        return 1;
    if (test__fixdfdi(-0.5, 0))
        return 1;
    if (test__fixdfdi(-0.99, 0))
        return 1;
    if (test__fixdfdi(-1.0, -1))
        return 1;
    if (test__fixdfdi(-1.5, -1))
        return 1;
    if (test__fixdfdi(-1.99, -1))
        return 1;
    if (test__fixdfdi(-2.0, -2))
        return 1;
    if (test__fixdfdi(-2.01, -2))
        return 1;

    if (test__fixdfdi(0x1.FFFFFEp+62, 0x7FFFFF8000000000LL))
        return 1;
    if (test__fixdfdi(0x1.FFFFFCp+62, 0x7FFFFF0000000000LL))
        return 1;

    if (test__fixdfdi(-0x1.FFFFFEp+62, 0x8000008000000000LL))
        return 1;
    if (test__fixdfdi(-0x1.FFFFFCp+62, 0x8000010000000000LL))
        return 1;

    if (test__fixdfdi(0x1.FFFFFFFFFFFFFp+62, 0x7FFFFFFFFFFFFC00LL))
        return 1;
    if (test__fixdfdi(0x1.FFFFFFFFFFFFEp+62, 0x7FFFFFFFFFFFF800LL))
        return 1;

    if (test__fixdfdi(-0x1.FFFFFFFFFFFFFp+62, 0x8000000000000400LL))
        return 1;
    if (test__fixdfdi(-0x1.FFFFFFFFFFFFEp+62, 0x8000000000000800LL))
        return 1;

   return 0;
}

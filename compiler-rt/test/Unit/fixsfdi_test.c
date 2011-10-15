//===-- fixsfdi_test.c - Test __fixsfdi -----------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file tests __fixsfdi for the compiler_rt library.
//
//===----------------------------------------------------------------------===//

#include "int_lib.h"
#include <stdio.h>

// Returns: convert a to a signed long long, rounding toward zero.

// Assumption: float is a IEEE 32 bit floating point type 
//             su_int is a 32 bit integral type
//             value in float is representable in di_int (no range checking performed)

// seee eeee emmm mmmm mmmm mmmm mmmm mmmm

di_int __fixsfdi(float a);

int test__fixsfdi(float a, di_int expected)
{
    di_int x = __fixsfdi(a);
    if (x != expected)
        printf("error in __fixsfdi(%A) = %llX, expected %llX\n", a, x, expected);
    return x != expected;
}

char assumption_1[sizeof(di_int) == 2*sizeof(si_int)] = {0};
char assumption_2[sizeof(su_int)*CHAR_BIT == 32] = {0};
char assumption_3[sizeof(float)*CHAR_BIT == 32] = {0};

int main()
{
    if (test__fixsfdi(0.0F, 0))
        return 1;

    if (test__fixsfdi(0.5F, 0))
        return 1;
    if (test__fixsfdi(0.99F, 0))
        return 1;
    if (test__fixsfdi(1.0F, 1))
        return 1;
    if (test__fixsfdi(1.5F, 1))
        return 1;
    if (test__fixsfdi(1.99F, 1))
        return 1;
    if (test__fixsfdi(2.0F, 2))
        return 1;
    if (test__fixsfdi(2.01F, 2))
        return 1;
    if (test__fixsfdi(-0.5F, 0))
        return 1;
    if (test__fixsfdi(-0.99F, 0))
        return 1;
    if (test__fixsfdi(-1.0F, -1))
        return 1;
    if (test__fixsfdi(-1.5F, -1))
        return 1;
    if (test__fixsfdi(-1.99F, -1))
        return 1;
    if (test__fixsfdi(-2.0F, -2))
        return 1;
    if (test__fixsfdi(-2.01F, -2))
        return 1;

    if (test__fixsfdi(0x1.FFFFFEp+62F, 0x7FFFFF8000000000LL))
        return 1;
    if (test__fixsfdi(0x1.FFFFFCp+62F, 0x7FFFFF0000000000LL))
        return 1;

    if (test__fixsfdi(-0x1.FFFFFEp+62F, 0x8000008000000000LL))
        return 1;
    if (test__fixsfdi(-0x1.FFFFFCp+62F, 0x8000010000000000LL))
        return 1;

   return 0;
}

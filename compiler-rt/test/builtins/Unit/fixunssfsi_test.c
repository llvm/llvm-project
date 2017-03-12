//===-- fixunssfsi_test.c - Test __fixunssfsi -----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file tests __fixunssfsi for the compiler_rt library.
//
//===----------------------------------------------------------------------===//

#include "int_lib.h"
#include <stdio.h>

// Returns: convert a to a unsigned int, rounding toward zero.
//          Negative values all become zero.

// Assumption: float is a IEEE 32 bit floating point type 
//             su_int is a 32 bit integral type
//             value in float is representable in su_int or is negative 
//                 (no range checking performed)

// seee eeee emmm mmmm mmmm mmmm mmmm mmmm

COMPILER_RT_ABI su_int __fixunssfsi(float a);

int test__fixunssfsi(float a, su_int expected)
{
    su_int x = __fixunssfsi(a);
    if (x != expected)
        printf("error in __fixunssfsi(%A) = %X, expected %X\n", a, x, expected);
    return x != expected;
}

char assumption_2[sizeof(su_int)*CHAR_BIT == 32] = {0};
char assumption_3[sizeof(float)*CHAR_BIT == 32] = {0};

int main()
{
    if (test__fixunssfsi(0.0F, 0))
        return 1;

    if (test__fixunssfsi(0.5F, 0))
        return 1;
    if (test__fixunssfsi(0.99F, 0))
        return 1;
    if (test__fixunssfsi(1.0F, 1))
        return 1;
    if (test__fixunssfsi(1.5F, 1))
        return 1;
    if (test__fixunssfsi(1.99F, 1))
        return 1;
    if (test__fixunssfsi(2.0F, 2))
        return 1;
    if (test__fixunssfsi(2.01F, 2))
        return 1;
    if (test__fixunssfsi(-0.5F, 0))
        return 1;
    if (test__fixunssfsi(-0.99F, 0))
        return 1;
#if !TARGET_LIBGCC
    if (test__fixunssfsi(-1.0F, 0))  // libgcc ignores "returns 0 for negative input" spec
        return 1;
    if (test__fixunssfsi(-1.5F, 0))
        return 1;
    if (test__fixunssfsi(-1.99F, 0))
        return 1;
    if (test__fixunssfsi(-2.0F, 0))
        return 1;
    if (test__fixunssfsi(-2.01F, 0))
        return 1;
#endif

    if (test__fixunssfsi(0x1.000000p+31F, 0x80000000))
        return 1;
    if (test__fixunssfsi(0x1.000000p+32F, 0xFFFFFFFF))
        return 1;
    if (test__fixunssfsi(0x1.FFFFFEp+31F, 0xFFFFFF00))
        return 1;
    if (test__fixunssfsi(0x1.FFFFFEp+30F, 0x7FFFFF80))
        return 1;
    if (test__fixunssfsi(0x1.FFFFFCp+30F, 0x7FFFFF00))
        return 1;

#if !TARGET_LIBGCC
    if (test__fixunssfsi(-0x1.FFFFFEp+30F, 0))
        return 1;
    if (test__fixunssfsi(-0x1.FFFFFCp+30F, 0))
        return 1;
#endif

   return 0;
}

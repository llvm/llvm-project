//===-- fixunssfdi_test.c - Test __fixunssfdi -----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file tests __fixunssfdi for the compiler_rt library.
//
//===----------------------------------------------------------------------===//

#include "int_lib.h"
#include <stdio.h>

// Returns: convert a to a unsigned long long, rounding toward zero.
//          Negative values all become zero.

// Assumption: float is a IEEE 32 bit floating point type 
//             du_int is a 64 bit integral type
//             value in float is representable in du_int or is negative 
//                 (no range checking performed)

// seee eeee emmm mmmm mmmm mmmm mmmm mmmm

du_int __fixunssfdi(float a);

int test__fixunssfdi(float a, du_int expected)
{
    du_int x = __fixunssfdi(a);
    if (x != expected)
        printf("error in __fixunssfdi(%A) = %llX, expected %llX\n",
               a, x, expected);
    return x != expected;
}

char assumption_1[sizeof(du_int) == 2*sizeof(si_int)] = {0};
char assumption_2[sizeof(su_int)*CHAR_BIT == 32] = {0};
char assumption_3[sizeof(float)*CHAR_BIT == 32] = {0};

int main()
{
    if (test__fixunssfdi(0.0F, 0))
        return 1;

    if (test__fixunssfdi(0.5F, 0))
        return 1;
    if (test__fixunssfdi(0.99F, 0))
        return 1;
    if (test__fixunssfdi(1.0F, 1))
        return 1;
    if (test__fixunssfdi(1.5F, 1))
        return 1;
    if (test__fixunssfdi(1.99F, 1))
        return 1;
    if (test__fixunssfdi(2.0F, 2))
        return 1;
    if (test__fixunssfdi(2.01F, 2))
        return 1;
    if (test__fixunssfdi(-0.5F, 0))
        return 1;
    if (test__fixunssfdi(-0.99F, 0))
        return 1;
#if !TARGET_LIBGCC
    if (test__fixunssfdi(-1.0F, 0))  // libgcc ignores "returns 0 for negative input" spec
        return 1;
    if (test__fixunssfdi(-1.5F, 0))
        return 1;
    if (test__fixunssfdi(-1.99F, 0))
        return 1;
    if (test__fixunssfdi(-2.0F, 0))
        return 1;
    if (test__fixunssfdi(-2.01F, 0))
        return 1;
#endif

    if (test__fixunssfdi(0x1.FFFFFEp+63F, 0xFFFFFF0000000000LL))
        return 1;
    if (test__fixunssfdi(0x1.000000p+63F, 0x8000000000000000LL))
        return 1;
    if (test__fixunssfdi(0x1.FFFFFEp+62F, 0x7FFFFF8000000000LL))
        return 1;
    if (test__fixunssfdi(0x1.FFFFFCp+62F, 0x7FFFFF0000000000LL))
        return 1;

#if !TARGET_LIBGCC
    if (test__fixunssfdi(-0x1.FFFFFEp+62F, 0x0000000000000000LL))
        return 1;
    if (test__fixunssfdi(-0x1.FFFFFCp+62F, 0x0000000000000000LL))
        return 1;
#endif

   return 0;
}

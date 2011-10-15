//===-- fixunsdfsi_test.c - Test __fixunsdfsi -----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file tests __fixunsdfsi for the compiler_rt library.
//
//===----------------------------------------------------------------------===//

#include "int_lib.h"
#include <stdio.h>

// Returns: convert a to a unsigned int, rounding toward zero.
//          Negative values all become zero.

// Assumption: double is a IEEE 64 bit floating point type 
//             su_int is a 32 bit integral type
//             value in double is representable in su_int or is negative 
//                 (no range checking performed)

// seee eeee eeee mmmm mmmm mmmm mmmm mmmm | mmmm mmmm mmmm mmmm mmmm mmmm mmmm mmmm

su_int __fixunsdfsi(double a);

int test__fixunsdfsi(double a, su_int expected)
{
    su_int x = __fixunsdfsi(a);
    if (x != expected)
        printf("error in __fixunsdfsi(%A) = %X, expected %X\n", a, x, expected);
    return x != expected;
}

char assumption_2[sizeof(su_int)*CHAR_BIT == 32] = {0};
char assumption_3[sizeof(double)*CHAR_BIT == 64] = {0};

int main()
{
    if (test__fixunsdfsi(0.0, 0))
        return 1;

    if (test__fixunsdfsi(0.5, 0))
        return 1;
    if (test__fixunsdfsi(0.99, 0))
        return 1;
    if (test__fixunsdfsi(1.0, 1))
        return 1;
    if (test__fixunsdfsi(1.5, 1))
        return 1;
    if (test__fixunsdfsi(1.99, 1))
        return 1;
    if (test__fixunsdfsi(2.0, 2))
        return 1;
    if (test__fixunsdfsi(2.01, 2))
        return 1;
    if (test__fixunsdfsi(-0.5, 0))
        return 1;
    if (test__fixunsdfsi(-0.99, 0))
        return 1;
#if !TARGET_LIBGCC
    if (test__fixunsdfsi(-1.0, 0))  // libgcc ignores "returns 0 for negative input" spec
        return 1;
    if (test__fixunsdfsi(-1.5, 0))
        return 1;
    if (test__fixunsdfsi(-1.99, 0))
        return 1;
    if (test__fixunsdfsi(-2.0, 0))
        return 1;
    if (test__fixunsdfsi(-2.01, 0))
        return 1;
#endif

    if (test__fixunsdfsi(0x1.000000p+31, 0x80000000))
        return 1;
    if (test__fixunsdfsi(0x1.FFFFFEp+31, 0xFFFFFF00))
        return 1;
    if (test__fixunsdfsi(0x1.FFFFFEp+30, 0x7FFFFF80))
        return 1;
    if (test__fixunsdfsi(0x1.FFFFFCp+30, 0x7FFFFF00))
        return 1;

#if !TARGET_LIBGCC
    if (test__fixunsdfsi(-0x1.FFFFFEp+30, 0))
        return 1;
    if (test__fixunsdfsi(-0x1.FFFFFCp+30, 0))
        return 1;
#endif

    if (test__fixunsdfsi(0x1.FFFFFFFEp+31, 0xFFFFFFFF))
        return 1;
    if (test__fixunsdfsi(0x1.FFFFFFFC00000p+30, 0x7FFFFFFF))
        return 1;
    if (test__fixunsdfsi(0x1.FFFFFFF800000p+30, 0x7FFFFFFE))
        return 1;

   return 0;
}

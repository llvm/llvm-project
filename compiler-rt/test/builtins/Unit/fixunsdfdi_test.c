// RUN: %clang_builtins %s %librt -o %t && %run %t
//===-- fixunsdfdi_test.c - Test __fixunsdfdi -----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file tests __fixunsdfdi for the compiler_rt library.
//
//===----------------------------------------------------------------------===//

#include "int_lib.h"
#include <stdio.h>

// Returns: convert a to a unsigned long long, rounding toward zero.
//          Negative values all become zero.

// Assumption: double is a IEEE 64 bit floating point type 
//             du_int is a 64 bit integral type
//             value in double is representable in du_int or is negative 
//                 (no range checking performed)

// seee eeee eeee mmmm mmmm mmmm mmmm mmmm | mmmm mmmm mmmm mmmm mmmm mmmm mmmm mmmm

COMPILER_RT_ABI du_int __fixunsdfdi(double a);

int test__fixunsdfdi(double a, du_int expected)
{
    du_int x = __fixunsdfdi(a);
    if (x != expected)
        printf("error in __fixunsdfdi(%A) = %llX, expected %llX\n", a, x, expected);
    return x != expected;
}

char assumption_1[sizeof(du_int) == 2*sizeof(su_int)] = {0};
char assumption_2[sizeof(su_int)*CHAR_BIT == 32] = {0};
char assumption_3[sizeof(double)*CHAR_BIT == 64] = {0};

int main()
{
    if (test__fixunsdfdi(0.0, 0))
        return 1;

    if (test__fixunsdfdi(0.5, 0))
        return 1;
    if (test__fixunsdfdi(0.99, 0))
        return 1;
    if (test__fixunsdfdi(1.0, 1))
        return 1;
    if (test__fixunsdfdi(1.5, 1))
        return 1;
    if (test__fixunsdfdi(1.99, 1))
        return 1;
    if (test__fixunsdfdi(2.0, 2))
        return 1;
    if (test__fixunsdfdi(2.01, 2))
        return 1;
    if (test__fixunsdfdi(-0.5, 0))
        return 1;
    if (test__fixunsdfdi(-0.99, 0))
        return 1;
#if !TARGET_LIBGCC
    if (test__fixunsdfdi(-1.0, 0))  // libgcc ignores "returns 0 for negative input" spec
        return 1;
    if (test__fixunsdfdi(-1.5, 0))
        return 1;
    if (test__fixunsdfdi(-1.99, 0))
        return 1;
    if (test__fixunsdfdi(-2.0, 0))
        return 1;
    if (test__fixunsdfdi(-2.01, 0))
        return 1;
#endif

    if (test__fixunsdfdi(0x1.FFFFFEp+62, 0x7FFFFF8000000000LL))
        return 1;
    if (test__fixunsdfdi(0x1.FFFFFCp+62, 0x7FFFFF0000000000LL))
        return 1;

#if !TARGET_LIBGCC
    if (test__fixunsdfdi(-0x1.FFFFFEp+62, 0))
        return 1;
    if (test__fixunsdfdi(-0x1.FFFFFCp+62, 0))
        return 1;
#endif

    if (test__fixunsdfdi(0x1.FFFFFFFFFFFFFp+63, 0xFFFFFFFFFFFFF800LL))
        return 1;
    if (test__fixunsdfdi(0x1.0000000000000p+63, 0x8000000000000000LL))
        return 1;
    if (test__fixunsdfdi(0x1.FFFFFFFFFFFFFp+62, 0x7FFFFFFFFFFFFC00LL))
        return 1;
    if (test__fixunsdfdi(0x1.FFFFFFFFFFFFEp+62, 0x7FFFFFFFFFFFF800LL))
        return 1;

#if !TARGET_LIBGCC
    if (test__fixunsdfdi(-0x1.FFFFFFFFFFFFFp+62, 0))
        return 1;
    if (test__fixunsdfdi(-0x1.FFFFFFFFFFFFEp+62, 0))
        return 1;
#endif

   return 0;
}

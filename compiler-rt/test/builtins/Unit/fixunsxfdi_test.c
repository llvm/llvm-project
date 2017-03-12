//===-- fixunsxfdi_test.c - Test __fixunsxfdi -----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file tests __fixunsxfdi for the compiler_rt library.
//
//===----------------------------------------------------------------------===//

#include "int_lib.h"
#include <stdio.h>
#include <limits.h>


#if HAS_80_BIT_LONG_DOUBLE
// Returns: convert a to a unsigned long long, rounding toward zero.
//          Negative values all become zero.

// Assumption: long double is an intel 80 bit floating point type padded with 6 bytes
//             du_int is a 64 bit integral type
//             value in long double is representable in du_int or is negative 
//                 (no range checking performed)

// gggg gggg gggg gggg gggg gggg gggg gggg | gggg gggg gggg gggg seee eeee eeee eeee |
// 1mmm mmmm mmmm mmmm mmmm mmmm mmmm mmmm | mmmm mmmm mmmm mmmm mmmm mmmm mmmm mmmm

COMPILER_RT_ABI du_int __fixunsxfdi(long double a);

int test__fixunsxfdi(long double a, du_int expected)
{
    du_int x = __fixunsxfdi(a);
    if (x != expected)
        printf("error in __fixunsxfdi(%LA) = %llX, expected %llX\n",
               a, x, expected);
    return x != expected;
}

char assumption_1[sizeof(du_int) == 2*sizeof(su_int)] = {0};
char assumption_2[sizeof(du_int)*CHAR_BIT == 64] = {0};
char assumption_3[sizeof(long double)*CHAR_BIT == 128] = {0};
#endif

int main()
{
#if HAS_80_BIT_LONG_DOUBLE
    if (test__fixunsxfdi(0.0, 0))
        return 1;

    if (test__fixunsxfdi(0.5, 0))
        return 1;
    if (test__fixunsxfdi(0.99, 0))
        return 1;
    if (test__fixunsxfdi(1.0, 1))
        return 1;
    if (test__fixunsxfdi(1.5, 1))
        return 1;
    if (test__fixunsxfdi(1.99, 1))
        return 1;
    if (test__fixunsxfdi(2.0, 2))
        return 1;
    if (test__fixunsxfdi(2.01, 2))
        return 1;
    if (test__fixunsxfdi(-0.5, 0))
        return 1;
    if (test__fixunsxfdi(-0.99, 0))
        return 1;
    if (test__fixunsxfdi(-1.0, 0))
        return 1;
    if (test__fixunsxfdi(-1.5, 0))
        return 1;
    if (test__fixunsxfdi(-1.99, 0))
        return 1;
    if (test__fixunsxfdi(-2.0, 0))
        return 1;
    if (test__fixunsxfdi(-2.01, 0))
        return 1;

    if (test__fixunsxfdi(0x1.FFFFFEp+62, 0x7FFFFF8000000000LL))
        return 1;
    if (test__fixunsxfdi(0x1.FFFFFCp+62, 0x7FFFFF0000000000LL))
        return 1;

    if (test__fixunsxfdi(-0x1.FFFFFEp+62, 0))
        return 1;
    if (test__fixunsxfdi(-0x1.FFFFFCp+62, 0))
        return 1;

    if (test__fixunsxfdi(0x1.FFFFFFFFFFFFFp+62, 0x7FFFFFFFFFFFFC00LL))
        return 1;
    if (test__fixunsxfdi(0x1.FFFFFFFFFFFFEp+62, 0x7FFFFFFFFFFFF800LL))
        return 1;

    if (test__fixunsxfdi(-0x1.FFFFFFFFFFFFFp+62, 0))
        return 1;
    if (test__fixunsxfdi(-0x1.FFFFFFFFFFFFEp+62, 0))
        return 1;

    if (test__fixunsxfdi(0x1.FFFFFFFFFFFFFFFEp+63L, 0xFFFFFFFFFFFFFFFFLL))
        return 1;
    if (test__fixunsxfdi(0x1.0000000000000002p+63L, 0x8000000000000001LL))
        return 1;
    if (test__fixunsxfdi(0x1.0000000000000000p+63L, 0x8000000000000000LL))
        return 1;
    if (test__fixunsxfdi(0x1.FFFFFFFFFFFFFFFCp+62L, 0x7FFFFFFFFFFFFFFFLL))
        return 1;
    if (test__fixunsxfdi(0x1.FFFFFFFFFFFFFFF8p+62L, 0x7FFFFFFFFFFFFFFELL))
        return 1;

    if (test__fixunsxfdi(-0x1.0000000000000000p+63L, 0))
        return 1;
    if (test__fixunsxfdi(-0x1.FFFFFFFFFFFFFFFCp+62L, 0))
        return 1;
    if (test__fixunsxfdi(-0x1.FFFFFFFFFFFFFFF8p+62L, 0))
        return 1;

#else
    printf("skipped\n");
#endif
   return 0;
}

// RUN: %clang_builtins %s %librt -o %t && %run %t
//===-- fixunsdfti_test.c - Test __fixunsdfti -----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file tests __fixunsdfti for the compiler_rt library.
//
//===----------------------------------------------------------------------===//

#include "int_lib.h"
#include <stdio.h>

// Returns: convert a to a unsigned long long, rounding toward zero.
//          Negative values all become zero.

// Assumption: double is a IEEE 64 bit floating point type 
//             tu_int is a 64 bit integral type
//             value in double is representable in tu_int or is negative 
//                 (no range checking performed)

// seee eeee eeee mmmm mmmm mmmm mmmm mmmm | mmmm mmmm mmmm mmmm mmmm mmmm mmmm mmmm

#ifdef CRT_HAS_128BIT

COMPILER_RT_ABI tu_int __fixunsdfti(double a);

int test__fixunsdfti(double a, tu_int expected)
{
    tu_int x = __fixunsdfti(a);
    if (x != expected)
    {
        utwords xt;
        xt.all = x;
        utwords expectedt;
        expectedt.all = expected;
        printf("error in __fixunsdfti(%A) = 0x%.16llX%.16llX, expected 0x%.16llX%.16llX\n",
               a, xt.s.high, xt.s.low, expectedt.s.high, expectedt.s.low);
    }
    return x != expected;
}

char assumption_1[sizeof(tu_int) == 2*sizeof(du_int)] = {0};
char assumption_2[sizeof(su_int)*CHAR_BIT == 32] = {0};
char assumption_3[sizeof(double)*CHAR_BIT == 64] = {0};

#endif

int main()
{
#ifdef CRT_HAS_128BIT
    if (test__fixunsdfti(0.0, 0))
        return 1;

    if (test__fixunsdfti(0.5, 0))
        return 1;
    if (test__fixunsdfti(0.99, 0))
        return 1;
    if (test__fixunsdfti(1.0, 1))
        return 1;
    if (test__fixunsdfti(1.5, 1))
        return 1;
    if (test__fixunsdfti(1.99, 1))
        return 1;
    if (test__fixunsdfti(2.0, 2))
        return 1;
    if (test__fixunsdfti(2.01, 2))
        return 1;
    if (test__fixunsdfti(-0.5, 0))
        return 1;
    if (test__fixunsdfti(-0.99, 0))
        return 1;
#if !TARGET_LIBGCC
    if (test__fixunsdfti(-1.0, 0))  // libgcc ignores "returns 0 for negative input" spec
        return 1;
    if (test__fixunsdfti(-1.5, 0))
        return 1;
    if (test__fixunsdfti(-1.99, 0))
        return 1;
    if (test__fixunsdfti(-2.0, 0))
        return 1;
    if (test__fixunsdfti(-2.01, 0))
        return 1;
#endif

    if (test__fixunsdfti(0x1.FFFFFEp+62, 0x7FFFFF8000000000LL))
        return 1;
    if (test__fixunsdfti(0x1.FFFFFCp+62, 0x7FFFFF0000000000LL))
        return 1;

#if !TARGET_LIBGCC
    if (test__fixunsdfti(-0x1.FFFFFEp+62, 0))
        return 1;
    if (test__fixunsdfti(-0x1.FFFFFCp+62, 0))
        return 1;
#endif

    if (test__fixunsdfti(0x1.FFFFFFFFFFFFFp+63, 0xFFFFFFFFFFFFF800LL))
        return 1;
    if (test__fixunsdfti(0x1.0000000000000p+63, 0x8000000000000000LL))
        return 1;
    if (test__fixunsdfti(0x1.FFFFFFFFFFFFFp+62, 0x7FFFFFFFFFFFFC00LL))
        return 1;
    if (test__fixunsdfti(0x1.FFFFFFFFFFFFEp+62, 0x7FFFFFFFFFFFF800LL))
        return 1;

    if (test__fixunsdfti(0x1.FFFFFFFFFFFFFp+127, make_ti(0xFFFFFFFFFFFFF800LL, 0)))
        return 1;
    if (test__fixunsdfti(0x1.0000000000000p+127, make_ti(0x8000000000000000LL, 0)))
        return 1;
    if (test__fixunsdfti(0x1.FFFFFFFFFFFFFp+126, make_ti(0x7FFFFFFFFFFFFC00LL, 0)))
        return 1;
    if (test__fixunsdfti(0x1.FFFFFFFFFFFFEp+126, make_ti(0x7FFFFFFFFFFFF800LL, 0)))
        return 1;
    if (test__fixunsdfti(0x1.0000000000000p+128, make_ti(0xFFFFFFFFFFFFFFFFLL,
                                                         0xFFFFFFFFFFFFFFFFLL)))
        return 1;

#if !TARGET_LIBGCC
    if (test__fixunsdfti(-0x1.FFFFFFFFFFFFFp+62, 0))
        return 1;
    if (test__fixunsdfti(-0x1.FFFFFFFFFFFFEp+62, 0))
        return 1;
#endif

#endif
   return 0;
}

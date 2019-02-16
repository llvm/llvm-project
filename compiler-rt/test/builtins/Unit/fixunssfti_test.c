// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: int128
//===-- fixunssfti_test.c - Test __fixunssfti -----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file tests __fixunssfti for the compiler_rt library.
//
//===----------------------------------------------------------------------===//

#include "int_lib.h"
#include <stdio.h>

#ifdef CRT_HAS_128BIT

// Returns: convert a to a unsigned long long, rounding toward zero.
//          Negative values all become zero.

// Assumption: float is a IEEE 32 bit floating point type 
//             tu_int is a 64 bit integral type
//             value in float is representable in tu_int or is negative 
//                 (no range checking performed)

// seee eeee emmm mmmm mmmm mmmm mmmm mmmm

COMPILER_RT_ABI tu_int __fixunssfti(float a);

int test__fixunssfti(float a, tu_int expected)
{
    tu_int x = __fixunssfti(a);
    if (x != expected)
    {
        utwords xt;
        xt.all = x;
        utwords expectedt;
        expectedt.all = expected;
        printf("error in __fixunssfti(%A) = 0x%.16llX%.16llX, expected 0x%.16llX%.16llX\n",
               a, xt.s.high, xt.s.low, expectedt.s.high, expectedt.s.low);
    }
    return x != expected;
}

char assumption_1[sizeof(tu_int) == 2*sizeof(di_int)] = {0};
char assumption_2[sizeof(su_int)*CHAR_BIT == 32] = {0};
char assumption_3[sizeof(float)*CHAR_BIT == 32] = {0};

#endif

int main()
{
#ifdef CRT_HAS_128BIT
    if (test__fixunssfti(0.0F, 0))
        return 1;

    if (test__fixunssfti(0.5F, 0))
        return 1;
    if (test__fixunssfti(0.99F, 0))
        return 1;
    if (test__fixunssfti(1.0F, 1))
        return 1;
    if (test__fixunssfti(1.5F, 1))
        return 1;
    if (test__fixunssfti(1.99F, 1))
        return 1;
    if (test__fixunssfti(2.0F, 2))
        return 1;
    if (test__fixunssfti(2.01F, 2))
        return 1;
    if (test__fixunssfti(-0.5F, 0))
        return 1;
    if (test__fixunssfti(-0.99F, 0))
        return 1;
#if !TARGET_LIBGCC
    if (test__fixunssfti(-1.0F, 0))  // libgcc ignores "returns 0 for negative input" spec
        return 1;
    if (test__fixunssfti(-1.5F, 0))
        return 1;
    if (test__fixunssfti(-1.99F, 0))
        return 1;
    if (test__fixunssfti(-2.0F, 0))
        return 1;
    if (test__fixunssfti(-2.01F, 0))
        return 1;
#endif

    if (test__fixunssfti(0x1.FFFFFEp+63F, 0xFFFFFF0000000000ULL))
        return 1;
    if (test__fixunssfti(0x1.000000p+63F, 0x8000000000000000ULL))
        return 1;
    if (test__fixunssfti(0x1.FFFFFEp+62F, 0x7FFFFF8000000000LL))
        return 1;
    if (test__fixunssfti(0x1.FFFFFCp+62F, 0x7FFFFF0000000000LL))
        return 1;

    if (test__fixunssfti(0x1.FFFFFEp+127F, make_ti(0xFFFFFF0000000000ULL, 0)))
        return 1;
    if (test__fixunssfti(0x1.000000p+127F, make_ti(0x8000000000000000ULL, 0)))
        return 1;
    if (test__fixunssfti(0x1.FFFFFEp+126F, make_ti(0x7FFFFF8000000000LL, 0)))
        return 1;
    if (test__fixunssfti(0x1.FFFFFCp+126F, make_ti(0x7FFFFF0000000000LL, 0)))
        return 1;

#if !TARGET_LIBGCC
    if (test__fixunssfti(-0x1.FFFFFEp+62F, 0x0000000000000000LL))
        return 1;
    if (test__fixunssfti(-0x1.FFFFFCp+62F, 0x0000000000000000LL))
        return 1;
    if (test__fixunssfti(-0x1.FFFFFEp+126F, 0x0000000000000000LL))
        return 1;
    if (test__fixunssfti(-0x1.FFFFFCp+126F, 0x0000000000000000LL))
        return 1;
#endif

#endif
   return 0;
}

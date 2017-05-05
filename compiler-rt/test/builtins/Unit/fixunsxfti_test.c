// RUN: %clang_builtins %s %librt -o %t && %run %t
// XFAIL: aarch64
// test fails for aarch64 (see pr32260)

// UNSUPPORTED: mips

//===-- fixunsxfti_test.c - Test __fixunsxfti -----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file tests __fixunsxfti for the compiler_rt library.
//
//===----------------------------------------------------------------------===//

#include "int_lib.h"
#include <stdio.h>

#ifdef CRT_HAS_128BIT

// Returns: convert a to a unsigned long long, rounding toward zero.
//          Negative values all become zero.

// Assumption: long double is an intel 80 bit floating point type padded with 6 bytes
//             tu_int is a 64 bit integral type
//             value in long double is representable in tu_int or is negative 
//                 (no range checking performed)

// gggg gggg gggg gggg gggg gggg gggg gggg | gggg gggg gggg gggg seee eeee eeee eeee |
// 1mmm mmmm mmmm mmmm mmmm mmmm mmmm mmmm | mmmm mmmm mmmm mmmm mmmm mmmm mmmm mmmm

COMPILER_RT_ABI tu_int __fixunsxfti(long double a);

int test__fixunsxfti(long double a, tu_int expected)
{
    tu_int x = __fixunsxfti(a);
    if (x != expected)
    {
        utwords xt;
        xt.all = x;
        utwords expectedt;
        expectedt.all = expected;
        printf("error in __fixunsxfti(%LA) = 0x%.16llX%.16llX, expected 0x%.16llX%.16llX\n",
               a, xt.s.high, xt.s.low, expectedt.s.high, expectedt.s.low);
    }
    return x != expected;
}

char assumption_1[sizeof(tu_int) == 2*sizeof(du_int)] = {0};
char assumption_2[sizeof(tu_int)*CHAR_BIT == 128] = {0};
char assumption_3[sizeof(long double)*CHAR_BIT == 128] = {0};

#endif

int main()
{
#ifdef CRT_HAS_128BIT
    if (test__fixunsxfti(0.0, 0))
        return 1;

    if (test__fixunsxfti(0.5, 0))
        return 1;
    if (test__fixunsxfti(0.99, 0))
        return 1;
    if (test__fixunsxfti(1.0, 1))
        return 1;
    if (test__fixunsxfti(1.5, 1))
        return 1;
    if (test__fixunsxfti(1.99, 1))
        return 1;
    if (test__fixunsxfti(2.0, 2))
        return 1;
    if (test__fixunsxfti(2.01, 2))
        return 1;
    if (test__fixunsxfti(-0.5, 0))
        return 1;
    if (test__fixunsxfti(-0.99, 0))
        return 1;
    if (test__fixunsxfti(-1.0, 0))
        return 1;
    if (test__fixunsxfti(-1.5, 0))
        return 1;
    if (test__fixunsxfti(-1.99, 0))
        return 1;
    if (test__fixunsxfti(-2.0, 0))
        return 1;
    if (test__fixunsxfti(-2.01, 0))
        return 1;

    if (test__fixunsxfti(0x1.FFFFFEp+62, 0x7FFFFF8000000000LL))
        return 1;
    if (test__fixunsxfti(0x1.FFFFFCp+62, 0x7FFFFF0000000000LL))
        return 1;

    if (test__fixunsxfti(-0x1.FFFFFEp+62, 0))
        return 1;
    if (test__fixunsxfti(-0x1.FFFFFCp+62, 0))
        return 1;

    if (test__fixunsxfti(0x1.FFFFFFFFFFFFFp+62, 0x7FFFFFFFFFFFFC00LL))
        return 1;
    if (test__fixunsxfti(0x1.FFFFFFFFFFFFEp+62, 0x7FFFFFFFFFFFF800LL))
        return 1;

    if (test__fixunsxfti(-0x1.FFFFFFFFFFFFFp+62, 0))
        return 1;
    if (test__fixunsxfti(-0x1.FFFFFFFFFFFFEp+62, 0))
        return 1;

    if (test__fixunsxfti(0x1.FFFFFFFFFFFFFFFEp+63L, 0xFFFFFFFFFFFFFFFFLL))
        return 1;
    if (test__fixunsxfti(0x1.0000000000000002p+63L, 0x8000000000000001LL))
        return 1;
    if (test__fixunsxfti(0x1.0000000000000000p+63L, 0x8000000000000000LL))
        return 1;
    if (test__fixunsxfti(0x1.FFFFFFFFFFFFFFFCp+62L, 0x7FFFFFFFFFFFFFFFLL))
        return 1;
    if (test__fixunsxfti(0x1.FFFFFFFFFFFFFFF8p+62L, 0x7FFFFFFFFFFFFFFELL))
        return 1;

    if (test__fixunsxfti(-0x1.0000000000000000p+63L, 0))
        return 1;
    if (test__fixunsxfti(-0x1.FFFFFFFFFFFFFFFCp+62L, 0))
        return 1;
    if (test__fixunsxfti(-0x1.FFFFFFFFFFFFFFF8p+62L, 0))
        return 1;

    if (test__fixunsxfti(0x1.FFFFFFFFFFFFFFFEp+127L, make_ti(0xFFFFFFFFFFFFFFFFLL, 0)))
        return 1;
    if (test__fixunsxfti(0x1.0000000000000002p+127L, make_ti(0x8000000000000001LL, 0)))
        return 1;
    if (test__fixunsxfti(0x1.0000000000000000p+127L, make_ti(0x8000000000000000LL, 0)))
        return 1;
    if (test__fixunsxfti(0x1.FFFFFFFFFFFFFFFCp+126L, make_ti(0x7FFFFFFFFFFFFFFFLL, 0)))
        return 1;
    if (test__fixunsxfti(0x1.FFFFFFFFFFFFFFF8p+126L, make_ti(0x7FFFFFFFFFFFFFFELL, 0)))
        return 1;

#else
    printf("skipped\n");
#endif
   return 0;
}

//===-- fixsfti_test.c - Test __fixsfti -----------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file tests __fixsfti for the compiler_rt library.
//
//===----------------------------------------------------------------------===//

#include "int_lib.h"
#include <stdio.h>

#ifdef CRT_HAS_128BIT

// Returns: convert a to a signed long long, rounding toward zero.

// Assumption: float is a IEEE 32 bit floating point type 
//             su_int is a 32 bit integral type
//             value in float is representable in ti_int (no range checking performed)

// seee eeee emmm mmmm mmmm mmmm mmmm mmmm

COMPILER_RT_ABI ti_int __fixsfti(float a);

int test__fixsfti(float a, ti_int expected)
{
    ti_int x = __fixsfti(a);
    if (x != expected)
    {
        twords xt;
        xt.all = x;
        twords expectedt;
        expectedt.all = expected;
        printf("error in __fixsfti(%A) = 0x%.16llX%.16llX, expected 0x%.16llX%.16llX\n",
        a, xt.s.high, xt.s.low, expectedt.s.high, expectedt.s.low);
    }
    return x != expected;
}

char assumption_1[sizeof(ti_int) == 2*sizeof(di_int)] = {0};
char assumption_2[sizeof(su_int)*CHAR_BIT == 32] = {0};
char assumption_3[sizeof(float)*CHAR_BIT == 32] = {0};

#endif

int main()
{
#ifdef CRT_HAS_128BIT
    if (test__fixsfti(0.0F, 0))
        return 1;

    if (test__fixsfti(0.5F, 0))
        return 1;
    if (test__fixsfti(0.99F, 0))
        return 1;
    if (test__fixsfti(1.0F, 1))
        return 1;
    if (test__fixsfti(1.5F, 1))
        return 1;
    if (test__fixsfti(1.99F, 1))
        return 1;
    if (test__fixsfti(2.0F, 2))
        return 1;
    if (test__fixsfti(2.01F, 2))
        return 1;
    if (test__fixsfti(-0.5F, 0))
        return 1;
    if (test__fixsfti(-0.99F, 0))
        return 1;
    if (test__fixsfti(-1.0F, -1))
        return 1;
    if (test__fixsfti(-1.5F, -1))
        return 1;
    if (test__fixsfti(-1.99F, -1))
        return 1;
    if (test__fixsfti(-2.0F, -2))
        return 1;
    if (test__fixsfti(-2.01F, -2))
        return 1;

    if (test__fixsfti(0x1.FFFFFEp+62F, 0x7FFFFF8000000000LL))
        return 1;
    if (test__fixsfti(0x1.FFFFFCp+62F, 0x7FFFFF0000000000LL))
        return 1;

    if (test__fixsfti(-0x1.FFFFFEp+62F, make_ti(0xFFFFFFFFFFFFFFFFLL,
                                                0x8000008000000000LL)))
        return 1;
    if (test__fixsfti(-0x1.FFFFFCp+62F, make_ti(0xFFFFFFFFFFFFFFFFLL,
                                                0x8000010000000000LL)))
        return 1;

    if (test__fixsfti(0x1.FFFFFEp+126F, make_ti(0x7FFFFF8000000000LL, 0)))
        return 1;
    if (test__fixsfti(0x1.FFFFFCp+126F, make_ti(0x7FFFFF0000000000LL, 0)))
        return 1;

    if (test__fixsfti(-0x1.FFFFFEp+126F, make_ti(0x8000008000000000LL, 0)))
        return 1;
    if (test__fixsfti(-0x1.FFFFFCp+126F, make_ti(0x8000010000000000LL, 0)))
        return 1;

#else
    printf("skipped\n");
#endif
   return 0;
}

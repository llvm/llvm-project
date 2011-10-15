//===-- fixxfti_test.c - Test __fixxfti -----------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file tests __fixxfti for the compiler_rt library.
//
//===----------------------------------------------------------------------===//

#if __x86_64

#include "int_lib.h"
#include <stdio.h>

// Returns: convert a to a signed long long, rounding toward zero.

// Assumption: long double is an intel 80 bit floating point type padded with 6 bytes
//             su_int is a 32 bit integral type
//             value in long double is representable in ti_int (no range checking performed)

// gggg gggg gggg gggg gggg gggg gggg gggg | gggg gggg gggg gggg seee eeee eeee eeee |
// 1mmm mmmm mmmm mmmm mmmm mmmm mmmm mmmm | mmmm mmmm mmmm mmmm mmmm mmmm mmmm mmmm

ti_int __fixxfti(long double a);

int test__fixxfti(long double a, ti_int expected)
{
    ti_int x = __fixxfti(a);
    if (x != expected)
    {
        utwords xt;
        xt.all = x;
        utwords expectedt;
        expectedt.all = expected;
        printf("error in __fixxfti(%LA) = 0x%.16llX%.16llX, expected 0x%.16llX%.16llX\n",
               a, xt.s.high, xt.s.low, expectedt.s.high, expectedt.s.low);
    }
    return x != expected;
}

char assumption_1[sizeof(ti_int) == 2*sizeof(di_int)] = {0};
char assumption_2[sizeof(su_int)*CHAR_BIT == 32] = {0};
char assumption_3[sizeof(long double)*CHAR_BIT == 128] = {0};

#endif

int main()
{
#if __x86_64
    if (test__fixxfti(0.0, 0))
        return 1;

    if (test__fixxfti(0.5, 0))
        return 1;
    if (test__fixxfti(0.99, 0))
        return 1;
    if (test__fixxfti(1.0, 1))
        return 1;
    if (test__fixxfti(1.5, 1))
        return 1;
    if (test__fixxfti(1.99, 1))
        return 1;
    if (test__fixxfti(2.0, 2))
        return 1;
    if (test__fixxfti(2.01, 2))
        return 1;
    if (test__fixxfti(-0.5, 0))
        return 1;
    if (test__fixxfti(-0.99, 0))
        return 1;
    if (test__fixxfti(-1.0, -1))
        return 1;
    if (test__fixxfti(-1.5, -1))
        return 1;
    if (test__fixxfti(-1.99, -1))
        return 1;
    if (test__fixxfti(-2.0, -2))
        return 1;
    if (test__fixxfti(-2.01, -2))
        return 1;

    if (test__fixxfti(0x1.FFFFFEp+62, 0x7FFFFF8000000000LL))
        return 1;
    if (test__fixxfti(0x1.FFFFFCp+62, 0x7FFFFF0000000000LL))
        return 1;

    if (test__fixxfti(-0x1.FFFFFEp+62, make_ti(0xFFFFFFFFFFFFFFFFLL,
                                               0x8000008000000000LL)))
        return 1;
    if (test__fixxfti(-0x1.FFFFFCp+62, make_ti(0xFFFFFFFFFFFFFFFFLL,
                                               0x8000010000000000LL)))
        return 1;

    if (test__fixxfti(0x1.FFFFFFFFFFFFFp+62, 0x7FFFFFFFFFFFFC00LL))
        return 1;
    if (test__fixxfti(0x1.FFFFFFFFFFFFEp+62, 0x7FFFFFFFFFFFF800LL))
        return 1;

    if (test__fixxfti(-0x1.FFFFFFFFFFFFFp+62, make_ti(0xFFFFFFFFFFFFFFFFLL,
                                                      0x8000000000000400LL)))
        return 1;
    if (test__fixxfti(-0x1.FFFFFFFFFFFFEp+62, make_ti(0xFFFFFFFFFFFFFFFFLL,
                                                      0x8000000000000800LL)))
        return 1;

    if (test__fixxfti(0x1.FFFFFFFFFFFFFFFCp+62L, 0x7FFFFFFFFFFFFFFFLL))
        return 1;
    if (test__fixxfti(0x1.FFFFFFFFFFFFFFF8p+62L, 0x7FFFFFFFFFFFFFFELL))
        return 1;

    if (test__fixxfti(-0x1.0000000000000000p+63L, make_ti(0xFFFFFFFFFFFFFFFFLL,
                                                          0x8000000000000000LL)))
        return 1;
    if (test__fixxfti(-0x1.FFFFFFFFFFFFFFFCp+62L, make_ti(0xFFFFFFFFFFFFFFFFLL,
                                                          0x8000000000000001LL)))
        return 1;
    if (test__fixxfti(-0x1.FFFFFFFFFFFFFFF8p+62L, make_ti(0xFFFFFFFFFFFFFFFFLL,
                                                          0x8000000000000002LL)))
        return 1;

    if (test__fixxfti(0x1.FFFFFFFFFFFFFFFEp+126L, make_ti(0x7FFFFFFFFFFFFFFFLL,
                                                          0x8000000000000000LL)))
        return 1;
    if (test__fixxfti(0x1.FFFFFFFFFFFFFFFCp+126L, make_ti(0x7FFFFFFFFFFFFFFFLL,
                                                          0x0000000000000000LL)))

        return 1;

    if (test__fixxfti(-0x1.0000000000000000p+127L, make_ti(0x8000000000000000LL,
                                                           0x0000000000000000LL)))
        return 1;
    if (test__fixxfti(-0x1.FFFFFFFFFFFFFFFEp+126L, make_ti(0x8000000000000000LL,
                                                           0x8000000000000000LL)))
        return 1;
    if (test__fixxfti(-0x1.FFFFFFFFFFFFFFFCp+126L, make_ti(0x8000000000000001LL,
                                                           0x0000000000000000LL)))
        return 1;
#else
    printf("skipped\n");
#endif
   return 0;
}

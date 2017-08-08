// RUN: %clang_builtins %s %librt -o %t && %run %t
//===--------------- divtf3_test.c - Test __divtf3 ------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file tests __divtf3 for the compiler_rt library.
//
//===----------------------------------------------------------------------===//

#include "int_lib.h"
#include <stdio.h>

#if __LDBL_MANT_DIG__ == 113

#include "fp_test.h"

// Returns: a / b
COMPILER_RT_ABI long double __divtf3(long double a, long double b);

int test__divtf3(long double a, long double b,
                 uint64_t expectedHi, uint64_t expectedLo)
{
    long double x = __divtf3(a, b);
    int ret = compareResultLD(x, expectedHi, expectedLo);

    if (ret){
        printf("error in test__divtf3(%.20Lf, %.20Lf) = %.20Lf, "
               "expected %.20Lf\n", a, b, x,
               fromRep128(expectedHi, expectedLo));
    }
    return ret;
}

char assumption_1[sizeof(long double) * CHAR_BIT == 128] = {0};

#endif

int main()
{
#if __LDBL_MANT_DIG__ == 113
    // qNaN / any = qNaN
    if (test__divtf3(makeQNaN128(),
                     0x1.23456789abcdefp+5L,
                     UINT64_C(0x7fff800000000000),
                     UINT64_C(0x0)))
        return 1;
    // NaN / any = NaN
    if (test__divtf3(makeNaN128(UINT64_C(0x800030000000)),
                     0x1.23456789abcdefp+5L,
                     UINT64_C(0x7fff800000000000),
                     UINT64_C(0x0)))
        return 1;
    // inf / any = inf
    if (test__divtf3(makeInf128(),
                     0x1.23456789abcdefp+5L,
                     UINT64_C(0x7fff000000000000),
                     UINT64_C(0x0)))
        return 1;
    // any / any
    if (test__divtf3(0x1.a23b45362464523375893ab4cdefp+5L,
                     0x1.eedcbaba3a94546558237654321fp-1L,
                     UINT64_C(0x4004b0b72924d407),
                     UINT64_C(0x0717e84356c6eba2)))
        return 1;
    if (test__divtf3(0x1.a2b34c56d745382f9abf2c3dfeffp-50L,
                     0x1.ed2c3ba15935332532287654321fp-9L,
                     UINT64_C(0x3fd5b2af3f828c9b),
                     UINT64_C(0x40e51f64cde8b1f2)))
        return 15;
    if (test__divtf3(0x1.2345f6aaaa786555f42432abcdefp+456L,
                     0x1.edacbba9874f765463544dd3621fp+6400L,
                     UINT64_C(0x28c62e15dc464466),
                     UINT64_C(0xb5a07586348557ac)))
        return 1;
    if (test__divtf3(0x1.2d3456f789ba6322bc665544edefp-234L,
                     0x1.eddcdba39f3c8b7a36564354321fp-4455L,
                     UINT64_C(0x507b38442b539266),
                     UINT64_C(0x22ce0f1d024e1252)))
        return 1;
    if (test__divtf3(0x1.2345f6b77b7a8953365433abcdefp+234L,
                     0x1.edcba987d6bb3aa467754354321fp-4055L,
                     UINT64_C(0x50bf2e02f0798d36),
                     UINT64_C(0x5e6fcb6b60044078)))
        return 1;

#else
    printf("skipped\n");

#endif
    return 0;
}

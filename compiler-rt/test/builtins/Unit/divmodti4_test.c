// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_divmodti4
// REQUIRES: int128
//===-- divmodti4_test.c - Test __divmodti4 -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file tests __divmodti4 for the compiler_rt library.
//
//===----------------------------------------------------------------------===//

#include "int_lib.h"
#include <stdio.h>

#ifdef CRT_HAS_128BIT

// Effects: if rem != 0, *rem = a % b
// Returns: a / b

COMPILER_RT_ABI ti_int __divmodti4(ti_int a, ti_int b, ti_int* rem);

int test__divmodti4(ti_int a, ti_int b, ti_int expected_q, ti_int expected_r) {
    ti_int r;
    ti_int q = __divmodti4(a, b, &r);
    if (q != expected_q || r != expected_r)
    {
        utwords at;
        at.all = a;
        utwords bt;
        bt.all = b;
        utwords expected_qt;
        expected_qt.all = expected_q;
        utwords expected_rt;
        expected_rt.all = expected_r;
        utwords qt;
        qt.all = q;
        utwords rt;
        rt.all = r;
        printf("error in __divmodti4: 0x%.16llX%.16llX / 0x%.16llX%.16llX = "
               "0x%.16llX%.16llX, R = 0x%.16llX%.16llX, expected 0x%.16llX%.16llX, "
               "0x%.16llX%.16llX\n",
               at.s.high, at.s.low, bt.s.high, bt.s.low, qt.s.high, qt.s.low,
               rt.s.high, rt.s.low, expected_qt.s.high, expected_qt.s.low,
               expected_rt.s.high, expected_rt.s.low);
    }
    return !(q == expected_q && r == expected_r);
}

char assumption_1[sizeof(ti_int) == 2*sizeof(di_int)] = {0};

tu_int tests[][4] =
{
{ (ti_int) 0,                              (ti_int) 1, (ti_int) 0,                                                  (ti_int) 0 },
{ (ti_int) 0,                              (ti_int)-1, (ti_int) 0,                                                  (ti_int) 0 },
{ (ti_int) 2,                              (ti_int) 1, (ti_int) 2,                                                  (ti_int) 0 },
{ (ti_int) 2,                              (ti_int)-1, (ti_int)-2,                                                  (ti_int) 0 },
{ (ti_int)-2,                              (ti_int) 1, (ti_int)-2,                                                  (ti_int) 0 },
{ (ti_int)-2,                              (ti_int)-1, (ti_int) 2,                                                  (ti_int) 0 },
{ (ti_int) 5,                              (ti_int) 3, (ti_int) 1,                                                  (ti_int) 2 },
{ (ti_int) 5,                              (ti_int)-3, (ti_int)-1,                                                  (ti_int) 2 },
{ (ti_int)-5,                              (ti_int) 3, (ti_int)-1,                                                  (ti_int)-2 },
{ (ti_int)-5,                              (ti_int)-3, (ti_int) 1,                                                  (ti_int)-2 },
{ (ti_int)0x8000000000000000ULL << 64 | 0, (ti_int) 1, (ti_int)0x8000000000000000ULL << 64 | 0,                     (ti_int)0x0LL },
{ (ti_int)0x8000000000000000ULL << 64 | 0, (ti_int)-1, (ti_int)0x8000000000000000ULL << 64 | 0,                     (ti_int)0x0LL },
{ (ti_int)0x8000000000000000ULL << 64 | 0, (ti_int)-2, (ti_int)0x4000000000000000ULL << 64 | 0,                     (ti_int)0x0LL },
{ (ti_int)0x8000000000000000ULL << 64 | 0, (ti_int) 2, (ti_int)0xC000000000000000ULL << 64 | 0,                     (ti_int)0x0LL },
{ (ti_int)0x8000000000000000ULL << 64 | 0, (ti_int)-3, (ti_int)0x2AAAAAAAAAAAAAAAULL << 64 | 0xAAAAAAAAAAAAAAAAULL, (ti_int)-2 },
{ (ti_int)0x8000000000000000ULL << 64 | 0, (ti_int) 3, (ti_int)0xD555555555555555ULL << 64 | 0x5555555555555556ULL, (ti_int)-2 },
};

#endif

int main()
{
#ifdef CRT_HAS_128BIT
    const unsigned N = sizeof(tests) / sizeof(tests[0]);
    unsigned i;
    for (i = 0; i < N; ++i)
        if (test__divmodti4(tests[i][0], tests[i][1], tests[i][2], tests[i][3]))
            return 1;


#else
    printf("skipped\n");
#endif
    return 0;
}

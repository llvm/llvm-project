// RUN: %clang_builtins %s %librt -o %t && %run %t
//===-- lshrti3_test.c - Test __lshrti3 -----------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file tests __lshrti3 for the compiler_rt library.
//
//===----------------------------------------------------------------------===//

#include "int_lib.h"
#include <stdio.h>

#ifdef CRT_HAS_128BIT

// Returns: logical a >> b

// Precondition:  0 <= b < bits_in_dword

COMPILER_RT_ABI ti_int __lshrti3(ti_int a, si_int b);

int test__lshrti3(ti_int a, si_int b, ti_int expected)
{
    ti_int x = __lshrti3(a, b);
    if (x != expected)
    {
        twords at;
        at.all = a;
        twords xt;
        xt.all = x;
        twords expectedt;
        expectedt.all = expected;
        printf("error in __lshrti3: 0x%llX%.16llX >> %d = 0x%llX%.16llX,"
               " expected 0x%llX%.16llX\n",
                at.s.high, at.s.low, b, xt.s.high, xt.s.low,
                expectedt.s.high, expectedt.s.low);
    }
    return x != expected;
}

char assumption_1[sizeof(ti_int) == 2*sizeof(di_int)] = {0};

#endif

int main()
{
#ifdef CRT_HAS_128BIT
    if (test__lshrti3(make_ti(0xFEDCBA9876543215LL, 0xFEDCBA9876543215LL), 0,
                      make_ti(0xFEDCBA9876543215LL, 0xFEDCBA9876543215LL)))
        return 1;
    if (test__lshrti3(make_ti(0xFEDCBA9876543215LL, 0xFEDCBA9876543215LL), 1,
                      make_ti(0x7F6E5D4C3B2A190ALL, 0xFF6E5D4C3B2A190ALL)))
        return 1;
    if (test__lshrti3(make_ti(0xFEDCBA9876543215LL, 0xFEDCBA9876543215LL), 2,
                      make_ti(0x3FB72EA61D950C85LL, 0x7FB72EA61D950C85LL)))
        return 1;
    if (test__lshrti3(make_ti(0xFEDCBA9876543215LL, 0xFEDCBA9876543215LL), 3,
                      make_ti(0x1FDB97530ECA8642LL, 0xBFDB97530ECA8642LL)))
        return 1;
    if (test__lshrti3(make_ti(0xFEDCBA9876543215LL, 0xFEDCBA9876543215LL), 4,
                      make_ti(0x0FEDCBA987654321LL, 0x5FEDCBA987654321LL)))
        return 1;

    if (test__lshrti3(make_ti(0xFEDCBA9876543215LL, 0xFEDCBA9876543215LL), 28,
                      make_ti(0x0000000FEDCBA987LL, 0x6543215FEDCBA987LL)))
        return 1;
    if (test__lshrti3(make_ti(0xFEDCBA9876543215LL, 0xFEDCBA9876543215LL), 29,
                      make_ti(0x00000007F6E5D4C3LL, 0xB2A190AFF6E5D4C3LL)))
        return 1;
    if (test__lshrti3(make_ti(0xFEDCBA9876543215LL, 0xFEDCBA9876543215LL), 30,
                      make_ti(0x00000003FB72EA61LL, 0xD950C857FB72EA61LL)))
        return 1;
    if (test__lshrti3(make_ti(0xFEDCBA9876543215LL, 0xFEDCBA9876543215LL), 31,
                      make_ti(0x00000001FDB97530LL, 0xECA8642BFDB97530LL)))
        return 1;

    if (test__lshrti3(make_ti(0xFEDCBA9876543215LL, 0xFEDCBA9876543215LL), 32,
                      make_ti(0x00000000FEDCBA98LL, 0x76543215FEDCBA98LL)))
        return 1;

    if (test__lshrti3(make_ti(0xFEDCBA9876543215LL, 0xFEDCBA9876543215LL), 33,
                      make_ti(0x000000007F6E5D4CLL, 0x3B2A190AFF6E5D4CLL)))
        return 1;
    if (test__lshrti3(make_ti(0xFEDCBA9876543215LL, 0xFEDCBA9876543215LL), 34,
                      make_ti(0x000000003FB72EA6LL, 0x1D950C857FB72EA6LL)))
        return 1;
    if (test__lshrti3(make_ti(0xFEDCBA9876543215LL, 0xFEDCBA9876543215LL), 35,
                      make_ti(0x000000001FDB9753LL, 0x0ECA8642BFDB9753LL)))
        return 1;
    if (test__lshrti3(make_ti(0xFEDCBA9876543215LL, 0xFEDCBA9876543215LL), 36,
                      make_ti(0x000000000FEDCBA9LL, 0x876543215FEDCBA9LL)))
        return 1;

    if (test__lshrti3(make_ti(0xFEDCBA9876543215LL, 0xFEDCBA9876543215LL), 60,
                      make_ti(0x000000000000000FLL, 0xEDCBA9876543215FLL)))
        return 1;
    if (test__lshrti3(make_ti(0xFEDCBA9876543215LL, 0xFEDCBA9876543215LL), 61,
                      make_ti(0x0000000000000007LL, 0xF6E5D4C3B2A190AFLL)))
        return 1;
    if (test__lshrti3(make_ti(0xFEDCBA9876543215LL, 0xFEDCBA9876543215LL), 62,
                      make_ti(0x0000000000000003LL, 0xFB72EA61D950C857LL)))
        return 1;
    if (test__lshrti3(make_ti(0xFEDCBA9876543215LL, 0xFEDCBA9876543215LL), 63,
                      make_ti(0x0000000000000001LL, 0xFDB97530ECA8642BLL)))
        return 1;

    if (test__lshrti3(make_ti(0xFEDCBA9876543215LL, 0xFEDCBA9876543215LL), 64,
                      make_ti(0x0000000000000000LL, 0xFEDCBA9876543215LL)))
        return 1;

    if (test__lshrti3(make_ti(0xFEDCBA9876543215LL, 0xFEDCBA9876543215LL), 65,
                      make_ti(0x0000000000000000LL, 0x7F6E5D4C3B2A190ALL)))
        return 1;
    if (test__lshrti3(make_ti(0xFEDCBA9876543215LL, 0xFEDCBA9876543215LL), 66,
                      make_ti(0x0000000000000000LL, 0x3FB72EA61D950C85LL)))
        return 1;
    if (test__lshrti3(make_ti(0xFEDCBA9876543215LL, 0xFEDCBA9876543215LL), 67,
                      make_ti(0x0000000000000000LL, 0x1FDB97530ECA8642LL)))
        return 1;
    if (test__lshrti3(make_ti(0xFEDCBA9876543215LL, 0xFEDCBA9876543215LL), 68,
                      make_ti(0x0000000000000000LL, 0x0FEDCBA987654321LL)))
        return 1;

    if (test__lshrti3(make_ti(0xFEDCBA9876543215LL, 0xFEDCBA9876543215LL), 92,
                      make_ti(0x0000000000000000LL, 0x0000000FEDCBA987LL)))
        return 1;
    if (test__lshrti3(make_ti(0xFEDCBA9876543215LL, 0xFEDCBA9876543215LL), 93,
                      make_ti(0x0000000000000000LL, 0x00000007F6E5D4C3LL)))
        return 1;
    if (test__lshrti3(make_ti(0xFEDCBA9876543215LL, 0xFEDCBA9876543215LL), 94,
                      make_ti(0x0000000000000000LL, 0x00000003FB72EA61LL)))
        return 1;
    if (test__lshrti3(make_ti(0xFEDCBA9876543215LL, 0xFEDCBA9876543215LL), 95,
                      make_ti(0x0000000000000000LL, 0x00000001FDB97530LL)))
        return 1;

    if (test__lshrti3(make_ti(0xFEDCBA9876543215LL, 0xFEDCBA9876543215LL), 96,
                      make_ti(0x0000000000000000LL, 0x00000000FEDCBA98LL)))
        return 1;

    if (test__lshrti3(make_ti(0xFEDCBA9876543215LL, 0xFEDCBA9876543215LL), 97,
                      make_ti(0x0000000000000000LL, 0x000000007F6E5D4CLL)))
        return 1;
    if (test__lshrti3(make_ti(0xFEDCBA9876543215LL, 0xFEDCBA9876543215LL), 98,
                      make_ti(0x0000000000000000LL, 0x000000003FB72EA6LL)))
        return 1;
    if (test__lshrti3(make_ti(0xFEDCBA9876543215LL, 0xFEDCBA9876543215LL), 99,
                      make_ti(0x0000000000000000LL, 0x000000001FDB9753LL)))
        return 1;
    if (test__lshrti3(make_ti(0xFEDCBA9876543215LL, 0xFEDCBA9876543215LL), 100,
                      make_ti(0x0000000000000000LL, 0x000000000FEDCBA9LL)))
        return 1;

    if (test__lshrti3(make_ti(0xFEDCBA9876543215LL, 0xFEDCBA9876543215LL), 124,
                      make_ti(0x0000000000000000LL, 0x000000000000000FLL)))
        return 1;
    if (test__lshrti3(make_ti(0xFEDCBA9876543215LL, 0xFEDCBA9876543215LL), 125,
                      make_ti(0x0000000000000000LL, 0x0000000000000007LL)))
        return 1;
    if (test__lshrti3(make_ti(0xFEDCBA9876543215LL, 0xFEDCBA9876543215LL), 126,
                      make_ti(0x0000000000000000LL, 0x0000000000000003LL)))
        return 1;
    if (test__lshrti3(make_ti(0xFEDCBA9876543215LL, 0xFEDCBA9876543215LL), 127,
                      make_ti(0x0000000000000000LL, 0x0000000000000001LL)))
        return 1;
#else
    printf("skipped\n");
#endif
    return 0;
}

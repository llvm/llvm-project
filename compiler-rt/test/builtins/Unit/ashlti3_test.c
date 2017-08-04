// RUN: %clang_builtins %s %librt -o %t && %run %t
//===-- ashlti3_test.c - Test __ashlti3 -----------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file tests __ashlti3 for the compiler_rt library.
//
//===----------------------------------------------------------------------===//

#include "int_lib.h"
#include <stdio.h>

#ifdef CRT_HAS_128BIT

// Returns: a << b

// Precondition:  0 <= b < bits_in_tword

COMPILER_RT_ABI ti_int __ashlti3(ti_int a, si_int b);

int test__ashlti3(ti_int a, si_int b, ti_int expected)
{
    ti_int x = __ashlti3(a, b);
    if (x != expected)
    {
        twords at;
        at.all = a;
        twords bt;
        bt.all = b;
        twords xt;
        xt.all = x;
        twords expectedt;
        expectedt.all = expected;
        printf("error in __ashlti3: 0x%llX%.16llX << %d = 0x%llX%.16llX,"
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
    if (test__ashlti3(make_ti(0xFEDCBA9876543215LL, 0xFEDCBA9876543215LL), 0,
                      make_ti(0xFEDCBA9876543215LL, 0xFEDCBA9876543215LL)))
        return 1;
    if (test__ashlti3(make_ti(0xFEDCBA9876543215LL, 0xFEDCBA9876543215LL), 1,
                      make_ti(0xFDB97530ECA8642BLL, 0xFDB97530ECA8642ALL)))
        return 1;
    if (test__ashlti3(make_ti(0xFEDCBA9876543215LL, 0xFEDCBA9876543215LL), 2,
                      make_ti(0xFB72EA61D950C857LL, 0XFB72EA61D950C854LL)))
        return 1;
    if (test__ashlti3(make_ti(0xFEDCBA9876543215LL, 0xFEDCBA9876543215LL), 3,
                      make_ti(0xF6E5D4C3B2A190AFLL, 0xF6E5D4C3B2A190A8LL)))
        return 1;
    if (test__ashlti3(make_ti(0xFEDCBA9876543215LL, 0xFEDCBA9876543215LL), 4,
                      make_ti(0xEDCBA9876543215FLL, 0xEDCBA98765432150LL)))
        return 1;

    if (test__ashlti3(make_ti(0xFEDCBA9876543215LL, 0xFEDCBA9876543215LL), 28,
                      make_ti(0x876543215FEDCBA9LL, 0x8765432150000000LL)))
        return 1;
    if (test__ashlti3(make_ti(0xFEDCBA9876543215LL, 0xFEDCBA9876543215LL), 29,
                      make_ti(0x0ECA8642BFDB9753LL, 0x0ECA8642A0000000LL)))
        return 1;
    if (test__ashlti3(make_ti(0xFEDCBA9876543215LL, 0xFEDCBA9876543215LL), 30,
                      make_ti(0x1D950C857FB72EA6LL, 0x1D950C8540000000LL)))
        return 1;
    if (test__ashlti3(make_ti(0xFEDCBA9876543215LL, 0xFEDCBA9876543215LL), 31,
                      make_ti(0x3B2A190AFF6E5D4CLL, 0x3B2A190A80000000LL)))
        return 1;

    if (test__ashlti3(make_ti(0xFEDCBA9876543215LL, 0xFEDCBA9876543215LL), 32,
                      make_ti(0x76543215FEDCBA98LL, 0x7654321500000000LL)))
        return 1;

    if (test__ashlti3(make_ti(0xFEDCBA9876543215LL, 0xFEDCBA9876543215LL), 33,
                      make_ti(0xECA8642BFDB97530LL, 0xECA8642A00000000LL)))
        return 1;
    if (test__ashlti3(make_ti(0xFEDCBA9876543215LL, 0xFEDCBA9876543215LL), 34,
                      make_ti(0xD950C857FB72EA61LL, 0xD950C85400000000LL)))
        return 1;
    if (test__ashlti3(make_ti(0xFEDCBA9876543215LL, 0xFEDCBA9876543215LL), 35,
                      make_ti(0xB2A190AFF6E5D4C3LL, 0xB2A190A800000000LL)))
        return 1;
    if (test__ashlti3(make_ti(0xFEDCBA9876543215LL, 0xFEDCBA9876543215LL), 36,
                      make_ti(0x6543215FEDCBA987LL, 0x6543215000000000LL)))
        return 1;

    if (test__ashlti3(make_ti(0xFEDCBA9876543215LL, 0xFEDCBA9876543215LL), 60,
                      make_ti(0x5FEDCBA987654321LL, 0x5000000000000000LL)))
        return 1;
    if (test__ashlti3(make_ti(0xFEDCBA9876543215LL, 0xFEDCBA9876543215LL), 61,
                      make_ti(0xBFDB97530ECA8642LL, 0xA000000000000000LL)))
        return 1;
    if (test__ashlti3(make_ti(0xFEDCBA9876543215LL, 0xFEDCBA9876543215LL), 62,
                      make_ti(0x7FB72EA61D950C85LL, 0x4000000000000000LL)))
        return 1;
    if (test__ashlti3(make_ti(0xFEDCBA9876543215LL, 0xFEDCBA9876543215LL), 63,
                      make_ti(0xFF6E5D4C3B2A190ALL, 0x8000000000000000LL)))
        return 1;

    if (test__ashlti3(make_ti(0xFEDCBA9876543215LL, 0xFEDCBA9876543215LL), 64,
                      make_ti(0xFEDCBA9876543215LL, 0x0000000000000000LL)))
        return 1;

    if (test__ashlti3(make_ti(0xFEDCBA9876543215LL, 0xFEDCBA9876543215LL), 65,
                      make_ti(0xFDB97530ECA8642ALL, 0x0000000000000000LL)))
        return 1;
    if (test__ashlti3(make_ti(0xFEDCBA9876543215LL, 0xFEDCBA9876543215LL), 66,
                      make_ti(0xFB72EA61D950C854LL, 0x0000000000000000LL)))
        return 1;
    if (test__ashlti3(make_ti(0xFEDCBA9876543215LL, 0xFEDCBA9876543215LL), 67,
                      make_ti(0xF6E5D4C3B2A190A8LL, 0x0000000000000000LL)))
        return 1;
    if (test__ashlti3(make_ti(0xFEDCBA9876543215LL, 0xFEDCBA9876543215LL), 68,
                      make_ti(0xEDCBA98765432150LL, 0x0000000000000000LL)))
        return 1;

    if (test__ashlti3(make_ti(0xFEDCBA9876543215LL, 0xFEDCBA9876543215LL), 92,
                      make_ti(0x8765432150000000LL, 0x0000000000000000LL)))
        return 1;
    if (test__ashlti3(make_ti(0xFEDCBA9876543215LL, 0xFEDCBA9876543215LL), 93,
                      make_ti(0x0ECA8642A0000000LL, 0x0000000000000000LL)))
        return 1;
    if (test__ashlti3(make_ti(0xFEDCBA9876543215LL, 0xFEDCBA9876543215LL), 94,
                      make_ti(0x1D950C8540000000LL, 0x0000000000000000LL)))
        return 1;
    if (test__ashlti3(make_ti(0xFEDCBA9876543215LL, 0xFEDCBA9876543215LL), 95,
                      make_ti(0x3B2A190A80000000LL, 0x0000000000000000LL)))
        return 1;

    if (test__ashlti3(make_ti(0xFEDCBA9876543215LL, 0xFEDCBA9876543215LL), 96,
                      make_ti(0x7654321500000000LL, 0x0000000000000000LL)))
        return 1;

    if (test__ashlti3(make_ti(0xFEDCBA9876543215LL, 0xFEDCBA9876543215LL), 97,
                      make_ti(0xECA8642A00000000LL, 0x0000000000000000LL)))
        return 1;
    if (test__ashlti3(make_ti(0xFEDCBA9876543215LL, 0xFEDCBA9876543215LL), 98,
                      make_ti(0xD950C85400000000LL, 0x0000000000000000LL)))
        return 1;
    if (test__ashlti3(make_ti(0xFEDCBA9876543215LL, 0xFEDCBA9876543215LL), 99,
                      make_ti(0xB2A190A800000000LL, 0x0000000000000000LL)))
        return 1;
    if (test__ashlti3(make_ti(0xFEDCBA9876543215LL, 0xFEDCBA9876543215LL), 100,
                      make_ti(0x6543215000000000LL, 0x0000000000000000LL)))
        return 1;

    if (test__ashlti3(make_ti(0xFEDCBA9876543215LL, 0xFEDCBA9876543215LL), 124,
                      make_ti(0x5000000000000000LL, 0x0000000000000000LL)))
        return 1;
    if (test__ashlti3(make_ti(0xFEDCBA9876543215LL, 0xFEDCBA9876543215LL), 125,
                      make_ti(0xA000000000000000LL, 0x0000000000000000LL)))
        return 1;
    if (test__ashlti3(make_ti(0xFEDCBA9876543215LL, 0xFEDCBA9876543215LL), 126,
                      make_ti(0x4000000000000000LL, 0x0000000000000000LL)))
        return 1;
    if (test__ashlti3(make_ti(0xFEDCBA9876543215LL, 0xFEDCBA9876543215LL), 127,
                      make_ti(0x8000000000000000LL, 0x0000000000000000LL)))
        return 1;
#else
    printf("skipped\n");
#endif
    return 0;
}

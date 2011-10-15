//===-- ucmpti2_test.c - Test __ucmpti2 -----------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file tests __ucmpti2 for the compiler_rt library.
//
//===----------------------------------------------------------------------===//

#if __x86_64

#include "int_lib.h"
#include <stdio.h>

// Returns:  if (a <  b) returns 0
//           if (a == b) returns 1
//           if (a >  b) returns 2

si_int __ucmpti2(tu_int a, tu_int b);

int test__ucmpti2(tu_int a, tu_int b, si_int expected)
{
    si_int x = __ucmpti2(a, b);
    if (x != expected)
    {
        utwords at;
        at.all = a;
        utwords bt;
        bt.all = b;
        printf("error in __ucmpti2(0x%.16llX%.16llX, 0x%.16llX%.16llX) = %d, "
               "expected %d\n",
               at.s.high, at.s.low, bt.s.high, bt.s.low, x, expected);
    }
    return x != expected;
}

#endif

int main()
{
#if __x86_64
    if (test__ucmpti2(0, 0, 1))
        return 1;
    if (test__ucmpti2(1, 1, 1))
        return 1;
    if (test__ucmpti2(2, 2, 1))
        return 1;
    if (test__ucmpti2(0x7FFFFFFF, 0x7FFFFFFF, 1))
        return 1;
    if (test__ucmpti2(0x80000000, 0x80000000, 1))
        return 1;
    if (test__ucmpti2(0x80000001, 0x80000001, 1))
        return 1;
    if (test__ucmpti2(0xFFFFFFFF, 0xFFFFFFFF, 1))
        return 1;
    if (test__ucmpti2(0x000000010000000LL, 0x000000010000000LL, 1))
        return 1;
    if (test__ucmpti2(0xFFFFFFFFFFFFFFFFLL, 0xFFFFFFFFFFFFFFFFLL, 1))
        return 1;

    if (test__ucmpti2(0x0000000200000002LL, 0x0000000300000001LL, 0))
        return 1;
    if (test__ucmpti2(0x0000000200000002LL, 0x0000000300000002LL, 0))
        return 1;
    if (test__ucmpti2(0x0000000200000002LL, 0x0000000300000003LL, 0))
        return 1;

    if (test__ucmpti2(0x0000000200000002LL, 0x0000000100000001LL, 2))
        return 1;
    if (test__ucmpti2(0x0000000200000002LL, 0x0000000100000002LL, 2))
        return 1;
    if (test__ucmpti2(0x0000000200000002LL, 0x0000000100000003LL, 2))
        return 1;

    if (test__ucmpti2(0x0000000200000002LL, 0x0000000200000001LL, 2))
        return 1;
    if (test__ucmpti2(0x0000000200000002LL, 0x0000000200000002LL, 1))
        return 1;
    if (test__ucmpti2(0x0000000200000002LL, 0x0000000200000003LL, 0))
        return 1;

    if (test__ucmpti2(make_tu(0x0000000000000001uLL, 0x0000000000000000uLL),
                      make_tu(0x0000000000000000uLL, 0xFFFFFFFFFFFFFFFFuLL), 2))
        return 1;
    if (test__ucmpti2(make_tu(0x0000000000000001uLL, 0x0000000000000000uLL),
                      make_tu(0x0000000000000001uLL, 0x0000000000000000uLL), 1))
        return 1;
    if (test__ucmpti2(make_tu(0x0000000000000001uLL, 0x0000000000000000uLL),
                      make_tu(0x0000000000000001uLL, 0x0000000000000001uLL), 0))
        return 1;

    if (test__ucmpti2(make_tu(0x8000000000000000uLL, 0x0000000000000000uLL),
                      make_tu(0x7FFFFFFFFFFFFFFFuLL, 0xFFFFFFFFFFFFFFFFuLL), 2))
        return 1;
    if (test__ucmpti2(make_tu(0x8000000000000000uLL, 0x0000000000000000uLL),
                      make_tu(0x8000000000000000uLL, 0x0000000000000000uLL), 1))
        return 1;
    if (test__ucmpti2(make_tu(0x8000000000000000uLL, 0x0000000000000000uLL),
                      make_tu(0x8000000000000000uLL, 0x0000000000000001uLL), 0))
        return 1;

    if (test__ucmpti2(make_tu(0xFFFFFFFFFFFFFFFFuLL, 0xFFFFFFFFFFFFFFFFuLL),
                      make_tu(0xFFFFFFFFFFFFFFFFuLL, 0xFFFFFFFFFFFFFFFEuLL), 2))
        return 1;
    if (test__ucmpti2(make_tu(0xFFFFFFFFFFFFFFFFuLL, 0xFFFFFFFFFFFFFFFFuLL),
                      make_tu(0xFFFFFFFFFFFFFFFFuLL, 0xFFFFFFFFFFFFFFFFuLL), 1))
        return 1;
#else
    printf("skipped\n");
#endif
   return 0;
}

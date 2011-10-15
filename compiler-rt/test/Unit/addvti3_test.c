//===-- addvti3_test.c - Test __addvti3 -----------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file tests __addvti3 for the compiler_rt library.
//
//===----------------------------------------------------------------------===//

#if __x86_64

#include "int_lib.h"
#include <stdio.h>

// Returns: a + b

// Effects: aborts if a + b overflows

ti_int __addvti3(ti_int a, ti_int b);

int test__addvti3(ti_int a, ti_int b)
{
    ti_int x = __addvti3(a, b);
    ti_int expected = a + b;
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
        printf("error in test__addvti3(0x%llX%.16llX, 0x%llX%.16llX) = "
               "0x%llX%.16llX, expected 0x%llX%.16llX\n",
                at.s.high, at.s.low, bt.s.high, bt.s.low, xt.s.high, xt.s.low,
                expectedt.s.high, expectedt.s.low);
    }
    return x != expected;
}

#endif

int main()
{
#if __x86_64
// should abort
//     test__addvti3(make_ti(0x8000000000000000LL, 0x0000000000000000LL),
//                   make_ti(0xFFFFFFFFFFFFFFFFLL, 0xFFFFFFFFFFFFFFFFLL));
// should abort
//     test__addvti3(make_ti(0xFFFFFFFFFFFFFFFFLL, 0xFFFFFFFFFFFFFFFFLL),
//                   make_ti(0x8000000000000000LL, 0x0000000000000000LL));
// should abort
//     test__addvti3(make_ti(0x0000000000000000LL, 0x0000000000000001LL),
//                   make_ti(0x7FFFFFFFFFFFFFFFLL, 0xFFFFFFFFFFFFFFFFLL));
// should abort
//     test__addvti3(make_ti(0x7FFFFFFFFFFFFFFFLL, 0xFFFFFFFFFFFFFFFFLL),
//                   make_ti(0x0000000000000000LL, 0x0000000000000001LL));

    if (test__addvti3(make_ti(0x8000000000000000LL, 0x0000000000000000LL),
                      make_ti(0x0000000000000000LL, 0x0000000000000001LL)))
        return 1;
    if (test__addvti3(make_ti(0x0000000000000000LL, 0x0000000000000001LL),
                      make_ti(0x8000000000000000LL, 0x0000000000000000LL)))
        return 1;
    if (test__addvti3(make_ti(0x8000000000000000LL, 0x0000000000000000LL),
                      make_ti(0x0000000000000000LL, 0x0000000000000000LL)))
        return 1;
    if (test__addvti3(make_ti(0x0000000000000000LL, 0x0000000000000000LL),
                      make_ti(0x8000000000000000LL, 0x0000000000000000LL)))
        return 1;
    if (test__addvti3(make_ti(0x7FFFFFFFFFFFFFFFLL, 0xFFFFFFFFFFFFFFFFLL),
                      make_ti(0xFFFFFFFFFFFFFFFFLL, 0xFFFFFFFFFFFFFFFFLL)))
        return 1;
    if (test__addvti3(make_ti(0xFFFFFFFFFFFFFFFFLL, 0xFFFFFFFFFFFFFFFFLL),
                      make_ti(0x7FFFFFFFFFFFFFFFLL, 0xFFFFFFFFFFFFFFFFLL)))
        return 1;
    if (test__addvti3(make_ti(0x7FFFFFFFFFFFFFFFLL, 0xFFFFFFFFFFFFFFFFLL),
                      make_ti(0x0000000000000000LL, 0x0000000000000000LL)))
        return 1;
    if (test__addvti3(make_ti(0x0000000000000000LL, 0x0000000000000000LL),
                      make_ti(0x7FFFFFFFFFFFFFFFLL, 0xFFFFFFFFFFFFFFFFLL)))
        return 1;

#else
    printf("skipped\n");
#endif
    return 0;
}

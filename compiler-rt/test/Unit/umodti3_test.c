//===-- umodti3_test.c - Test __umodti3 -----------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file tests __umodti3 for the compiler_rt library.
//
//===----------------------------------------------------------------------===//

#if __x86_64

#include "int_lib.h"
#include <stdio.h>

// Returns: a % b

tu_int __umodti3(tu_int a, tu_int b);

int test__umodti3(tu_int a, tu_int b, tu_int expected_r)
{
    tu_int r = __umodti3(a, b);
    if (r != expected_r)
    {
        utwords at;
        at.all = a;
        utwords bt;
        bt.all = b;
        utwords rt;
        rt.all = r;
        utwords expected_rt;
        expected_rt.all = expected_r;
        printf("error in __umodti3: 0x%llX%.16llX %% 0x%llX%.16llX = "
               "0x%llX%.16llX, expected 0x%llX%.16llX\n",
               at.s.high, at.s.low, bt.s.high, bt.s.low, rt.s.high, rt.s.low,
               expected_rt.s.high, expected_rt.s.low);
    }
    return r != expected_r;
}

#endif

int main()
{
#if __x86_64
    if (test__umodti3(0, 1, 0))
        return 1;
    if (test__umodti3(2, 1, 0))
        return 1;
    if (test__umodti3(make_tu(0x8000000000000000uLL, 0), 1, 0x0uLL))
        return 1;
    if (test__umodti3(make_tu(0x8000000000000000uLL, 0), 2, 0x0uLL))
        return 1;
    if (test__umodti3(make_tu(0xFFFFFFFFFFFFFFFFuLL, 0xFFFFFFFFFFFFFFFFuLL),
                      2, 0x1uLL))
        return 1;

#else
    printf("skipped\n");
#endif
    return 0;
}

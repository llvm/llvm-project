//===-------------- trunctfdf2_test.c - Test __trunctfdf2 -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file tests __trunctfdf2 for the compiler_rt library.
//
//===----------------------------------------------------------------------===//

#include "int_lib.h"
#include <stdio.h>

#if __LDBL_MANT_DIG__ == 113

#include "fp_test.h"

COMPILER_RT_ABI double __trunctfdf2(long double a);

int test__trunctfdf2(long double a, uint64_t expected)
{
    double x = __trunctfdf2(a);
    int ret = compareResultD(x, expected);

    if (ret)
    {
        printf("error in test__trunctfdf2(%.20Lf) = %lf, "
               "expected %lf\n", a, x, fromRep64(expected));
    }
    return ret;
}

char assumption_1[sizeof(long double) * CHAR_BIT == 128] = {0};

#endif

int main()
{
#if __LDBL_MANT_DIG__ == 113
    // qNaN
    if (test__trunctfdf2(makeQNaN128(),
                         UINT64_C(0x7ff8000000000000)))
        return 1;
    // NaN
    if (test__trunctfdf2(makeNaN128(UINT64_C(0x810000000000)),
                         UINT64_C(0x7ff8100000000000)))
        return 1;
    // inf
    if (test__trunctfdf2(makeInf128(),
                         UINT64_C(0x7ff0000000000000)))
        return 1;
    // zero
    if (test__trunctfdf2(0.0L, UINT64_C(0x0)))
        return 1;

    if (test__trunctfdf2(0x1.af23456789bbaaab347645365cdep+5L,
                         UINT64_C(0x404af23456789bbb)))
        return 1;
    if (test__trunctfdf2(0x1.dedafcff354b6ae9758763545432p-9L,
                         UINT64_C(0x3f6dedafcff354b7)))
        return 1;
    if (test__trunctfdf2(0x1.2f34dd5f437e849b4baab754cdefp+4534L,
                         UINT64_C(0x7ff0000000000000)))
        return 1;
    if (test__trunctfdf2(0x1.edcbff8ad76ab5bf46463233214fp-435L,
                         UINT64_C(0x24cedcbff8ad76ab)))
        return 1;

#else
    printf("skipped\n");

#endif
    return 0;
}

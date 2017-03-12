//===--------------- subtf3_test.c - Test __subtf3 ------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file tests __subtf3 for the compiler_rt library.
//
//===----------------------------------------------------------------------===//

#include <stdio.h>

#if __LDBL_MANT_DIG__ == 113

#include "fp_test.h"

// Returns: a - b
COMPILER_RT_ABI long double __subtf3(long double a, long double b);

int test__subtf3(long double a, long double b,
                 uint64_t expectedHi, uint64_t expectedLo)
{
    long double x = __subtf3(a, b);
    int ret = compareResultLD(x, expectedHi, expectedLo);

    if (ret){
        printf("error in test__subtf3(%.20Lf, %.20Lf) = %.20Lf, "
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
    // qNaN - any = qNaN
    if (test__subtf3(makeQNaN128(),
                     0x1.23456789abcdefp+5L,
                     UINT64_C(0x7fff800000000000),
                     UINT64_C(0x0)))
        return 1;
    // NaN - any = NaN
    if (test__subtf3(makeNaN128(UINT64_C(0x800030000000)),
                     0x1.23456789abcdefp+5L,
                     UINT64_C(0x7fff800000000000),
                     UINT64_C(0x0)))
        return 1;
    // inf - any = inf
    if (test__subtf3(makeInf128(),
                     0x1.23456789abcdefp+5L,
                     UINT64_C(0x7fff000000000000),
                     UINT64_C(0x0)))
        return 1;
    // any - any
    if (test__subtf3(0x1.234567829a3bcdef5678ade36734p+5L,
                     0x1.ee9d7c52354a6936ab8d7654321fp-1L,
                     UINT64_C(0x40041b8af1915166),
                     UINT64_C(0xa44a7bca780a166c)))
        return 1;

#else
    printf("skipped\n");

#endif
    return 0;
}

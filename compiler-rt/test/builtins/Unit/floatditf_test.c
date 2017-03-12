//===-- floatditf_test.c - Test __floatditf -------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file tests __floatditf for the compiler_rt library.
//
//===----------------------------------------------------------------------===//

#include "int_lib.h"
#include <math.h>
#include <complex.h>
#include <stdio.h>

#if __LDBL_MANT_DIG__ == 113

#include "fp_test.h"

// Returns: long integer converted to long double

COMPILER_RT_ABI long double __floatditf(long long a);

int test__floatditf(long long a, uint64_t expectedHi, uint64_t expectedLo)
{
    long double x = __floatditf(a);
    int ret = compareResultLD(x, expectedHi, expectedLo);

    if (ret)
        printf("error in __floatditf(%Ld) = %.20Lf, "
               "expected %.20Lf\n", a, x, fromRep128(expectedHi, expectedLo));
    return ret;
}

char assumption_1[sizeof(long double) * CHAR_BIT == 128] = {0};

#endif

int main()
{
#if __LDBL_MANT_DIG__ == 113
    if (test__floatditf(0x7fffffffffffffff, UINT64_C(0x403dffffffffffff), UINT64_C(0xfffc000000000000)))
        return 1;
    if (test__floatditf(0x123456789abcdef1, UINT64_C(0x403b23456789abcd), UINT64_C(0xef10000000000000)))
        return 1;
    if (test__floatditf(0x2, UINT64_C(0x4000000000000000), UINT64_C(0x0)))
        return 1;
    if (test__floatditf(0x1, UINT64_C(0x3fff000000000000), UINT64_C(0x0)))
        return 1;
    if (test__floatditf(0x0, UINT64_C(0x0), UINT64_C(0x0)))
        return 1;
    if (test__floatditf(0xffffffffffffffff, UINT64_C(0xbfff000000000000), UINT64_C(0x0)))
        return 1;
    if (test__floatditf(0xfffffffffffffffe, UINT64_C(0xc000000000000000), UINT64_C(0x0)))
        return 1;
    if (test__floatditf(-0x123456789abcdef1, UINT64_C(0xc03b23456789abcd), UINT64_C(0xef10000000000000)))
        return 1;
    if (test__floatditf(0x8000000000000000, UINT64_C(0xc03e000000000000), UINT64_C(0x0)))
        return 1;

#else
    printf("skipped\n");

#endif
    return 0;
}

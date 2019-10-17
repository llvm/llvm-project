// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_eqdf2vfp

//===-- eqdf2vfp_test.c - Test __eqdf2vfp ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file tests __eqdf2vfp for the compiler_rt library.
//
//===----------------------------------------------------------------------===//

#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <math.h>


extern int __eqdf2vfp(double a, double b);

#if __arm__ && __VFP_FP__
int test__eqdf2vfp(double a, double b)
{
    int actual = __eqdf2vfp(a, b);
	int expected = (a == b) ? 1 : 0;
    if (actual != expected)
        printf("error in __eqdf2vfp(%f, %f) = %d, expected %d\n",
               a, b, actual, expected);
    return actual != expected;
}
#endif

int main()
{
#if __arm__ && __VFP_FP__
    if (test__eqdf2vfp(0.0, 0.0))
        return 1;
    if (test__eqdf2vfp(1.0, 1.0))
        return 1;
    if (test__eqdf2vfp(0.0, 1.0))
        return 1;
    if (test__eqdf2vfp(-1.0, -1.0))
        return 1;
    if (test__eqdf2vfp(-1.0, 0.0))
        return 1;
    if (test__eqdf2vfp(HUGE_VAL, 1.0))
        return 1;
    if (test__eqdf2vfp(1.0, HUGE_VAL))
        return 1;
#else
    printf("skipped\n");
#endif
    return 0;
}

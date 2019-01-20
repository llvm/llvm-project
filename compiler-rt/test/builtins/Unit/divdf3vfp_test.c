// RUN: %clang_builtins %s %librt -o %t && %run %t
//===-- divdf3vfp_test.c - Test __divdf3vfp -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file tests __divdf3vfp for the compiler_rt library.
//
//===----------------------------------------------------------------------===//

#include "int_lib.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>


#if __arm__ && __VFP_FP__
extern COMPILER_RT_ABI double __divdf3vfp(double a, double b);

int test__divdf3vfp(double a, double b)
{
    double actual = __divdf3vfp(a, b);
    double expected = a / b;
    if (actual != expected)
        printf("error in test__divdf3vfp(%f, %f) = %f, expected %f\n",
               a, b, actual, expected);
    return actual != expected;
}
#endif

int main()
{
#if __arm__ && __VFP_FP__
    if (test__divdf3vfp(1.0, 1.0))
        return 1;
    if (test__divdf3vfp(12345.678, 1.23))
        return 1;
    if (test__divdf3vfp(-10.0, 0.25))
        return 1;
    if (test__divdf3vfp(10.0, -2.0))
        return 1;
#else
    printf("skipped\n");
#endif
    return 0;
}

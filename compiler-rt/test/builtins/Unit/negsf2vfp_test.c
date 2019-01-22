// RUN: %clang_builtins %s %librt -o %t && %run %t
//===-- negsf2vfp_test.c - Test __negsf2vfp -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file tests __negsf2vfp for the compiler_rt library.
//
//===----------------------------------------------------------------------===//

#include "int_lib.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>


extern COMPILER_RT_ABI float __negsf2vfp(float a);

#if __arm__ && __VFP_FP__
int test__negsf2vfp(float a)
{
    float actual = __negsf2vfp(a);
    float expected = -a;
    if (actual != expected)
        printf("error in test__negsf2vfp(%f) = %f, expected %f\n",
               a, actual, expected);
    return actual != expected;
}
#endif

int main()
{
#if __arm__ && __VFP_FP__
    if (test__negsf2vfp(1.0))
        return 1;
    if (test__negsf2vfp(HUGE_VALF))
        return 1;
    if (test__negsf2vfp(0.0))
        return 1;
    if (test__negsf2vfp(-1.0))
        return 1;
#else
    printf("skipped\n");
#endif
    return 0;
}

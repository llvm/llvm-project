// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_unordsf2vfp

//===-- unordsf2vfp_test.c - Test __unordsf2vfp ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file tests __unordsf2vfp for the compiler_rt library.
//
//===----------------------------------------------------------------------===//

#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <math.h>


extern int __unordsf2vfp(float a, float b);

#if __arm__ && __VFP_FP__
int test__unordsf2vfp(float a, float b)
{
    int actual = __unordsf2vfp(a, b);
	int expected = (isnan(a) || isnan(b)) ? 1 : 0;
    if (actual != expected)
        printf("error in __unordsf2vfp(%f, %f) = %d, expected %d\n",
               a, b, actual, expected);
    return actual != expected;
}
#endif

int main()
{
#if __arm__ && __VFP_FP__
    if (test__unordsf2vfp(0.0, NAN))
        return 1;
    if (test__unordsf2vfp(NAN, 1.0))
        return 1;
    if (test__unordsf2vfp(NAN, NAN))
        return 1;
    if (test__unordsf2vfp(1.0, 1.0))
        return 1;
#else
    printf("skipped\n");
#endif
    return 0;
}

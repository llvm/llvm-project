// RUN: %clang_builtins %s %librt -o %t && %run %t
//===-- floatsisfvfp_test.c - Test __floatsisfvfp -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file tests __floatsisfvfp for the compiler_rt library.
//
//===----------------------------------------------------------------------===//

#include "int_lib.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>


extern COMPILER_RT_ABI float __floatsisfvfp(int a);

#if __arm__ && __VFP_FP__
int test__floatsisfvfp(int a)
{
    float actual = __floatsisfvfp(a);
    float expected = a;
    if (actual != expected)
        printf("error in test__floatsisfvfp(%d) = %f, expected %f\n",
               a, actual, expected);
    return actual != expected;
}
#endif

int main()
{
#if __arm__ && __VFP_FP__
    if (test__floatsisfvfp(0))
        return 1;
    if (test__floatsisfvfp(1))
        return 1;
    if (test__floatsisfvfp(-1))
        return 1;
    if (test__floatsisfvfp(0x7FFFFFFF))
        return 1;
    if (test__floatsisfvfp(0x80000000))
        return 1;
#else
    printf("skipped\n");
#endif
    return 0;
}

// RUN: %clang_builtins %s %librt -o %t && %run %t
//===-- extendsfdf2vfp_test.c - Test __extendsfdf2vfp ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file tests __extendsfdf2vfp for the compiler_rt library.
//
//===----------------------------------------------------------------------===//

#include "int_lib.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>


extern COMPILER_RT_ABI double __extendsfdf2vfp(float a);

#if __arm__ && __VFP_FP__
int test__extendsfdf2vfp(float a)
{
    double actual = __extendsfdf2vfp(a);
    double expected = a;
    if (actual != expected)
        printf("error in test__extendsfdf2vfp(%f) = %f, expected %f\n",
               a, actual, expected);
    return actual != expected;
}
#endif

int main()
{
#if __arm__ && __VFP_FP__
    if (test__extendsfdf2vfp(0.0))
        return 1;
    if (test__extendsfdf2vfp(1.0))
        return 1;
    if (test__extendsfdf2vfp(-1.0))
        return 1;
    if (test__extendsfdf2vfp(3.1415926535))
        return 1;
#else
    printf("skipped\n");
#endif
    return 0;
}

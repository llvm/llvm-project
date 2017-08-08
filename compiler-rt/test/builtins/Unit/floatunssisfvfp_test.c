// RUN: %clang_builtins %s %librt -o %t && %run %t
//===-- floatunssisfvfp_test.c - Test __floatunssisfvfp -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file tests __floatunssisfvfp for the compiler_rt library.
//
//===----------------------------------------------------------------------===//

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "int_lib.h"

extern COMPILER_RT_ABI float __floatunssisfvfp(unsigned int a);

#if __arm__ && __VFP_FP__
int test__floatunssisfvfp(unsigned int a)
{
    float actual = __floatunssisfvfp(a);
    float expected = a;
    if (actual != expected)
        printf("error in test__floatunssisfvfp(%u) = %f, expected %f\n",
               a, actual, expected);
    return actual != expected;
}
#endif

int main()
{
#if __arm__ && __VFP_FP__
    if (test__floatunssisfvfp(0))
        return 1;
    if (test__floatunssisfvfp(1))
        return 1;
    if (test__floatunssisfvfp(0x7FFFFFFF))
        return 1;
    if (test__floatunssisfvfp(0x80000000))
        return 1;
    if (test__floatunssisfvfp(0xFFFFFFFF))
        return 1;
#else
    printf("skipped\n");
#endif
    return 0;
}

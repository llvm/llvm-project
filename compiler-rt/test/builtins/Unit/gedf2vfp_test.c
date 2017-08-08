// RUN: %clang_builtins %s %librt -o %t && %run %t

//===-- gedf2vfp_test.c - Test __gedf2vfp ---------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file tests __gedf2vfp for the compiler_rt library.
//
//===----------------------------------------------------------------------===//

#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <math.h>


extern int __gedf2vfp(double a, double b);

#if __arm__ && __VFP_FP__
int test__gedf2vfp(double a, double b)
{
    int actual = __gedf2vfp(a, b);
	int expected = (a >= b) ? 1 : 0;
    if (actual != expected)
        printf("error in __gedf2vfp(%f, %f) = %d, expected %d\n",
               a, b, actual, expected);
    return actual != expected;
}
#endif

int main()
{
#if __arm__ && __VFP_FP__
    if (test__gedf2vfp(0.0, 0.0))
        return 1;
    if (test__gedf2vfp(1.0, 0.0))
        return 1;
    if (test__gedf2vfp(-1.0, -2.0))
        return 1;
    if (test__gedf2vfp(-2.0, -1.0))
        return 1;
    if (test__gedf2vfp(HUGE_VAL, 1.0))
        return 1;
    if (test__gedf2vfp(1.0, HUGE_VAL))
        return 1;
#else
    printf("skipped\n");
#endif
    return 0;
}

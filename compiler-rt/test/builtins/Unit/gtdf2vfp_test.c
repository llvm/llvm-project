// RUN: %clang_builtins %s %librt -o %t && %run %t
// XFAIL: armhf-target-arch
// This test fails for armhf (see pr32261)

//===-- gtdf2vfp_test.c - Test __gtdf2vfp ---------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file tests __gtdf2vfp for the compiler_rt library.
//
//===----------------------------------------------------------------------===//

#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <math.h>


extern int __gtdf2vfp(double a, double b);

#if __arm__ && __VFP_FP__
int test__gtdf2vfp(double a, double b)
{
    int actual = __gtdf2vfp(a, b);
	int expected = (a > b) ? 1 : 0;
    if (actual != expected)
        printf("error in __gtdf2vfp(%f, %f) = %d, expected %d\n",
               a, b, actual, expected);
    return actual != expected;
}
#endif

int main()
{
#if __arm__ && __VFP_FP__
    if (test__gtdf2vfp(0.0, 0.0))
        return 1;
    if (test__gtdf2vfp(1.0, 0.0))
        return 1;
    if (test__gtdf2vfp(-1.0, -2.0))
        return 1;
    if (test__gtdf2vfp(-2.0, -1.0))
        return 1;
    if (test__gtdf2vfp(HUGE_VALF, 1.0))
        return 1;
    if (test__gtdf2vfp(1.0, HUGE_VALF))
        return 1;
#else
    printf("skipped\n");
#endif
    return 0;
}

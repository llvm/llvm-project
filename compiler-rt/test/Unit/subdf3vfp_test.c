//===-- subdf3vfp_test.c - Test __subdf3vfp -------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file tests __subdf3vfp for the compiler_rt library.
//
//===----------------------------------------------------------------------===//

#include <stdio.h>
#include <stdlib.h>
#include <math.h>


#if __arm__
extern double __subdf3vfp(double a, double b);

int test__subdf3vfp(double a, double b)
{
    double actual = __subdf3vfp(a, b);
    double expected = a - b;
    if (actual != expected)
        printf("error in test__subdf3vfp(%f, %f) = %f, expected %f\n",
               a, b, actual, expected);
    return actual != expected;
}
#endif

int main()
{
#if __arm__
    if (test__subdf3vfp(1.0, 1.0))
        return 1;
    if (test__subdf3vfp(1234.567, 765.4321))
        return 1;
    if (test__subdf3vfp(-123.0, -678.0))
        return 1;
    if (test__subdf3vfp(0.0, -0.0))
        return 1;
#else
    printf("skipped\n");
#endif
    return 0;
}

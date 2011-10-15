//===-- divdf3vfp_test.c - Test __divdf3vfp -------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file tests __divdf3vfp for the compiler_rt library.
//
//===----------------------------------------------------------------------===//

#include <stdio.h>
#include <stdlib.h>
#include <math.h>


#if __arm__
extern double __divdf3vfp(double a, double b);

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
#if __arm__
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

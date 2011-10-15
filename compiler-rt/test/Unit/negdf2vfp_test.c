//===-- negdf2vfp_test.c - Test __negdf2vfp -------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file tests __negdf2vfp for the compiler_rt library.
//
//===----------------------------------------------------------------------===//

#include <stdio.h>
#include <stdlib.h>
#include <math.h>


extern double __negdf2vfp(double a);

#if __arm__
int test__negdf2vfp(double a)
{
    double actual = __negdf2vfp(a);
    double expected = -a;
    if (actual != expected)
        printf("error in test__negdf2vfp(%f) = %f, expected %f\n",
               a, actual, expected);
    return actual != expected;
}
#endif

int main()
{
#if __arm__
    if (test__negdf2vfp(1.0))
        return 1;
    if (test__negdf2vfp(HUGE_VALF))
        return 1;
    if (test__negdf2vfp(0.0))
        return 1;
    if (test__negdf2vfp(-1.0))
        return 1;
#else
    printf("skipped\n");
#endif
    return 0;
}

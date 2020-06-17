// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_muldf3vfp

#include "int_lib.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>


#if __arm__ && __VFP_FP__
extern COMPILER_RT_ABI double __muldf3vfp(double a, double b);

int test__muldf3vfp(double a, double b)
{
    double actual = __muldf3vfp(a, b);
    double expected = a * b;
    if (actual != expected)
        printf("error in test__muldf3vfp(%f, %f) = %f, expected %f\n",
               a, b, actual, expected);
    return actual != expected;
}
#endif

int main()
{
#if __arm__ && __VFP_FP__
    if (test__muldf3vfp(0.5, 10.0))
        return 1;
    if (test__muldf3vfp(-0.5, -2.0))
        return 1;
    if (test__muldf3vfp(HUGE_VALF, 0.25))
        return 1;
    if (test__muldf3vfp(-0.125, HUGE_VALF))
        return 1;
    if (test__muldf3vfp(0.0, -0.0))
		return 1;
#else
    printf("skipped\n");
#endif
    return 0;
}

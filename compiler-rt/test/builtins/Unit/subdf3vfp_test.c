// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_subdf3vfp

#include "int_lib.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>


#if __arm__ && __VFP_FP__
extern COMPILER_RT_ABI double __subdf3vfp(double a, double b);

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
#if __arm__ && __VFP_FP__
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

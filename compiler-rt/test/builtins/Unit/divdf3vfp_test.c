// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_divdf3vfp

#include "int_lib.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>


#if defined(__arm__) && defined(__ARM_FP) && (__ARM_FP & 0x8)
extern COMPILER_RT_ABI double __divdf3vfp(double a, double b);

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
#if defined(__arm__) && defined(__ARM_FP) && (__ARM_FP & 0x8)
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

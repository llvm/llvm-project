// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_mulsf3vfp

#include "int_lib.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>


extern COMPILER_RT_ABI float __mulsf3vfp(float a, float b);

#if defined(__arm__) && defined(__ARM_FP) && (__ARM_FP & 0x4)
int test__mulsf3vfp(float a, float b)
{
    float actual = __mulsf3vfp(a, b);
    float expected = a * b;
    if (actual != expected)
        printf("error in test__mulsf3vfp(%f, %f) = %f, expected %f\n",
               a, b, actual, expected);
    return actual != expected;
}
#endif

int main()
{
#if defined(__arm__) && defined(__ARM_FP) && (__ARM_FP & 0x4)
    if (test__mulsf3vfp(0.5, 10.0))
        return 1;
    if (test__mulsf3vfp(-0.5, -2.0))
        return 1;
    if (test__mulsf3vfp(HUGE_VALF, 0.25))
        return 1;
    if (test__mulsf3vfp(-0.125, HUGE_VALF))
        return 1;
    if (test__mulsf3vfp(0.0, -0.0))
        return 1;
#else
    printf("skipped\n");
#endif
    return 0;
}

// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_divsf3vfp

#include "int_lib.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>


extern COMPILER_RT_ABI float __divsf3vfp(float a, float b);

#if defined(__arm__) && defined(__ARM_FP) && (__ARM_FP & 0x4)
int test__divsf3vfp(float a, float b)
{
    float actual = __divsf3vfp(a, b);
    float expected = a / b;
    if (actual != expected)
        printf("error in test__divsf3vfp(%f, %f) = %f, expected %f\n",
               a, b, actual, expected);
    return actual != expected;
}
#endif

int main()
{
#if defined(__arm__) && defined(__ARM_FP) && (__ARM_FP & 0x4)
    if (test__divsf3vfp(1.0, 1.0))
        return 1;
    if (test__divsf3vfp(12345.678, 1.23))
        return 1;
    if (test__divsf3vfp(0.0, HUGE_VALF))
        return 1;
    if (test__divsf3vfp(10.0, -2.0))
        return 1;
#else
    printf("skipped\n");
#endif
    return 0;
}

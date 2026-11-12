// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_subsf3vfp

#include "int_lib.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>


extern COMPILER_RT_ABI float __subsf3vfp(float a, float b);

#if defined(__arm__) && defined(__ARM_FP) && (__ARM_FP & 0x4)
int test__subsf3vfp(float a, float b)
{
    float actual = __subsf3vfp(a, b);
    float expected = a - b;
    if (actual != expected)
        printf("error in test__subsf3vfp(%f, %f) = %f, expected %f\n",
               a, b, actual, expected);
    return actual != expected;
}
#endif

int main()
{
#if defined(__arm__) && defined(__ARM_FP) && (__ARM_FP & 0x4)
    if (test__subsf3vfp(1.0, 1.0))
        return 1;
    if (test__subsf3vfp(1234.567, 765.4321))
        return 1;
    if (test__subsf3vfp(-123.0, -678.0))
        return 1;
    if (test__subsf3vfp(0.0, -0.0))
        return 1;
#else
    printf("skipped\n");
#endif
    return 0;
}

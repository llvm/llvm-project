// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_negsf2vfp

#include "int_lib.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>


extern COMPILER_RT_ABI float __negsf2vfp(float a);

#if __arm__ && __VFP_FP__
int test__negsf2vfp(float a)
{
    float actual = __negsf2vfp(a);
    float expected = -a;
    if (actual != expected)
        printf("error in test__negsf2vfp(%f) = %f, expected %f\n",
               a, actual, expected);
    return actual != expected;
}
#endif

int main()
{
#if __arm__ && __VFP_FP__
    if (test__negsf2vfp(1.0))
        return 1;
    if (test__negsf2vfp(HUGE_VALF))
        return 1;
    if (test__negsf2vfp(0.0))
        return 1;
    if (test__negsf2vfp(-1.0))
        return 1;
#else
    printf("skipped\n");
#endif
    return 0;
}

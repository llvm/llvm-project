// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_negdf2vfp

#include "int_lib.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>


#if __arm__ && __VFP_FP__
extern COMPILER_RT_ABI double __negdf2vfp(double a);

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
#if __arm__ && __VFP_FP__
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

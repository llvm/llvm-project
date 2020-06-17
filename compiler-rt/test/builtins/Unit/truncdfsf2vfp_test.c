// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_truncdfsf2vfp

#include "int_lib.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>


extern COMPILER_RT_ABI float __truncdfsf2vfp(double a);

#if __arm__ && __VFP_FP__
int test__truncdfsf2vfp(double a)
{
    float actual = __truncdfsf2vfp(a);
    float expected = a;
    if (actual != expected)
        printf("error in test__truncdfsf2vfp(%f) = %f, expected %f\n",
               a, actual, expected);
    return actual != expected;
}
#endif

int main()
{
#if __arm__ && __VFP_FP__
    if (test__truncdfsf2vfp(0.0))
        return 1;
    if (test__truncdfsf2vfp(1.0))
        return 1;
    if (test__truncdfsf2vfp(-1.0))
        return 1;
    if (test__truncdfsf2vfp(3.1415926535))
        return 1;
    if (test__truncdfsf2vfp(123.456))
        return 1;
#else
    printf("skipped\n");
#endif
    return 0;
}

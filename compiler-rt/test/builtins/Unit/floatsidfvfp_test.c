// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_floatsidfvfp

#include "int_lib.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>


extern COMPILER_RT_ABI double __floatsidfvfp(int a);

#if __arm__ && __VFP_FP__
int test__floatsidfvfp(int a)
{
    double actual = __floatsidfvfp(a);
    double expected = a;
    if (actual != expected)
        printf("error in test__ floatsidfvfp(%d) = %f, expected %f\n",
               a, actual, expected);
    return actual != expected;
}
#endif

int main()
{
#if __arm__ && __VFP_FP__
    if (test__floatsidfvfp(0))
        return 1;
    if (test__floatsidfvfp(1))
        return 1;
    if (test__floatsidfvfp(-1))
        return 1;
    if (test__floatsidfvfp(0x7FFFFFFF))
        return 1;
    if (test__floatsidfvfp(0x80000000))
        return 1;
#else
    printf("skipped\n");
#endif
    return 0;
}

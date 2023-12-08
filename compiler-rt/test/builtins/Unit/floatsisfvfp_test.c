// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_floatsisfvfp

#include "int_lib.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>


extern COMPILER_RT_ABI float __floatsisfvfp(int a);

#if defined(__arm__) && defined(__ARM_FP) && (__ARM_FP & 0x4)
int test__floatsisfvfp(int a)
{
    float actual = __floatsisfvfp(a);
    float expected = a;
    if (actual != expected)
        printf("error in test__floatsisfvfp(%d) = %f, expected %f\n",
               a, actual, expected);
    return actual != expected;
}
#endif

int main()
{
#if defined(__arm__) && defined(__ARM_FP) && (__ARM_FP & 0x4)
    if (test__floatsisfvfp(0))
        return 1;
    if (test__floatsisfvfp(1))
        return 1;
    if (test__floatsisfvfp(-1))
        return 1;
    if (test__floatsisfvfp(0x7FFFFFFF))
        return 1;
    if (test__floatsisfvfp(0x80000000))
        return 1;
#else
    printf("skipped\n");
#endif
    return 0;
}

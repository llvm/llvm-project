// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_floatunssisfvfp

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "int_lib.h"

extern COMPILER_RT_ABI float __floatunssisfvfp(unsigned int a);

#if defined(__arm__) && defined(__ARM_FP) && (__ARM_FP & 0x4)
int test__floatunssisfvfp(unsigned int a)
{
    float actual = __floatunssisfvfp(a);
    float expected = a;
    if (actual != expected)
        printf("error in test__floatunssisfvfp(%u) = %f, expected %f\n",
               a, actual, expected);
    return actual != expected;
}
#endif

int main()
{
#if defined(__arm__) && defined(__ARM_FP) && (__ARM_FP & 0x4)
    if (test__floatunssisfvfp(0))
        return 1;
    if (test__floatunssisfvfp(1))
        return 1;
    if (test__floatunssisfvfp(0x7FFFFFFF))
        return 1;
    if (test__floatunssisfvfp(0x80000000))
        return 1;
    if (test__floatunssisfvfp(0xFFFFFFFF))
        return 1;
#else
    printf("skipped\n");
#endif
    return 0;
}

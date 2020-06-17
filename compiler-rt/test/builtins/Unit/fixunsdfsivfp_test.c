// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_fixunsdfsivfp

#include "int_lib.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>


extern COMPILER_RT_ABI unsigned int __fixunsdfsivfp(double a);

#if __arm__ && __VFP_FP__
int test__fixunsdfsivfp(double a)
{
    unsigned int actual = __fixunsdfsivfp(a);
    unsigned int expected = a;
    if (actual != expected)
        printf("error in test__fixunsdfsivfp(%f) = %u, expected %u\n",
               a, actual, expected);
    return actual != expected;
}
#endif

int main()
{
#if __arm__ && __VFP_FP__
    if (test__fixunsdfsivfp(0.0))
        return 1;
    if (test__fixunsdfsivfp(1.0))
        return 1;
    if (test__fixunsdfsivfp(-1.0))
        return 1;
    if (test__fixunsdfsivfp(4294967295.0))
        return 1;
    if (test__fixunsdfsivfp(65536.0))
        return 1;
#else
    printf("skipped\n");
#endif
    return 0;
}

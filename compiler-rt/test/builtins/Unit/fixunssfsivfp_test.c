// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_fixunssfsivfp

#include <stdio.h>
#include <stdlib.h>
#include <math.h>


extern unsigned int __fixunssfsivfp(float a);

#if __arm__ && __VFP_FP__
int test__fixunssfsivfp(float a)
{
    unsigned int actual = __fixunssfsivfp(a);
    unsigned int expected = a;
    if (actual != expected)
        printf("error in test__fixunssfsivfp(%f) = %u, expected %u\n",
               a, actual, expected);
    return actual != expected;
}
#endif

int main()
{
#if __arm__ && __VFP_FP__
    if (test__fixunssfsivfp(0.0))
        return 1;
    if (test__fixunssfsivfp(1.0))
        return 1;
    if (test__fixunssfsivfp(-1.0))
        return 1;
    if (test__fixunssfsivfp(4294967295.0))
        return 1;
    if (test__fixunssfsivfp(65536.0))
        return 1;
#else
    printf("skipped\n");
#endif
    return 0;
}

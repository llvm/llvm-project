// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_lesf2vfp

#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <math.h>


extern int __lesf2vfp(float a, float b);

#if __arm__ && __VFP_FP__
int test__lesf2vfp(float a, float b)
{
    int actual = __lesf2vfp(a, b);
	int expected = (a <= b) ? 1 : 0;
    if (actual != expected)
        printf("error in __lesf2vfp(%f, %f) = %d, expected %d\n",
               a, b, actual, expected);
    return actual != expected;
}
#endif

int main()
{
#if __arm__ && __VFP_FP__
    if (test__lesf2vfp(0.0, 0.0))
        return 1;
    if (test__lesf2vfp(1.0, 1.0))
        return 1;
    if (test__lesf2vfp(-1.0, -2.0))
        return 1;
    if (test__lesf2vfp(-2.0, -1.0))
        return 1;
    if (test__lesf2vfp(HUGE_VALF, 1.0))
        return 1;
    if (test__lesf2vfp(1.0, HUGE_VALF))
        return 1;
#else
    printf("skipped\n");
#endif
    return 0;
}

// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_gtsf2vfp

#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <math.h>


extern int __gtsf2vfp(float a, float b);

#if defined(__arm__) && defined(__ARM_FP) && (__ARM_FP & 0x4)
int test__gtsf2vfp(float a, float b)
{
    int actual = __gtsf2vfp(a, b);
	int expected = (a > b) ? 1 : 0;
    if (actual != expected)
        printf("error in __gtsf2vfp(%f, %f) = %d, expected %d\n",
               a, b, actual, expected);
    return actual != expected;
}
#endif

int main()
{
#if defined(__arm__) && defined(__ARM_FP) && (__ARM_FP & 0x4)
    if (test__gtsf2vfp(0.0, 0.0))
        return 1;
    if (test__gtsf2vfp(1.0, 0.0))
        return 1;
    if (test__gtsf2vfp(-1.0, -2.0))
        return 1;
    if (test__gtsf2vfp(-2.0, -1.0))
        return 1;
    if (test__gtsf2vfp(HUGE_VALF, 1.0))
        return 1;
    if (test__gtsf2vfp(1.0, HUGE_VALF))
        return 1;
#else
    printf("skipped\n");
#endif
    return 0;
}

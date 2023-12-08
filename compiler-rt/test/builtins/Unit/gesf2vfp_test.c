// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_gesf2vfp

#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <math.h>


extern int __gesf2vfp(float a, float b);

#if defined(__arm__) && defined(__ARM_FP) && (__ARM_FP & 0x4)
int test__gesf2vfp(float a, float b)
{
    int actual = __gesf2vfp(a, b);
	int expected = (a >= b) ? 1 : 0;
    if (actual != expected)
        printf("error in __gesf2vfp(%f, %f) = %d, expected %d\n",
               a, b, actual, expected);
    return actual != expected;
}
#endif

int main()
{
#if defined(__arm__) && defined(__ARM_FP) && (__ARM_FP & 0x4)
    if (test__gesf2vfp(0.0, 0.0))
        return 1;
    if (test__gesf2vfp(1.1, 1.0))
        return 1;
    if (test__gesf2vfp(-1.0, -2.0))
        return 1;
    if (test__gesf2vfp(-2.0, -1.0))
        return 1;
    if (test__gesf2vfp(HUGE_VALF, 1.0))
        return 1;
    if (test__gesf2vfp(1.0, HUGE_VALF))
        return 1;
#else
    printf("skipped\n");
#endif
    return 0;
}

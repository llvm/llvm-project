// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_ltdf2vfp

#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <math.h>


extern int __ltdf2vfp(double a, double b);

#if __arm__ && __VFP_FP__
int test__ltdf2vfp(double a, double b)
{
    int actual = __ltdf2vfp(a, b);
	int expected = (a < b) ? 1 : 0;
    if (actual != expected)
        printf("error in __ltdf2vfp(%f, %f) = %d, expected %d\n",
               a, b, actual, expected);
    return actual != expected;
}
#endif

int main()
{
#if __arm__ && __VFP_FP__
    if (test__ltdf2vfp(0.0, 0.0))
        return 1;
    if (test__ltdf2vfp(1.0, 1.0))
        return 1;
    if (test__ltdf2vfp(-1.0, -1.0))
        return 1;
    if (test__ltdf2vfp(HUGE_VAL, 1.0))
        return 1;
    if (test__ltdf2vfp(1.0, HUGE_VAL))
        return 1;
#else
    printf("skipped\n");
#endif
    return 0;
}

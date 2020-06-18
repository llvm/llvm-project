// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_unorddf2vfp

#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <math.h>


extern int __unorddf2vfp(double a, double b);

#if __arm__ && __VFP_FP__
int test__unorddf2vfp(double a, double b)
{
    int actual = __unorddf2vfp(a, b);
	int expected = (isnan(a) || isnan(b)) ? 1 : 0;
    if (actual != expected)
        printf("error in __unorddf2vfp(%f, %f) = %d, expected %d\n",
               a, b, actual, expected);
    return actual != expected;
}
#endif

int main()
{
#if __arm__ && __VFP_FP__
    if (test__unorddf2vfp(0.0, NAN))
        return 1;
    if (test__unorddf2vfp(NAN, 1.0))
        return 1;
    if (test__unorddf2vfp(NAN, NAN))
        return 1;
    if (test__unorddf2vfp(1.0, 1.0))
        return 1;
#else
    printf("skipped\n");
#endif
    return 0;
}

//===-- negsf2vfp_test.c - Test __negsf2vfp -------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file tests __negsf2vfp for the compiler_rt library.
//
//===----------------------------------------------------------------------===//

#include <stdio.h>
#include <stdlib.h>
#include <math.h>


extern float __negsf2vfp(float a);

#if __arm__
int test__negsf2vfp(float a)
{
    float actual = __negsf2vfp(a);
    float expected = -a;
    if (actual != expected)
        printf("error in test__negsf2vfp(%f) = %f, expected %f\n",
               a, actual, expected);
    return actual != expected;
}
#endif

int main()
{
#if __arm__
    if (test__negsf2vfp(1.0))
        return 1;
    if (test__negsf2vfp(HUGE_VALF))
        return 1;
    if (test__negsf2vfp(0.0))
        return 1;
    if (test__negsf2vfp(-1.0))
        return 1;
#else
    printf("skipped\n");
#endif
    return 0;
}

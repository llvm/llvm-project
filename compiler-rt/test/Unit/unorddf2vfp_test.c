//===-- unorddf2vfp_test.c - Test __unorddf2vfp ---------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file tests __unorddf2vfp for the compiler_rt library.
//
//===----------------------------------------------------------------------===//

#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <math.h>


extern int __unorddf2vfp(double a, double b);

#if __arm__
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
#if __arm__
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

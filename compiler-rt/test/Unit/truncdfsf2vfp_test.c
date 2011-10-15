//===-- truncdfsf2vfp_test.c - Test __truncdfsf2vfp -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file tests __truncdfsf2vfp for the compiler_rt library.
//
//===----------------------------------------------------------------------===//

#include <stdio.h>
#include <stdlib.h>
#include <math.h>


extern float __truncdfsf2vfp(double a);

#if __arm__
int test__truncdfsf2vfp(double a)
{
    float actual = __truncdfsf2vfp(a);
    float expected = a;
    if (actual != expected)
        printf("error in test__truncdfsf2vfp(%f) = %f, expected %f\n",
               a, actual, expected);
    return actual != expected;
}
#endif

int main()
{
#if __arm__
    if (test__truncdfsf2vfp(0.0))
        return 1;
    if (test__truncdfsf2vfp(1.0))
        return 1;
    if (test__truncdfsf2vfp(-1.0))
        return 1;
    if (test__truncdfsf2vfp(3.1415926535))
        return 1;
    if (test__truncdfsf2vfp(123.456))
        return 1;
#else
    printf("skipped\n");
#endif
    return 0;
}

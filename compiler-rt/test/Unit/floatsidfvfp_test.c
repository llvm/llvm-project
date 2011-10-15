//===-- floatsidfvfp_test.c - Test __floatsidfvfp -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file tests __floatsidfvfp for the compiler_rt library.
//
//===----------------------------------------------------------------------===//

#include <stdio.h>
#include <stdlib.h>
#include <math.h>


extern double __floatsidfvfp(int a);

#if __arm__
int test__floatsidfvfp(int a)
{
    double actual = __floatsidfvfp(a);
    double expected = a;
    if (actual != expected)
        printf("error in test__ floatsidfvfp(%d) = %f, expected %f\n",
               a, actual, expected);
    return actual != expected;
}
#endif

int main()
{
#if __arm__
    if (test__floatsidfvfp(0))
        return 1;
    if (test__floatsidfvfp(1))
        return 1;
    if (test__floatsidfvfp(-1))
        return 1;
    if (test__floatsidfvfp(0x7FFFFFFF))
        return 1;
    if (test__floatsidfvfp(0x80000000))
        return 1;
#else
    printf("skipped\n");
#endif
    return 0;
}

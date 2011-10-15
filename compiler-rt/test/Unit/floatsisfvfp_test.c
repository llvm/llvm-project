//===-- floatsisfvfp_test.c - Test __floatsisfvfp -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file tests __floatsisfvfp for the compiler_rt library.
//
//===----------------------------------------------------------------------===//

#include <stdio.h>
#include <stdlib.h>
#include <math.h>


extern float __floatsisfvfp(int a);

#if __arm__
int test__floatsisfvfp(int a)
{
    float actual = __floatsisfvfp(a);
    float expected = a;
    if (actual != expected)
        printf("error in test__floatsisfvfp(%d) = %f, expected %f\n",
               a, actual, expected);
    return actual != expected;
}
#endif

int main()
{
#if __arm__
    if (test__floatsisfvfp(0))
        return 1;
    if (test__floatsisfvfp(1))
        return 1;
    if (test__floatsisfvfp(-1))
        return 1;
    if (test__floatsisfvfp(0x7FFFFFFF))
        return 1;
    if (test__floatsisfvfp(0x80000000))
        return 1;
#else
    printf("skipped\n");
#endif
    return 0;
}

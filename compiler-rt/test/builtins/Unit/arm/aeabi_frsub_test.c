// REQUIRES-ANY: arm-target-arch,armv6m-target-arch
// RUN: %clang_builtins %s %librt -o %t && %run %t
//===-- aeabi_frsub.c - Test __aeabi_frsub --------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file tests __aeabi_frsub for the compiler_rt library.
//
//===----------------------------------------------------------------------===//

#include <stdio.h>
#include <stdlib.h>
#include <math.h>


#if __arm__
extern __attribute__((pcs("aapcs"))) float __aeabi_frsub(float a, float b);

int test__aeabi_frsub(float a, float b, float expected)
{
    float actual = __aeabi_frsub(a, b);
    if (actual != expected)
        printf("error in __aeabi_frsub(%f, %f) = %f, expected %f\n",
               a, b, actual, expected);
    return actual != expected;
}
#endif

int main()
{
#if __arm__
    if (test__aeabi_frsub(1.0, 1.0, 0.0))
        return 1;
    if (test__aeabi_frsub(1234.567, 765.4321, -469.134900))
        return 1;
    if (test__aeabi_frsub(-123.0, -678.0, -555.0))
        return 1;
    if (test__aeabi_frsub(0.0, -0.0, 0.0))
        return 1;
#else
    printf("skipped\n");
#endif
    return 0;
}

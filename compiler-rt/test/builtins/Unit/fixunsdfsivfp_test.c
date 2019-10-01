// RUN: %clang_builtins %s %librt -o %t && %run %t
//===-- fixunsdfsivfp_test.c - Test __fixunsdfsivfp -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file tests __fixunsdfsivfp for the compiler_rt library.
//
//===----------------------------------------------------------------------===//

#include "int_lib.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>


extern COMPILER_RT_ABI unsigned int __fixunsdfsivfp(double a);

#if __arm__ && __VFP_FP__
int test__fixunsdfsivfp(double a)
{
    unsigned int actual = __fixunsdfsivfp(a);
    unsigned int expected = a;
    if (actual != expected)
        printf("error in test__fixunsdfsivfp(%f) = %u, expected %u\n",
               a, actual, expected);
    return actual != expected;
}
#endif

int main()
{
#if __arm__ && __VFP_FP__
    if (test__fixunsdfsivfp(0.0))
        return 1;
    if (test__fixunsdfsivfp(1.0))
        return 1;
    if (test__fixunsdfsivfp(-1.0))
        return 1;
    if (test__fixunsdfsivfp(4294967295.0))
        return 1;
    if (test__fixunsdfsivfp(65536.0))
        return 1;
#else
    printf("skipped\n");
#endif
    return 0;
}

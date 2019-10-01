// RUN: %clang_builtins %s %librt -o %t && %run %t

//===-- fixsfsivfp_test.c - Test __fixsfsivfp -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file tests __fixsfsivfp for the compiler_rt library.
//
//===----------------------------------------------------------------------===//

#include <stdio.h>
#include <stdlib.h>
#include <math.h>


extern int __fixsfsivfp(float a);

#if __arm__ && __VFP_FP__
int test__fixsfsivfp(float a)
{
	int actual = __fixsfsivfp(a);
	int expected = a;
    if (actual != expected)
        printf("error in test__fixsfsivfp(%f) = %u, expected %u\n",
               a, actual, expected);
    return actual != expected;
}
#endif

int main()
{
#if __arm__ && __VFP_FP__
    if (test__fixsfsivfp(0.0))
        return 1;
    if (test__fixsfsivfp(1.0))
        return 1;
    if (test__fixsfsivfp(-1.0))
        return 1;
    if (test__fixsfsivfp(2147483647.0))
        return 1;
    if (test__fixsfsivfp(-2147483648.0))
        return 1;
    if (test__fixsfsivfp(65536.0))
        return 1;
#else
    printf("skipped\n");
#endif
    return 0;
}

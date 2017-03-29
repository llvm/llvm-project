// RUN: %clang_builtins %s %librt -o %t && %run %t

//===-- fixsfsivfp_test.c - Test __fixsfsivfp -----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
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

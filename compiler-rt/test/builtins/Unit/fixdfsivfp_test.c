// RUN: %clang_builtins %s %librt -o %t && %run %t
// XFAIL: armhf-target-arch
// This test fails for armhf (see pr32261)

//===-- fixdfsivfp_test.c - Test __fixdfsivfp -----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file tests __fixdfsivfp for the compiler_rt library.
//
//===----------------------------------------------------------------------===//

#include <stdio.h>
#include <stdlib.h>
#include <math.h>


extern int __fixdfsivfp(double a);

#if __arm__ && __VFP_FP__
int test__fixdfsivfp(double a)
{
	int actual = __fixdfsivfp(a);
	int expected = a;
    if (actual != expected)
        printf("error in test__fixdfsivfp(%f) = %d, expected %d\n",
               a, actual, expected);
    return actual != expected;
}
#endif

int main()
{
#if __arm__ && __VFP_FP__
    if (test__fixdfsivfp(0.0))
        return 1;
    if (test__fixdfsivfp(1.0))
        return 1;
    if (test__fixdfsivfp(-1.0))
        return 1;
    if (test__fixdfsivfp(2147483647))
        return 1;
    if (test__fixdfsivfp(-2147483648.0))
        return 1;
#else
    printf("skipped\n");
#endif
    return 0;
}

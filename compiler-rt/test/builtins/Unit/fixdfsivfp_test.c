// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_fixdfsivfp

#include <stdio.h>
#include <stdlib.h>
#include <math.h>


extern int __fixdfsivfp(double a);

#if defined(__arm__) && defined(__ARM_FP) && (__ARM_FP & 0x8)
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
#if defined(__arm__) && defined(__ARM_FP) && (__ARM_FP & 0x8)
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

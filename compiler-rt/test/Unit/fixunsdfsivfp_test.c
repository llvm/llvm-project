//===-- fixunsdfsivfp_test.c - Test __fixunsdfsivfp -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file tests __fixunsdfsivfp for the compiler_rt library.
//
//===----------------------------------------------------------------------===//

#include <stdio.h>
#include <stdlib.h>
#include <math.h>


extern unsigned int __fixunsdfsivfp(double a);

#if __arm__
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
#if __arm__
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

// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_powixf2
// UNSUPPORTED: target=powerpc64{{.*}}
// REQUIRES: x86-target-arch

#if !_ARCH_PPC

#include "int_lib.h"
#include <stdio.h>
#include <math.h>

// Returns: a ^ b

COMPILER_RT_ABI long double __powixf2(long double a, int b);

int test__powixf2(long double a, int b, long double expected)
{
    long double x = __powixf2(a, b);
    int correct = (x == expected) && (signbit(x) == signbit(expected));
    if (!correct)
        printf("error in __powixf2(%Lf, %d) = %Lf, expected %Lf\n",
               a, b, x, expected);
    return !correct;
}

#endif

int main()
{
#if !_ARCH_PPC
    if (test__powixf2(0, 0, 1))
        return 1;
    if (test__powixf2(1, 0, 1))
        return 1;
    if (test__powixf2(1.5, 0, 1))
        return 1;
    if (test__powixf2(2, 0, 1))
        return 1;
    if (test__powixf2(INFINITY, 0, 1))
        return 1;

    if (test__powixf2(-0., 0, 1))
        return 1;
    if (test__powixf2(-1, 0, 1))
        return 1;
    if (test__powixf2(-1.5, 0, 1))
        return 1;
    if (test__powixf2(-2, 0, 1))
        return 1;
    if (test__powixf2(-INFINITY, 0, 1))
        return 1;

    if (test__powixf2(0, 1, 0))
        return 1;
    if (test__powixf2(0, 2, 0))
        return 1;
    if (test__powixf2(0, 3, 0))
        return 1;
    if (test__powixf2(0, 4, 0))
        return 1;
    if (test__powixf2(0, INT_MAX - 1, 0))
        return 1;
    if (test__powixf2(0, INT_MAX, 0))
        return 1;

    if (test__powixf2(-0., 1, -0.))
        return 1;
    if (test__powixf2(-0., 2, 0))
        return 1;
    if (test__powixf2(-0., 3, -0.))
        return 1;
    if (test__powixf2(-0., 4, 0))
        return 1;
    if (test__powixf2(-0., INT_MAX - 1, 0))
        return 1;
    if (test__powixf2(-0., INT_MAX, -0.))
        return 1;

    if (test__powixf2(1, 1, 1))
        return 1;
    if (test__powixf2(1, 2, 1))
        return 1;
    if (test__powixf2(1, 3, 1))
        return 1;
    if (test__powixf2(1, 4, 1))
        return 1;
    if (test__powixf2(1, INT_MAX - 1, 1))
        return 1;
    if (test__powixf2(1, INT_MAX, 1))
        return 1;

    if (test__powixf2(INFINITY, 1, INFINITY))
        return 1;
    if (test__powixf2(INFINITY, 2, INFINITY))
        return 1;
    if (test__powixf2(INFINITY, 3, INFINITY))
        return 1;
    if (test__powixf2(INFINITY, 4, INFINITY))
        return 1;
    if (test__powixf2(INFINITY, INT_MAX - 1, INFINITY))
        return 1;
    if (test__powixf2(INFINITY, INT_MAX, INFINITY))
        return 1;

    if (test__powixf2(-INFINITY, 1, -INFINITY))
        return 1;
    if (test__powixf2(-INFINITY, 2, INFINITY))
        return 1;
    if (test__powixf2(-INFINITY, 3, -INFINITY))
        return 1;
    if (test__powixf2(-INFINITY, 4, INFINITY))
        return 1;
    if (test__powixf2(-INFINITY, INT_MAX - 1, INFINITY))
        return 1;
    if (test__powixf2(-INFINITY, INT_MAX, -INFINITY))
        return 1;

    if (test__powixf2(0, -1, INFINITY))
        return 1;
    if (test__powixf2(0, -2, INFINITY))
        return 1;
    if (test__powixf2(0, -3, INFINITY))
        return 1;
    if (test__powixf2(0, -4, INFINITY))
        return 1;
    if (test__powixf2(0, INT_MIN + 2, INFINITY))
        return 1;
    if (test__powixf2(0, INT_MIN + 1, INFINITY))
        return 1;
    if (test__powixf2(0, INT_MIN, INFINITY))
        return 1;

    if (test__powixf2(-0., -1, -INFINITY))
        return 1;
    if (test__powixf2(-0., -2, INFINITY))
        return 1;
    if (test__powixf2(-0., -3, -INFINITY))
        return 1;
    if (test__powixf2(-0., -4, INFINITY))
        return 1;
    if (test__powixf2(-0., INT_MIN + 2, INFINITY))
        return 1;
    if (test__powixf2(-0., INT_MIN + 1, -INFINITY))
        return 1;
    if (test__powixf2(-0., INT_MIN, INFINITY))
        return 1;

    if (test__powixf2(1, -1, 1))
        return 1;
    if (test__powixf2(1, -2, 1))
        return 1;
    if (test__powixf2(1, -3, 1))
        return 1;
    if (test__powixf2(1, -4, 1))
        return 1;
    if (test__powixf2(1, INT_MIN + 2, 1))
        return 1;
    if (test__powixf2(1, INT_MIN + 1, 1))
        return 1;
    if (test__powixf2(1, INT_MIN, 1))
        return 1;

    if (test__powixf2(INFINITY, -1, 0))
        return 1;
    if (test__powixf2(INFINITY, -2, 0))
        return 1;
    if (test__powixf2(INFINITY, -3, 0))
        return 1;
    if (test__powixf2(INFINITY, -4, 0))
        return 1;
    if (test__powixf2(INFINITY, INT_MIN + 2, 0))
        return 1;
    if (test__powixf2(INFINITY, INT_MIN + 1, 0))
        return 1;
    if (test__powixf2(INFINITY, INT_MIN, 0))
        return 1;

    if (test__powixf2(-INFINITY, -1, -0.))
        return 1;
    if (test__powixf2(-INFINITY, -2, 0))
        return 1;
    if (test__powixf2(-INFINITY, -3, -0.))
        return 1;
    if (test__powixf2(-INFINITY, -4, 0))
        return 1;
    if (test__powixf2(-INFINITY, INT_MIN + 2, 0))
        return 1;
    if (test__powixf2(-INFINITY, INT_MIN + 1, -0.))
        return 1;
    if (test__powixf2(-INFINITY, INT_MIN, 0))
        return 1;

    if (test__powixf2(2, 10, 1024.))
        return 1;
    if (test__powixf2(-2, 10, 1024.))
        return 1;
    if (test__powixf2(2, -10, 1/1024.))
        return 1;
    if (test__powixf2(-2, -10, 1/1024.))
        return 1;

    if (test__powixf2(2, 19, 524288.))
        return 1;
    if (test__powixf2(-2, 19, -524288.))
        return 1;
    if (test__powixf2(2, -19, 1/524288.))
        return 1;
    if (test__powixf2(-2, -19, -1/524288.))
        return 1;

    if (test__powixf2(2, 31, 2147483648.))
        return 1;
    if (test__powixf2(-2, 31, -2147483648.))
        return 1;
    if (test__powixf2(2, -31, 1/2147483648.))
        return 1;
    if (test__powixf2(-2, -31, -1/2147483648.))
        return 1;

#else
    printf("skipped\n");
#endif
    return 0;
}

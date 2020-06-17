// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_powitf2

#include <stdio.h>

#if __LDBL_MANT_DIG__ == 113

#include "int_lib.h"
#include <math.h>

// Returns: a ^ b

COMPILER_RT_ABI long double __powitf2(long double a, si_int b);

int test__powitf2(long double a, si_int b, long double expected)
{
    long double x = __powitf2(a, b);
    int correct = (x == expected) && (signbit(x) == signbit(expected));
    if (!correct)
        printf("error in __powitf2(%Lf, %d) = %Lf, expected %Lf\n",
               a, b, x, expected);
    return !correct;
}

#endif

int main()
{
#if __LDBL_MANT_DIG__ == 113
    if (test__powitf2(0, 0, 1))
        return 1;
    if (test__powitf2(1, 0, 1))
        return 1;
    if (test__powitf2(1.5, 0, 1))
        return 1;
    if (test__powitf2(2, 0, 1))
        return 1;
    if (test__powitf2(INFINITY, 0, 1))
        return 1;

    if (test__powitf2(-0., 0, 1))
        return 1;
    if (test__powitf2(-1, 0, 1))
        return 1;
    if (test__powitf2(-1.5, 0, 1))
        return 1;
    if (test__powitf2(-2, 0, 1))
        return 1;
    if (test__powitf2(-INFINITY, 0, 1))
        return 1;

    if (test__powitf2(0, 1, 0))
        return 1;
    if (test__powitf2(0, 2, 0))
        return 1;
    if (test__powitf2(0, 3, 0))
        return 1;
    if (test__powitf2(0, 4, 0))
        return 1;
    if (test__powitf2(0, 0x7FFFFFFE, 0))
        return 1;
    if (test__powitf2(0, 0x7FFFFFFF, 0))
        return 1;

    if (test__powitf2(-0., 1, -0.))
        return 1;
    if (test__powitf2(-0., 2, 0))
        return 1;
    if (test__powitf2(-0., 3, -0.))
        return 1;
    if (test__powitf2(-0., 4, 0))
        return 1;
    if (test__powitf2(-0., 0x7FFFFFFE, 0))
        return 1;
    if (test__powitf2(-0., 0x7FFFFFFF, -0.))
        return 1;

    if (test__powitf2(1, 1, 1))
        return 1;
    if (test__powitf2(1, 2, 1))
        return 1;
    if (test__powitf2(1, 3, 1))
        return 1;
    if (test__powitf2(1, 4, 1))
        return 1;
    if (test__powitf2(1, 0x7FFFFFFE, 1))
        return 1;
    if (test__powitf2(1, 0x7FFFFFFF, 1))
        return 1;

    if (test__powitf2(INFINITY, 1, INFINITY))
        return 1;
    if (test__powitf2(INFINITY, 2, INFINITY))
        return 1;
    if (test__powitf2(INFINITY, 3, INFINITY))
        return 1;
    if (test__powitf2(INFINITY, 4, INFINITY))
        return 1;
    if (test__powitf2(INFINITY, 0x7FFFFFFE, INFINITY))
        return 1;
    if (test__powitf2(INFINITY, 0x7FFFFFFF, INFINITY))
        return 1;

    if (test__powitf2(-INFINITY, 1, -INFINITY))
        return 1;
    if (test__powitf2(-INFINITY, 2, INFINITY))
        return 1;
    if (test__powitf2(-INFINITY, 3, -INFINITY))
        return 1;
    if (test__powitf2(-INFINITY, 4, INFINITY))
        return 1;
    if (test__powitf2(-INFINITY, 0x7FFFFFFE, INFINITY))
        return 1;
    if (test__powitf2(-INFINITY, 0x7FFFFFFF, -INFINITY))
        return 1;

    if (test__powitf2(0, -1, INFINITY))
        return 1;
    if (test__powitf2(0, -2, INFINITY))
        return 1;
    if (test__powitf2(0, -3, INFINITY))
        return 1;
    if (test__powitf2(0, -4, INFINITY))
        return 1;
    if (test__powitf2(0, 0x80000002, INFINITY))
        return 1;
    if (test__powitf2(0, 0x80000001, INFINITY))
        return 1;
    if (test__powitf2(0, 0x80000000, INFINITY))
        return 1;

    if (test__powitf2(-0., -1, -INFINITY))
        return 1;
    if (test__powitf2(-0., -2, INFINITY))
        return 1;
    if (test__powitf2(-0., -3, -INFINITY))
        return 1;
    if (test__powitf2(-0., -4, INFINITY))
        return 1;
    if (test__powitf2(-0., 0x80000002, INFINITY))
        return 1;
    if (test__powitf2(-0., 0x80000001, -INFINITY))
        return 1;
    if (test__powitf2(-0., 0x80000000, INFINITY))
        return 1;

    if (test__powitf2(1, -1, 1))
        return 1;
    if (test__powitf2(1, -2, 1))
        return 1;
    if (test__powitf2(1, -3, 1))
        return 1;
    if (test__powitf2(1, -4, 1))
        return 1;
    if (test__powitf2(1, 0x80000002, 1))
        return 1;
    if (test__powitf2(1, 0x80000001, 1))
        return 1;
    if (test__powitf2(1, 0x80000000, 1))
        return 1;

    if (test__powitf2(INFINITY, -1, 0))
        return 1;
    if (test__powitf2(INFINITY, -2, 0))
        return 1;
    if (test__powitf2(INFINITY, -3, 0))
        return 1;
    if (test__powitf2(INFINITY, -4, 0))
        return 1;
    if (test__powitf2(INFINITY, 0x80000002, 0))
        return 1;
    if (test__powitf2(INFINITY, 0x80000001, 0))
        return 1;
    if (test__powitf2(INFINITY, 0x80000000, 0))
        return 1;

    if (test__powitf2(-INFINITY, -1, -0.))
        return 1;
    if (test__powitf2(-INFINITY, -2, 0))
        return 1;
    if (test__powitf2(-INFINITY, -3, -0.))
        return 1;
    if (test__powitf2(-INFINITY, -4, 0))
        return 1;
    if (test__powitf2(-INFINITY, 0x80000002, 0))
        return 1;
    if (test__powitf2(-INFINITY, 0x80000001, -0.))
        return 1;
    if (test__powitf2(-INFINITY, 0x80000000, 0))
        return 1;

    if (test__powitf2(2, 10, 1024.))
        return 1;
    if (test__powitf2(-2, 10, 1024.))
        return 1;
    if (test__powitf2(2, -10, 1/1024.))
        return 1;
    if (test__powitf2(-2, -10, 1/1024.))
        return 1;

    if (test__powitf2(2, 19, 524288.))
        return 1;
    if (test__powitf2(-2, 19, -524288.))
        return 1;
    if (test__powitf2(2, -19, 1/524288.))
        return 1;
    if (test__powitf2(-2, -19, -1/524288.))
        return 1;

    if (test__powitf2(2, 31, 2147483648.))
        return 1;
    if (test__powitf2(-2, 31, -2147483648.))
        return 1;
    if (test__powitf2(2, -31, 1/2147483648.))
        return 1;
    if (test__powitf2(-2, -31, -1/2147483648.))
        return 1;

#else
    printf("skipped\n");
#endif
    return 0;
}

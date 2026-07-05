// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_fixunsxfsi

#include "int_lib.h"
#include <stdio.h>

#if HAS_80_BIT_LONG_DOUBLE
// Returns: convert a to a unsigned int, rounding toward zero.
//          Negative values all become zero.

// Assumption: long double is an intel 80 bit floating point type padded with 6 bytes
//             su_int is a 32 bit integral type
//             value in long double is representable in su_int or is negative 
//                 (no range checking performed)

// gggg gggg gggg gggg gggg gggg gggg gggg | gggg gggg gggg gggg seee eeee eeee eeee |
// 1mmm mmmm mmmm mmmm mmmm mmmm mmmm mmmm | mmmm mmmm mmmm mmmm mmmm mmmm mmmm mmmm

COMPILER_RT_ABI su_int __fixunsxfsi(long double a);

int test__fixunsxfsi(long double a, su_int expected)
{
    su_int x = __fixunsxfsi(a);
    if (x != expected)
        printf("error in __fixunsxfsi(%LA) = %X, expected %X\n", a, x, expected);
    return x != expected;
}

char assumption_2[sizeof(su_int)*CHAR_BIT == 32] = {0};
char assumption_3[sizeof(long double)*CHAR_BIT == 128] = {0};
#endif

int main()
{
#if HAS_80_BIT_LONG_DOUBLE
    if (test__fixunsxfsi(0.0, 0))
        return 1;

    if (test__fixunsxfsi(0.5, 0))
        return 1;
    if (test__fixunsxfsi(0.99, 0))
        return 1;
    if (test__fixunsxfsi(1.0, 1))
        return 1;
    if (test__fixunsxfsi(1.5, 1))
        return 1;
    if (test__fixunsxfsi(1.99, 1))
        return 1;
    if (test__fixunsxfsi(2.0, 2))
        return 1;
    if (test__fixunsxfsi(2.01, 2))
        return 1;
    if (test__fixunsxfsi(-0.5, 0))
        return 1;
    if (test__fixunsxfsi(-0.99, 0))
        return 1;
#if !TARGET_LIBGCC
    if (test__fixunsxfsi(-1.0, 0))  // libgcc ignores "returns 0 for negative input" spec
        return 1;
    if (test__fixunsxfsi(-1.5, 0))
        return 1;
    if (test__fixunsxfsi(-1.99, 0))
        return 1;
    if (test__fixunsxfsi(-2.0, 0))
        return 1;
    if (test__fixunsxfsi(-2.01, 0))
        return 1;
#endif

    if (test__fixunsxfsi(0x1.000000p+31, 0x80000000))
        return 1;
    if (test__fixunsxfsi(0x1.FFFFFEp+31, 0xFFFFFF00))
        return 1;
    if (test__fixunsxfsi(0x1.FFFFFEp+30, 0x7FFFFF80))
        return 1;
    if (test__fixunsxfsi(0x1.FFFFFCp+30, 0x7FFFFF00))
        return 1;

#if !TARGET_LIBGCC
    if (test__fixunsxfsi(-0x1.FFFFFEp+30, 0))
        return 1;
    if (test__fixunsxfsi(-0x1.FFFFFCp+30, 0))
        return 1;
#endif

    if (test__fixunsxfsi(0x1.FFFFFFFEp+31, 0xFFFFFFFF))
        return 1;
    if (test__fixunsxfsi(0x1.FFFFFFFC00000p+30, 0x7FFFFFFF))
        return 1;
    if (test__fixunsxfsi(0x1.FFFFFFF800000p+30, 0x7FFFFFFE))
        return 1;

#endif
   return 0;
}

// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_subvdi3

#include "int_lib.h"
#include <stdio.h>
#include <stdlib.h>

// Returns: a - b

// Effects: aborts if a - b overflows

COMPILER_RT_ABI di_int __subvdi3(di_int a, di_int b);

int test__subvdi3(di_int a, di_int b)
{
    di_int x = __subvdi3(a, b);
    di_int expected = a - b;
    if (x != expected)
        printf("error in test__subvsi3(0x%llX, 0x%llX) = %lld, expected %lld\n",
               a, b, x, expected);
    return x != expected;
}

int main()
{
//     test__subvdi3(0x8000000000000000LL, 1);  // should abort
//     test__subvdi3(0, 0x8000000000000000LL);  // should abort
//     test__subvdi3(1, 0x8000000000000000LL);  // should abort
//     test__subvdi3(0x7FFFFFFFFFFFFFFFLL, -1);  // should abort
//     test__subvdi3(-2, 0x7FFFFFFFFFFFFFFFLL);  // should abort

    if (test__subvdi3(0x8000000000000000LL, -1))
        return 1;
    if (test__subvdi3(0x8000000000000000LL, 0))
        return 1;
    if (test__subvdi3(-1, 0x8000000000000000LL))
        return 1;
    if (test__subvdi3(0x7FFFFFFFFFFFFFFLL, 1))
        return 1;
    if (test__subvdi3(0x7FFFFFFFFFFFFFFFLL, 0))
        return 1;
    if (test__subvdi3(1, 0x7FFFFFFFFFFFFFFLL))
        return 1;
    if (test__subvdi3(0, 0x7FFFFFFFFFFFFFFFLL))
        return 1;
    if (test__subvdi3(-1, 0x7FFFFFFFFFFFFFFFLL))
        return 1;

    return 0;
}

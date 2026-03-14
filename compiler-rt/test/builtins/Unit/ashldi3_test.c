// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_ashldi3

#include "int_lib.h"
#include <stdio.h>

// Returns: a << b

// Precondition:  0 <= b < bits_in_dword

COMPILER_RT_ABI di_int __ashldi3(di_int a, int b);

int test__ashldi3(di_int a, int b, di_int expected)
{
    di_int x = __ashldi3(a, b);
    if (x != expected)
        printf("error in __ashldi3: %llX << %d = %llX, expected %llX\n",
               a, b, __ashldi3(a, b), expected);
    return x != expected;
}

char assumption_1[sizeof(di_int) == 2*sizeof(si_int)] = {0};

int main()
{
    if (test__ashldi3(0x0123456789ABCDEFLL, 0, 0x123456789ABCDEFLL))
        return 1;
    if (test__ashldi3(0x0123456789ABCDEFLL, 1, 0x2468ACF13579BDELL))
        return 1;
    if (test__ashldi3(0x0123456789ABCDEFLL, 2, 0x48D159E26AF37BCLL))
        return 1;
    if (test__ashldi3(0x0123456789ABCDEFLL, 3, 0x91A2B3C4D5E6F78LL))
        return 1;
    if (test__ashldi3(0x0123456789ABCDEFLL, 4, 0x123456789ABCDEF0LL))
        return 1;

    if (test__ashldi3(0x0123456789ABCDEFLL, 28, 0x789ABCDEF0000000LL))
        return 1;
    if (test__ashldi3(0x0123456789ABCDEFLL, 29, 0xF13579BDE0000000LL))
        return 1;
    if (test__ashldi3(0x0123456789ABCDEFLL, 30, 0xE26AF37BC0000000LL))
        return 1;
    if (test__ashldi3(0x0123456789ABCDEFLL, 31, 0xC4D5E6F780000000LL))
        return 1;

    if (test__ashldi3(0x0123456789ABCDEFLL, 32, 0x89ABCDEF00000000LL))
        return 1;

    if (test__ashldi3(0x0123456789ABCDEFLL, 33, 0x13579BDE00000000LL))
        return 1;
    if (test__ashldi3(0x0123456789ABCDEFLL, 34, 0x26AF37BC00000000LL))
        return 1;
    if (test__ashldi3(0x0123456789ABCDEFLL, 35, 0x4D5E6F7800000000LL))
        return 1;
    if (test__ashldi3(0x0123456789ABCDEFLL, 36, 0x9ABCDEF000000000LL))
        return 1;

    if (test__ashldi3(0x0123456789ABCDEFLL, 60, 0xF000000000000000LL))
        return 1;
    if (test__ashldi3(0x0123456789ABCDEFLL, 61, 0xE000000000000000LL))
        return 1;
    if (test__ashldi3(0x0123456789ABCDEFLL, 62, 0xC000000000000000LL))
        return 1;
    if (test__ashldi3(0x0123456789ABCDEFLL, 63, 0x8000000000000000LL))
        return 1;
    return 0;
}

// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_paritydi2

#include "int_lib.h"
#include <stdio.h>
#include <stdlib.h>

// Returns: 1 if number of bits is odd else returns 0

COMPILER_RT_ABI int __paritydi2(di_int a);

int naive_parity(di_int a)
{
    int r = 0;
    for (; a; a = a & (a - 1))
        r = ~r;
    return r & 1;
}

int test__paritydi2(di_int a)
{
    si_int x = __paritydi2(a);
    si_int expected = naive_parity(a);
    if (x != expected)
        printf("error in __paritydi2(0x%llX) = %d, expected %d\n",
               a, x, expected);
    return x != expected;
}

char assumption_1[sizeof(di_int) == 2*sizeof(si_int)] = {0};
char assumption_2[sizeof(si_int)*CHAR_BIT == 32] = {0};

int main()
{
    int i;
    for (i = 0; i < 10000; ++i)
        if (test__paritydi2(((di_int)rand() << 32) + rand()))
            return 1;

   return 0;
}

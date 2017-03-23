// UNSUPPORTED: armv6m-target-arch
// RUN: %clang_builtins %s %librt -o %t && %run %t
//===-- bswapsi2_test.c - Test __bswapsi2 ---------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file tests __bswapsi2 for the compiler_rt library.
//
//===----------------------------------------------------------------------===//

#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <math.h>


extern uint32_t __bswapsi2(uint32_t);

#if __arm__
int test__bswapsi2(uint32_t a, uint32_t expected)
{
    uint32_t actual = __bswapsi2(a);
    if (actual != expected)
        printf("error in test__bswapsi2(0x%0X) = 0x%0X, expected 0x%0X\n",
               a, actual, expected);
    return actual != expected;
}
#endif

int main()
{
#if __arm__
    if (test__bswapsi2(0x12345678, 0x78563412))
        return 1;
    if (test__bswapsi2(0x00000001, 0x01000000))
        return 1;
#else
    printf("skipped\n");
#endif
    return 0;
}

/* ===-- powitf2.cpp - Implement __powitf2 ---------------------------------===
 *
 *                     The LLVM Compiler Infrastructure
 *
 * This file is dual licensed under the MIT and the University of Illinois Open
 * Source Licenses. See LICENSE.TXT for details.
 *
 * ===----------------------------------------------------------------------===
 *
 * This file implements __powitf2 for the compiler_rt library.
 *
 * ===----------------------------------------------------------------------===
 */

#if _ARCH_PPC

#include "int_lib.h"

/* Returns: a ^ b */

long double
__powitf2(long double a, si_int b)
{
    const int recip = b < 0;
    long double r = 1;
    while (1)
    {
        if (b & 1)
            r *= a;
        b /= 2;
        if (b == 0)
            break;
        a *= a;
    }
    return recip ? 1/r : r;
}

#endif

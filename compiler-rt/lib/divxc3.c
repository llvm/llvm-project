/* ===-- divxc3.c - Implement __divxc3 -------------------------------------===
 *
 *                     The LLVM Compiler Infrastructure
 *
 * This file is dual licensed under the MIT and the University of Illinois Open
 * Source Licenses. See LICENSE.TXT for details.
 *
 * ===----------------------------------------------------------------------===
 *
 * This file implements __divxc3 for the compiler_rt library.
 *
 */

#if !_ARCH_PPC

#include "int_lib.h"
#include <math.h>
#include <complex.h>

/* Returns: the quotient of (a + ib) / (c + id) */

long double _Complex
__divxc3(long double __a, long double __b, long double __c, long double __d)
{
    int __ilogbw = 0;
    long double __logbw = logbl(fmaxl(fabsl(__c), fabsl(__d)));
    if (isfinite(__logbw))
    {
        __ilogbw = (int)__logbw;
        __c = scalbnl(__c, -__ilogbw);
        __d = scalbnl(__d, -__ilogbw);
    }
    long double __denom = __c * __c + __d * __d;
    long double _Complex z;
    __real__ z = scalbnl((__a * __c + __b * __d) / __denom, -__ilogbw);
    __imag__ z = scalbnl((__b * __c - __a * __d) / __denom, -__ilogbw);
    if (isnan(__real__ z) && isnan(__imag__ z))
    {
        if ((__denom == 0) && (!isnan(__a) || !isnan(__b)))
        {
            __real__ z = copysignl(INFINITY, __c) * __a;
            __imag__ z = copysignl(INFINITY, __c) * __b;
        }
        else if ((isinf(__a) || isinf(__b)) && isfinite(__c) && isfinite(__d))
        {
            __a = copysignl(isinf(__a) ? 1 : 0, __a);
            __b = copysignl(isinf(__b) ? 1 : 0, __b);
            __real__ z = INFINITY * (__a * __c + __b * __d);
            __imag__ z = INFINITY * (__b * __c - __a * __d);
        }
        else if (isinf(__logbw) && __logbw > 0 && isfinite(__a) && isfinite(__b))
        {
            __c = copysignl(isinf(__c) ? 1 : 0, __c);
            __d = copysignl(isinf(__d) ? 1 : 0, __d);
            __real__ z = 0 * (__a * __c + __b * __d);
            __imag__ z = 0 * (__b * __c - __a * __d);
        }
    }
    return z;
}

#endif

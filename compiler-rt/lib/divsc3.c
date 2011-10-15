/*===-- divsc3.c - Implement __divsc3 -------------------------------------===
 *
 *                     The LLVM Compiler Infrastructure
 *
 * This file is dual licensed under the MIT and the University of Illinois Open
 * Source Licenses. See LICENSE.TXT for details.
 *
 * ===----------------------------------------------------------------------===
 *
 * This file implements __divsc3 for the compiler_rt library.
 *
 *===----------------------------------------------------------------------===
 */

#include "int_lib.h"
#include <math.h>
#include <complex.h>

/* Returns: the quotient of (a + ib) / (c + id) */

float _Complex
__divsc3(float __a, float __b, float __c, float __d)
{
    int __ilogbw = 0;
    float __logbw = logbf(fmaxf(fabsf(__c), fabsf(__d)));
    if (isfinite(__logbw))
    {
        __ilogbw = (int)__logbw;
        __c = scalbnf(__c, -__ilogbw);
        __d = scalbnf(__d, -__ilogbw);
    }
    float __denom = __c * __c + __d * __d;
    float _Complex z;
    __real__ z = scalbnf((__a * __c + __b * __d) / __denom, -__ilogbw);
    __imag__ z = scalbnf((__b * __c - __a * __d) / __denom, -__ilogbw);
    if (isnan(__real__ z) && isnan(__imag__ z))
    {
        if ((__denom == 0) && (!isnan(__a) || !isnan(__b)))
        {
            __real__ z = copysignf(INFINITY, __c) * __a;
            __imag__ z = copysignf(INFINITY, __c) * __b;
        }
        else if ((isinf(__a) || isinf(__b)) && isfinite(__c) && isfinite(__d))
        {
            __a = copysignf(isinf(__a) ? 1 : 0, __a);
            __b = copysignf(isinf(__b) ? 1 : 0, __b);
            __real__ z = INFINITY * (__a * __c + __b * __d);
            __imag__ z = INFINITY * (__b * __c - __a * __d);
        }
        else if (isinf(__logbw) && __logbw > 0 && isfinite(__a) && isfinite(__b))
        {
            __c = copysignf(isinf(__c) ? 1 : 0, __c);
            __d = copysignf(isinf(__d) ? 1 : 0, __d);
            __real__ z = 0 * (__a * __c + __b * __d);
            __imag__ z = 0 * (__b * __c - __a * __d);
        }
    }
    return z;
}

/* ===-- divdc3.c - Implement __divdc3 -------------------------------------===
 *
 *                     The LLVM Compiler Infrastructure
 *
 * This file is dual licensed under the MIT and the University of Illinois Open
 * Source Licenses. See LICENSE.TXT for details.
 *
 * ===----------------------------------------------------------------------===
 *
 * This file implements __divdc3 for the compiler_rt library.
 *
 * ===----------------------------------------------------------------------===
 */

#include "int_lib.h"
#include <math.h>
#include <complex.h>

/* Returns: the quotient of (a + ib) / (c + id) */

double _Complex
__divdc3(double __a, double __b, double __c, double __d)
{
    int __ilogbw = 0;
    double __logbw = logb(fmax(fabs(__c), fabs(__d)));
    if (isfinite(__logbw))
    {
        __ilogbw = (int)__logbw;
        __c = scalbn(__c, -__ilogbw);
        __d = scalbn(__d, -__ilogbw);
    }
    double __denom = __c * __c + __d * __d;
    double _Complex z;
    __real__ z = scalbn((__a * __c + __b * __d) / __denom, -__ilogbw);
    __imag__ z = scalbn((__b * __c - __a * __d) / __denom, -__ilogbw);
    if (isnan(__real__ z) && isnan(__imag__ z))
    {
        if ((__denom == 0.0) && (!isnan(__a) || !isnan(__b)))
        {
            __real__ z = copysign(INFINITY, __c) * __a;
            __imag__ z = copysign(INFINITY, __c) * __b;
        }
        else if ((isinf(__a) || isinf(__b)) && isfinite(__c) && isfinite(__d))
        {
            __a = copysign(isinf(__a) ? 1.0 : 0.0, __a);
            __b = copysign(isinf(__b) ? 1.0 : 0.0, __b);
            __real__ z = INFINITY * (__a * __c + __b * __d);
            __imag__ z = INFINITY * (__b * __c - __a * __d);
        }
        else if (isinf(__logbw) && __logbw > 0.0 && isfinite(__a) && isfinite(__b))
        {
            __c = copysign(isinf(__c) ? 1.0 : 0.0, __c);
            __d = copysign(isinf(__d) ? 1.0 : 0.0, __d);
            __real__ z = 0.0 * (__a * __c + __b * __d);
            __imag__ z = 0.0 * (__b * __c - __a * __d);
        }
    }
    return z;
}

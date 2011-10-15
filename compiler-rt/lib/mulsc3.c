/* ===-- mulsc3.c - Implement __mulsc3 -------------------------------------===
 *
 *                     The LLVM Compiler Infrastructure
 *
 * This file is dual licensed under the MIT and the University of Illinois Open
 * Source Licenses. See LICENSE.TXT for details.
 *
 * ===----------------------------------------------------------------------===
 *
 * This file implements __mulsc3 for the compiler_rt library.
 *
 * ===----------------------------------------------------------------------===
 */

#include "int_lib.h"
#include <math.h>
#include <complex.h>

/* Returns: the product of a + ib and c + id */

float _Complex
__mulsc3(float __a, float __b, float __c, float __d)
{
    float __ac = __a * __c;
    float __bd = __b * __d;
    float __ad = __a * __d;
    float __bc = __b * __c;
    float _Complex z;
    __real__ z = __ac - __bd;
    __imag__ z = __ad + __bc;
    if (isnan(__real__ z) && isnan(__imag__ z))
    {
        int __recalc = 0;
        if (isinf(__a) || isinf(__b))
        {
            __a = copysignf(isinf(__a) ? 1 : 0, __a);
            __b = copysignf(isinf(__b) ? 1 : 0, __b);
            if (isnan(__c))
                __c = copysignf(0, __c);
            if (isnan(__d))
                __d = copysignf(0, __d);
            __recalc = 1;
        }
        if (isinf(__c) || isinf(__d))
        {
            __c = copysignf(isinf(__c) ? 1 : 0, __c);
            __d = copysignf(isinf(__d) ? 1 : 0, __d);
            if (isnan(__a))
                __a = copysignf(0, __a);
            if (isnan(__b))
                __b = copysignf(0, __b);
            __recalc = 1;
        }
        if (!__recalc && (isinf(__ac) || isinf(__bd) ||
                          isinf(__ad) || isinf(__bc)))
        {
            if (isnan(__a))
                __a = copysignf(0, __a);
            if (isnan(__b))
                __b = copysignf(0, __b);
            if (isnan(__c))
                __c = copysignf(0, __c);
            if (isnan(__d))
                __d = copysignf(0, __d);
            __recalc = 1;
        }
        if (__recalc)
        {
            __real__ z = INFINITY * (__a * __c - __b * __d);
            __imag__ z = INFINITY * (__a * __d + __b * __c);
        }
    }
    return z;
}

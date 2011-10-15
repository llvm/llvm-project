/* ===-- mulxc3.c - Implement __mulxc3 -------------------------------------===
 *
 *                     The LLVM Compiler Infrastructure
 *
 * This file is dual licensed under the MIT and the University of Illinois Open
 * Source Licenses. See LICENSE.TXT for details.
 *
 * ===----------------------------------------------------------------------===
 *
 * This file implements __mulxc3 for the compiler_rt library.
 *
 * ===----------------------------------------------------------------------===
 */

#if !_ARCH_PPC

#include "int_lib.h"
#include <math.h>
#include <complex.h>

/* Returns: the product of a + ib and c + id */

long double _Complex
__mulxc3(long double __a, long double __b, long double __c, long double __d)
{
    long double __ac = __a * __c;
    long double __bd = __b * __d;
    long double __ad = __a * __d;
    long double __bc = __b * __c;
    long double _Complex z;
    __real__ z = __ac - __bd;
    __imag__ z = __ad + __bc;
    if (isnan(__real__ z) && isnan(__imag__ z))
    {
        int __recalc = 0;
        if (isinf(__a) || isinf(__b))
        {
            __a = copysignl(isinf(__a) ? 1 : 0, __a);
            __b = copysignl(isinf(__b) ? 1 : 0, __b);
            if (isnan(__c))
                __c = copysignl(0, __c);
            if (isnan(__d))
                __d = copysignl(0, __d);
            __recalc = 1;
        }
        if (isinf(__c) || isinf(__d))
        {
            __c = copysignl(isinf(__c) ? 1 : 0, __c);
            __d = copysignl(isinf(__d) ? 1 : 0, __d);
            if (isnan(__a))
                __a = copysignl(0, __a);
            if (isnan(__b))
                __b = copysignl(0, __b);
            __recalc = 1;
        }
        if (!__recalc && (isinf(__ac) || isinf(__bd) ||
                          isinf(__ad) || isinf(__bc)))
        {
            if (isnan(__a))
                __a = copysignl(0, __a);
            if (isnan(__b))
                __b = copysignl(0, __b);
            if (isnan(__c))
                __c = copysignl(0, __c);
            if (isnan(__d))
                __d = copysignl(0, __d);
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

#endif

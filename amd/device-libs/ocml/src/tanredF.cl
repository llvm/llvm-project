/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"
#include "trigredF.h"

CONSTATTR INLINEATTR float
MATH_PRIVATE(tanred)(float x, int regn)
{
    // Core Remez [1,2] approximation to tan(x) on the interval [0,pi/4].
    float r = x * x;

    float a = MATH_MAD(r, -0.0172032480471481694693109f, 0.385296071263995406715129f);

    float b = MATH_MAD(r,
                       MATH_MAD(r, 0.01844239256901656082986661f, -0.51396505478854532132342f),
                       1.15588821434688393452299f);

    float t = MATH_MAD(x*r, MATH_FAST_DIV(a, b), x);
    float tr = -MATH_FAST_RCP(t);

    return regn & 1 ? tr : t;
}


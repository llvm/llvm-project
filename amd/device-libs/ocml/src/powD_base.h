/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"

extern CONSTATTR double2 MATH_PRIVATE(epln)(double);
extern CONSTATTR double MATH_PRIVATE(expep)(double2);

#define DOUBLE_SPECIALIZATION
#include "ep.h"

static bool
samesign(double x, double y)
{
    uint xh = AS_UINT2(x).hi;
    uint yh = AS_UINT2(y).hi;
    return ((xh ^ yh) & 0x80000000U) == 0;
}

// Check if a double is an integral value, and whether it's even or
// odd.
//
// status: 0=not integer, 1=odd, 2=even
static int classify_integer(double ay)
{
    double tay = BUILTIN_TRUNC_F64(ay);
    int inty = ay == tay;
    inty += inty & (BUILTIN_FRACTION_F64(tay*0.5) == 0.0);
    return inty;
}

#if defined(COMPILING_POW)

CONSTATTR double
MATH_MANGLE(pow)(double x, double y)
{
    double ax = BUILTIN_ABS_F64(x);
    double expylnx = MATH_PRIVATE(expep)(omul(y, MATH_PRIVATE(epln)(ax)));

    double ay = BUILTIN_ABS_F64(y);
    int inty = classify_integer(ay);
    double ret = BUILTIN_COPYSIGN_F64(expylnx, ((inty == 1) & (x < 0.0)) ? -0.0 : 0.0);

    // Now all the edge cases
    if (x < 0.0 && !inty)
        ret = QNAN_F64;

    if (BUILTIN_ISINF_F64(ay))
        ret = ax == 1.0 ? ax : (samesign(y, ax - 1.0) ? ay : 0.0);

    if (BUILTIN_ISINF_F64(ax) || x == 0.0)
        ret = BUILTIN_COPYSIGN_F64((x == 0.0) ^ (y < 0.0) ? 0.0 : PINF_F64,
                                   inty == 1 ? x : 0.0);

    if (BUILTIN_ISUNORDERED_F64(x, y))
        ret = QNAN_F64;

    if (x == 1.0 || y == 0.0)
        ret = 1.0;

    return ret;
}


#elif defined(COMPILING_POWR)

CONSTATTR double
MATH_MANGLE(powr)(double x, double y)
{
    double ax = BUILTIN_ABS_F64(x);
    double expylnx = MATH_PRIVATE(expep)(omul(y, MATH_PRIVATE(epln)(ax)));

    double ay = BUILTIN_ABS_F64(y);
    int inty = classify_integer(ay);

    double ret = BUILTIN_COPYSIGN_F64(expylnx, ((inty == 1) & (x < 0.0)) ? -0.0 : 0.0);

    // Now all the edge cases
    double iz = y < 0.0 ? PINF_F64 : 0.0;
    double zi = y < 0.0 ? 0.0 : PINF_F64;

    if (x == 0.0)
        ret = iz;

    if (BUILTIN_ISINF_F64(x))
        ret = zi;

    if (BUILTIN_ISINF_F64(y))
        ret = ax < 1.0 ? iz : zi;

    if (y == 0.0)
        ret = x == 0.0 || BUILTIN_ISINF_F64(x) ? QNAN_F64 : 1.0;

    if (x == 1.0)
        ret = BUILTIN_ISINF_F64(y) ? QNAN_F64 : 1.0;

    if (x < 0.0 || BUILTIN_ISUNORDERED_F64(x, y))
        ret = QNAN_F64;

    return ret;
}

#elif defined(COMPILING_POWN)

CONSTATTR double
MATH_MANGLE(pown)(double x, int ny)
{
    double y = (double) ny;

    double ax = BUILTIN_ABS_F64(x);
    double expylnx = MATH_PRIVATE(expep)(omul(y, MATH_PRIVATE(epln)(ax)));

    // y status: 0=not integer, 1=odd, 2=even
    int inty = 2 - (ny & 1);

    double ret = BUILTIN_COPYSIGN_F64(expylnx, ((inty == 1) & (x < 0.0)) ? -0.0 : 0.0);

    // Now all the edge cases
    if (BUILTIN_ISINF_F64(ax) || x == 0.0)
        ret = BUILTIN_COPYSIGN_F64((x == 0.0) ^ (ny < 0) ? 0.0 : PINF_F64,
                                   inty == 1 ? x : 0.0);

    if (BUILTIN_ISNAN_F64(x))
        ret = QNAN_F64;

    if (ny == 0)
        ret = 1.0;

    return ret;
}

#elif defined(COMPILING_ROOTN)

CONSTATTR double
MATH_MANGLE(rootn)(double x, int ny)
{
    double2 y = rcp((double)ny);

    double ax = BUILTIN_ABS_F64(x);
    double expylnx = MATH_PRIVATE(expep)(omul(y, MATH_PRIVATE(epln)(ax)));

    // y status: 0=not integer, 1=odd, 2=even
    int inty = 2 - (ny & 1);

    double ret = BUILTIN_COPYSIGN_F64(expylnx, ((inty == 1) & (x < 0.0)) ? -0.0 : 0.0);

    // Now all the edge cases
    if (BUILTIN_ISINF_F64(ax) || x == 0.0)
        ret = BUILTIN_COPYSIGN_F64((x == 0.0) ^ (ny < 0) ? 0.0 : PINF_F64,
                                   inty == 1 ? x : 0.0);

    if ((x < 0.0 && inty != 1) || ny == 0)
        ret = QNAN_F64;

    return ret;
}

#else
#error missing function macro
#endif


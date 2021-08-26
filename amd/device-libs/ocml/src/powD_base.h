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

CONSTATTR double
#if defined(COMPILING_POWR)
MATH_MANGLE(powr)(double x, double y)
#elif defined(COMPILING_POWN)
MATH_MANGLE(pown)(double x, int ny)
#elif defined(COMPILING_ROOTN)
MATH_MANGLE(rootn)(double x, int ny)
#else
MATH_MANGLE(pow)(double x, double y)
#endif
{
#if defined(COMPILING_POWN)
    double y = (double) ny;
#elif defined(COMPILING_ROOTN)
    double2 y = rcp((double)ny);
#endif

    double ax = BUILTIN_ABS_F64(x);
    double expylnx = MATH_PRIVATE(expep)(omul(y, MATH_PRIVATE(epln)(ax)));

    // y status: 0=not integer, 1=odd, 2=even
#if defined(COMPILING_POWN) | defined(COMPILING_ROOTN)
    int inty = 2 - (ny & 1);
#else
    double ay = BUILTIN_ABS_F64(y);
    int inty;
    {
        double tay = BUILTIN_TRUNC_F64(ay);
        inty = ay == tay;
        inty += inty & (BUILTIN_FRACTION_F64(tay*0.5) == 0.0);
    }
#endif

    double ret = BUILTIN_COPYSIGN_F64(expylnx, ((inty == 1) & (x < 0.0)) ? -0.0 : 0.0);

    // Now all the edge cases
#if defined COMPILING_POWR
    double iz = y < 0.0 ? AS_DOUBLE(PINFBITPATT_DP64) : 0.0;
    double zi = y < 0.0 ? 0.0 : AS_DOUBLE(PINFBITPATT_DP64);

    if (x == 0.0)
        ret = iz;

    if (BUILTIN_ISINF_F64(x))
        ret = zi;

    if (BUILTIN_ISINF_F64(y))
        ret = ax < 1.0 ? iz : zi;

    if (y == 0.0)
        ret = x == 0.0 || BUILTIN_ISINF_F64(x) ? AS_DOUBLE(QNANBITPATT_DP64) : 1.0;

    if (x == 1.0)
        ret = BUILTIN_ISINF_F64(y) ? AS_DOUBLE(QNANBITPATT_DP64) : 1.0;

    if (x < 0.0 || BUILTIN_ISNAN_F64(x) || BUILTIN_ISNAN_F64(y))
        ret = AS_DOUBLE(QNANBITPATT_DP64);
#elif defined COMPILING_POWN
    if (BUILTIN_ISINF_F64(ax) || x == 0.0)
        ret = BUILTIN_COPYSIGN_F64((x == 0.0) ^ (ny < 0) ? 0.0 : AS_DOUBLE(PINFBITPATT_DP64), inty == 1 ? x : 0.0);

    if (BUILTIN_ISNAN_F64(x))
        ret = AS_DOUBLE(QNANBITPATT_DP64);
    
    if (ny == 0)
        ret = 1.0;
#elif defined COMPILING_ROOTN
    if (BUILTIN_ISINF_F64(ax) || x == 0.0)
        ret = BUILTIN_COPYSIGN_F64((x == 0.0) ^ (ny < 0) ? 0.0 : AS_DOUBLE(PINFBITPATT_DP64), inty == 1 ? x : 0.0);

    if ((x < 0.0 && inty != 1) || ny == 0)
        ret = AS_DOUBLE(QNANBITPATT_DP64);
#else
    if (x < 0.0 && !inty)
        ret = AS_DOUBLE(QNANBITPATT_DP64);

    if (BUILTIN_ISINF_F64(ay))
        ret = ax == 1.0 ? ax : (samesign(y, ax - 1.0) ? ay : 0.0);

    if (BUILTIN_ISINF_F64(ax) || x == 0.0)
        ret = BUILTIN_COPYSIGN_F64((x == 0.0) ^ (y < 0.0) ? 0.0 : AS_DOUBLE(PINFBITPATT_DP64), inty == 1 ? x : 0.0);

    if (BUILTIN_ISNAN_F64(x) || BUILTIN_ISNAN_F64(y))
        ret = AS_DOUBLE(QNANBITPATT_DP64);

    if (x == 1.0 || y == 0.0)
        ret = 1.0;
#endif

    return ret;
}


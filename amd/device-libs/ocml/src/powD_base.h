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

static bool is_integer(double ay)
{
    return BUILTIN_TRUNC_F64(ay) == ay;
}

static bool is_even_integer(double ay) {
    // Even integers are still integers after division by 2.
    return is_integer(0.5 * ay);
}

static bool is_odd_integer(double ay) {
    return is_integer(ay) && !is_even_integer(ay);
}

#if defined(COMPILING_POW)

CONSTATTR double
MATH_MANGLE(pow)(double x, double y)
{
    double ax = BUILTIN_ABS_F64(x);
    double expylnx = MATH_PRIVATE(expep)(omul(y, MATH_PRIVATE(epln)(ax)));

    double ay = BUILTIN_ABS_F64(y);
    bool is_odd_y = is_odd_integer(ay);

    double ret = BUILTIN_COPYSIGN_F64(expylnx, (is_odd_y & (x < 0.0)) ? -1.0 : 1.0);

    // Now all the edge cases
    if (x < 0.0 && !is_integer(ay))
        ret = QNAN_F64;

    if (BUILTIN_ISINF_F64(ay))
        ret = ax == 1.0 ? ax : (samesign(y, ax - 1.0) ? ay : 0.0);

    if (BUILTIN_ISINF_F64(ax) || x == 0.0)
        ret = BUILTIN_COPYSIGN_F64((x == 0.0) ^ (y < 0.0) ? 0.0 : PINF_F64,
                                   is_odd_y ? x : 0.0);

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
    double ret = BUILTIN_COPYSIGN_F64(expylnx, (is_odd_integer(ay) & (x < 0.0)) ? -1.0 : 1.0);

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

    bool is_odd_y = ny & 1;

    double ret = BUILTIN_COPYSIGN_F64(expylnx, (is_odd_y & (x < 0.0)) ? -1.0 : 1.0);

    // Now all the edge cases
    if (BUILTIN_ISINF_F64(ax) || x == 0.0)
        ret = BUILTIN_COPYSIGN_F64((x == 0.0) ^ (ny < 0) ? 0.0 : PINF_F64,
                                   is_odd_y ? x : 0.0);

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

    bool is_odd_y = ny & 1;

    double ret = BUILTIN_COPYSIGN_F64(expylnx, (is_odd_y & (x < 0.0)) ? -1.0 : 1.0);

    // Now all the edge cases
    if (BUILTIN_ISINF_F64(ax) || x == 0.0)
        ret = BUILTIN_COPYSIGN_F64((x == 0.0) ^ (ny < 0) ? 0.0 : PINF_F64,
                                   is_odd_y ? x : 0.0);

    if ((x < 0.0 && !is_odd_y) || ny == 0)
        ret = QNAN_F64;

    return ret;
}

#else
#error missing function macro
#endif


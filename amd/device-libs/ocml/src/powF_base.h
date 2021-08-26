/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

#define FLOAT_SPECIALIZATION
#include "ep.h"

extern CONSTATTR float2 MATH_PRIVATE(epln)(float);
extern CONSTATTR float MATH_PRIVATE(expep)(float2);

static bool
samesign(float x, float y)
{
    return ((AS_UINT(x) ^ AS_UINT(y)) & 0x80000000) == 0;
}

CONSTATTR float
#if defined(COMPILING_POWR)
MATH_MANGLE(powr)(float x, float y)
#elif defined(COMPILING_POWN)
MATH_MANGLE(pown)(float x, int ny)
#elif defined(COMPILING_ROOTN)
MATH_MANGLE(rootn)(float x, int ny)
#else
MATH_MANGLE(pow)(float x, float y)
#endif
{
    float ax = BUILTIN_ABS_F32(x);
    float expylnx;

    if (UNSAFE_MATH_OPT()) {
#if defined COMPILING_POWN
        float y = (float)ny;
#elif defined COMPILING_ROOTN
        float y = MATH_FAST_RCP((float)ny);
#endif
        if (DAZ_OPT()) {
            expylnx = BUILTIN_EXP2_F32(y * BUILTIN_LOG2_F32(ax));
        } else {
            bool b = ax < 0x1.0p-126f;
            float ylnx = y * (BUILTIN_LOG2_F32(ax * (b ? 0x1.0p+24f : 1.0f)) - (b ? 24.0f : 0.0f));
            b = ylnx < -126.0f;
            expylnx = BUILTIN_EXP2_F32(ylnx + (b ? 24.0f : 0.0f)) * (b ? 0x1.0p-24f : 1.0f);
        }
    } else {
#if defined COMPILING_POWN || defined COMPILING_ROOTN
        int nyh = ny & 0xffff0000;
        float2 y = fadd((float)nyh, (float)(ny - nyh));
#if defined(COMPILING_ROOTN)
        y = rcp(y);
#endif
#endif

        expylnx = MATH_PRIVATE(expep)(omul(y, MATH_PRIVATE(epln)(ax)));
    }

    // y status: 0=not integer, 1=odd, 2=even
#if defined(COMPILING_POWN) || defined(COMPILING_ROOTN)
    int inty = 2 - (ny & 1);
#else
    float ay = BUILTIN_ABS_F32(y);
    int inty;
    {
        float tay = BUILTIN_TRUNC_F32(ay);
        inty = ay == tay;
        inty += inty & (BUILTIN_FRACTION_F32(tay*0.5f) == 0.0f);
    }
#endif

    float ret = BUILTIN_COPYSIGN_F32(expylnx, ((inty == 1) & (x < 0.0f)) ? -0.0f : 0.0f);

    // Now all the edge cases
#if defined COMPILING_POWR
    float iz = y < 0.0f ? AS_FLOAT(PINFBITPATT_SP32) : 0.0f;
    float zi = y < 0.0f ? 0.0f : AS_FLOAT(PINFBITPATT_SP32);

    if (x == 0.0f)
        ret = iz;

    if (BUILTIN_ISINF_F32(x))
        ret = zi;

    if (BUILTIN_ISINF_F32(y))
        ret = ax < 1.0f ? iz : zi;

    if (y == 0.0f)
        ret = x == 0.0f || BUILTIN_ISINF_F32(x) ? AS_FLOAT(QNANBITPATT_SP32) : 1.0f;

    if (x == 1.0f)
        ret = BUILTIN_ISINF_F32(y) ? AS_FLOAT(QNANBITPATT_SP32) : 1.0f;

    if (x < 0.0f || BUILTIN_ISNAN_F32(x) || BUILTIN_ISNAN_F32(y))
        ret = AS_FLOAT(QNANBITPATT_SP32);
#elif defined COMPILING_POWN
    if (BUILTIN_ISINF_F32(ax) || x == 0.0f)
        ret = BUILTIN_COPYSIGN_F32((x == 0.0f) ^ (ny < 0) ? 0.0f : AS_FLOAT(PINFBITPATT_SP32), inty == 1 ? x : 0.0f);

    if (BUILTIN_ISNAN_F32(x))
        ret = AS_FLOAT(QNANBITPATT_SP32);
    
    if (ny == 0)
        ret = 1.0f;
#elif defined COMPILING_ROOTN
    if (BUILTIN_ISINF_F32(ax) || x == 0.0f)
        ret = BUILTIN_COPYSIGN_F32((x == 0.0f) ^ (ny < 0) ? 0.0f : AS_FLOAT(PINFBITPATT_SP32), inty == 1 ? x : 0.0f);

    if ((x < 0.0f && inty != 1) || ny == 0)
        ret = AS_FLOAT(QNANBITPATT_SP32);
#else
    if (x < 0.0f && !inty)
        ret = AS_FLOAT(QNANBITPATT_SP32);

    if (BUILTIN_ISINF_F32(ay))
        ret = ax == 1.0f ? ax : (samesign(y, ax - 1.0f) ? ay : 0.0f);

    if (BUILTIN_ISINF_F32(ax) || x == 0.0f)
        ret = BUILTIN_COPYSIGN_F32((x == 0.0f) ^ (y < 0.0f) ? 0.0f : AS_FLOAT(PINFBITPATT_SP32), inty == 1 ? x : 0.0f);

    if (BUILTIN_ISNAN_F32(x) || BUILTIN_ISNAN_F32(y))
        ret = AS_FLOAT(QNANBITPATT_SP32);

    if (x == 1.0f || y == 0.0f)
        ret = 1.0f;
#endif


    return ret;
}


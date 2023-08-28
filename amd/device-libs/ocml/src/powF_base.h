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

static float fast_expylnx(float ax, float y)
{
    return BUILTIN_EXP2_F32(y * BUILTIN_LOG2_F32(ax));
}

static float compute_expylnx_int(float ax, int ny)
{
    if (UNSAFE_MATH_OPT())
        return fast_expylnx(ax, (float)ny);

    int nyh = ny & 0xffff0000;
    float2 y = fadd((float)nyh, (float)(ny - nyh));
    return MATH_PRIVATE(expep)(omul(y, MATH_PRIVATE(epln)(ax)));
}

// root version of compute_expylnx_int
static float compute_exp_inverse_y_lnx_int(float ax, int ny)
{
    if (UNSAFE_MATH_OPT()) {
        float y = MATH_FAST_RCP((float)ny);
        return fast_expylnx(ax, y);
    }

    int nyh = ny & 0xffff0000;
    float2 y = fadd((float)nyh, (float)(ny - nyh));
    y = rcp(y);
    return MATH_PRIVATE(expep)(omul(y, MATH_PRIVATE(epln)(ax)));
}

static float compute_expylnx_float(float ax, float y)
{
    if (UNSAFE_MATH_OPT())
        return fast_expylnx(ax, y);
    return MATH_PRIVATE(expep)(omul(y, MATH_PRIVATE(epln)(ax)));
}

static bool is_integer(float ay)
{
    return BUILTIN_TRUNC_F32(ay) == ay;
}

static bool is_even_integer(float ay) {
    // Even integers are still integers after division by 2.
    return is_integer(0.5f * ay);
}

static bool is_odd_integer(float ay) {
    return is_integer(ay) && !is_even_integer(ay);
}

#if defined(COMPILING_POW)

CONSTATTR float
MATH_MANGLE(pow)(float x, float y)
{
    if (x == 1.0f)
        y = 1.0f;
    if (y == 0.0f)
        x = 1.0f;

    float ax = BUILTIN_ABS_F32(x);
    float expylnx = compute_expylnx_float(ax, y);

    bool is_odd_y = is_odd_integer(y);

    float ret = BUILTIN_COPYSIGN_F32(expylnx, is_odd_y ? x : 1.0f);

    // Now all the edge cases
    if (x < 0.0f && !is_integer(y))
        ret = QNAN_F32;

    float ay = BUILTIN_ABS_F32(y);
    if (BUILTIN_ISINF_F32(ay)) {
        // FIXME: Missing backend optimization to save on
        // materialization cost of mixed sign constant infinities.
        bool y_is_neg_inf = y != ay;
        ret = ax == 1.0f ? ax : ((ax < 1.0f) ^ y_is_neg_inf ? 0.0f : ay);
    }

    if (BUILTIN_ISINF_F32(ax) || x == 0.0f)
        ret = BUILTIN_COPYSIGN_F32((x == 0.0f) ^ (y < 0.0f) ? 0.0f : PINF_F32,
                                   is_odd_y ? x : 0.0f);

    if (BUILTIN_ISUNORDERED_F32(x, y))
        ret = QNAN_F32;

    return ret;
}

#elif defined(COMPILING_POWR)

CONSTATTR float
MATH_MANGLE(powr)(float x, float y)
{
    if (x < 0.0f)
        x = QNAN_F32;

    float ret = compute_expylnx_float(x, y);

    // Now all the edge cases
    float iz = y < 0.0f ? PINF_F32 : 0.0f;
    float zi = y < 0.0f ? 0.0f : PINF_F32;

    if (x == 0.0f)
        ret = y == 0.0f ? QNAN_F32 : iz;

    if (x == PINF_F32 && y != 0.0f)
        ret = zi;

    if (BUILTIN_ISINF_F32(y) && x != 1.0f)
        ret = x < 1.0f ? iz : zi;

    if (BUILTIN_ISUNORDERED_F32(x, y))
        ret = QNAN_F32;

    return ret;
}

#elif defined(COMPILING_POWN)

CONSTATTR float
MATH_MANGLE(pown)(float x, int ny)
{
    if (ny == 0)
        x = 1.0f;

    float ax = BUILTIN_ABS_F32(x);
    float expylnx = compute_expylnx_int(ax, ny);

    bool is_odd_y = ny & 1;

    float ret = BUILTIN_COPYSIGN_F32(expylnx, is_odd_y ? x : 1.0f);

    // Now all the edge cases
    if (BUILTIN_ISINF_F32(ax) || x == 0.0f)
        ret = BUILTIN_COPYSIGN_F32((x == 0.0f) ^ (ny < 0) ? 0.0f : PINF_F32,
                                   is_odd_y ? x : 0.0f);
    return ret;
}

#elif defined(COMPILING_ROOTN)

CONSTATTR float
MATH_MANGLE(rootn)(float x, int ny)
{
    float ax = BUILTIN_ABS_F32(x);
    float expylnx = compute_exp_inverse_y_lnx_int(ax, ny);

    bool is_odd_y = ny & 1;

    float ret = BUILTIN_COPYSIGN_F32(expylnx, is_odd_y ? x : 1.0f);

    // Now all the edge cases
    if (BUILTIN_ISINF_F32(ax) || x == 0.0f)
        ret = BUILTIN_COPYSIGN_F32((x == 0.0f) ^ (ny < 0) ? 0.0f : PINF_F32,
                                   is_odd_y ? x : 0.0f);

    if ((x < 0.0f && !is_odd_y) || ny == 0)
        ret = QNAN_F32;

    return ret;
}

#else
#error missing function macro
#endif


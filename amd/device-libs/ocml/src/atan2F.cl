
#include "mathF.h"

#ifndef TABLE_BASED_ATAN2
CONSTATTR float
MATH_MANGLE(atan2)(float y, float x)
{
    const float pi = 0x1.921fb6p+1f;
    const float piby2 = 0x1.921fb6p+0f;
    const float piby4 = 0x1.921fb6p-1f;
    const float threepiby4 = 0x1.2d97c8p+1f;

    float ax = BUILTIN_ABS_F32(x);
    float ay = BUILTIN_ABS_F32(y);
    float v = BUILTIN_MIN_F32(ax, ay);
    float u = BUILTIN_MAX_F32(ax, ay);

    float vbyu;

    if (DAZ_OPT()) {
        float s = u < 0x1.0p-96f ? 0x1.0p+32f : 1.0f;
        s = u > 0x1.0p+96f ? 0x1.0p-32f : s;
        vbyu = s * MATH_FAST_DIV(v, s*u);
    } else {
        vbyu = MATH_DIV(v, u);
    }

    float vbyu2 = vbyu * vbyu;

#define USE_2_2_APPROXIMATION
#if defined USE_2_2_APPROXIMATION
    float p = MATH_MAD(vbyu2, MATH_MAD(vbyu2, -0x1.7e1f78p-9f, -0x1.7d1b98p-3f), -0x1.5554d0p-2f) * vbyu2 * vbyu;
    float q = MATH_MAD(vbyu2, MATH_MAD(vbyu2, 0x1.1a714cp-2f, 0x1.287c56p+0f), 1.0f);
#else
    float p = MATH_MAD(vbyu2, MATH_MAD(vbyu2, -0x1.55cd22p-5f, -0x1.26cf76p-2f), -0x1.55554ep-2f) * vbyu2 * vbyu;
    float q = MATH_MAD(vbyu2, MATH_MAD(vbyu2, MATH_MAD(vbyu2, 0x1.9f1304p-5f, 0x1.2656fap-1f), 0x1.76b4b8p+0f), 1.0f);
#endif

    // Octant 0 result
    float a = MATH_MAD(p, MATH_FAST_RCP(q), vbyu);

    // Fix up 3 other octants
    float at = piby2 - a;
    a = ay > ax ? at : a;
    at = pi - a;
    a = x < 0.0f ? at : a;

    // y == 0 => 0 for x > 0, pi for x < 0
    at = AS_INT(x) < 0 ? pi : 0.0f;
    a = y == 0.0f ? at : a;

    if (!FINITE_ONLY_OPT()) {
        // x and y are +- Inf
        at = x > 0.0f ? piby4 : threepiby4;
        a = BUILTIN_CLASS_F32(x, CLASS_PINF|CLASS_NINF) &
            BUILTIN_CLASS_F32(y, CLASS_PINF|CLASS_NINF) ?
            at : a;

        // x or y is NaN
        a = BUILTIN_CLASS_F32(x, CLASS_SNAN|CLASS_QNAN) |
            BUILTIN_CLASS_F32(y, CLASS_SNAN|CLASS_QNAN) ?
            AS_FLOAT(QNANBITPATT_SP32) : a;
    }

    // Fixup sign and return
    return BUILTIN_COPYSIGN_F32(a, y);
}
#else
PUREATTR float
MATH_MANGLE(atan2)(float y, float x)
{
    USE_TABLE(float, p_tbl, M32_ATAN2_JBY256);

    // Explicitly flush arguments
    x = FTZ(x);
    y = FTZ(y);

    uint uy = AS_UINT(y);
    uint ux = AS_UINT(x);
    uint aux = ux & EXSIGNBIT_SP32;
    uint auy = uy & EXSIGNBIT_SP32;

    // General case: take absolute values of arguments
    float u = AS_FLOAT(aux);
    float v = AS_FLOAT(auy);

    // Swap u and v if necessary to obtain 0 < v < u
    int swap_vu = u < v;
    float uu = u;
    u = swap_vu ? v : u;
    v = swap_vu ? uu : v;

    // Use full range division here because the reciprocal of u could be subnormal
    float vbyu = v / u;

    // Handle large quotient with table and polynomial approximation
    int big = vbyu > 0.0625f;

    int index = (int) MATH_MAD(vbyu, 256.0f, 0.5f);
    float findex = (float)index;
    float r = MATH_FAST_DIV(MATH_MAD(vbyu, 256.0f, -findex), MATH_MAD(vbyu, findex, 256.0f));
    float s = r * r;
    index = clamp(index-16, 0, 240);
    float qbig = MATH_MAD(r*s, -0.33333333333224095522f, r) + p_tbl[index];

    // Handle small quotient with a series expansion
    s = vbyu * vbyu;
    float q = MATH_MAD(s, -MATH_MAD(s, -0.14285713561807169030f, 0.19999999999393223405f), 0.33333333333333170500f);
    q = MATH_MAD(vbyu*s, -q, vbyu);
    q = big ? qbig : q;

    // Tidy-up according to which quadrant the arguments lie in
    const float piby2 = 1.5707963267948966e+00f;
    float qt = piby2 - q;
    q = swap_vu ? qt : q;

    int xneg = ux != aux;
    const float pi = 3.1415926535897932e+00f;
    qt = pi - q;
    q = xneg ? qt : q;

    uint ysign = uy ^ auy;
    q = AS_FLOAT(ysign | AS_UINT(q));

    // Now handle a few special cases
    // Zero y gives +-0 for positive x and +-pi for negative x
    qt = AS_FLOAT(ysign | AS_UINT(pi));
    qt = xneg ? qt : y;
    q = y == 0.0f ? qt : q;

    if (!FINITE_ONLY_OPT()) {
        // If abs(x) and abs(y) are both infinity return +-pi/4 or +- 3pi/4 according to signs
        const float piby4 = 7.8539816339744831e-01f;
        const float three_piby4 = 2.3561944901923449e+00f;
        qt = xneg ? three_piby4 : piby4;
        qt = AS_FLOAT(ysign | AS_UINT(qt));
        q = auy == PINFBITPATT_SP32 & aux == PINFBITPATT_SP32 ? qt : q;
    
        // If either arg was NaN, return it
        q = aux > PINFBITPATT_SP32 ? x : q;
        q = auy > PINFBITPATT_SP32 ? y : q;
    }

    return q;
}
#endif


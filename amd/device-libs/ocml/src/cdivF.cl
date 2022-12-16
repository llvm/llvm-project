/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

#define CP(A,B,C,D) ({ \
    float _a = A; \
    float _b = B; \
    float _c = C; \
    float _d = D; \
    float _bd = _b * _d; \
    float _e = BUILTIN_FMA_F32(_b, _d, -_bd); \
    float _f = BUILTIN_FMA_F32(_a, _c, _bd); \
    _f + _e; \
})


CONSTATTR float2
MATH_MANGLE(cdiv)(float2 zn, float2 zd)
{
    float zdx = zd.x;
    float zdy = zd.y;
    bool g = BUILTIN_ABS_F32(zdx) > BUILTIN_ABS_F32(zdy);
    int ed = BUILTIN_FREXP_EXP_F32(g ? zdx : zdy);
    int en = BUILTIN_FREXP_EXP_F32(BUILTIN_MAX_F32(BUILTIN_ABS_F32(zn.x), BUILTIN_ABS_F32(zn.y)));
    int es1 = 126 - ed;
    int es2 = 126 - ed - ed;
    int es3 = 126 - ed - en;
    int es = BUILTIN_MIN_S32(BUILTIN_MIN_S32(es1, es2), es3) >> 1;

    zdx = BUILTIN_FLDEXP_F32(zdx, es);
    zdy = BUILTIN_FLDEXP_F32(zdy, es);
    float u = g ? zdx : zdy;
    float v = g ? zdy : zdx;
    float d2 = BUILTIN_FMA_F32(u, u, v*v);

    zdx = BUILTIN_FLDEXP_F32(zdx, es);
    zdy = BUILTIN_FLDEXP_F32(zdy, es);
    float tr = CP(zn.x,  zn.y, zdx, zdy);
    float ti = CP(zn.y, -zn.x, zdx, zdy);
    float rr = MATH_DIV(tr, d2);
    float ri = MATH_DIV(ti, d2);

    if (!FINITE_ONLY_OPT()) {
        if (BUILTIN_ISNAN_F32(rr) && BUILTIN_ISNAN_F32(ri)) {
            if (d2 == 0.0f && (!BUILTIN_ISNAN_F32(zn.x) || !BUILTIN_ISNAN_F32(zn.y))) {
                float i = BUILTIN_COPYSIGN_F32(PINF_F32, zd.x);
                rr = i * zn.x;
                ri = i * zn.y;
            } else if ((BUILTIN_ISINF_F32(zn.x) || BUILTIN_ISINF_F32(zn.y)) &&
                       (BUILTIN_ISFINITE_F32(zd.x) && BUILTIN_ISFINITE_F32(zd.y))) {
                float znx = BUILTIN_COPYSIGN_F32(BUILTIN_ISINF_F32(zn.x) ? 1.0f : 0.0f, zn.x);
                float zny = BUILTIN_COPYSIGN_F32(BUILTIN_ISINF_F32(zn.y) ? 1.0f : 0.0f, zn.y);
                rr = PINF_F32 * MATH_MAD(znx, zd.x,   zny * zd.y);
                ri = PINF_F32 * MATH_MAD(zny, zd.x,  -znx * zd.y);
            } else if ((BUILTIN_ISINF_F32(zd.x) || BUILTIN_ISINF_F32(zd.y)) &&
                       (BUILTIN_ISFINITE_F32(zn.x) && BUILTIN_ISFINITE_F32(zn.y))) {
                zdx = BUILTIN_COPYSIGN_F32(BUILTIN_ISINF_F32(zd.x) ? 1.0f : 0.0f, zd.x);
                zdy = BUILTIN_COPYSIGN_F32(BUILTIN_ISINF_F32(zd.y) ? 1.0f : 0.0f, zd.y);
                rr = 0.0f * MATH_MAD(zn.x, zdx,  zn.y * zdy);
                ri = 0.0f * MATH_MAD(zn.y, zdx, -zn.x * zdy);
            }
        }
    }

    return (float2)(rr, ri);
}


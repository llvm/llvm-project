/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"
#include "trigredF.h"

#define FMUL(A, AHI, ALO, B, BHI, BLO, CHI, CLO) \
    do { \
        CHI = A * B; \
        CLO = MATH_MAD(ALO, BLO, MATH_MAD(ALO, BHI, MATH_MAD(AHI, BLO, MATH_MAD(AHI, BHI, -CHI)))); \
    } while(0)

#define FNMA(A, AHI, ALO, B, BHI, BLO, C, D) \
    do { \
        float __ph, __pt; \
        FMUL(A, AHI, ALO, B, BHI, BLO, __ph, __pt); \
        float __t = C - __ph; \
        D = __t + (((C - __t) - __ph) - __pt); \
    } while(0)

static inline struct redret
mad_reduce(float x)
{
#if defined EXTRA_PRECISION
#error Not implemented
#else
    const float twobypi = 0x1.45f306p-1f;

    const float piby2_h = 0x1.921fb4p+0f;
    const float piby2_hh = 0x1.92p+0f;
    const float piby2_hl = 0x1.fb4p-12f;

    const float piby2_m = 0x1.4442d0p-24f;
    const float piby2_mh = 0x1.444p-24f;
    const float piby2_ml = 0x1.680p-39f;

    const float piby2_l = 0x1.846988p-48f;
    const float piby2_lh = 0x1.846p-48f;
    const float piby2_ll = 0x1.310p-61f;


    float fn = BUILTIN_RINT_F32(x * twobypi);
    float fnh = AS_FLOAT(AS_UINT(fn) & 0xfffff000U);
    float fnl = fn - fnh;

    float r;
    FNMA(fn, fnh, fnl, piby2_h, piby2_hh, piby2_hl, x, r);
    FNMA(fn, fnh, fnl, piby2_m, piby2_mh, piby2_ml, r, r);

    struct redret ret;
    ret.hi = MATH_MAD(-piby2_l, fn, r);
    ret.i = (int)fn & 0x3;
    return ret;
#endif
}

static inline struct redret
fma_reduce(float x)
{
    const float twobypi = 0x1.45f306p-1f;
    const float piby2_h = 0x1.921fb4p+0f;
    const float piby2_m = 0x1.4442d0p-24f;
    const float piby2_l = 0x1.846988p-48f;

    float fn = BUILTIN_RINT_F32(x * twobypi);

    struct redret ret;

#if defined EXTRA_PRECISION
    float xt = BUILTIN_FMA_F32(fn, -piby2_h, x);
    float yh = BUILTIN_FMA_F32(fn, -piby2_m, xt);
    float ph = fn * piby2_m;
    float pt = BUILTIN_FMA_F32(fn, piby2_m, -ph);
    float th = xt - ph;
    float tt = (xt - th) - ph;
    float yt = BUILTIN_FMA_F32(fn, -piby2_l, ((th - yh) + tt) - pt);
    float rh = yh + yt;
    float rt = yt - (rh - yh);
    ret.hi = rh;
    ret.lo = rt;
#else
    float r = BUILTIN_FMA_F32(fn, -piby2_l, BUILTIN_FMA_F32(fn, -piby2_m, BUILTIN_FMA_F32(fn, -piby2_h, x)));
    ret.hi = r;
#endif

    ret.i =(int)fn & 0x3;
    return ret;
}

CONSTATTR struct redret
MATH_PRIVATE(trigredsmall)(float x)
{
    if (HAVE_FAST_FMA32()) {
	return fma_reduce(x);
    } else {
	return mad_reduce(x);
    }
}


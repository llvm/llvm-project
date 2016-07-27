/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"
#include "trigredF.h"

#define FULL_MUL(A, AHI, ALO, B, BHI, BLO, CHI, CLO) \
    do { \
        CHI = A * B; \
        CLO = MATH_MAD(ALO, BLO, MATH_MAD(ALO, BHI, MATH_MAD(AHI, BLO, MATH_MAD(AHI, BHI, -CHI)))); \
    } while(0)

static inline int
#if defined EXTRA_PRECISION
mad_reduce(__private float *hi, __private float *lo, float x)
#else
mad_reduce(__private float *hi, float x)
#endif
{
#if !defined EXTRA_PRECISION
    // 72 bits of pi/2
    // We'll probably need these for extra precision, if implemented
    const float piby2_h = (float) 0xC90FDA / 0x1.0p+23f;
    const float piby2_hh = (float) 0xC90 / 0x1.0p+11f;
    const float piby2_ht = (float) 0xFDA / 0x1.0p+23f;

    const float piby2_m = (float) 0xA22168 / 0x1.0p+47f;
    const float piby2_mh = (float) 0xA22 / 0x1.0p+35f;
    const float piby2_mt = (float) 0x168 / 0x1.0p+47f;

    const float piby2_t = (float) 0xC234C4 / 0x1.0p+71f;
    // const float piby2_th = (float) 0xC23 / 0x1.0p+59f;
    // const float piby2_tt = (float) 0x4C4 / 0x1.0p+71f;
#endif

    const float twobypi = 0x1.45f306p-1f;

    float fn = BUILTIN_RINT_F32(x * twobypi);
    float fnh = AS_FLOAT(AS_UINT(fn) & 0xfffff000U);
    float fnt = fn - fnh;

    // subtract n * pi/2 from x

#if defined EXTRA_PRECISION
#error This has not been implemented
#else
    float ph, pt;
    FULL_MUL(fn, fnh, fnt, piby2_h, piby2_hh, piby2_ht, ph, pt);
    float d = x - ph;
    float r = d + (((x - d) - ph) - pt);

    FULL_MUL(fn, fnh, fnt, piby2_m, piby2_mh, piby2_mt, ph, pt);
    d = r - ph;
    r = d + (((r - d) - ph) - pt);

    // FULL_MUL(fn, fnh, fnt, piby2_t, piby2_th, piby2_tt, ph, pt);
    ph = fn * piby2_t;
    d = r - ph;
    *hi = d + ((r - d) - ph);
#endif
    return (int)fn & 0x3;
}

static inline int
#if defined EXTRA_PRECISION
fma_reduce(__private float *hi, __private float *lo, float x)
#else
fma_reduce(__private float *hi, float x)
#endif
{
    const float twobypi = 0x1.45f306p-1f;
    const float piby2_h = 0x1.921fb4p+0f;
    const float piby2_m = 0x1.4442d0p-24f;
    const float piby2_t = 0x1.846988p-48f;

#if defined EXTRA_PRECISION
    const float shift = 0x1.800000p+23f;
    float fn = BUILTIN_FMA_F32(x, twobypi, shift) - shift;
    float xt = BUILTIN_FMA_F32(fn, -piby2_h, x);
    float yh = BUILTIN_FMA_F32(fn, -piby2_m, xt);
    float ph = fn * piby2_m;
    float pt = BUILTIN_FMA_F32(fn, piby2_m, -ph);
    float th = xt - ph;
    float tt = (xt - th) - ph;
    float yt = BUILTIN_FMA_F32(fn, -piby2_t, ((th - yh) + tt) - pt);
    float rh = yh + yt;
    float rt = yt - (rh - yh);

    *hi = rh;
    *lo = rt;
#else
    float fn = BUILTIN_RINT_F32(x * twobypi);
    float r = BUILTIN_FMA_F32(fn, -piby2_t, BUILTIN_FMA_F32(fn, -piby2_m, BUILTIN_FMA_F32(fn, -piby2_h, x)));
    *hi = r;
#endif
    return (int)fn & 0x3;
}

INLINEATTR int
#if defined EXTRA_PRECISION
MATH_PRIVATE(trigredsmall)(__private float *r, __private float *rr, float x)
#else
MATH_PRIVATE(trigredsmall)(__private float *r, float x)
#endif
{
    if (HAVE_FAST_FMA32()) {
#if defined EXTRA_PRECISION
	return fma_reduce(r, rr, x);
#else
	return fma_reduce(r, x);
#endif
    } else {
#if defined EXTRA_PRECISION
        return mad_reduce(r, rr, x);
#else
	return mad_reduce(r, x);
#endif
    }
}


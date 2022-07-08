/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"
#include "trigredF.h"


CONSTATTR struct redret
MATH_PRIVATE(trigredlarge)(float x)
{
    int xe = (int)(AS_UINT(x) >> 23) - 127;
    uint xm = 0x00800000U | (AS_UINT(x) & 0x7fffffU);

    // 224 bits of 2/PI: . A2F9836E 4E441529 FC2757D1 F534DDC0 DB629599 3C439041 FE5163AB
    const uint b6 = 0xA2F9836EU;
    const uint b5 = 0x4E441529U;
    const uint b4 = 0xFC2757D1U;
    const uint b3 = 0xF534DDC0U;
    const uint b2 = 0xDB629599U;
    const uint b1 = 0x3C439041U;
    const uint b0 = 0xFE5163ABU;

    uint p0, p1, p2, p3, p4, p5, p6, p7;
    ulong a;

    a = (ulong)xm * (ulong)b0;      p0 = a; a >>= 32;
    a = (ulong)xm * (ulong)b1 + a;  p1 = a; a >>= 32;
    a = (ulong)xm * (ulong)b2 + a;  p2 = a; a >>= 32;
    a = (ulong)xm * (ulong)b3 + a;  p3 = a; a >>= 32;
    a = (ulong)xm * (ulong)b4 + a;  p4 = a; a >>= 32;
    a = (ulong)xm * (ulong)b5 + a;  p5 = a; a >>= 32;
    a = (ulong)xm * (ulong)b6 + a;  p6 = a; p7 = a >> 32;

    uint fbits = 224 + 23 - xe;

    // shift amount to get 2 lsb of integer part at top 2 bits
    //   min: 25 (xe=18) max: 134 (xe=127)
    uint shift = 256U - 2 - fbits;

    // Shift by up to 134/32 = 4 words
    int c = shift > 63;
    p7 = c ? p5 : p7;
    p6 = c ? p4 : p6;
    p5 = c ? p3 : p5;
    p4 = c ? p2 : p4;
    p3 = c ? p1 : p3;
    p2 = c ? p0 : p2;
    shift -= (-c) & 64;

    c = shift > 31;
    p7 = c ? p6 : p7;
    p6 = c ? p5 : p6;
    p5 = c ? p4 : p5;
    p4 = c ? p3 : p4;
    p3 = c ? p2 : p3;
    shift -= (-c) & 32;

    c = shift > 31;
    p7 = c ? p6 : p7;
    p6 = c ? p5 : p6;
    p5 = c ? p4 : p5;
    p4 = c ? p3 : p4;
    shift -= (-c) & 32;

    // BUILTIN_BITALIGN_B32 cannot handle a shift of 32
    c = shift > 0;
    shift = 32 - shift;
    uint t7 = BUILTIN_BITALIGN_B32(p7, p6, shift);
    uint t6 = BUILTIN_BITALIGN_B32(p6, p5, shift);
    uint t5 = BUILTIN_BITALIGN_B32(p5, p4, shift);
    p7 = c ? t7 : p7;
    p6 = c ? t6 : p6;
    p5 = c ? t5 : p5;

    // Get 2 lsb of int part and msb of fraction
    int i = p7 >> 29;

    // Scoot up 2 more bits so only fraction remains
    p7 = BUILTIN_BITALIGN_B32(p7, p6, 30);
    p6 = BUILTIN_BITALIGN_B32(p6, p5, 30);
    p5 = BUILTIN_BITALIGN_B32(p5, p4, 30);

    // Subtract 1 if msb of fraction is 1, i.e. fraction >= 0.5
    uint flip = i & 1 ? 0xffffffffU : 0U;
    uint sign = i & 1 ? 0x80000000U : 0U;
    p7 = p7 ^ flip;
    p6 = p6 ^ flip;
    p5 = p5 ^ flip;

    // Find exponent and shift away leading zeroes and hidden bit
    xe = BUILTIN_CLZ_U32(p7) + 1;
    shift = 32 - xe;
    p7 = BUILTIN_BITALIGN_B32(p7, p6, shift);
    p6 = BUILTIN_BITALIGN_B32(p6, p5, shift);

    // Most significant part of fraction
    float q1 = AS_FLOAT(sign | ((127 - xe) << 23) | (p7 >> 9));

    // Shift out bits we captured on q1
    p7 = BUILTIN_BITALIGN_B32(p7, p6, 32-23);

    // Get 24 more bits of fraction in another float, there are not long strings of zeroes here
    int xxe = BUILTIN_CLZ_U32(p7) + 1;
    p7 = BUILTIN_BITALIGN_B32(p7, p6, 32-xxe);
    float q0 = AS_FLOAT(sign | ((127 - (xe + 23 + xxe)) << 23) | (p7 >> 9));

    // At this point, the fraction q1 + q0 is correct to at least 48 bits
    // Now we need to multiply the fraction by pi/2
    // This loses us about 4 bits
    // pi/2 = C90 FDA A22 168 C23 4C4

    const float pio2h = (float)0xc90fda / 0x1.0p+23f;
    const float pio2hh = (float)0xc90 / 0x1.0p+11f;
    const float pio2ht = (float)0xfda / 0x1.0p+23f;
    const float pio2t = (float)0xa22168 / 0x1.0p+47f;

    float rh, rt;

    if (HAVE_FAST_FMA32()) {
        rh = q1 * pio2h;
        rt = BUILTIN_FMA_F32(q0, pio2h, BUILTIN_FMA_F32(q1, pio2t, BUILTIN_FMA_F32(q1, pio2h, -rh)));
    } else {
        float q1h = AS_FLOAT(AS_UINT(q1) & 0xfffff000);
        float q1t = q1 - q1h;
        rh = q1 * pio2h;
        rt = MATH_MAD(q1t, pio2ht, MATH_MAD(q1t, pio2hh, MATH_MAD(q1h, pio2ht, MATH_MAD(q1h, pio2hh, -rh)))) +
             MATH_MAD(q0, pio2h, q1*pio2t);
    }

    struct redret ret;
#if defined EXTRA_PRECISION
    float t = rh + rt;
    rt = rt - (t - rh);

    ret.hi = t;
    ret.lo = rt;
#else
    ret.hi  = rh + rt;
#endif

    ret.i = ((i >> 1) + (i & 1)) & 0x3;
    return ret;
}


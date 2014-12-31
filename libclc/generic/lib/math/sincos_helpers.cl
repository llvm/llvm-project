/*
 * Copyright (c) 2014 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include <clc/clc.h>

#include "math.h"
#include "sincos_helpers.h"

uint bitalign(uint hi, uint lo, uint shift)
{
        return (hi << (32 - shift)) | (lo >> shift);
}

float sinf_piby4(float x, float y)
{
    // Taylor series for sin(x) is x - x^3/3! + x^5/5! - x^7/7! ...
    // = x * (1 - x^2/3! + x^4/5! - x^6/7! ...
    // = x * f(w)
    // where w = x*x and f(w) = (1 - w/3! + w^2/5! - w^3/7! ...
    // We use a minimax approximation of (f(w) - 1) / w
    // because this produces an expansion in even powers of x.

    const float c1 = -0.1666666666e0f;
    const float c2 = 0.8333331876e-2f;
    const float c3 = -0.198400874e-3f;
    const float c4 = 0.272500015e-5f;
    const float c5 = -2.5050759689e-08f; // 0xb2d72f34
    const float c6 = 1.5896910177e-10f;	 // 0x2f2ec9d3

    float z = x * x;
    float v = z * x;
    float r = mad(z, mad(z, mad(z, mad(z, c6, c5), c4), c3), c2);
    float ret = x - mad(v, -c1, mad(z, mad(y, 0.5f, -v*r), -y));

    return ret;
}

float cosf_piby4(float x, float y)
{
    // Taylor series for cos(x) is 1 - x^2/2! + x^4/4! - x^6/6! ...
    // = f(w)
    // where w = x*x and f(w) = (1 - w/2! + w^2/4! - w^3/6! ...
    // We use a minimax approximation of (f(w) - 1 + w/2) / (w*w)
    // because this produces an expansion in even powers of x.

    const float c1 = 0.416666666e-1f;
    const float c2 = -0.138888876e-2f;
    const float c3 = 0.248006008e-4f;
    const float c4 = -0.2730101334e-6f;
    const float c5 = 2.0875723372e-09f;	 // 0x310f74f6
    const float c6 = -1.1359647598e-11f; // 0xad47d74e

    float z = x * x;
    float r = z * mad(z, mad(z, mad(z, mad(z, mad(z, c6,  c5), c4), c3), c2), c1);

    // if |x| < 0.3
    float qx = 0.0f;

    int ix = as_int(x) & EXSIGNBIT_SP32;

    //  0.78125 > |x| >= 0.3
    float xby4 = as_float(ix - 0x01000000);
    qx = (ix >= 0x3e99999a) & (ix <= 0x3f480000) ? xby4 : qx;

    // x > 0.78125
    qx = ix > 0x3f480000 ? 0.28125f : qx;

    float hz = mad(z, 0.5f, -qx);
    float a = 1.0f - qx;
    float ret = a - (hz - mad(z, r, -x*y));
    return ret;
}

void fullMulS(float *hi, float *lo, float a, float b, float bh, float bt)
{
    if (HAVE_HW_FMA32()) {
        float ph = a * b;
        *hi = ph;
        *lo = fma(a, b, -ph);
    } else {
        float ah = as_float(as_uint(a) & 0xfffff000U);
        float at = a - ah;
        float ph = a * b;
        float pt = mad(at, bt, mad(at, bh, mad(ah, bt, mad(ah, bh, -ph))));
        *hi = ph;
        *lo = pt;
    }
}

float removePi2S(float *hi, float *lo, float x)
{
    // 72 bits of pi/2
    const float fpiby2_1 = (float) 0xC90FDA / 0x1.0p+23f;
    const float fpiby2_1_h = (float) 0xC90 / 0x1.0p+11f;
    const float fpiby2_1_t = (float) 0xFDA / 0x1.0p+23f;

    const float fpiby2_2 = (float) 0xA22168 / 0x1.0p+47f;
    const float fpiby2_2_h = (float) 0xA22 / 0x1.0p+35f;
    const float fpiby2_2_t = (float) 0x168 / 0x1.0p+47f;

    const float fpiby2_3 = (float) 0xC234C4 / 0x1.0p+71f;
    const float fpiby2_3_h = (float) 0xC23 / 0x1.0p+59f;
    const float fpiby2_3_t = (float) 0x4C4 / 0x1.0p+71f;

    const float twobypi = 0x1.45f306p-1f;

    float fnpi2 = trunc(mad(x, twobypi, 0.5f));

    // subtract n * pi/2 from x
    float rhead, rtail;
    fullMulS(&rhead, &rtail, fnpi2, fpiby2_1, fpiby2_1_h, fpiby2_1_t);
    float v = x - rhead;
    float rem = v + (((x - v) - rhead) - rtail);

    float rhead2, rtail2;
    fullMulS(&rhead2, &rtail2, fnpi2, fpiby2_2, fpiby2_2_h, fpiby2_2_t);
    v = rem - rhead2;
    rem = v + (((rem - v) - rhead2) - rtail2);

    float rhead3, rtail3;
    fullMulS(&rhead3, &rtail3, fnpi2, fpiby2_3, fpiby2_3_h, fpiby2_3_t);
    v = rem - rhead3;

    *hi = v + ((rem - v) - rhead3);
    *lo = -rtail3;
    return fnpi2;
}

int argReductionSmallS(float *r, float *rr, float x)
{
    float fnpi2 = removePi2S(r, rr, x);
    return (int)fnpi2 & 0x3;
}

#define FULL_MUL(A, B, HI, LO) \
    LO = A * B; \
    HI = mul_hi(A, B)

#define FULL_MAD(A, B, C, HI, LO) \
    LO = ((A) * (B) + (C)); \
    HI = mul_hi(A, B); \
    HI += LO < C

int argReductionLargeS(float *r, float *rr, float x)
{
    int xe = (int)(as_uint(x) >> 23) - 127;
    uint xm = 0x00800000U | (as_uint(x) & 0x7fffffU);

    // 224 bits of 2/PI: . A2F9836E 4E441529 FC2757D1 F534DDC0 DB629599 3C439041 FE5163AB
    const uint b6 = 0xA2F9836EU;
    const uint b5 = 0x4E441529U;
    const uint b4 = 0xFC2757D1U;
    const uint b3 = 0xF534DDC0U;
    const uint b2 = 0xDB629599U;
    const uint b1 = 0x3C439041U;
    const uint b0 = 0xFE5163ABU;

    uint p0, p1, p2, p3, p4, p5, p6, p7, c0, c1;

    FULL_MUL(xm, b0, c0, p0);
    FULL_MAD(xm, b1, c0, c1, p1);
    FULL_MAD(xm, b2, c1, c0, p2);
    FULL_MAD(xm, b3, c0, c1, p3);
    FULL_MAD(xm, b4, c1, c0, p4);
    FULL_MAD(xm, b5, c0, c1, p5);
    FULL_MAD(xm, b6, c1, p7, p6);

    uint fbits = 224 + 23 - xe;

    // shift amount to get 2 lsb of integer part at top 2 bits
    //   min: 25 (xe=18) max: 134 (xe=127)
    uint shift = 256U - 2 - fbits;

    // Shift by up to 134/32 = 4 words
    int c = shift > 31;
    p7 = c ? p6 : p7;
    p6 = c ? p5 : p6;
    p5 = c ? p4 : p5;
    p4 = c ? p3 : p4;
    p3 = c ? p2 : p3;
    p2 = c ? p1 : p2;
    p1 = c ? p0 : p1;
    shift -= (-c) & 32;

    c = shift > 31;
    p7 = c ? p6 : p7;
    p6 = c ? p5 : p6;
    p5 = c ? p4 : p5;
    p4 = c ? p3 : p4;
    p3 = c ? p2 : p3;
    p2 = c ? p1 : p2;
    shift -= (-c) & 32;

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

    // bitalign cannot handle a shift of 32
    c = shift > 0;
    shift = 32 - shift;
    uint t7 = bitalign(p7, p6, shift);
    uint t6 = bitalign(p6, p5, shift);
    uint t5 = bitalign(p5, p4, shift);
    p7 = c ? t7 : p7;
    p6 = c ? t6 : p6;
    p5 = c ? t5 : p5;

    // Get 2 lsb of int part and msb of fraction
    int i = p7 >> 29;

    // Scoot up 2 more bits so only fraction remains
    p7 = bitalign(p7, p6, 30);
    p6 = bitalign(p6, p5, 30);
    p5 = bitalign(p5, p4, 30);

    // Subtract 1 if msb of fraction is 1, i.e. fraction >= 0.5
    uint flip = i & 1 ? 0xffffffffU : 0U;
    uint sign = i & 1 ? 0x80000000U : 0U;
    p7 = p7 ^ flip;
    p6 = p6 ^ flip;
    p5 = p5 ^ flip;

    // Find exponent and shift away leading zeroes and hidden bit
    xe = clz(p7) + 1;
    shift = 32 - xe;
    p7 = bitalign(p7, p6, shift);
    p6 = bitalign(p6, p5, shift);

    // Most significant part of fraction
    float q1 = as_float(sign | ((127 - xe) << 23) | (p7 >> 9));

    // Shift out bits we captured on q1
    p7 = bitalign(p7, p6, 32-23);

    // Get 24 more bits of fraction in another float, there are not long strings of zeroes here
    int xxe = clz(p7) + 1;
    p7 = bitalign(p7, p6, 32-xxe);
    float q0 = as_float(sign | ((127 - (xe + 23 + xxe)) << 23) | (p7 >> 9));

    // At this point, the fraction q1 + q0 is correct to at least 48 bits
    // Now we need to multiply the fraction by pi/2
    // This loses us about 4 bits
    // pi/2 = C90 FDA A22 168 C23 4C4

    const float pio2h = (float)0xc90fda / 0x1.0p+23f;
    const float pio2hh = (float)0xc90 / 0x1.0p+11f;
    const float pio2ht = (float)0xfda / 0x1.0p+23f;
    const float pio2t = (float)0xa22168 / 0x1.0p+47f;

    float rh, rt;

    if (HAVE_HW_FMA32()) {
        rh = q1 * pio2h;
        rt = fma(q0, pio2h, fma(q1, pio2t, fma(q1, pio2h, -rh)));
    } else {
        float q1h = as_float(as_uint(q1) & 0xfffff000);
        float q1t = q1 - q1h;
        rh = q1 * pio2h;
        rt = mad(q1t, pio2ht, mad(q1t, pio2hh, mad(q1h, pio2ht, mad(q1h, pio2hh, -rh))));
        rt = mad(q0, pio2h, mad(q1, pio2t, rt));
    }

    float t = rh + rt;
    rt = rt - (t - rh);

    *r = t;
    *rr = rt;
    return ((i >> 1) + (i & 1)) & 0x3;
}

int argReductionS(float *r, float *rr, float x)
{
    if (x < 0x1.0p+23f)
        return argReductionSmallS(r, rr, x);
    else
        return argReductionLargeS(r, rr, x);
}


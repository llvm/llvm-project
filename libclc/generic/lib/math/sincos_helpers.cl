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

#include "sincos_helpers.h"
#include <clc/clc.h>
#include <clc/integer/clc_clz.h>
#include <clc/integer/clc_mul_hi.h>
#include <clc/math/clc_mad.h>
#include <clc/math/clc_trunc.h>
#include <clc/math/math.h>
#include <clc/math/tables.h>
#include <clc/shared/clc_max.h>

#define bitalign(hi, lo, shift) ((hi) << (32 - (shift))) | ((lo) >> (shift));

#define bytealign(src0, src1, src2)                                            \
  ((uint)(((((long)(src0)) << 32) | (long)(src1)) >> (((src2) & 3) * 8)))

_CLC_DEF float __clc_sinf_piby4(float x, float y) {
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
  const float c6 = 1.5896910177e-10f;  // 0x2f2ec9d3

  float z = x * x;
  float v = z * x;
  float r = __clc_mad(
      z, __clc_mad(z, __clc_mad(z, __clc_mad(z, c6, c5), c4), c3), c2);
  float ret =
      x - __clc_mad(v, -c1, __clc_mad(z, __clc_mad(y, 0.5f, -v * r), -y));

  return ret;
}

_CLC_DEF float __clc_cosf_piby4(float x, float y) {
  // Taylor series for cos(x) is 1 - x^2/2! + x^4/4! - x^6/6! ...
  // = f(w)
  // where w = x*x and f(w) = (1 - w/2! + w^2/4! - w^3/6! ...
  // We use a minimax approximation of (f(w) - 1 + w/2) / (w*w)
  // because this produces an expansion in even powers of x.

  const float c1 = 0.416666666e-1f;
  const float c2 = -0.138888876e-2f;
  const float c3 = 0.248006008e-4f;
  const float c4 = -0.2730101334e-6f;
  const float c5 = 2.0875723372e-09f;  // 0x310f74f6
  const float c6 = -1.1359647598e-11f; // 0xad47d74e

  float z = x * x;
  float r =
      z *
      __clc_mad(
          z,
          __clc_mad(z, __clc_mad(z, __clc_mad(z, __clc_mad(z, c6, c5), c4), c3),
                    c2),
          c1);

  // if |x| < 0.3
  float qx = 0.0f;

  int ix = as_int(x) & EXSIGNBIT_SP32;

  //  0.78125 > |x| >= 0.3
  float xby4 = as_float(ix - 0x01000000);
  qx = (ix >= 0x3e99999a) & (ix <= 0x3f480000) ? xby4 : qx;

  // x > 0.78125
  qx = ix > 0x3f480000 ? 0.28125f : qx;

  float hz = __clc_mad(z, 0.5f, -qx);
  float a = 1.0f - qx;
  float ret = a - (hz - __clc_mad(z, r, -x * y));
  return ret;
}

_CLC_DEF float __clc_tanf_piby4(float x, int regn) {
  // Core Remez [1,2] approximation to tan(x) on the interval [0,pi/4].
  float r = x * x;

  float a =
      __clc_mad(r, -0.0172032480471481694693109f, 0.385296071263995406715129f);

  float b = __clc_mad(
      r,
      __clc_mad(r, 0.01844239256901656082986661f, -0.51396505478854532132342f),
      1.15588821434688393452299f);

  float t = __clc_mad(x * r, native_divide(a, b), x);
  float tr = -MATH_RECIP(t);

  return regn & 1 ? tr : t;
}

_CLC_DEF void __clc_fullMulS(float *hi, float *lo, float a, float b, float bh,
                             float bt) {
  if (HAVE_HW_FMA32()) {
    float ph = a * b;
    *hi = ph;
    *lo = fma(a, b, -ph);
  } else {
    float ah = as_float(as_uint(a) & 0xfffff000U);
    float at = a - ah;
    float ph = a * b;
    float pt = __clc_mad(
        at, bt, __clc_mad(at, bh, __clc_mad(ah, bt, __clc_mad(ah, bh, -ph))));
    *hi = ph;
    *lo = pt;
  }
}

_CLC_DEF float __clc_removePi2S(float *hi, float *lo, float x) {
  // 72 bits of pi/2
  const float fpiby2_1 = (float)0xC90FDA / 0x1.0p+23f;
  const float fpiby2_1_h = (float)0xC90 / 0x1.0p+11f;
  const float fpiby2_1_t = (float)0xFDA / 0x1.0p+23f;

  const float fpiby2_2 = (float)0xA22168 / 0x1.0p+47f;
  const float fpiby2_2_h = (float)0xA22 / 0x1.0p+35f;
  const float fpiby2_2_t = (float)0x168 / 0x1.0p+47f;

  const float fpiby2_3 = (float)0xC234C4 / 0x1.0p+71f;
  const float fpiby2_3_h = (float)0xC23 / 0x1.0p+59f;
  const float fpiby2_3_t = (float)0x4C4 / 0x1.0p+71f;

  const float twobypi = 0x1.45f306p-1f;

  float fnpi2 = __clc_trunc(__clc_mad(x, twobypi, 0.5f));

  // subtract n * pi/2 from x
  float rhead, rtail;
  __clc_fullMulS(&rhead, &rtail, fnpi2, fpiby2_1, fpiby2_1_h, fpiby2_1_t);
  float v = x - rhead;
  float rem = v + (((x - v) - rhead) - rtail);

  float rhead2, rtail2;
  __clc_fullMulS(&rhead2, &rtail2, fnpi2, fpiby2_2, fpiby2_2_h, fpiby2_2_t);
  v = rem - rhead2;
  rem = v + (((rem - v) - rhead2) - rtail2);

  float rhead3, rtail3;
  __clc_fullMulS(&rhead3, &rtail3, fnpi2, fpiby2_3, fpiby2_3_h, fpiby2_3_t);
  v = rem - rhead3;

  *hi = v + ((rem - v) - rhead3);
  *lo = -rtail3;
  return fnpi2;
}

_CLC_DEF int __clc_argReductionSmallS(float *r, float *rr, float x) {
  float fnpi2 = __clc_removePi2S(r, rr, x);
  return (int)fnpi2 & 0x3;
}

#define FULL_MUL(A, B, HI, LO)                                                 \
  LO = A * B;                                                                  \
  HI = __clc_mul_hi(A, B)

#define FULL_MAD(A, B, C, HI, LO)                                              \
  LO = ((A) * (B) + (C));                                                      \
  HI = __clc_mul_hi(A, B);                                                     \
  HI += LO < C

_CLC_DEF int __clc_argReductionLargeS(float *r, float *rr, float x) {
  int xe = (int)(as_uint(x) >> 23) - 127;
  uint xm = 0x00800000U | (as_uint(x) & 0x7fffffU);

  // 224 bits of 2/PI: . A2F9836E 4E441529 FC2757D1 F534DDC0 DB629599 3C439041
  // FE5163AB
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
  xe = __clc_clz(p7) + 1;
  shift = 32 - xe;
  p7 = bitalign(p7, p6, shift);
  p6 = bitalign(p6, p5, shift);

  // Most significant part of fraction
  float q1 = as_float(sign | ((127 - xe) << 23) | (p7 >> 9));

  // Shift out bits we captured on q1
  p7 = bitalign(p7, p6, 32 - 23);

  // Get 24 more bits of fraction in another float, there are not long strings
  // of zeroes here
  int xxe = __clc_clz(p7) + 1;
  p7 = bitalign(p7, p6, 32 - xxe);
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
    rt = __clc_mad(
        q1t, pio2ht,
        __clc_mad(q1t, pio2hh,
                  __clc_mad(q1h, pio2ht, __clc_mad(q1h, pio2hh, -rh))));
    rt = __clc_mad(q0, pio2h, __clc_mad(q1, pio2t, rt));
  }

  float t = rh + rt;
  rt = rt - (t - rh);

  *r = t;
  *rr = rt;
  return ((i >> 1) + (i & 1)) & 0x3;
}

_CLC_DEF int __clc_argReductionS(float *r, float *rr, float x) {
  if (x < 0x1.0p+23f)
    return __clc_argReductionSmallS(r, rr, x);
  else
    return __clc_argReductionLargeS(r, rr, x);
}

#ifdef cl_khr_fp64

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

// Reduction for medium sized arguments
_CLC_DEF void __clc_remainder_piby2_medium(double x, double *r, double *rr,
                                           int *regn) {
  // How many pi/2 is x a multiple of?
  const double two_by_pi = 0x1.45f306dc9c883p-1;
  double dnpi2 = __clc_trunc(fma(x, two_by_pi, 0.5));

  const double piby2_h = -7074237752028440.0 / 0x1.0p+52;
  const double piby2_m = -2483878800010755.0 / 0x1.0p+105;
  const double piby2_t = -3956492004828932.0 / 0x1.0p+158;

  // Compute product of npi2 with 159 bits of 2/pi
  double p_hh = piby2_h * dnpi2;
  double p_ht = fma(piby2_h, dnpi2, -p_hh);
  double p_mh = piby2_m * dnpi2;
  double p_mt = fma(piby2_m, dnpi2, -p_mh);
  double p_th = piby2_t * dnpi2;
  double p_tt = fma(piby2_t, dnpi2, -p_th);

  // Reduce to 159 bits
  double ph = p_hh;
  double pm = p_ht + p_mh;
  double t = p_mh - (pm - p_ht);
  double pt = p_th + t + p_mt + p_tt;
  t = ph + pm;
  pm = pm - (t - ph);
  ph = t;
  t = pm + pt;
  pt = pt - (t - pm);
  pm = t;

  // Subtract from x
  t = x + ph;
  double qh = t + pm;
  double qt = pm - (qh - t) + pt;

  *r = qh;
  *rr = qt;
  *regn = (int)(long)dnpi2 & 0x3;
}

// Given positive argument x, reduce it to the range [-pi/4,pi/4] using
// extra precision, and return the result in r, rr.
// Return value "regn" tells how many lots of pi/2 were subtracted
// from x to put it in the range [-pi/4,pi/4], mod 4.

_CLC_DEF void __clc_remainder_piby2_large(double x, double *r, double *rr,
                                          int *regn) {

  long ux = as_long(x);
  int e = (int)(ux >> 52) - 1023;
  int i = __clc_max(23, (e >> 3) + 17);
  int j = 150 - i;
  int j16 = j & ~0xf;
  double fract_temp;

  // The following extracts 192 consecutive bits of 2/pi aligned on an arbitrary
  // byte boundary
  uint4 q0 = USE_TABLE(pibits_tbl, j16);
  uint4 q1 = USE_TABLE(pibits_tbl, (j16 + 16));
  uint4 q2 = USE_TABLE(pibits_tbl, (j16 + 32));

  int k = (j >> 2) & 0x3;
  int4 c = (int4)k == (int4)(0, 1, 2, 3);

  uint u0, u1, u2, u3, u4, u5, u6;

  u0 = c.s1 ? q0.s1 : q0.s0;
  u0 = c.s2 ? q0.s2 : u0;
  u0 = c.s3 ? q0.s3 : u0;

  u1 = c.s1 ? q0.s2 : q0.s1;
  u1 = c.s2 ? q0.s3 : u1;
  u1 = c.s3 ? q1.s0 : u1;

  u2 = c.s1 ? q0.s3 : q0.s2;
  u2 = c.s2 ? q1.s0 : u2;
  u2 = c.s3 ? q1.s1 : u2;

  u3 = c.s1 ? q1.s0 : q0.s3;
  u3 = c.s2 ? q1.s1 : u3;
  u3 = c.s3 ? q1.s2 : u3;

  u4 = c.s1 ? q1.s1 : q1.s0;
  u4 = c.s2 ? q1.s2 : u4;
  u4 = c.s3 ? q1.s3 : u4;

  u5 = c.s1 ? q1.s2 : q1.s1;
  u5 = c.s2 ? q1.s3 : u5;
  u5 = c.s3 ? q2.s0 : u5;

  u6 = c.s1 ? q1.s3 : q1.s2;
  u6 = c.s2 ? q2.s0 : u6;
  u6 = c.s3 ? q2.s1 : u6;

  uint v0 = bytealign(u1, u0, j);
  uint v1 = bytealign(u2, u1, j);
  uint v2 = bytealign(u3, u2, j);
  uint v3 = bytealign(u4, u3, j);
  uint v4 = bytealign(u5, u4, j);
  uint v5 = bytealign(u6, u5, j);

  // Place those 192 bits in 4 48-bit doubles along with correct exponent
  // If i > 1018 we would get subnormals so we scale p up and x down to get the
  // same product
  i = 2 + 8 * i;
  x *= i > 1018 ? 0x1.0p-136 : 1.0;
  i -= i > 1018 ? 136 : 0;

  uint ua = (uint)(1023 + 52 - i) << 20;
  double a = as_double((uint2)(0, ua));
  double p0 = as_double((uint2)(v0, ua | (v1 & 0xffffU))) - a;
  ua += 0x03000000U;
  a = as_double((uint2)(0, ua));
  double p1 = as_double((uint2)((v2 << 16) | (v1 >> 16), ua | (v2 >> 16))) - a;
  ua += 0x03000000U;
  a = as_double((uint2)(0, ua));
  double p2 = as_double((uint2)(v3, ua | (v4 & 0xffffU))) - a;
  ua += 0x03000000U;
  a = as_double((uint2)(0, ua));
  double p3 = as_double((uint2)((v5 << 16) | (v4 >> 16), ua | (v5 >> 16))) - a;

  // Exact multiply
  double f0h = p0 * x;
  double f0l = fma(p0, x, -f0h);
  double f1h = p1 * x;
  double f1l = fma(p1, x, -f1h);
  double f2h = p2 * x;
  double f2l = fma(p2, x, -f2h);
  double f3h = p3 * x;
  double f3l = fma(p3, x, -f3h);

  // Accumulate product into 4 doubles
  double s, t;

  double f3 = f3h + f2h;
  t = f2h - (f3 - f3h);
  s = f3l + t;
  t = t - (s - f3l);

  double f2 = s + f1h;
  t = f1h - (f2 - s) + t;
  s = f2l + t;
  t = t - (s - f2l);

  double f1 = s + f0h;
  t = f0h - (f1 - s) + t;
  s = f1l + t;

  double f0 = s + f0l;

  // Strip off unwanted large integer bits
  f3 = 0x1.0p+10 * fract(f3 * 0x1.0p-10, &fract_temp);
  f3 += f3 + f2 < 0.0 ? 0x1.0p+10 : 0.0;

  // Compute least significant integer bits
  t = f3 + f2;
  double di = t - fract(t, &fract_temp);
  i = (float)di;

  // Shift out remaining integer part
  f3 -= di;
  s = f3 + f2;
  t = f2 - (s - f3);
  f3 = s;
  f2 = t;
  s = f2 + f1;
  t = f1 - (s - f2);
  f2 = s;
  f1 = t;
  f1 += f0;

  // Subtract 1 if fraction is >= 0.5, and update regn
  int g = f3 >= 0.5;
  i += g;
  f3 -= (float)g;

  // Shift up bits
  s = f3 + f2;
  t = f2 - (s - f3);
  f3 = s;
  f2 = t + f1;

  // Multiply precise fraction by pi/2 to get radians
  const double p2h = 7074237752028440.0 / 0x1.0p+52;
  const double p2t = 4967757600021510.0 / 0x1.0p+106;

  double rhi = f3 * p2h;
  double rlo = fma(f2, p2h, fma(f3, p2t, fma(f3, p2h, -rhi)));

  *r = rhi + rlo;
  *rr = rlo - (*r - rhi);
  *regn = i & 0x3;
}

_CLC_DEF double2 __clc_sincos_piby4(double x, double xx) {
  // Taylor series for sin(x) is x - x^3/3! + x^5/5! - x^7/7! ...
  //                      = x * (1 - x^2/3! + x^4/5! - x^6/7! ...
  //                      = x * f(w)
  // where w = x*x and f(w) = (1 - w/3! + w^2/5! - w^3/7! ...
  // We use a minimax approximation of (f(w) - 1) / w
  // because this produces an expansion in even powers of x.
  // If xx (the tail of x) is non-zero, we add a correction
  // term g(x,xx) = (1-x*x/2)*xx to the result, where g(x,xx)
  // is an approximation to cos(x)*sin(xx) valid because
  // xx is tiny relative to x.

  // Taylor series for cos(x) is 1 - x^2/2! + x^4/4! - x^6/6! ...
  //                      = f(w)
  // where w = x*x and f(w) = (1 - w/2! + w^2/4! - w^3/6! ...
  // We use a minimax approximation of (f(w) - 1 + w/2) / (w*w)
  // because this produces an expansion in even powers of x.
  // If xx (the tail of x) is non-zero, we subtract a correction
  // term g(x,xx) = x*xx to the result, where g(x,xx)
  // is an approximation to sin(x)*sin(xx) valid because
  // xx is tiny relative to x.

  const double sc1 = -0.166666666666666646259241729;
  const double sc2 = 0.833333333333095043065222816e-2;
  const double sc3 = -0.19841269836761125688538679e-3;
  const double sc4 = 0.275573161037288022676895908448e-5;
  const double sc5 = -0.25051132068021699772257377197e-7;
  const double sc6 = 0.159181443044859136852668200e-9;

  const double cc1 = 0.41666666666666665390037e-1;
  const double cc2 = -0.13888888888887398280412e-2;
  const double cc3 = 0.248015872987670414957399e-4;
  const double cc4 = -0.275573172723441909470836e-6;
  const double cc5 = 0.208761463822329611076335e-8;
  const double cc6 = -0.113826398067944859590880e-10;

  double x2 = x * x;
  double x3 = x2 * x;
  double r = 0.5 * x2;
  double t = 1.0 - r;

  double sp = fma(fma(fma(fma(sc6, x2, sc5), x2, sc4), x2, sc3), x2, sc2);

  double cp =
      t + fma(fma(fma(fma(fma(fma(cc6, x2, cc5), x2, cc4), x2, cc3), x2, cc2),
                  x2, cc1),
              x2 * x2, fma(x, xx, (1.0 - t) - r));

  double2 ret;
  ret.lo = x - fma(-x3, sc1, fma(fma(-x3, sp, 0.5 * xx), x2, -xx));
  ret.hi = cp;

  return ret;
}

#endif

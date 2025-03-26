//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "sincos_helpers.h"
#include <clc/clc.h>
#include <clc/integer/clc_clz.h>
#include <clc/integer/clc_mul_hi.h>
#include <clc/math/clc_fma.h>
#include <clc/math/clc_mad.h>
#include <clc/math/clc_trunc.h>
#include <clc/math/math.h>
#include <clc/math/tables.h>
#include <clc/shared/clc_max.h>

#define bytealign(src0, src1, src2)                                            \
  ((uint)(((((long)(src0)) << 32) | (long)(src1)) >> (((src2) & 3) * 8)))

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

#ifdef cl_khr_fp64

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

// Reduction for medium sized arguments
_CLC_DEF void __clc_remainder_piby2_medium(double x, private double *r,
                                           private double *rr,
                                           private int *regn) {
  // How many pi/2 is x a multiple of?
  const double two_by_pi = 0x1.45f306dc9c883p-1;
  double dnpi2 = __clc_trunc(__clc_fma(x, two_by_pi, 0.5));

  const double piby2_h = -7074237752028440.0 / 0x1.0p+52;
  const double piby2_m = -2483878800010755.0 / 0x1.0p+105;
  const double piby2_t = -3956492004828932.0 / 0x1.0p+158;

  // Compute product of npi2 with 159 bits of 2/pi
  double p_hh = piby2_h * dnpi2;
  double p_ht = __clc_fma(piby2_h, dnpi2, -p_hh);
  double p_mh = piby2_m * dnpi2;
  double p_mt = __clc_fma(piby2_m, dnpi2, -p_mh);
  double p_th = piby2_t * dnpi2;
  double p_tt = __clc_fma(piby2_t, dnpi2, -p_th);

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

_CLC_DEF void __clc_remainder_piby2_large(double x, private double *r,
                                          private double *rr,
                                          private int *regn) {

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
  double f0l = __clc_fma(p0, x, -f0h);
  double f1h = p1 * x;
  double f1l = __clc_fma(p1, x, -f1h);
  double f2h = p2 * x;
  double f2l = __clc_fma(p2, x, -f2h);
  double f3h = p3 * x;
  double f3l = __clc_fma(p3, x, -f3h);

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
  double rlo = __clc_fma(f2, p2h, __clc_fma(f3, p2t, __clc_fma(f3, p2h, -rhi)));

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

  double sp = __clc_fma(
      __clc_fma(__clc_fma(__clc_fma(sc6, x2, sc5), x2, sc4), x2, sc3), x2, sc2);

  double cp =
      t +
      __clc_fma(__clc_fma(__clc_fma(__clc_fma(__clc_fma(__clc_fma(cc6, x2, cc5),
                                                        x2, cc4),
                                              x2, cc3),
                                    x2, cc2),
                          x2, cc1),
                x2 * x2, __clc_fma(x, xx, (1.0 - t) - r));

  double2 ret;
  ret.lo =
      x - __clc_fma(-x3, sc1, __clc_fma(__clc_fma(-x3, sp, 0.5 * xx), x2, -xx));
  ret.hi = cp;

  return ret;
}

#endif

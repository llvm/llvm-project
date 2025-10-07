//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Reduction for medium sized arguments
_CLC_DEF _CLC_OVERLOAD void
__clc_remainder_piby2_medium(__CLC_DOUBLEN x, private __CLC_DOUBLEN *r,
                             private __CLC_DOUBLEN *rr,
                             private __CLC_INTN *regn) {
  // How many pi/2 is x a multiple of?
  const __CLC_DOUBLEN two_by_pi = 0x1.45f306dc9c883p-1;
  __CLC_DOUBLEN dnpi2 = __clc_trunc(__clc_fma(x, two_by_pi, 0.5));

  const __CLC_DOUBLEN piby2_h = -7074237752028440.0 / 0x1.0p+52;
  const __CLC_DOUBLEN piby2_m = -2483878800010755.0 / 0x1.0p+105;
  const __CLC_DOUBLEN piby2_t = -3956492004828932.0 / 0x1.0p+158;

  // Compute product of npi2 with 159 bits of 2/pi
  __CLC_DOUBLEN p_hh = piby2_h * dnpi2;
  __CLC_DOUBLEN p_ht = __clc_fma(piby2_h, dnpi2, -p_hh);
  __CLC_DOUBLEN p_mh = piby2_m * dnpi2;
  __CLC_DOUBLEN p_mt = __clc_fma(piby2_m, dnpi2, -p_mh);
  __CLC_DOUBLEN p_th = piby2_t * dnpi2;
  __CLC_DOUBLEN p_tt = __clc_fma(piby2_t, dnpi2, -p_th);

  // Reduce to 159 bits
  __CLC_DOUBLEN ph = p_hh;
  __CLC_DOUBLEN pm = p_ht + p_mh;
  __CLC_DOUBLEN t = p_mh - (pm - p_ht);
  __CLC_DOUBLEN pt = p_th + t + p_mt + p_tt;
  t = ph + pm;
  pm = pm - (t - ph);
  ph = t;
  t = pm + pt;
  pt = pt - (t - pm);
  pm = t;

  // Subtract from x
  t = x + ph;
  __CLC_DOUBLEN qh = t + pm;
  __CLC_DOUBLEN qt = pm - (qh - t) + pt;

  *r = qh;
  *rr = qt;
  *regn = __CLC_CONVERT_INTN(__CLC_CONVERT_LONGN(dnpi2)) & 0x3;
}

// Given positive argument x, reduce it to the range [-pi/4,pi/4] using
// extra precision, and return the result in r, rr.
// Return value "regn" tells how many lots of pi/2 were subtracted
// from x to put it in the range [-pi/4,pi/4], mod 4.
_CLC_DEF _CLC_OVERLOAD void
__clc_remainder_piby2_large(__CLC_DOUBLEN x, private __CLC_DOUBLEN *r,
                            private __CLC_DOUBLEN *rr,
                            private __CLC_INTN *regn) {

  __CLC_LONGN ux = __CLC_AS_LONGN(x);
  __CLC_INTN e = __CLC_CONVERT_INTN(ux >> EXPSHIFTBITS_DP64) - EXPBIAS_DP64;
  __CLC_INTN i = __clc_max(23, (e >> 3) + 17);
  __CLC_INTN j = 150 - i;
  __CLC_INTN j16 = j & ~0xf;
  __CLC_DOUBLEN fract_temp;

  // The following extracts 192 consecutive bits of 2/pi aligned on an arbitrary
  // byte boundary
  __CLC_ULONGN q0 = __CLC_USE_TABLE(pibits_tbl, j16);
  __CLC_ULONGN q1 = __CLC_USE_TABLE(pibits_tbl, (j16 + 8));
  __CLC_ULONGN q2 = __CLC_USE_TABLE(pibits_tbl, (j16 + 16));
  __CLC_ULONGN q3 = __CLC_USE_TABLE(pibits_tbl, (j16 + 24));
  __CLC_ULONGN q4 = __CLC_USE_TABLE(pibits_tbl, (j16 + 32));

  __CLC_UINTN q0s0 = __CLC_CONVERT_UINTN(q0);
  __CLC_UINTN q0s1 = __CLC_CONVERT_UINTN(q0 >> 32);

  __CLC_UINTN q1s0 = __CLC_CONVERT_UINTN(q1);
  __CLC_UINTN q1s1 = __CLC_CONVERT_UINTN(q1 >> 32);

  __CLC_UINTN q2s0 = __CLC_CONVERT_UINTN(q2);
  __CLC_UINTN q2s1 = __CLC_CONVERT_UINTN(q2 >> 32);

  __CLC_UINTN q3s0 = __CLC_CONVERT_UINTN(q3);
  __CLC_UINTN q3s1 = __CLC_CONVERT_UINTN(q3 >> 32);

  __CLC_UINTN q4s0 = __CLC_CONVERT_UINTN(q4);
  __CLC_UINTN q4s1 = __CLC_CONVERT_UINTN(q4 >> 32);

  __CLC_INTN k = (j >> 2) & 0x3;
  __CLC_INTN c0 = k == 0;
  __CLC_INTN c1 = k == 1;
  __CLC_INTN c2 = k == 2;
  __CLC_INTN c3 = k == 3;

  __CLC_UINTN u0, u1, u2, u3, u4, u5, u6;

  u0 = c1 ? q0s1 : q0s0;
  u0 = c2 ? q1s0 : u0;
  u0 = c3 ? q1s1 : u0;

  u1 = c1 ? q1s0 : q0s1;
  u1 = c2 ? q1s1 : u1;
  u1 = c3 ? q2s0 : u1;

  u2 = c1 ? q1s1 : q1s0;
  u2 = c2 ? q2s0 : u2;
  u2 = c3 ? q2s1 : u2;

  u3 = c1 ? q2s0 : q1s1;
  u3 = c2 ? q2s1 : u3;
  u3 = c3 ? q3s0 : u3;

  u4 = c1 ? q2s1 : q2s0;
  u4 = c2 ? q3s0 : u4;
  u4 = c3 ? q3s1 : u4;

  u5 = c1 ? q3s0 : q2s1;
  u5 = c2 ? q3s1 : u5;
  u5 = c3 ? q4s0 : u5;

  u6 = c1 ? q3s1 : q3s0;
  u6 = c2 ? q4s0 : u6;
  u6 = c3 ? q4s1 : u6;

  __CLC_UINTN v0 = bytealign(u1, u0, j);
  __CLC_UINTN v1 = bytealign(u2, u1, j);
  __CLC_UINTN v2 = bytealign(u3, u2, j);
  __CLC_UINTN v3 = bytealign(u4, u3, j);
  __CLC_UINTN v4 = bytealign(u5, u4, j);
  __CLC_UINTN v5 = bytealign(u6, u5, j);

  // Place those 192 bits in 4 48-bit doubles along with correct exponent
  // If i > 1018 we would get subnormals so we scale p up and x down to get the
  // same product
  i = 2 + 8 * i;
  x *= __CLC_CONVERT_BIT_INTN(i > 1018) ? 0x1.0p-136 : 1.0;
  i -= i > 1018 ? 136 : 0;

#define doublen_lohi(x, y)                                                     \
  __CLC_AS_DOUBLEN(__CLC_CONVERT_ULONGN((x)) & 0xFFFFFFFF |                    \
                   __CLC_CONVERT_ULONGN((y)) << 32)

  __CLC_UINTN ua = __CLC_CONVERT_UINTN(EXPBIAS_DP64 + EXPSHIFTBITS_DP64 - i)
                   << 20;
  __CLC_DOUBLEN a = doublen_lohi((__CLC_ULONGN)0, ua);
  __CLC_DOUBLEN p0 = doublen_lohi(v0, ua | (v1 & 0xffffU)) - a;
  ua += 0x03000000U;
  a = doublen_lohi((__CLC_ULONGN)0, ua);
  __CLC_DOUBLEN p1 =
      doublen_lohi(((v2 << 16) | (v1 >> 16)), (ua | (v2 >> 16))) - a;
  ua += 0x03000000U;
  a = doublen_lohi((__CLC_ULONGN)0, ua);
  __CLC_DOUBLEN p2 = doublen_lohi(v3, (ua | (v4 & 0xffffU))) - a;
  ua += 0x03000000U;
  a = doublen_lohi((__CLC_ULONGN)0, ua);
  __CLC_DOUBLEN p3 =
      doublen_lohi(((v5 << 16) | (v4 >> 16)), (ua | (v5 >> 16))) - a;

#undef doublen_lohi

  // Exact multiply
  __CLC_DOUBLEN f0h = p0 * x;
  __CLC_DOUBLEN f0l = __clc_fma(p0, x, -f0h);
  __CLC_DOUBLEN f1h = p1 * x;
  __CLC_DOUBLEN f1l = __clc_fma(p1, x, -f1h);
  __CLC_DOUBLEN f2h = p2 * x;
  __CLC_DOUBLEN f2l = __clc_fma(p2, x, -f2h);
  __CLC_DOUBLEN f3h = p3 * x;
  __CLC_DOUBLEN f3l = __clc_fma(p3, x, -f3h);

  // Accumulate product into 4 doubles
  __CLC_DOUBLEN s, t;

  __CLC_DOUBLEN f3 = f3h + f2h;
  t = f2h - (f3 - f3h);
  s = f3l + t;
  t = t - (s - f3l);

  __CLC_DOUBLEN f2 = s + f1h;
  t = f1h - (f2 - s) + t;
  s = f2l + t;
  t = t - (s - f2l);

  __CLC_DOUBLEN f1 = s + f0h;
  t = f0h - (f1 - s) + t;
  s = f1l + t;

  __CLC_DOUBLEN f0 = s + f0l;

  // Strip off unwanted large integer bits
  f3 = 0x1.0p+10 * __clc_fract(f3 * 0x1.0p-10, &fract_temp);
  f3 += f3 + f2 < 0.0 ? 0x1.0p+10 : 0.0;

  // Compute least significant integer bits
  t = f3 + f2;
  __CLC_DOUBLEN di = t - __clc_fract(t, &fract_temp);
  i = __CLC_CONVERT_INTN(__CLC_CONVERT_FLOATN(di));

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
  __CLC_INTN g = __CLC_CONVERT_INTN(f3 >= 0.5 ? 1L : 0L);
  i += g;
  f3 -= __CLC_CONVERT_DOUBLEN(__CLC_CONVERT_FLOATN(g));

  // Shift up bits
  s = f3 + f2;
  t = f2 - (s - f3);
  f3 = s;
  f2 = t + f1;

  // Multiply precise fraction by pi/2 to get radians
  const __CLC_DOUBLEN p2h = 7074237752028440.0 / 0x1.0p+52;
  const __CLC_DOUBLEN p2t = 4967757600021510.0 / 0x1.0p+106;

  __CLC_DOUBLEN rhi = f3 * p2h;
  __CLC_DOUBLEN rlo =
      __clc_fma(f2, p2h, __clc_fma(f3, p2t, __clc_fma(f3, p2h, -rhi)));

  *r = rhi + rlo;
  *rr = rlo - (*r - rhi);
  *regn = i & 0x3;
}

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma OPENCL FP_CONTRACT OFF

_CLC_DEF _CLC_OVERLOAD __CLC_SINCOS_RET_GENTYPE
__clc_sincos_reduced_eval(__CLC_DOUBLEN x, __CLC_DOUBLEN y) {
  const __CLC_DOUBLEN S0 = -0x1.5555555555555p-3;
  const __CLC_DOUBLEN S1 = 0x1.1111111110bb3p-7;
  const __CLC_DOUBLEN S2 = -0x1.a01a019e83e5cp-13;
  const __CLC_DOUBLEN S3 = 0x1.71de3796cde01p-19;
  const __CLC_DOUBLEN S4 = -0x1.ae600b42fdfa7p-26;
  const __CLC_DOUBLEN S5 = 0x1.5e0b2f9a43bb8p-33;

  const __CLC_DOUBLEN C0 = 0x1.5555555555555p-5;
  const __CLC_DOUBLEN C1 = -0x1.6c16c16c16967p-10;
  const __CLC_DOUBLEN C2 = 0x1.a01a019f4ec90p-16;
  const __CLC_DOUBLEN C3 = -0x1.27e4fa17f65f6p-22;
  const __CLC_DOUBLEN C4 = 0x1.1eeb69037ab78p-29;
  const __CLC_DOUBLEN C5 = -0x1.907db46cc5e42p-37;

  __CLC_DOUBLEN x2 = x * x;
  __CLC_DOUBLEN x3 = x * x2;
  __CLC_DOUBLEN r = 0.5 * x2;
  __CLC_DOUBLEN t = 1.0 - r;
  __CLC_DOUBLEN u = 1.0 - t;
  __CLC_DOUBLEN v = u - r;

  __CLC_DOUBLEN cxy = t + __clc_mad(x2 * x2, __clc_mad(x2, __clc_mad(x2, __clc_mad(x2, __clc_mad(x2, __clc_mad(x2, C5, C4), C3), C2), C1), C0), __clc_mad(x, -y, v));
  __CLC_DOUBLEN sxy = __clc_mad(x2, __clc_mad(x2, __clc_mad(x2, __clc_mad(x2, S5, S4), S3), S2), S1);
  sxy = x - __clc_mad(-x3, S0, __clc_mad(x2, __clc_mad(-x3, sxy, __CLC_FP_LIT(0.5) * y), -y));

  __CLC_SINCOS_RET_GENTYPE ret;
  ret.cos = cxy;
  ret.sin = sxy;
  return ret;
}

_CLC_DEF _CLC_OVERLOAD __CLC_DOUBLEN __clc_tan_reduced_eval(__CLC_DOUBLEN x,
                                                            __CLC_DOUBLEN xx,
                                                            __CLC_INTN is_odd) {
  __CLC_DOUBLEN s = __clc_ep_sqr(__clc_ep_make_pair(x, xx)).hi;
  __CLC_DOUBLEN p = s * __clc_mad(s, __clc_mad(s, __clc_mad(s, __clc_mad(s,
                 __clc_mad(s, __clc_mad(s, __clc_mad(s, __clc_mad(s,
                 __clc_mad(s, __clc_mad(s, __clc_mad(s, __clc_mad(s,
                 __clc_mad(s,
                     0x1.5e089c751c08cp-16, -0x1.78809a9a29f71p-15),
                     0x1.7746f90a8aaep-14), -0x1.bb44da6fbf144p-16),
                     0x1.1e634a7943acfp-13), 0x1.d250fdeb68febp-13),
                     0x1.37fd9b58c4d95p-11), 0x1.7d5af15120e2cp-10),
                     0x1.d6d93e09491dfp-9), 0x1.226e12033784dp-7),
                     0x1.664f49ac36ae2p-6), 0x1.ba1ba1b451c21p-5),
                     0x1.11111111185b7p-3), 0x1.55555555554eep-2);
  __CLC_EP_PAIR t = __clc_ep_fast_add(__clc_ep_make_pair(x, xx), __clc_ep_mul(x, p));
  __CLC_EP_PAIR tr = __clc_ep_fast_recip(t);
  return __CLC_CONVERT_LONGN(is_odd) ? -tr.hi : t.hi;
}

_CLC_DEF _CLC_OVERLOAD void __clc_sincos_piby4(__CLC_DOUBLEN x,
                                               __CLC_DOUBLEN xx,
                                               private __CLC_DOUBLEN *sinval,
                                               private __CLC_DOUBLEN *cosval) {
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

  const __CLC_DOUBLEN sc1 = -0.166666666666666646259241729;
  const __CLC_DOUBLEN sc2 = 0.833333333333095043065222816e-2;
  const __CLC_DOUBLEN sc3 = -0.19841269836761125688538679e-3;
  const __CLC_DOUBLEN sc4 = 0.275573161037288022676895908448e-5;
  const __CLC_DOUBLEN sc5 = -0.25051132068021699772257377197e-7;
  const __CLC_DOUBLEN sc6 = 0.159181443044859136852668200e-9;

  const __CLC_DOUBLEN cc1 = 0.41666666666666665390037e-1;
  const __CLC_DOUBLEN cc2 = -0.13888888888887398280412e-2;
  const __CLC_DOUBLEN cc3 = 0.248015872987670414957399e-4;
  const __CLC_DOUBLEN cc4 = -0.275573172723441909470836e-6;
  const __CLC_DOUBLEN cc5 = 0.208761463822329611076335e-8;
  const __CLC_DOUBLEN cc6 = -0.113826398067944859590880e-10;

  __CLC_DOUBLEN x2 = x * x;
  __CLC_DOUBLEN x3 = x2 * x;
  __CLC_DOUBLEN r = __CLC_FP_LIT(0.5) * x2;
  __CLC_DOUBLEN t = __CLC_FP_LIT(1.0) - r;

  __CLC_DOUBLEN sp = __clc_fma(
      __clc_fma(__clc_fma(__clc_fma(sc6, x2, sc5), x2, sc4), x2, sc3), x2, sc2);

  __CLC_DOUBLEN cp =
      t +
      __clc_fma(__clc_fma(__clc_fma(__clc_fma(__clc_fma(__clc_fma(cc6, x2, cc5),
                                                        x2, cc4),
                                              x2, cc3),
                                    x2, cc2),
                          x2, cc1),
                x2 * x2, __clc_fma(x, xx, (__CLC_FP_LIT(1.0) - t) - r));

  *sinval = x - __clc_fma(-x3, sc1,
                          __clc_fma(__clc_fma(-x3, sp, __CLC_FP_LIT(0.5) * xx),
                                    x2, -xx));
  *cosval = cp;
}

_CLC_DEF _CLC_OVERLOAD void __clc_tan_piby4(__CLC_DOUBLEN x, __CLC_DOUBLEN xx,
                                            private __CLC_DOUBLEN *leadval,
                                            private __CLC_DOUBLEN *tailval) {
  // 0x3fe921fb54442d18
  const __CLC_DOUBLEN piby4_lead = 7.85398163397448278999e-01;
  // 0x3c81a62633145c06
  const __CLC_DOUBLEN piby4_tail = 3.06161699786838240164e-17;

  // In order to maintain relative precision transform using the identity:
  // tan(pi/4-x) = (1-tan(x))/(1+tan(x)) for arguments close to pi/4.
  // Similarly use tan(x-pi/4) = (tan(x)-1)/(tan(x)+1) close to -pi/4.

  __CLC_LONGN ca = x > 0.68;
  __CLC_LONGN cb = x < -0.68;
  __CLC_DOUBLEN transform = ca ? 1.0 : 0.0;
  transform = cb ? -1.0 : transform;

  __CLC_DOUBLEN tx = __clc_fma(-transform, x, piby4_lead) +
                     __clc_fma(-transform, xx, piby4_tail);
  __CLC_LONGN c = ca | cb;
  x = c ? tx : x;
  xx = c ? 0.0 : xx;

  // Core Remez [2,3] approximation to tan(x+xx) on the interval [0,0.68].
  __CLC_DOUBLEN t1 = x;
  __CLC_DOUBLEN r = __clc_fma(2.0, x * xx, x * x);

  __CLC_DOUBLEN a = __clc_fma(r,
                              __clc_fma(r, 0.224044448537022097264602535574e-3,
                                        -0.229345080057565662883358588111e-1),
                              0.372379159759792203640806338901e0);

  __CLC_DOUBLEN b =
      __clc_fma(r,
                __clc_fma(r,
                          __clc_fma(r, -0.232371494088563558304549252913e-3,
                                    0.260656620398645407524064091208e-1),
                          -0.515658515729031149329237816945e0),
                0.111713747927937668539901657944e1);

  __CLC_DOUBLEN t2 = __clc_fma(MATH_DIVIDE(a, b), x * r, xx);

  __CLC_DOUBLEN tp = t1 + t2;

  // Compute -1.0/(t1 + t2) accurately
  __CLC_DOUBLEN z1 =
      __CLC_AS_GENTYPE(__CLC_AS_ULONGN(tp) & 0xffffffff00000000L);
  __CLC_DOUBLEN z2 = t2 - (z1 - t1);
  __CLC_DOUBLEN trec = -MATH_RECIP(tp);
  __CLC_DOUBLEN trec_top =
      __CLC_AS_GENTYPE(__CLC_AS_ULONGN(trec) & 0xffffffff00000000L);

  __CLC_DOUBLEN tpr = __clc_fma(
      __clc_fma(trec_top, z2, __clc_fma(trec_top, z1, 1.0)), trec, trec_top);

  __CLC_DOUBLEN tpt = transform * (1.0 - MATH_DIVIDE(2.0 * tp, 1.0 + tp));
  __CLC_DOUBLEN tptr = transform * (MATH_DIVIDE(2.0 * tp, tp - 1.0) - 1.0);

  *leadval = c ? tpt : tp;
  *tailval = c ? tptr : tpr;
}

// Reduction for small sized arguments
_CLC_DEF _CLC_OVERLOAD __CLC_INTN __clc_remainder_piby2_small(
    __CLC_DOUBLEN x, private __CLC_DOUBLEN *rh, private __CLC_DOUBLEN *rt) {
  // How many pi/2 is x a multiple of?
  const __CLC_DOUBLEN twobypi = 0x1.45f306dc9c883p-1;
  const __CLC_DOUBLEN piby2_h = 0x1.921fb54442d18p+0;
  const __CLC_DOUBLEN piby2_m = 0x1.1a62633145c00p-54;
  const __CLC_DOUBLEN piby2_t = 0x1.b839a252049c0p-104;

  __CLC_DOUBLEN dn_pi2 = __clc_rint(x * twobypi);

  // Compute product of npi2 with 159 bits of 2/pi
  __CLC_DOUBLEN xt = __clc_fma(dn_pi2, -piby2_h, x);
  __CLC_DOUBLEN yh = __clc_fma(dn_pi2, -piby2_m, xt);
  __CLC_DOUBLEN ph = dn_pi2 * piby2_m;
  __CLC_DOUBLEN pt = __clc_fma(dn_pi2, piby2_m, -ph);

  // Reduce to 159 bits
  __CLC_DOUBLEN th = xt - ph;
  __CLC_DOUBLEN tt = (xt - th) - ph;
  __CLC_DOUBLEN yt = __clc_fma(dn_pi2, -piby2_t, ((th - yh) + tt) - pt);

  __CLC_EP_PAIR result = __clc_ep_fast_add(yh, yt);
  *rh = result.lo;
  *rt = result.hi;

  return __CLC_CONVERT_INTN(dn_pi2) & 0x3;
}

// Given positive argument x, reduce it to the range [-pi/4,pi/4] using
// extra precision, and return the result in r, rr.
// Return value "regn" tells how many lots of pi/2 were subtracted
// from x to put it in the range [-pi/4,pi/4], mod 4.
_CLC_DEF _CLC_OVERLOAD __CLC_INTN __clc_remainder_piby2_large(
    __CLC_DOUBLEN x, private __CLC_DOUBLEN *r, private __CLC_DOUBLEN *rr) {
  // Scale x by relevant part of 2/pi
  __CLC_DOUBLEN p0 = __clc_get_twobypi_bits(x, 2);
  __CLC_DOUBLEN p1 = __clc_get_twobypi_bits(x, 1);
  __CLC_DOUBLEN p2 = __clc_get_twobypi_bits(x, 0);

  x = __clc_ldexp(x, __CLC_CONVERT_INTN(x >= 0x1.0p+945) ? -128 : 0);

  // Exact multiply
  __CLC_EP_PAIR rp0 = __clc_ep_mul(p0, x);
  __CLC_EP_PAIR rp1 = __clc_ep_mul(p1, x);
  __CLC_EP_PAIR rp2 = __clc_ep_mul(p2, x);

  // Accumulate product into 3 doubles
  __CLC_EP_PAIR v2 = __clc_ep_add(rp2.lo, rp1.hi);
  __CLC_EP_PAIR v1 = __clc_ep_add(rp1.lo, rp0.hi);
  __CLC_EP_PAIR w2 = __clc_ep_add(v2.lo, v1.hi);

  __CLC_DOUBLEN e3 = rp2.hi;
  __CLC_DOUBLEN e2 = v2.hi;
  __CLC_DOUBLEN e1 = w2.hi;
  __CLC_DOUBLEN e0 = w2.lo + v1.lo + rp0.lo;

  __CLC_EP_PAIR e32 = __clc_ep_fast_add(e3, e2);
  __CLC_EP_PAIR e21 = __clc_ep_fast_add(e32.lo, e1);
  __CLC_EP_PAIR e10 = __clc_ep_fast_add(e21.lo, e0);

  __CLC_DOUBLEN f2 = e32.hi;
  __CLC_DOUBLEN f1 = e21.hi;
  __CLC_DOUBLEN f0 = e10.hi;

  __CLC_DOUBLEN f2_scale = __clc_ldexp(f2, -2);

  // Compute least significant integer bits
  __CLC_DOUBLEN unused;
  __CLC_DOUBLEN f2_frac = __clc_fract(f2_scale, &unused);

  f2 = __clc_ldexp(f2_frac, 2);

  f2 += (f2 + f1) < 0.0 ? 4.0 : 0.0;

  __CLC_INTN i = __CLC_CONVERT_INTN(f2 + f1);
  f2 -= __CLC_CONVERT_DOUBLEN(i);

  __CLC_EP_PAIR f21 = __clc_ep_fast_add(f2, f1);
  __CLC_EP_PAIR f10 = __clc_ep_fast_add(f21.lo, f0);
  f2 = f21.hi;
  f1 = f10.hi;
  f0 = f10.lo;

  __CLC_S_GENTYPE g = f2 >= __CLC_FP_LIT(0.5);
  i += __CLC_CONVERT_INTN(g) ? 1 : 0;
  f2 -= g ? __CLC_FP_LIT(1.0) : __CLC_FP_LIT(0.0);

  __CLC_EP_PAIR rf = __clc_ep_fast_add(f2, f1);
  f2 = rf.hi;
  f1 = rf.lo;

  // Multiply precise fraction by pi/2 to get radians
  const __CLC_DOUBLEN p2h = 0x1.921fb54442d18p+0;
  const __CLC_DOUBLEN p2t = 0x1.1a62633145c07p-54;

  __CLC_DOUBLEN rh = f2 * p2h;
  __CLC_DOUBLEN rt =
      __clc_fma(f1, p2h, __clc_fma(f2, p2t, __clc_fma(f2, p2h, -rh)));

  __CLC_EP_PAIR result = __clc_ep_fast_add(rh, rt);
  *r = result.lo;
  *rr = result.hi;

  return i & 0x3;
}

_CLC_DEF _CLC_OVERLOAD __CLC_INTN __clc_argReductionS(
    private __CLC_DOUBLEN *r_lo, private __CLC_DOUBLEN *r_hi, __CLC_DOUBLEN x) {
  __CLC_LONGN is_large = x >= (__CLC_DOUBLEN)0x1.0p+30;

#ifdef __CLC_SCALAR1
  if (is_large)
    return __clc_remainder_piby2_large(x, r_lo, r_hi);
  return __clc_remainder_piby2_small(x, r_lo, r_hi);
#else
  __CLC_DOUBLEN rlo_1, rlo_2;
  __CLC_DOUBLEN rhi_1, rhi_2;
  __CLC_INTN ret1 = __clc_remainder_piby2_small(x, &rlo_1, &rhi_1);
  __CLC_INTN ret2 = __clc_remainder_piby2_large(x, &rlo_2, &rhi_2);
  *r_lo = is_large ? rlo_2 : rlo_1;
  *r_hi = is_large ? rhi_2 : rhi_1;
  return __CLC_CONVERT_INTN(is_large) ? ret2 : ret1;
#endif
}

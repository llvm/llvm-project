//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if __CLC_FPSIZE == 32

// Evaluate single precisions in and cos of value in interval [-pi/4, pi/4]
_CLC_INLINE _CLC_OVERLOAD void
__clc_sincos_piby4(__CLC_GENTYPE x, private __CLC_GENTYPE *sinval,
                   private __CLC_GENTYPE *cosval) {
  // Taylor series for sin(x) is x - x^3/3! + x^5/5! - x^7/7! ...
  // = x * (1 - x^2/3! + x^4/5! - x^6/7! ...
  // = x * f(w)
  // where w = x*x and f(w) = (1 - w/3! + w^2/5! - w^3/7! ...
  // We use a minimax approximation of (f(w) - 1) / w
  // because this produces an expansion in even powers of x.

  // Taylor series for cos(x) is 1 - x^2/2! + x^4/4! - x^6/6! ...
  // = f(w)
  // where w = x*x and f(w) = (1 - w/2! + w^2/4! - w^3/6! ...
  // We use a minimax approximation of (f(w) - 1 + w/2) / (w*w)
  // because this produces an expansion in even powers of x.

  const __CLC_GENTYPE sc1 = -0.166666666638608441788607926e0F;
  const __CLC_GENTYPE sc2 = 0.833333187633086262120839299e-2F;
  const __CLC_GENTYPE sc3 = -0.198400874359527693921333720e-3F;
  const __CLC_GENTYPE sc4 = 0.272500015145584081596826911e-5F;

  const __CLC_GENTYPE cc1 = 0.41666666664325175238031e-1F;
  const __CLC_GENTYPE cc2 = -0.13888887673175665567647e-2F;
  const __CLC_GENTYPE cc3 = 0.24800600878112441958053e-4F;
  const __CLC_GENTYPE cc4 = -0.27301013343179832472841e-6F;

  __CLC_GENTYPE x2 = x * x;

  *sinval = __clc_mad(
      x * x2, __clc_mad(x2, __clc_mad(x2, __clc_mad(x2, sc4, sc3), sc2), sc1),
      x);
  *cosval = __clc_mad(
      x2 * x2, __clc_mad(x2, __clc_mad(x2, __clc_mad(x2, cc4, cc3), cc2), cc1),
      __clc_mad(x2, -0.5f, 1.0f));
}

#elif __CLC_FPSIZE == 64

_CLC_INLINE _CLC_OVERLOAD void
__clc_sincos_piby4(__CLC_GENTYPE x, __CLC_GENTYPE xx,
                   private __CLC_GENTYPE *sinval,
                   private __CLC_GENTYPE *cosval) {
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

  const __CLC_GENTYPE sc1 = -0.166666666666666646259241729;
  const __CLC_GENTYPE sc2 = 0.833333333333095043065222816e-2;
  const __CLC_GENTYPE sc3 = -0.19841269836761125688538679e-3;
  const __CLC_GENTYPE sc4 = 0.275573161037288022676895908448e-5;
  const __CLC_GENTYPE sc5 = -0.25051132068021699772257377197e-7;
  const __CLC_GENTYPE sc6 = 0.159181443044859136852668200e-9;

  const __CLC_GENTYPE cc1 = 0.41666666666666665390037e-1;
  const __CLC_GENTYPE cc2 = -0.13888888888887398280412e-2;
  const __CLC_GENTYPE cc3 = 0.248015872987670414957399e-4;
  const __CLC_GENTYPE cc4 = -0.275573172723441909470836e-6;
  const __CLC_GENTYPE cc5 = 0.208761463822329611076335e-8;
  const __CLC_GENTYPE cc6 = -0.113826398067944859590880e-10;

  __CLC_GENTYPE x2 = x * x;
  __CLC_GENTYPE x3 = x2 * x;
  __CLC_GENTYPE r = (__CLC_GENTYPE)0.5 * x2;
  __CLC_GENTYPE t = (__CLC_GENTYPE)1.0 - r;

  __CLC_GENTYPE sp = __clc_fma(
      __clc_fma(__clc_fma(__clc_fma(sc6, x2, sc5), x2, sc4), x2, sc3), x2, sc2);

  __CLC_GENTYPE cp =
      t +
      __clc_fma(__clc_fma(__clc_fma(__clc_fma(__clc_fma(__clc_fma(cc6, x2, cc5),
                                                        x2, cc4),
                                              x2, cc3),
                                    x2, cc2),
                          x2, cc1),
                x2 * x2, __clc_fma(x, xx, (1.0 - t) - r));

  *sinval =
      x - __clc_fma(-x3, sc1, __clc_fma(__clc_fma(-x3, sp, 0.5 * xx), x2, -xx));
  *cosval = cp;
}

_CLC_INLINE _CLC_OVERLOAD void __clc_tan_piby4(__CLC_GENTYPE x,
                                               __CLC_GENTYPE xx,
                                               private __CLC_GENTYPE *leadval,
                                               private __CLC_GENTYPE *tailval) {
  // 0x3fe921fb54442d18
  const __CLC_GENTYPE piby4_lead = 7.85398163397448278999e-01;
  // 0x3c81a62633145c06
  const __CLC_GENTYPE piby4_tail = 3.06161699786838240164e-17;

  // In order to maintain relative precision transform using the identity:
  // tan(pi/4-x) = (1-tan(x))/(1+tan(x)) for arguments close to pi/4.
  // Similarly use tan(x-pi/4) = (tan(x)-1)/(tan(x)+1) close to -pi/4.

  __CLC_LONGN ca = x > 0.68;
  __CLC_LONGN cb = x < -0.68;
  __CLC_GENTYPE transform = ca ? 1.0 : 0.0;
  transform = cb ? -1.0 : transform;

  __CLC_GENTYPE tx = __clc_fma(-transform, x, piby4_lead) +
                     __clc_fma(-transform, xx, piby4_tail);
  __CLC_LONGN c = ca | cb;
  x = c ? tx : x;
  xx = c ? 0.0 : xx;

  // Core Remez [2,3] approximation to tan(x+xx) on the interval [0,0.68].
  __CLC_GENTYPE t1 = x;
  __CLC_GENTYPE r = __clc_fma(2.0, x * xx, x * x);

  __CLC_GENTYPE a = __clc_fma(r,
                              __clc_fma(r, 0.224044448537022097264602535574e-3,
                                        -0.229345080057565662883358588111e-1),
                              0.372379159759792203640806338901e0);

  __CLC_GENTYPE b =
      __clc_fma(r,
                __clc_fma(r,
                          __clc_fma(r, -0.232371494088563558304549252913e-3,
                                    0.260656620398645407524064091208e-1),
                          -0.515658515729031149329237816945e0),
                0.111713747927937668539901657944e1);

  __CLC_GENTYPE t2 = __clc_fma(MATH_DIVIDE(a, b), x * r, xx);

  __CLC_GENTYPE tp = t1 + t2;

  // Compute -1.0/(t1 + t2) accurately
  __CLC_GENTYPE z1 =
      __CLC_AS_GENTYPE(__CLC_AS_ULONGN(tp) & 0xffffffff00000000L);
  __CLC_GENTYPE z2 = t2 - (z1 - t1);
  __CLC_GENTYPE trec = -MATH_RECIP(tp);
  __CLC_GENTYPE trec_top =
      __CLC_AS_GENTYPE(__CLC_AS_ULONGN(trec) & 0xffffffff00000000L);

  __CLC_GENTYPE tpr = __clc_fma(
      __clc_fma(trec_top, z2, __clc_fma(trec_top, z1, 1.0)), trec, trec_top);

  __CLC_GENTYPE tpt = transform * (1.0 - MATH_DIVIDE(2.0 * tp, 1.0 + tp));
  __CLC_GENTYPE tptr = transform * (MATH_DIVIDE(2.0 * tp, tp - 1.0) - 1.0);

  *leadval = c ? tpt : tp;
  *tailval = c ? tptr : tpr;
}

#endif

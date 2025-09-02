//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

_CLC_DEF _CLC_OVERLOAD __CLC_FLOATN __clc_sinf_piby4(__CLC_FLOATN x,
                                                     __CLC_FLOATN y) {
  // Taylor series for sin(x) is x - x^3/3! + x^5/5! - x^7/7! ...
  // = x * (1 - x^2/3! + x^4/5! - x^6/7! ...
  // = x * f(w)
  // where w = x*x and f(w) = (1 - w/3! + w^2/5! - w^3/7! ...
  // We use a minimax approximation of (f(w) - 1) / w
  // because this produces an expansion in even powers of x.

  const __CLC_FLOATN c1 = -0.1666666666e0f;
  const __CLC_FLOATN c2 = 0.8333331876e-2f;
  const __CLC_FLOATN c3 = -0.198400874e-3f;
  const __CLC_FLOATN c4 = 0.272500015e-5f;
  const __CLC_FLOATN c5 = -2.5050759689e-08f; // 0xb2d72f34
  const __CLC_FLOATN c6 = 1.5896910177e-10f;  // 0x2f2ec9d3

  __CLC_FLOATN z = x * x;
  __CLC_FLOATN v = z * x;
  __CLC_FLOATN r = __clc_mad(
      z, __clc_mad(z, __clc_mad(z, __clc_mad(z, c6, c5), c4), c3), c2);
  __CLC_FLOATN ret =
      x - __clc_mad(v, -c1, __clc_mad(z, __clc_mad(y, 0.5f, -v * r), -y));

  return ret;
}

_CLC_DEF _CLC_OVERLOAD __CLC_FLOATN __clc_cosf_piby4(__CLC_FLOATN x,
                                                     __CLC_FLOATN y) {
  // Taylor series for cos(x) is 1 - x^2/2! + x^4/4! - x^6/6! ...
  // = f(w)
  // where w = x*x and f(w) = (1 - w/2! + w^2/4! - w^3/6! ...
  // We use a minimax approximation of (f(w) - 1 + w/2) / (w*w)
  // because this produces an expansion in even powers of x.

  const __CLC_FLOATN c1 = 0.416666666e-1f;
  const __CLC_FLOATN c2 = -0.138888876e-2f;
  const __CLC_FLOATN c3 = 0.248006008e-4f;
  const __CLC_FLOATN c4 = -0.2730101334e-6f;
  const __CLC_FLOATN c5 = 2.0875723372e-09f;  // 0x310f74f6
  const __CLC_FLOATN c6 = -1.1359647598e-11f; // 0xad47d74e

  __CLC_FLOATN z = x * x;
  __CLC_FLOATN r =
      z *
      __clc_mad(
          z,
          __clc_mad(z, __clc_mad(z, __clc_mad(z, __clc_mad(z, c6, c5), c4), c3),
                    c2),
          c1);

  // if |x| < 0.3
  __CLC_FLOATN qx = 0.0f;

  __CLC_INTN ix = __CLC_AS_INTN(x) & EXSIGNBIT_SP32;

  //  0.78125 > |x| >= 0.3
  __CLC_FLOATN xby4 = __CLC_AS_FLOATN(ix - 0x01000000);
  qx = ((ix >= 0x3e99999a) & (ix <= 0x3f480000)) ? xby4 : qx;

  // x > 0.78125
  qx = ix > 0x3f480000 ? 0.28125f : qx;

  __CLC_FLOATN hz = __clc_mad(z, 0.5f, -qx);
  __CLC_FLOATN a = 1.0f - qx;
  __CLC_FLOATN ret = a - (hz - __clc_mad(z, r, -x * y));
  return ret;
}

_CLC_DECL _CLC_OVERLOAD __CLC_FLOATN __clc_tanf_piby4(__CLC_FLOATN x,
                                                      __CLC_INTN regn) {
  // Core Remez [1,2] approximation to tan(x) on the interval [0,pi/4].
  __CLC_FLOATN r = x * x;

  __CLC_FLOATN a =
      __clc_mad(r, -0.0172032480471481694693109f, 0.385296071263995406715129f);

  __CLC_FLOATN b = __clc_mad(
      r,
      __clc_mad(r, 0.01844239256901656082986661f, -0.51396505478854532132342f),
      1.15588821434688393452299f);

  __CLC_FLOATN t = __clc_mad(x * r, __clc_native_divide(a, b), x);
  __CLC_FLOATN tr = -MATH_RECIP(t);

  return (regn & 1) != 0 ? tr : t;
}

_CLC_DEF _CLC_OVERLOAD void __clc_fullMulS(private __CLC_FLOATN *hi,
                                           private __CLC_FLOATN *lo,
                                           __CLC_FLOATN a, __CLC_FLOATN b,
                                           __CLC_FLOATN bh, __CLC_FLOATN bt) {
  if (__CLC_HAVE_HW_FMA32()) {
    __CLC_FLOATN ph = a * b;
    *hi = ph;
    *lo = __clc_fma(a, b, -ph);
  } else {
    __CLC_FLOATN ah = __CLC_AS_FLOATN(__CLC_AS_UINTN(a) & 0xfffff000U);
    __CLC_FLOATN at = a - ah;
    __CLC_FLOATN ph = a * b;
    __CLC_FLOATN pt = __clc_mad(
        at, bt, __clc_mad(at, bh, __clc_mad(ah, bt, __clc_mad(ah, bh, -ph))));
    *hi = ph;
    *lo = pt;
  }
}

_CLC_DEF _CLC_OVERLOAD __CLC_FLOATN __clc_removePi2S(private __CLC_FLOATN *hi,
                                                     private __CLC_FLOATN *lo,
                                                     __CLC_FLOATN x) {
  // 72 bits of pi/2
  const __CLC_FLOATN fpiby2_1 = (__CLC_FLOATN)0xC90FDA / 0x1.0p+23f;
  const __CLC_FLOATN fpiby2_1_h = (__CLC_FLOATN)0xC90 / 0x1.0p+11f;
  const __CLC_FLOATN fpiby2_1_t = (__CLC_FLOATN)0xFDA / 0x1.0p+23f;

  const __CLC_FLOATN fpiby2_2 = (__CLC_FLOATN)0xA22168 / 0x1.0p+47f;
  const __CLC_FLOATN fpiby2_2_h = (__CLC_FLOATN)0xA22 / 0x1.0p+35f;
  const __CLC_FLOATN fpiby2_2_t = (__CLC_FLOATN)0x168 / 0x1.0p+47f;

  const __CLC_FLOATN fpiby2_3 = (__CLC_FLOATN)0xC234C4 / 0x1.0p+71f;
  const __CLC_FLOATN fpiby2_3_h = (__CLC_FLOATN)0xC23 / 0x1.0p+59f;
  const __CLC_FLOATN fpiby2_3_t = (__CLC_FLOATN)0x4C4 / 0x1.0p+71f;

  const __CLC_FLOATN twobypi = 0x1.45f306p-1f;

  __CLC_FLOATN fnpi2 = __clc_trunc(__clc_mad(x, twobypi, 0.5f));

  // subtract n * pi/2 from x
  __CLC_FLOATN rhead, rtail;
  __clc_fullMulS(&rhead, &rtail, fnpi2, fpiby2_1, fpiby2_1_h, fpiby2_1_t);
  __CLC_FLOATN v = x - rhead;
  __CLC_FLOATN rem = v + (((x - v) - rhead) - rtail);

  __CLC_FLOATN rhead2, rtail2;
  __clc_fullMulS(&rhead2, &rtail2, fnpi2, fpiby2_2, fpiby2_2_h, fpiby2_2_t);
  v = rem - rhead2;
  rem = v + (((rem - v) - rhead2) - rtail2);

  __CLC_FLOATN rhead3, rtail3;
  __clc_fullMulS(&rhead3, &rtail3, fnpi2, fpiby2_3, fpiby2_3_h, fpiby2_3_t);
  v = rem - rhead3;

  *hi = v + ((rem - v) - rhead3);
  *lo = -rtail3;
  return fnpi2;
}

_CLC_DEF _CLC_OVERLOAD __CLC_INTN __clc_argReductionSmallS(
    private __CLC_FLOATN *r, private __CLC_FLOATN *rr, __CLC_FLOATN x) {
  __CLC_FLOATN fnpi2 = __clc_removePi2S(r, rr, x);
  return __CLC_CONVERT_INTN(fnpi2) & 0x3;
}

_CLC_DEF _CLC_OVERLOAD __CLC_INTN __clc_argReductionLargeS(
    private __CLC_FLOATN *r, private __CLC_FLOATN *rr, __CLC_FLOATN x) {
  __CLC_INTN xe = __CLC_AS_INTN((__CLC_AS_UINTN(x) >> 23) - 127);
  __CLC_UINTN xm = 0x00800000U | (__CLC_AS_UINTN(x) & 0x7fffffU);

  // 224 bits of 2/PI: . A2F9836E 4E441529 FC2757D1 F534DDC0 DB629599 3C439041
  // FE5163AB
  const __CLC_UINTN b6 = 0xA2F9836EU;
  const __CLC_UINTN b5 = 0x4E441529U;
  const __CLC_UINTN b4 = 0xFC2757D1U;
  const __CLC_UINTN b3 = 0xF534DDC0U;
  const __CLC_UINTN b2 = 0xDB629599U;
  const __CLC_UINTN b1 = 0x3C439041U;
  const __CLC_UINTN b0 = 0xFE5163ABU;

  __CLC_UINTN p0, p1, p2, p3, p4, p5, p6, p7, c0, c1;

  __CLC_FULL_MUL(xm, b0, c0, p0);
  __CLC_FULL_MAD(xm, b1, c0, c1, p1);
  __CLC_FULL_MAD(xm, b2, c1, c0, p2);
  __CLC_FULL_MAD(xm, b3, c0, c1, p3);
  __CLC_FULL_MAD(xm, b4, c1, c0, p4);
  __CLC_FULL_MAD(xm, b5, c0, c1, p5);
  __CLC_FULL_MAD(xm, b6, c1, p7, p6);

  __CLC_UINTN fbits = (__CLC_UINTN)224 + (__CLC_UINTN)23 - __CLC_AS_UINTN(xe);

  // shift amount to get 2 lsb of integer part at top 2 bits
  //   min: 25 (xe=18) max: 134 (xe=127)
  __CLC_UINTN shift = 256U - 2 - fbits;

  // Shift by up to 134/32 = 4 words
  __CLC_INTN c = shift > 31;
  p7 = c ? p6 : p7;
  p6 = c ? p5 : p6;
  p5 = c ? p4 : p5;
  p4 = c ? p3 : p4;
  p3 = c ? p2 : p3;
  p2 = c ? p1 : p2;
  p1 = c ? p0 : p1;
  shift -= (c ? 32U : 0U);

  c = shift > 31;
  p7 = c ? p6 : p7;
  p6 = c ? p5 : p6;
  p5 = c ? p4 : p5;
  p4 = c ? p3 : p4;
  p3 = c ? p2 : p3;
  p2 = c ? p1 : p2;
  shift -= (c ? 32U : 0U);

  c = shift > 31;
  p7 = c ? p6 : p7;
  p6 = c ? p5 : p6;
  p5 = c ? p4 : p5;
  p4 = c ? p3 : p4;
  p3 = c ? p2 : p3;
  shift -= (c ? 32U : 0U);

  c = shift > 31;
  p7 = c ? p6 : p7;
  p6 = c ? p5 : p6;
  p5 = c ? p4 : p5;
  p4 = c ? p3 : p4;
  shift -= (c ? 32U : 0U);

  // bitalign cannot handle a shift of 32
  c = shift > 0;
  shift = 32 - shift;
  __CLC_UINTN t7 = bitalign(p7, p6, shift);
  __CLC_UINTN t6 = bitalign(p6, p5, shift);
  __CLC_UINTN t5 = bitalign(p5, p4, shift);
  p7 = c ? t7 : p7;
  p6 = c ? t6 : p6;
  p5 = c ? t5 : p5;

  // Get 2 lsb of int part and msb of fraction
  __CLC_INTN i = __CLC_AS_INTN(p7 >> 29U);

  // Scoot up 2 more bits so only fraction remains
  p7 = bitalign(p7, p6, 30);
  p6 = bitalign(p6, p5, 30);
  p5 = bitalign(p5, p4, 30);

  // Subtract 1 if msb of fraction is 1, i.e. fraction >= 0.5
  __CLC_UINTN flip = (i & 1) != 0 ? 0xFFFFFFFFU : 0U;
  __CLC_UINTN sign = (i & 1) != 0 ? 0x80000000U : 0U;
  p7 = p7 ^ flip;
  p6 = p6 ^ flip;
  p5 = p5 ^ flip;

  // Find exponent and shift away leading zeroes and hidden bit
  xe = __CLC_AS_INTN(__clc_clz(p7)) + 1;
  shift = 32 - __CLC_AS_UINTN(xe);
  p7 = bitalign(p7, p6, shift);
  p6 = bitalign(p6, p5, shift);

  // Most significant part of fraction
  __CLC_FLOATN q1 =
      __CLC_AS_FLOATN(sign | ((127U - __CLC_AS_UINTN(xe)) << 23U) | p7 >> 9);

  // Shift out bits we captured on q1
  p7 = bitalign(p7, p6, 32 - 23);

  // Get 24 more bits of fraction in another float, there are not long strings
  // of zeroes here
  __CLC_INTN xxe = __CLC_AS_INTN(__clc_clz(p7)) + 1;
  p7 = bitalign(p7, p6, 32 - xxe);
  __CLC_FLOATN q0 = __CLC_AS_FLOATN(
      sign | ((127U - __CLC_AS_UINTN(xe + 23 + xxe)) << 23U) | p7 >> 9);

  // At this point, the fraction q1 + q0 is correct to at least 48 bits
  // Now we need to multiply the fraction by pi/2
  // This loses us about 4 bits
  // pi/2 = C90 FDA A22 168 C23 4C4

  const __CLC_FLOATN pio2h = (__CLC_FLOATN)0xc90fda / 0x1.0p+23f;
  const __CLC_FLOATN pio2hh = (__CLC_FLOATN)0xc90 / 0x1.0p+11f;
  const __CLC_FLOATN pio2ht = (__CLC_FLOATN)0xfda / 0x1.0p+23f;
  const __CLC_FLOATN pio2t = (__CLC_FLOATN)0xa22168 / 0x1.0p+47f;

  __CLC_FLOATN rh, rt;

  if (__CLC_HAVE_HW_FMA32()) {
    rh = q1 * pio2h;
    rt = __clc_fma(q0, pio2h, __clc_fma(q1, pio2t, __clc_fma(q1, pio2h, -rh)));
  } else {
    __CLC_FLOATN q1h = __CLC_AS_FLOATN(__CLC_AS_UINTN(q1) & 0xfffff000);
    __CLC_FLOATN q1t = q1 - q1h;
    rh = q1 * pio2h;
    rt = __clc_mad(
        q1t, pio2ht,
        __clc_mad(q1t, pio2hh,
                  __clc_mad(q1h, pio2ht, __clc_mad(q1h, pio2hh, -rh))));
    rt = __clc_mad(q0, pio2h, __clc_mad(q1, pio2t, rt));
  }

  __CLC_FLOATN t = rh + rt;
  rt = rt - (t - rh);

  *r = t;
  *rr = rt;
  return ((i >> 1) + (i & 1)) & 0x3;
}

_CLC_DEF _CLC_OVERLOAD __CLC_INTN __clc_argReductionS(private __CLC_FLOATN *r,
                                                      private __CLC_FLOATN *rr,
                                                      __CLC_FLOATN x) {
  __CLC_INTN is_small = x < (__CLC_FLOATN)0x1.0p+23f;
#ifdef __CLC_SCALAR
  if (is_small)
    return __clc_argReductionSmallS(r, rr, x);
  else
    return __clc_argReductionLargeS(r, rr, x);
#else
  __CLC_FLOATN r1, rr1, r2, rr2;
  __CLC_INTN ret1 = __clc_argReductionSmallS(&r1, &rr1, x);
  __CLC_INTN ret2 = __clc_argReductionLargeS(&r2, &rr2, x);
  *r = is_small ? r1 : r2;
  *rr = is_small ? rr1 : rr2;
  return is_small ? ret1 : ret2;
#endif
}

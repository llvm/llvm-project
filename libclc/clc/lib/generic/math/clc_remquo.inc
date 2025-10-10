//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define _sHighMask 0xfffff000u
#define _iMaxQExp 0xbu
// To prevent YLow to be denormal it should be checked
// that Exp(Y) <= -127+23 (worst case when only last bit is non zero)
//      Exp(Y) < -103 -> Y < 0x0C000000
// That value is used to construct _iYSub by setting up first bit to 1.
// _iYCmp is get from max acceptable value 0x797fffff:
//   0x797fffff - 0x8C000000 = 0x(1)ED7FFFFF
#define _iYSub 0x8C000000u
#define _iYCmp 0xED7FFFFFu
#define _iOne 0x00000001u

static _CLC_INLINE _CLC_OVERLOAD int
internal_remquo(float x, float y, float *r, __CLC_ADDRESS_SPACE uint *q) {
  uint signif_x, signif_y, rem_bit, quo_bit, tmp_x, tmp_y;
  int exp_x, exp_y, i, j;
  uint expabs_diff, special_op = 0, signbit_x, signbit_y, sign = 1;
  float result, abs_x, abs_y;
  float zero = 0.0f;
  int nRet = 0;
  // Remove sign bits
  tmp_x = ((*(int *)&x)) & EXSIGNBIT_SP32;
  tmp_y = ((*(int *)&y)) & EXSIGNBIT_SP32;
  signbit_x = (uint)((*(int *)&x) >> 31);
  signbit_y = (uint)((*(int *)&y) >> 31);
  if (signbit_x ^ signbit_y)
    sign = (-sign);
  // Get float absolute values
  abs_x = *(float *)&tmp_x;
  abs_y = *(float *)&tmp_y;
  // Remove exponent bias
  exp_x = (int)((tmp_x & (0x7F800000)) >> 23) - 127;
  exp_y = (int)((tmp_y & (0x7F800000)) >> 23) - 127;
  // Test for NaNs, Infs, and Zeros
  if ((exp_x == (0x00000080L)) || (exp_y == (0x00000080L)) ||
      (tmp_x == (0x00000000L)) || (tmp_y == (0x00000000L)))
    special_op++;
  // Get significands
  signif_x = (tmp_x & (0x007FFFFFL));
  signif_y = (tmp_y & (0x007FFFFFL));
  // Process NaNs, Infs, and Zeros
  if (special_op) {
    (*q) = 0;
    // x is NaN
    if ((signif_x != (0x00000000L)) && (exp_x == (0x00000080L)))
      result = x * 1.7f;
    // y is NaN
    else if ((signif_y != (0x00000000L)) && (exp_y == (0x00000080L)))
      result = y * 1.7f;
    // y is zero
    else if (abs_y == zero) {
      result = zero / zero;
      nRet = 1;
    }
    // x is zero
    else if (abs_x == zero)
      result = x;
    // x is Inf
    else if ((signif_x == (0x00000000L)) && (exp_x == (0x00000080L)))
      result = zero / zero;
    // y is Inf
    else
      result = x;
    (*r) = (result);
    return nRet;
  }
  // If x < y, fast return
  if (abs_x <= abs_y) {
    (*q) = 1 * sign;
    if (abs_x == abs_y) {
      (*r) = (zero * x);
      return nRet;
    }
    // Is x too big to scale up by 2.0f?
    if (exp_x != 127) {
      if ((2.0f * abs_x) <= abs_y) {
        (*q) = 0;
        (*r) = x;
        return nRet;
      }
    }
    result = abs_x - abs_y;
    if (signbit_x) {
      result = -result;
    }
    (*r) = (result);
    return nRet;
  }
  // Check for denormal x and y, adjust and normalize
  if ((exp_x == -127) && (signif_x != (0x00000000L))) {
    exp_x = -126;
    while (signif_x <= (0x007FFFFFL)) {
      exp_x--;
      signif_x <<= 1;
    };
  } else
    signif_x = (signif_x | (0x00800000L));
  if ((exp_y == -127) && (signif_y != (0x00000000L))) {
    exp_y = -126;
    while (signif_y <= (0x007FFFFFL)) {
      exp_y--;
      signif_y <<= 1;
    };
  } else
    signif_y = (signif_y | (0x00800000L));
  //
  // Main computational path
  //
  // Calculate exponent difference
  expabs_diff = (exp_x - exp_y) + 1;
  rem_bit = signif_x;
  quo_bit = 0;
  for (i = 0; i < expabs_diff; i++) {
    quo_bit = quo_bit << 1;
    if (rem_bit >= signif_y) {
      rem_bit -= signif_y;
      quo_bit++;
    }
    rem_bit <<= 1;
  }
  // Zero remquo ... return immediately with sign of x
  if (rem_bit == (0x00000000L)) {
    (*q) = ((uint)(0x7FFFFFFFL & quo_bit)) * sign;
    (*r) = (zero * x);
    return nRet;
  }
  // Adjust remquo
  rem_bit >>= 1;
  // Set exponent base, unbiased
  j = exp_y;
  // Calculate normalization shift
  while (rem_bit <= (0x007FFFFFL)) {
    j--;
    rem_bit <<= 1;
  };
  // Prepare normal results
  if (j >= -126) {
    // Remove explicit 1
    rem_bit &= (0x007FFFFFL);
    // Set final exponent ... add exponent bias
    j = j + 127;
  }
  // Prepare denormal results
  else {
    // Determine denormalization shift count
    j = -j - 126;
    // Denormalization
    rem_bit >>= j;
    // Set final exponent ... denorms are 0
    j = 0;
  }
  rem_bit = (((uint)(j)) << 23) | rem_bit;
  // Create float result and adjust if >= .5 * divisor
  result = *(float *)&rem_bit;
  if ((2.0f * result) >= abs_y) {
    if ((2.0f * result) == abs_y) {
      if (quo_bit & 0x01) {
        result = -result;
        quo_bit++;
      }
    } else {
      result = result - abs_y;
      quo_bit++;
    }
  }
  // Final adjust for sign of input
  (*q) = ((uint)(0x7FFFFFFF & (quo_bit))) * sign;
  if (signbit_x)
    result = -result;
  (*r) = (result);
  return nRet;
}

_CLC_DEF _CLC_OVERLOAD float __clc_remquo(float x, float y,
                                          __CLC_ADDRESS_SPACE int *quo) {
  float vr1;
  uint vr2;

  float sZero = __clc_as_float(0);
  /* Absolute values */
  float sAbsMask = __clc_as_float(EXSIGNBIT_SP32);
  float sXAbs = __clc_as_float(__clc_as_uint(x) & __clc_as_uint(sAbsMask));
  float sYAbs = __clc_as_float(__clc_as_uint(y) & __clc_as_uint(sAbsMask));
  uint iXExp = __clc_as_uint(sXAbs);
  uint iYExp = __clc_as_uint(sYAbs);
  uint iYSub = (_iYSub);
  uint iYCmp = (_iYCmp);
  uint iYSpec = (iYExp - iYSub);
  iYSpec = ((uint)(-(int)((int)iYSpec > (int)iYCmp)));
  iXExp = ((uint)(iXExp) >> (23));
  iYExp = ((uint)(iYExp) >> (23));
  uint iQExp = (iXExp - iYExp);
  uint iMaxQExp = (_iMaxQExp);
  uint iRangeMask = (uint)(-(int)((int)iQExp > (int)iMaxQExp));
  iRangeMask = (iRangeMask | iYSpec);
  uint vm = iRangeMask;
  if (__builtin_expect((vm) != 0, 0)) {
    internal_remquo(x, y, &vr1, &vr2);
  } else {
    /* yhi = y & highmask */
    float sHighMask = __clc_as_float(_sHighMask);
    float sYHi = __clc_as_float(__clc_as_uint(y) & __clc_as_uint(sHighMask));
    /* ylo = y - yhi */
    float sYLo = (y - sYHi);
    /* q = x/y */
    float sQ = (x / y);
    /* iq = trunc(q) */
    sQ = __clc_rint(sQ);
    uint iQ = ((int)(sQ));
    /* yhi*iq */
    float sQYHi = (sQ * sYHi);
    /* ylo*iq */
    float sQYLo = (sQ * sYLo);
    /* res = x - yhi*iq */
    float sRes = (x - sQYHi);
    /* res = res - ylo*iq */
    sRes = (sRes - sQYLo);
    /* get result's abs value and sign */
    float sSignMask = __clc_as_float(SIGNBIT_SP32);
    float sResSign =
        __clc_as_float(__clc_as_uint(sRes) & __clc_as_uint(sSignMask));
    float sAbsRes =
        __clc_as_float(__clc_as_uint(sRes) & __clc_as_uint(sAbsMask));
    float sYSign = __clc_as_float(__clc_as_uint(y) & __clc_as_uint(sSignMask));
    float sQSign =
        __clc_as_float(__clc_as_uint(sResSign) ^ __clc_as_uint(sYSign));
    float sXSign = __clc_as_float(__clc_as_uint(x) & __clc_as_uint(sSignMask));
    /* prepare integer correction term */
    uint iOne = (_iOne);
    uint iCorrValue = __clc_as_uint(sQSign);
    iCorrValue = ((int)iCorrValue >> (31));
    iCorrValue = (iCorrValue | iOne);
    /* |y|*0.5 */
    float sHalf = __clc_as_float(HALFEXPBITS_SP32);
    float sCorr = (sYAbs * sHalf);
    /* if |res| > |y|*0.5 */
    sCorr = __clc_as_float((uint)(-(int)(sAbsRes > sCorr)));
    uint iCorr = __clc_as_uint(sCorr);
    iCorr = (iCorr & iCorrValue);
    /* then res = res - sign(res)*|y| */
    sCorr = __clc_as_float(__clc_as_uint(sCorr) & __clc_as_uint(sYAbs));
    sCorr = __clc_as_float(__clc_as_uint(sCorr) | __clc_as_uint(sResSign));
    sRes = (sRes - sCorr);
    vr2 = (iQ + iCorr);
    float sZeroRes = __clc_as_float((uint)(-(int)(sRes == sZero)));
    sZeroRes = __clc_as_float(__clc_as_uint(sZeroRes) & __clc_as_uint(sXSign));
    vr1 = __clc_as_float(__clc_as_uint(sRes) | __clc_as_uint(sZeroRes));
  }

  ((__CLC_ADDRESS_SPACE uint *)quo)[0] = vr2;
  return vr1;
}

// remquo signature is special, we don't have macro for this
#define __CLC_VEC_REMQUO(TYPE, VEC_SIZE, HALF_VEC_SIZE)                        \
  _CLC_DEF _CLC_OVERLOAD TYPE##VEC_SIZE __clc_remquo(                          \
      TYPE##VEC_SIZE x, TYPE##VEC_SIZE y,                                      \
      __CLC_ADDRESS_SPACE int##VEC_SIZE *quo) {                                \
    int##HALF_VEC_SIZE lo, hi;                                                 \
    TYPE##VEC_SIZE ret;                                                        \
    ret.lo = __clc_remquo(x.lo, y.lo, &lo);                                    \
    ret.hi = __clc_remquo(x.hi, y.hi, &hi);                                    \
    (*quo).lo = lo;                                                            \
    (*quo).hi = hi;                                                            \
    return ret;                                                                \
  }

#define __CLC_VEC3_REMQUO(TYPE)                                                \
  _CLC_DEF _CLC_OVERLOAD TYPE##3 __clc_remquo(                                 \
      TYPE##3 x, TYPE##3 y, __CLC_ADDRESS_SPACE int##3 * quo) {                \
    int2 lo;                                                                   \
    int hi;                                                                    \
    TYPE##3 ret;                                                               \
    ret.s01 = __clc_remquo(x.s01, y.s01, &lo);                                 \
    ret.s2 = __clc_remquo(x.s2, y.s2, &hi);                                    \
    (*quo).s01 = lo;                                                           \
    (*quo).s2 = hi;                                                            \
    return ret;                                                                \
  }
__CLC_VEC_REMQUO(float, 2, )
__CLC_VEC3_REMQUO(float)
__CLC_VEC_REMQUO(float, 4, 2)
__CLC_VEC_REMQUO(float, 8, 4)
__CLC_VEC_REMQUO(float, 16, 8)

#ifdef cl_khr_fp64

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

_CLC_DEF _CLC_OVERLOAD double __clc_remquo(double x, double y,
                                           __CLC_ADDRESS_SPACE int *pquo) {
  ulong ux = __clc_as_ulong(x);
  ulong ax = ux & ~SIGNBIT_DP64;
  ulong xsgn = ux ^ ax;
  double dx = __clc_as_double(ax);
  int xexp = __clc_convert_int(ax >> EXPSHIFTBITS_DP64);
  int xexp1 = 11 - (int)__clc_clz(ax & MANTBITS_DP64);
  xexp1 = xexp < 1 ? xexp1 : xexp;

  ulong uy = __clc_as_ulong(y);
  ulong ay = uy & ~SIGNBIT_DP64;
  double dy = __clc_as_double(ay);
  int yexp = __clc_convert_int(ay >> EXPSHIFTBITS_DP64);
  int yexp1 = 11 - (int)__clc_clz(ay & MANTBITS_DP64);
  yexp1 = yexp < 1 ? yexp1 : yexp;

  int qsgn = ((ux ^ uy) & SIGNBIT_DP64) == 0UL ? 1 : -1;

  // First assume |x| > |y|

  // Set ntimes to the number of times we need to do a
  // partial remainder. If the exponent of x is an exact multiple
  // of 53 larger than the exponent of y, and the mantissa of x is
  // less than the mantissa of y, ntimes will be one too large
  // but it doesn't matter - it just means that we'll go round
  // the loop below one extra time.
  int ntimes = __clc_max(0, (xexp1 - yexp1) / 53);
  double w = __clc_ldexp(dy, ntimes * 53);
  w = ntimes == 0 ? dy : w;
  double scale = ntimes == 0 ? 1.0 : 0x1.0p-53;

  // Each time round the loop we compute a partial remainder.
  // This is done by subtracting a large multiple of w
  // from x each time, where w is a scaled up version of y.
  // The subtraction must be performed exactly in quad
  // precision, though the result at each stage can
  // fit exactly in a double precision number.
  int i;
  double t, v, p, pp;

  for (i = 0; i < ntimes; i++) {
    // Compute integral multiplier
    t = __clc_trunc(dx / w);

    // Compute w * t in quad precision
    p = w * t;
    pp = __clc_fma(w, t, -p);

    // Subtract w * t from dx
    v = dx - p;
    dx = v + (((dx - v) - p) - pp);

    // If t was one too large, dx will be negative. Add back one w.
    dx += dx < 0.0 ? w : 0.0;

    // Scale w down by 2^(-53) for the next iteration
    w *= scale;
  }

  // One more time
  // Variable todd says whether the integer t is odd or not
  t = __clc_floor(dx / w);
  long lt = (long)t;
  int todd = lt & 1;

  p = w * t;
  pp = __clc_fma(w, t, -p);
  v = dx - p;
  dx = v + (((dx - v) - p) - pp);
  i = dx < 0.0;
  todd ^= i;
  dx += i ? w : 0.0;

  lt -= i;

  // At this point, dx lies in the range [0,dy)

  // For the remainder function, we need to adjust dx
  // so that it lies in the range (-y/2, y/2] by carefully
  // subtracting w (== dy == y) if necessary. The rigmarole
  // with todd is to get the correct sign of the result
  // when x/y lies exactly half way between two integers,
  // when we need to choose the even integer.

  int al = (2.0 * dx > w) | (todd & (2.0 * dx == w));
  double dxl = dx - (al ? w : 0.0);

  int ag = (dx > 0.5 * w) | (todd & (dx == 0.5 * w));
  double dxg = dx - (ag ? w : 0.0);

  dx = dy < 0x1.0p+1022 ? dxl : dxg;
  lt += dy < 0x1.0p+1022 ? al : ag;
  int quo = ((int)lt & 0x7f) * qsgn;

  double ret = __clc_as_double(xsgn ^ __clc_as_ulong(dx));
  dx = __clc_as_double(ax);

  // Now handle |x| == |y|
  int c = dx == dy;
  t = __clc_as_double(xsgn);
  quo = c ? qsgn : quo;
  ret = c ? t : ret;

  // Next, handle |x| < |y|
  c = dx < dy;
  quo = c ? 0 : quo;
  ret = c ? x : ret;

  c &= (yexp < 1023 & 2.0 * dx > dy) | (dx > 0.5 * dy);
  quo = c ? qsgn : quo;
  // we could use a conversion here instead since qsgn = +-1
  p = qsgn == 1 ? -1.0 : 1.0;
  t = __clc_fma(y, p, x);
  ret = c ? t : ret;

  // We don't need anything special for |x| == 0

  // |y| is 0
  c = dy == 0.0;
  quo = c ? 0 : quo;
  ret = c ? __clc_as_double(QNANBITPATT_DP64) : ret;

  // y is +-Inf, NaN
  c = yexp > BIASEDEMAX_DP64;
  quo = c ? 0 : quo;
  t = y == y ? x : y;
  ret = c ? t : ret;

  // x is +=Inf, NaN
  c = xexp > BIASEDEMAX_DP64;
  quo = c ? 0 : quo;
  ret = c ? __clc_as_double(QNANBITPATT_DP64) : ret;

  *pquo = quo;
  return ret;
}
__CLC_VEC_REMQUO(double, 2, )
__CLC_VEC3_REMQUO(double)
__CLC_VEC_REMQUO(double, 4, 2)
__CLC_VEC_REMQUO(double, 8, 4)
__CLC_VEC_REMQUO(double, 16, 8)

#endif

#ifdef cl_khr_fp16

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

_CLC_OVERLOAD _CLC_DEF half __clc_remquo(half x, half y,
                                         __CLC_ADDRESS_SPACE int *pquo) {
  return (half)__clc_remquo((float)x, (float)y, pquo);
}
__CLC_VEC_REMQUO(half, 2, )
__CLC_VEC3_REMQUO(half)
__CLC_VEC_REMQUO(half, 4, 2)
__CLC_VEC_REMQUO(half, 8, 4)
__CLC_VEC_REMQUO(half, 16, 8)

#endif

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Computes pow using log and exp
//
//   x^y = exp(y * log(x))
//
// We take care not to lose precision in the intermediate steps.
//
// When computing log, calculate it in splits:
//
//   r = f * (p_invead + p_inv_tail)
//   r = rh + rt
//
// Calculate log polynomial using r, in end addition, do:
//
//   poly = poly + ((rh-r) + rt)
//
//   lth = -r
//   ltt = ((xexp * log2_t) - poly) + logT
//   lt = lth + ltt
//
//   lh = (xexp * log2_h) + logH
//   l = lh + lt
//
// Calculate final log answer as gh and gt:
//
//   gh = l & higher-half bits
//   gt = (((ltt - (lt - lth)) + ((lh - l) + lt)) + (l - gh))
//
//   yh = y & higher-half bits
//   yt = y - yh
//
// Before entering computation of exp:
//
//   vs = ((yt*gt + yt*gh) + yh*gt)
//   v = vs + yh*gh
//   vt = ((yh*gh - v) + vs)
//
// In calculation of exp, add vt to r that is used for poly.
//
// At the end of exp, do:
//
//   ((((expT * poly) + expT) + expH*poly) + expH)
//
//===----------------------------------------------------------------------===//

#ifdef __CLC_SCALAR

#ifdef __CLC_COMPILING_POW

_CLC_OVERLOAD _CLC_CONST static bool is_integer(__CLC_GENTYPE ay) {
  return __clc_trunc(ay) == ay;
}

_CLC_OVERLOAD _CLC_CONST static bool is_even_integer(__CLC_GENTYPE ay) {
  // Even integers are still integers after division by 2.
  return is_integer(__CLC_FP_LIT(0.5) * ay);
}

_CLC_OVERLOAD _CLC_CONST static bool is_odd_integer(__CLC_GENTYPE ay) {
  return is_integer(ay) && !is_even_integer(ay);
}
#endif

#if __CLC_FPSIZE == 32

_CLC_CONST
static __CLC_GENTYPE fast_expylnx(__CLC_GENTYPE x, __CLC_GENTYPE y) {
  __CLC_GENTYPE ax = __clc_fabs(x);
  return __clc_exp2(y * __clc_log2(ax));
}

#if defined(__CLC_COMPILING_POW) || defined(__CLC_COMPILING_POWR)

_CLC_CONST
static __CLC_GENTYPE compute_expylnx_float(__CLC_GENTYPE x, __CLC_GENTYPE y) {
  __CLC_GENTYPE ax = __clc_fabs(x);
  return __clc_ep_exp(__clc_ep_mul_overflow(y, __clc_ep_ln(ax)));
}
#endif

#if defined(__CLC_COMPILING_POW)

_CLC_CONST
static __CLC_GENTYPE pow_fixup(__CLC_GENTYPE x, __CLC_GENTYPE y,
                               __CLC_GENTYPE expylnx) {
  __CLC_GENTYPE ax = __clc_fabs(x);
  bool is_odd_y = is_odd_integer(y);

  __CLC_GENTYPE ret = __clc_copysign(expylnx, is_odd_y ? x : 1.0f);

  // Now all the edge cases
  if (x < 0.0f && !is_integer(y))
    ret = FLT_NAN;

  __CLC_GENTYPE ay = __clc_fabs(y);
  if (__clc_isinf(ay)) {
    // FIXME: Missing backend optimization to save on
    // materialization cost of mixed sign constant infinities.
    bool y_is_neg_inf = y != ay;
    ret = ax == 1.0f ? ax : ((ax < 1.0f) ^ y_is_neg_inf ? 0.0f : ay);
  }

  if (__clc_isinf(ax) || x == 0.0f)
    ret = __clc_copysign((x == 0.0f) ^ (y < 0.0f) ? 0.0f : __CLC_GENTYPE_INF,
                         is_odd_y ? x : 0.0f);

  if (__clc_isunordered(x, y))
    ret = __CLC_GENTYPE_NAN;

  return ret;
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST __CLC_GENTYPE __clc_pow(__CLC_GENTYPE x,
                                                          __CLC_GENTYPE y) {
  if (x == 1.0f)
    y = 1.0f;
  if (y == 0.0f)
    x = 1.0f;

  __CLC_GENTYPE expylnx = compute_expylnx_float(x, y);
  return pow_fixup(x, y, expylnx);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST __CLC_GENTYPE
__clc_pow_fast(__CLC_GENTYPE x, __CLC_GENTYPE y) {
  if (x == 1.0f)
    y = 1.0f;
  if (y == 0.0f)
    x = 1.0f;

  __CLC_GENTYPE expylnx = fast_expylnx(x, y);
  return pow_fixup(x, y, expylnx);
}

#elif defined(__CLC_COMPILING_POWR)

_CLC_CONST
static __CLC_GENTYPE powr_fixup(__CLC_GENTYPE x, __CLC_GENTYPE y,
                                __CLC_GENTYPE expylnx) {
  __CLC_GENTYPE ret = expylnx;

  // Now all the edge cases
  __CLC_GENTYPE iz = y < 0.0f ? __CLC_GENTYPE_INF : 0.0f;
  __CLC_GENTYPE zi = y < 0.0f ? 0.0f : __CLC_GENTYPE_INF;

  if (x == 0.0f)
    ret = y == 0.0f ? __CLC_GENTYPE_NAN : iz;

  if (x == __CLC_GENTYPE_INF && y != 0.0f)
    ret = zi;

  if (__clc_isinf(y) && x != 1.0f)
    ret = x < 1.0f ? iz : zi;

  if (__clc_isunordered(x, y))
    ret = __CLC_GENTYPE_NAN;

  return ret;
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST __CLC_GENTYPE __clc_powr(__CLC_GENTYPE x,
                                                           __CLC_GENTYPE y) {
  if (x < 0.0f)
    x = __CLC_GENTYPE_NAN;

  __CLC_GENTYPE expylnx = compute_expylnx_float(x, y);
  return powr_fixup(x, y, expylnx);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST __CLC_GENTYPE
__clc_powr_fast(__CLC_GENTYPE x, __CLC_GENTYPE y) {
  if (x < 0.0f)
    x = __CLC_GENTYPE_NAN;

  __CLC_GENTYPE expylnx = fast_expylnx(x, y);
  return powr_fixup(x, y, expylnx);
}

#elif defined(__CLC_COMPILING_POWN)

_CLC_CONST
static __CLC_GENTYPE compute_expylnx_int(__CLC_GENTYPE x, __CLC_INTN ny) {
  __CLC_GENTYPE ax = __clc_fabs(x);
  __CLC_INTN nyh = ny & 0xffff0000;
  __CLC_EP_PAIR y = __clc_ep_fast_add(__CLC_CONVERT_GENTYPE(nyh),
                                      __CLC_CONVERT_GENTYPE(ny - nyh));
  return __clc_ep_exp(__clc_ep_mul_overflow(y, __clc_ep_ln(ax)));
}

_CLC_CONST
static __CLC_GENTYPE pown_fixup(__CLC_GENTYPE x, __CLC_INTN ny,
                                __CLC_GENTYPE expylnx) {
  bool is_odd_y = ny & 1;

  __CLC_GENTYPE ret = __clc_copysign(expylnx, is_odd_y ? x : 1.0f);

  // Now all the edge cases
  if (__clc_isinf(x) || x == 0.0f)
    ret = __clc_copysign((x == 0.0f) ^ (ny < 0) ? 0.0f : __CLC_GENTYPE_INF,
                         is_odd_y ? x : 0.0f);
  return ret;
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST __CLC_GENTYPE __clc_pown(__CLC_GENTYPE x,
                                                           __CLC_INTN ny) {
  if (ny == 0)
    x = 1.0f;

  __CLC_GENTYPE expylnx = compute_expylnx_int(x, ny);
  return pown_fixup(x, ny, expylnx);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST __CLC_GENTYPE __clc_pown_fast(__CLC_GENTYPE x,
                                                                __CLC_INTN ny) {
  if (ny == 0)
    x = 1.0f;

  __CLC_GENTYPE expylnx = fast_expylnx(x, __CLC_CONVERT_GENTYPE(ny));
  return pown_fixup(x, ny, expylnx);
}

#elif defined(__CLC_COMPILING_ROOTN)

// root version of compute_expylnx_int
_CLC_CONST
static __CLC_GENTYPE compute_exp_inverse_y_lnx_int(__CLC_GENTYPE x,
                                                   __CLC_INTN ny) {
  __CLC_GENTYPE ax = __clc_fabs(x);
  __CLC_INTN nyh = ny & 0xffff0000;
  __CLC_EP_PAIR y = __clc_ep_fast_add(__CLC_CONVERT_GENTYPE(nyh),
                                      __CLC_CONVERT_GENTYPE(ny - nyh));
  y = __clc_ep_recip(y);
  return __clc_ep_exp(__clc_ep_mul_overflow(y, __clc_ep_ln(ax)));
}

_CLC_CONST
static __CLC_GENTYPE rootn_fixup(__CLC_GENTYPE x, __CLC_INTN ny,
                                 __CLC_GENTYPE expylnx) {
  bool is_odd_y = ny & 1;

  __CLC_GENTYPE ret = __clc_copysign(expylnx, is_odd_y ? x : 1.0f);

  // Now all the edge cases
  if (__clc_isinf(x) || x == 0.0f)
    ret = __clc_copysign((x == 0.0f) ^ (ny < 0) ? 0.0f : __CLC_GENTYPE_INF,
                         is_odd_y ? x : 0.0f);

  if ((x < 0.0f && !is_odd_y) || ny == 0)
    ret = __CLC_GENTYPE_NAN;

  return ret;
}

_CLC_CONST
_CLC_DEF _CLC_OVERLOAD _CLC_CONST __CLC_GENTYPE __clc_rootn(__CLC_GENTYPE x,
                                                            __CLC_INTN ny) {
  __CLC_GENTYPE expylnx = compute_exp_inverse_y_lnx_int(x, ny);
  return rootn_fixup(x, ny, expylnx);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST __CLC_GENTYPE
__clc_rootn_fast(__CLC_GENTYPE x, __CLC_INTN ny) {
  __CLC_GENTYPE y = __clc_recip_fast(__CLC_CONVERT_GENTYPE(ny));
  __CLC_GENTYPE expylnx = fast_expylnx(x, y);
  return rootn_fixup(x, ny, expylnx);
}

#else
#error missing function macro
#endif

#elif __CLC_FPSIZE == 64

#if defined(__CLC_COMPILING_POW)

_CLC_DEF _CLC_OVERLOAD _CLC_CONST __CLC_GENTYPE __clc_pow(__CLC_GENTYPE x,
                                                          __CLC_GENTYPE y) {
  if (x == 1.0)
    y = 1.0;
  if (y == 0.0)
    x = 1.0;

  __CLC_GENTYPE ax = __clc_fabs(x);
  __CLC_GENTYPE expylnx =
      __clc_ep_exp(__clc_ep_mul_overflow(y, __clc_ep_ln(ax)));

  bool is_odd_y = is_odd_integer(y);

  __CLC_GENTYPE ret = __clc_copysign(expylnx, is_odd_y ? x : 1.0);

  // Now all the edge cases
  if (x < 0.0 && !is_integer(y))
    ret = __CLC_GENTYPE_NAN;

  __CLC_GENTYPE ay = __clc_fabs(y);
  if (__clc_isinf(ay)) {
    // FIXME: Missing backend optimization to save on
    // materialization cost of mixed sign constant infinities.
    bool y_is_neg_inf = y != ay;
    ret = ax == 1.0 ? ax : ((ax < 1.0) ^ y_is_neg_inf ? 0.0 : ay);
  }

  if (__clc_isinf(ax) || x == 0.0)
    ret = __clc_copysign((x == 0.0) ^ (y < 0.0) ? 0.0 : __CLC_GENTYPE_INF,
                         is_odd_y ? x : 0.0);

  if (__clc_isunordered(x, y))
    ret = __CLC_GENTYPE_NAN;

  return ret;
}

#elif defined(__CLC_COMPILING_POWR)

_CLC_DEF _CLC_OVERLOAD _CLC_CONST __CLC_GENTYPE __clc_powr(__CLC_GENTYPE x,
                                                           __CLC_GENTYPE y) {
  if (x < 0.0)
    x = __CLC_GENTYPE_NAN;

  __CLC_GENTYPE ret = __clc_ep_exp(__clc_ep_mul_overflow(y, __clc_ep_ln(x)));

  // Now all the edge cases
  __CLC_GENTYPE iz = y < 0.0 ? __CLC_GENTYPE_INF : 0.0;
  __CLC_GENTYPE zi = y < 0.0 ? 0.0 : __CLC_GENTYPE_INF;

  if (x == 0.0)
    ret = y == 0.0 ? __CLC_GENTYPE_NAN : iz;

  if (x == __CLC_GENTYPE_INF && y != 0.0)
    ret = zi;

  if (__clc_isinf(y) && x != 1.0)
    ret = x < 1.0 ? iz : zi;

  if (y == 0.0)
    ret = x == 0.0 || __clc_isinf(x) ? __CLC_GENTYPE_NAN : 1.0;

  if (__clc_isunordered(x, y))
    ret = __CLC_GENTYPE_NAN;

  return ret;
}

#elif defined(__CLC_COMPILING_POWN)

_CLC_DEF _CLC_OVERLOAD _CLC_CONST __CLC_GENTYPE __clc_pown(__CLC_GENTYPE x,
                                                           __CLC_INTN ny) {
  if (ny == 0)
    x = 1.0;

  __CLC_GENTYPE y = __CLC_CONVERT_GENTYPE(ny);

  __CLC_GENTYPE ax = __clc_fabs(x);
  __CLC_GENTYPE expylnx =
      __clc_ep_exp(__clc_ep_mul_overflow(y, __clc_ep_ln(ax)));

  bool is_odd_y = ny & 1;

  __CLC_GENTYPE ret = __clc_copysign(expylnx, is_odd_y ? x : 1.0);

  // Now all the edge cases
  if (__clc_isinf(ax) || x == 0.0)
    ret = __clc_copysign((x == 0.0) ^ (ny < 0) ? 0.0 : __CLC_GENTYPE_INF,
                         is_odd_y ? x : 0.0);

  return ret;
}

#elif defined(__CLC_COMPILING_ROOTN)

_CLC_DEF _CLC_OVERLOAD _CLC_CONST __CLC_GENTYPE __clc_rootn(__CLC_GENTYPE x,
                                                            __CLC_INTN ny) {
  __CLC_EP_PAIR y = __clc_ep_recip(__CLC_CONVERT_GENTYPE(ny));

  __CLC_GENTYPE ax = __clc_fabs(x);
  __CLC_GENTYPE expylnx =
      __clc_ep_exp(__clc_ep_mul_overflow(y, __clc_ep_ln(ax)));

  bool is_odd_y = ny & 1;

  __CLC_GENTYPE ret = __clc_copysign(expylnx, is_odd_y ? x : 1.0);

  // Now all the edge cases
  if (__clc_isinf(ax) || x == 0.0)
    ret = __clc_copysign((x == 0.0) ^ (ny < 0) ? 0.0 : __CLC_GENTYPE_INF,
                         is_odd_y ? x : 0.0);

  if ((x < 0.0 && !is_odd_y) || ny == 0)
    ret = __CLC_GENTYPE_NAN;

  return ret;
}

#else
#error missing function macro
#endif

#elif __CLC_FPSIZE == 16

#if defined(__CLC_COMPILING_POW) || defined(__CLC_COMPILING_POWR)

_CLC_CONST
static __CLC_GENTYPE compute_expylnx_f16(__CLC_GENTYPE ax, __CLC_GENTYPE y) {
  __CLC_FLOATN x_float = __CLC_CONVERT_FLOATN(ax);
  __CLC_FLOATN y_float = __CLC_CONVERT_FLOATN(y);
  __CLC_FLOATN result = __clc_exp2_fast(y_float * __clc_log2_fast(x_float));
  return __CLC_CONVERT_GENTYPE(result);
}

#endif // defined(__CLC_COMPILING_POW) || defined(__CLC_COMPILING_POWR)

#if defined(__CLC_COMPILING_POW)

_CLC_DEF _CLC_OVERLOAD _CLC_CONST __CLC_GENTYPE __clc_pow(__CLC_GENTYPE x,
                                                          __CLC_GENTYPE y) {
  if (x == 1.0h)
    y = 1.0h;
  if (y == 0.0h)
    x = 1.0h;

  __CLC_GENTYPE ax = __clc_fabs(x);
  __CLC_GENTYPE p = compute_expylnx_f16(ax, y);

  bool is_odd_y = is_odd_integer(y);
  __CLC_GENTYPE ret = __clc_copysign(p, is_odd_y ? x : 1.0h);

  // Now all the edge cases
  if (x < 0.0h && !is_integer(y))
    ret = __CLC_GENTYPE_NAN;

  __CLC_GENTYPE ay = __clc_fabs(y);
  if (__clc_isinf(ay)) {
    // FIXME: Missing backend optimization to save on
    // materialization cost of mixed sign constant infinities.
    bool y_is_neg_inf = y != ay;
    ret = ax == 1.0h ? ax : ((ax < 1.0h) ^ y_is_neg_inf ? 0.0h : ay);
  }

  if (__clc_isinf(ax) || x == 0.0h) {
    ret = __clc_copysign((x == 0.0h) ^ (y < 0.0h) ? 0.0h : __CLC_GENTYPE_INF,
                         is_odd_y ? x : 0.0h);
  }

  if (__clc_isunordered(x, y))
    ret = __CLC_GENTYPE_NAN;

  return ret;
}

#elif defined(__CLC_COMPILING_POWR)

_CLC_DEF _CLC_OVERLOAD _CLC_CONST __CLC_GENTYPE __clc_powr(__CLC_GENTYPE x,
                                                           __CLC_GENTYPE y) {
  if (x < 0.0h)
    x = __CLC_GENTYPE_NAN;

  __CLC_GENTYPE ret = compute_expylnx_f16(x, y);

  // Now all the edge cases
  __CLC_GENTYPE iz = y < 0.0h ? __CLC_GENTYPE_INF : 0.0h;
  __CLC_GENTYPE zi = y < 0.0h ? 0.0h : __CLC_GENTYPE_INF;

  if (x == 0.0h)
    ret = y == 0.0h ? __CLC_GENTYPE_NAN : iz;

  if (x == __CLC_GENTYPE_INF && y != 0.0h)
    ret = zi;

  if (__clc_isinf(y) && x != 1.0h)
    ret = x < 1.0h ? iz : zi;

  if (__clc_isunordered(x, y))
    ret = __CLC_GENTYPE_NAN;

  return ret;
}

#elif defined(__CLC_COMPILING_POWN)

_CLC_DEF _CLC_OVERLOAD _CLC_CONST __CLC_GENTYPE __clc_pown(__CLC_GENTYPE x,
                                                           __CLC_INTN ny) {
  if (ny == 0)
    x = 1.0h;

  __CLC_GENTYPE ax = __clc_fabs(x);
  __CLC_FLOATN fy = __CLC_CONVERT_FLOATN(ny);
  __CLC_FLOATN p =
      __clc_exp2_fast(fy * __clc_log2_fast(__CLC_CONVERT_FLOATN(ax)));

  bool is_odd_y = ny & 1;

  __CLC_GENTYPE ret =
      __clc_copysign(__CLC_CONVERT_GENTYPE(p), is_odd_y ? x : 1.0h);

  // Now all the edge cases
  if (__clc_isinf(ax) || x == 0.0h)
    ret = __clc_copysign((x == 0.0h) ^ (ny < 0) ? 0.0h : __CLC_GENTYPE_INF,
                         is_odd_y ? x : 0.0h);

  return ret;
}

#elif defined(__CLC_COMPILING_ROOTN)

_CLC_DEF _CLC_OVERLOAD _CLC_CONST __CLC_GENTYPE __clc_rootn(__CLC_GENTYPE x,
                                                            __CLC_INTN ny) {
  __CLC_GENTYPE ax = __clc_fabs(x);

  __CLC_FLOATN fy = __clc_recip_fast(__CLC_CONVERT_FLOATN(ny));

  __CLC_FLOATN p =
      __clc_exp2_fast(fy * __clc_log2_fast(__CLC_CONVERT_FLOATN(ax)));

  bool is_odd_y = ny & 1;

  __CLC_GENTYPE ret =
      __clc_copysign(__CLC_CONVERT_GENTYPE(p), is_odd_y ? x : 1.0h);

  // Now all the edge cases
  if (__clc_isinf(ax) || x == 0.0h)
    ret = __clc_copysign((x == 0.0h) ^ (ny < 0) ? 0.0h : __CLC_GENTYPE_INF,
                         is_odd_y ? x : 0.0h);

  if ((x < 0.0h && !is_odd_y) || ny == 0)
    ret = __CLC_GENTYPE_NAN;

  return ret;
}

#else
#error missing function macro
#endif

#endif
#endif // __CLC_SCALAR

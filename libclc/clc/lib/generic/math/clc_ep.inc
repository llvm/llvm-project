//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if __CLC_FPSIZE == 16
#define __CLC_EP_USE_FMA true
#elif __CLC_FPSIZE == 32
#if defined(FP_FAST_FMAF)
#define __CLC_EP_USE_FMA true
#else
#define __CLC_EP_USE_FMA false
#endif

#elif __CLC_FPSIZE == 64
#if defined(FP_FAST_FMA)
#define __CLC_EP_USE_FMA true
#else
#define __CLC_EP_USE_FMA false
#endif
#endif

#pragma OPENCL FP_CONTRACT OFF

#if __CLC_FPSIZE == 16
_CLC_DEF _CLC_OVERLOAD _CLC_CONST static __CLC_GENTYPE
ep_high_fp_bits(__CLC_GENTYPE x) {
  return __CLC_AS_GENTYPE(__CLC_CONVERT_U_GENTYPE(
      (__CLC_AS_U_GENTYPE(x) & (__CLC_U_GENTYPE)0xffc0U)));
}

#elif __CLC_FPSIZE == 32

_CLC_DEF _CLC_OVERLOAD _CLC_CONST static __CLC_GENTYPE
ep_high_fp_bits(__CLC_GENTYPE x) {
  return __CLC_AS_GENTYPE(__CLC_AS_U_GENTYPE(x) & 0xfffff000U);
}

#elif __CLC_FPSIZE == 64

_CLC_DEF _CLC_OVERLOAD _CLC_CONST static __CLC_GENTYPE
ep_high_fp_bits(__CLC_GENTYPE x) {
  return __CLC_AS_GENTYPE(__CLC_AS_U_GENTYPE(x) & 0xfffffffff8000000UL);
}

#endif

typedef struct __CLC_XCONCAT(__clc_ep_int_pair_, __CLC_GENTYPE) {
  __CLC_INTN lo, hi;
} __CLC_XCONCAT(__clc_ep_int_pair_, __CLC_GENTYPE);

#define __CLC_EP_INT_PAIR __CLC_XCONCAT(__clc_ep_int_pair_, __CLC_GENTYPE)

_CLC_OVERLOAD
static __CLC_S_GENTYPE samesign(__CLC_GENTYPE x, __CLC_GENTYPE y) {
  return __clc_signbit(x) == __clc_signbit(y);
}

_CLC_DEF
_CLC_OVERLOAD _CLC_CONST __CLC_EP_PAIR __clc_ep_absv(__CLC_EP_PAIR a) {
  __CLC_EP_PAIR ret;
  __CLC_EP_PAIR neg_a = __clc_ep_neg(a);
  ret.lo = __clc_signbit(a.hi) ? neg_a.lo : a.lo;
  ret.hi = __clc_signbit(a.hi) ? neg_a.hi : a.hi;
  return ret;
}

_CLC_DEF
_CLC_OVERLOAD _CLC_CONST __CLC_EP_PAIR __clc_ep_neg(__CLC_EP_PAIR a) {
  a.lo = -a.lo;
  a.hi = -a.hi;
  return a;
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST __CLC_EP_PAIR
__clc_ep_conditional_sign_match(__CLC_EP_PAIR a, __CLC_EP_PAIR b) {
  __CLC_S_GENTYPE ss = samesign(a.hi, b.hi);
  __CLC_EP_PAIR neg_a = __clc_ep_neg(a);

  __CLC_EP_PAIR ret = {ss ? a.lo : neg_a.lo, ss ? a.hi : neg_a.hi};
  return ret;
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST __CLC_EP_PAIR
__clc_ep_make_pair(__CLC_GENTYPE a, __CLC_GENTYPE b) {
  __CLC_EP_PAIR ret = {b, a};
  return ret;
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST __CLC_EP_PAIR
__clc_ep_fast_add(__CLC_GENTYPE a, __CLC_GENTYPE b) {
  __CLC_GENTYPE s = a + b;
  return __clc_ep_make_pair(s, b - (s - a));
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST __CLC_EP_PAIR
__clc_ep_normalize(__CLC_EP_PAIR a) {
  return __clc_ep_fast_add(a.hi, a.lo);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST __CLC_EP_PAIR
__clc_ep_normalize_overflow(__CLC_EP_PAIR a) {
  __CLC_GENTYPE s = a.hi + a.lo;
  __CLC_GENTYPE t = a.lo - (s - a.hi);
  s = __clc_isinf(a.hi) ? a.hi : s;
  return __clc_ep_make_pair(s, __clc_isinf(s) ? __CLC_FP_LIT(0.0) : t);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST __CLC_EP_PAIR
__clc_ep_fast_sub(__CLC_GENTYPE a, __CLC_GENTYPE b) {
  __CLC_GENTYPE d = a - b;
  return __clc_ep_make_pair(d, (a - d) - b);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST __CLC_EP_PAIR __clc_ep_add(__CLC_GENTYPE a,
                                                             __CLC_GENTYPE b) {
  __CLC_GENTYPE s = a + b;
  __CLC_GENTYPE d = s - a;
  return __clc_ep_make_pair(s, (a - (s - d)) + (b - d));
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST __CLC_EP_PAIR __clc_ep_sub(__CLC_GENTYPE a,
                                                             __CLC_GENTYPE b) {
  __CLC_GENTYPE d = a - b;
  __CLC_GENTYPE e = d - a;
  return __clc_ep_make_pair(d, (a - (d - e)) - (b + e));
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST __CLC_EP_PAIR __clc_ep_mul(__CLC_GENTYPE a,
                                                             __CLC_GENTYPE b) {
  __CLC_GENTYPE p = a * b;
  if (__CLC_EP_USE_FMA) {
    return __clc_ep_make_pair(p, __clc_fma(a, b, -p));
  }

  __CLC_GENTYPE ah = ep_high_fp_bits(a);
  __CLC_GENTYPE al = a - ah;
  __CLC_GENTYPE bh = ep_high_fp_bits(b);
  __CLC_GENTYPE bl = b - bh;
  return __clc_ep_make_pair(p, ((ah * bh - p) + ah * bl + al * bh) + al * bl);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST __CLC_EP_PAIR __clc_ep_sqr(__CLC_GENTYPE a) {
  __CLC_GENTYPE p = a * a;
  if (__CLC_EP_USE_FMA)
    return __clc_ep_make_pair(p, __clc_fma(a, a, -p));

  __CLC_GENTYPE ah = ep_high_fp_bits(a);
  __CLC_GENTYPE al = a - ah;
  return __clc_ep_make_pair(p, ((ah * ah - p) + __CLC_FP_LIT(2.0) * ah * al) +
                                   al * al);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST __CLC_EP_PAIR __clc_ep_add(__CLC_EP_PAIR a,
                                                             __CLC_GENTYPE b) {
  __CLC_EP_PAIR s = __clc_ep_add(a.hi, b);
  s.lo += a.lo;
  return __clc_ep_normalize(s);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST __CLC_EP_PAIR
__clc_ep_fast_add(__CLC_EP_PAIR a, __CLC_GENTYPE b) {
  __CLC_EP_PAIR s = __clc_ep_fast_add(a.hi, b);
  s.lo += a.lo;
  return __clc_ep_normalize(s);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST __CLC_EP_PAIR __clc_ep_add(__CLC_GENTYPE a,
                                                             __CLC_EP_PAIR b) {
  __CLC_EP_PAIR s = __clc_ep_add(a, b.hi);
  s.lo += b.lo;
  return __clc_ep_normalize(s);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST __CLC_EP_PAIR
__clc_ep_fast_add(__CLC_GENTYPE a, __CLC_EP_PAIR b) {
  __CLC_EP_PAIR s = __clc_ep_fast_add(a, b.hi);
  s.lo += b.lo;
  return __clc_ep_normalize(s);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST __CLC_EP_PAIR __clc_ep_add(__CLC_EP_PAIR a,
                                                             __CLC_EP_PAIR b) {
  __CLC_EP_PAIR s = __clc_ep_add(a.hi, b.hi);
  __CLC_EP_PAIR t = __clc_ep_add(a.lo, b.lo);
  s.lo += t.hi;
  s = __clc_ep_normalize(s);
  s.lo += t.lo;
  return __clc_ep_normalize(s);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST __CLC_EP_PAIR
__clc_ep_fast_add(__CLC_EP_PAIR a, __CLC_EP_PAIR b) {
  __CLC_EP_PAIR s = __clc_ep_fast_add(a.hi, b.hi);
  s.lo += a.lo + b.lo;
  return __clc_ep_normalize(s);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST __CLC_EP_PAIR __clc_ep_sub(__CLC_EP_PAIR a,
                                                             __CLC_GENTYPE b) {
  __CLC_EP_PAIR d = __clc_ep_sub(a.hi, b);
  d.lo += a.lo;
  return __clc_ep_normalize(d);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST __CLC_EP_PAIR
__clc_ep_fast_sub(__CLC_EP_PAIR a, __CLC_GENTYPE b) {
  __CLC_EP_PAIR d = __clc_ep_fast_sub(a.hi, b);
  d.lo += a.lo;
  return __clc_ep_normalize(d);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST __CLC_EP_PAIR __clc_ep_sub(__CLC_GENTYPE a,
                                                             __CLC_EP_PAIR b) {
  __CLC_EP_PAIR d = __clc_ep_sub(a, b.hi);
  d.lo -= b.lo;
  return __clc_ep_normalize(d);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST __CLC_EP_PAIR
__clc_ep_fast_sub(__CLC_GENTYPE a, __CLC_EP_PAIR b) {
  __CLC_EP_PAIR d = __clc_ep_fast_sub(a, b.hi);
  d.lo -= b.lo;
  return __clc_ep_normalize(d);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST __CLC_EP_PAIR __clc_ep_sub(__CLC_EP_PAIR a,
                                                             __CLC_EP_PAIR b) {
  __CLC_EP_PAIR d = __clc_ep_sub(a.hi, b.hi);
  __CLC_EP_PAIR e = __clc_ep_sub(a.lo, b.lo);
  d.lo += e.hi;
  d = __clc_ep_normalize(d);
  d.lo += e.lo;
  return __clc_ep_normalize(d);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST __CLC_EP_PAIR
__clc_ep_fast_sub(__CLC_EP_PAIR a, __CLC_EP_PAIR b) {
  __CLC_EP_PAIR d = __clc_ep_fast_sub(a.hi, b.hi);
  d.lo = d.lo + a.lo - b.lo;
  return __clc_ep_normalize(d);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST __CLC_EP_PAIR __clc_ep_ldexp(__CLC_EP_PAIR a,
                                                               __CLC_INTN e) {
  return __clc_ep_make_pair(__clc_ldexp(a.hi, e), __clc_ldexp(a.lo, e));
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST __CLC_EP_PAIR __clc_ep_mul(__CLC_EP_PAIR a,
                                                             __CLC_GENTYPE b) {
  __CLC_EP_PAIR p = __clc_ep_mul(a.hi, b);
  if (__CLC_EP_USE_FMA) {
    p.lo = __clc_fma(a.lo, b, p.lo);
  } else {
    p.lo += a.lo * b;
  }
  return __clc_ep_normalize(p);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST __CLC_EP_PAIR
__clc_ep_mul_overflow(__CLC_EP_PAIR a, __CLC_GENTYPE b) {
  __CLC_EP_PAIR p = __clc_ep_mul(a.hi, b);
  if (__CLC_EP_USE_FMA) {
    p.lo = __clc_fma(a.lo, b, p.lo);
  } else {
    p.lo += a.lo * b;
  }
  return __clc_ep_normalize_overflow(p);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST __CLC_EP_PAIR __clc_ep_mul(__CLC_GENTYPE a,
                                                             __CLC_EP_PAIR b) {
  __CLC_EP_PAIR p = __clc_ep_mul(a, b.hi);
  if (__CLC_EP_USE_FMA) {
    p.lo = __clc_fma(a, b.lo, p.lo);
  } else {
    p.lo += a * b.lo;
  }
  return __clc_ep_normalize(p);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST __CLC_EP_PAIR
__clc_ep_mul_overflow(__CLC_GENTYPE a, __CLC_EP_PAIR b) {
  __CLC_EP_PAIR p = __clc_ep_mul(a, b.hi);
  if (__CLC_EP_USE_FMA) {
    p.lo = __clc_fma(a, b.lo, p.lo);
  } else {
    p.lo += a * b.lo;
  }
  return __clc_ep_normalize_overflow(p);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST __CLC_EP_PAIR __clc_ep_mul(__CLC_EP_PAIR a,
                                                             __CLC_EP_PAIR b) {
  __CLC_EP_PAIR p = __clc_ep_mul(a.hi, b.hi);
  if (__CLC_EP_USE_FMA) {
    p.lo = __clc_fma(a.lo, b.hi, __clc_fma(a.hi, b.lo, p.lo));
  } else {
    p.lo += a.hi * b.lo + a.lo * b.hi;
  }
  return __clc_ep_normalize(p);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST __CLC_EP_PAIR
__clc_ep_mul_overflow(__CLC_EP_PAIR a, __CLC_EP_PAIR b) {
  __CLC_EP_PAIR p = __clc_ep_mul(a.hi, b.hi);
  if (__CLC_EP_USE_FMA) {
    p.lo += __clc_fma(a.hi, b.lo, a.lo * b.hi);
  } else {
    p.lo += a.hi * b.lo + a.lo * b.hi;
  }
  return __clc_ep_normalize_overflow(p);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST __CLC_EP_PAIR __clc_ep_div(__CLC_GENTYPE a,
                                                             __CLC_GENTYPE b) {
  __CLC_GENTYPE r = __clc_recip_fast(b);
  __CLC_GENTYPE qhi = a * r;
  __CLC_EP_PAIR p = __clc_ep_mul(qhi, b);
  __CLC_EP_PAIR d = __clc_ep_fast_sub(a, p.hi);
  d.lo -= p.lo;
  __CLC_GENTYPE qlo = (d.hi + d.lo) * r;
  return __clc_ep_fast_add(qhi, qlo);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST __CLC_EP_PAIR __clc_ep_div(__CLC_EP_PAIR a,
                                                             __CLC_GENTYPE b) {
  __CLC_GENTYPE r = __clc_recip_fast(b);
  __CLC_GENTYPE qhi = a.hi * r;
  __CLC_EP_PAIR p = __clc_ep_mul(qhi, b);
  __CLC_EP_PAIR d = __clc_ep_fast_sub(a.hi, p.hi);
  d.lo = d.lo + a.lo - p.lo;
  __CLC_GENTYPE qlo = (d.hi + d.lo) * r;
  return __clc_ep_fast_add(qhi, qlo);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST __CLC_EP_PAIR __clc_ep_div(__CLC_GENTYPE a,
                                                             __CLC_EP_PAIR b) {
  __CLC_GENTYPE r = __clc_recip_fast(b.hi);
  __CLC_GENTYPE qhi = a * r;
  __CLC_EP_PAIR p = __clc_ep_mul(qhi, b);
  __CLC_EP_PAIR d = __clc_ep_fast_sub(a, p.hi);
  d.lo -= p.lo;
  __CLC_GENTYPE qlo = (d.hi + d.lo) * r;
  return __clc_ep_fast_add(qhi, qlo);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST __CLC_EP_PAIR
__clc_ep_fast_div(__CLC_EP_PAIR a, __CLC_EP_PAIR b) {
  __CLC_GENTYPE r = __clc_recip_fast(b.hi);
  __CLC_GENTYPE qhi = a.hi * r;
  __CLC_EP_PAIR p = __clc_ep_mul(qhi, b);
  __CLC_EP_PAIR d = __clc_ep_fast_sub(a.hi, p.hi);
  d.lo = d.lo - p.lo + a.lo;
  __CLC_GENTYPE qlo = (d.hi + d.lo) * r;
  return __clc_ep_fast_add(qhi, qlo);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST __CLC_EP_PAIR __clc_ep_div(__CLC_EP_PAIR a,
                                                             __CLC_EP_PAIR b) {
  __CLC_GENTYPE y = __clc_recip_fast(b.hi);
  __CLC_GENTYPE qhi = a.hi * y;
  __CLC_EP_PAIR r = __clc_ep_fast_sub(a, __clc_ep_mul(qhi, b));
  __CLC_GENTYPE qmi = r.hi * y;
  r = __clc_ep_fast_sub(r, __clc_ep_mul(qmi, b));
  __CLC_GENTYPE qlo = r.hi * y;
  __CLC_EP_PAIR q = __clc_ep_fast_add(qhi, qmi);
  q.lo += qlo;
  return __clc_ep_normalize(q);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST __CLC_EP_PAIR
__clc_ep_recip(__CLC_GENTYPE b) {
  __CLC_GENTYPE qhi = __clc_recip_fast(b);
  __CLC_EP_PAIR p = __clc_ep_mul(qhi, b);
  __CLC_EP_PAIR d = __clc_ep_fast_sub(__CLC_FP_LIT(1.0), p.hi);
  d.lo -= p.lo;
  __CLC_GENTYPE qlo = (d.hi + d.lo) * qhi;
  return __clc_ep_fast_add(qhi, qlo);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST __CLC_EP_PAIR
__clc_ep_fast_recip(__CLC_EP_PAIR b) {
  __CLC_GENTYPE qhi = __clc_recip_fast(b.hi);
  __CLC_EP_PAIR p = __clc_ep_mul(qhi, b);
  __CLC_EP_PAIR d = __clc_ep_fast_sub(__CLC_FP_LIT(1.0), p.hi);
  d.lo -= p.lo;
  __CLC_GENTYPE qlo = (d.hi + d.lo) * qhi;
  return __clc_ep_fast_add(qhi, qlo);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST __CLC_EP_PAIR
__clc_ep_recip(__CLC_EP_PAIR b) {
  __CLC_GENTYPE qhi = __clc_recip_fast(b.hi);
  __CLC_EP_PAIR r = __clc_ep_fast_sub(__CLC_FP_LIT(1.0), __clc_ep_mul(qhi, b));
  __CLC_GENTYPE qmi = r.hi * qhi;
  r = __clc_ep_fast_sub(r, __clc_ep_mul(qmi, b));
  __CLC_GENTYPE qlo = r.hi * qhi;
  __CLC_EP_PAIR q = __clc_ep_fast_add(qhi, qmi);
  q.lo += qlo;
  return __clc_ep_normalize(q);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST __CLC_EP_PAIR __clc_ep_sqr(__CLC_EP_PAIR a) {
  __CLC_EP_PAIR p = __clc_ep_sqr(a.hi);
  if (__CLC_EP_USE_FMA) {
    p.lo = __clc_fma(a.hi, __CLC_FP_LIT(2.0) * a.lo, p.lo);
  } else {
    p.lo = p.lo + __CLC_FP_LIT(2.0) * a.lo * a.hi;
  }
  return __clc_ep_normalize(p);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST __CLC_EP_PAIR __clc_ep_sqrt(__CLC_GENTYPE a) {
  __CLC_GENTYPE shi = __clc_sqrt_fast(a);
  __CLC_EP_PAIR e = __clc_ep_fast_sub(a, __clc_ep_sqr(shi));
  __CLC_GENTYPE slo = __clc_div_fast(e.hi, __CLC_FP_LIT(2.0) * shi);
  return __clc_ep_fast_add(shi,
                           a == __CLC_FP_LIT(0.0) ? __CLC_FP_LIT(0.0) : slo);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST __CLC_EP_PAIR __clc_ep_sqrt(__CLC_EP_PAIR a) {
  __CLC_GENTYPE shi = __clc_sqrt_fast(a.hi);
  __CLC_EP_PAIR e = __clc_ep_fast_sub(a, __clc_ep_sqr(shi));
  __CLC_GENTYPE slo = __clc_div_fast(e.hi, __CLC_FP_LIT(2.0) * shi);
  return __clc_ep_fast_add(shi,
                           a.hi == __CLC_FP_LIT(0.0) ? __CLC_FP_LIT(0.0) : slo);
}

#if __CLC_FPSIZE == 32

_CLC_DEF _CLC_OVERLOAD _CLC_CONST __CLC_GENTYPE __clc_ep_exp(__CLC_EP_PAIR x) {
  __CLC_GENTYPE d = x.hi == 0x1.62e430p+6f ? 0x1.0p-17f : 0.0f;
  x.hi -= d;
  x.lo += d;
  __CLC_GENTYPE z = __clc_exp(x.hi);
  __CLC_GENTYPE zz = __clc_fma(z, x.lo, z);
  return __clc_isinf(z) ? z : zz;
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST __CLC_EP_PAIR
__clc_ep_exp_extended(__CLC_EP_PAIR x) {
  const __CLC_FLOATN LOG2E = 0x1.715476p+0f;
  const __CLC_FLOATN LN2_HI = 0x1.62e400p-1f;
  const __CLC_FLOATN LN2_MED = 0x1.7f7800p-20f;
  const __CLC_FLOATN LN2_LO = 0x1.473de6p-34f;

  const __CLC_FLOATN POLY_C1 = 0x1.6850e4p-10f;
  const __CLC_FLOATN POLY_C2 = 0x1.123bccp-7f;
  const __CLC_FLOATN POLY_C3 = 0x1.555b98p-5f;
  const __CLC_FLOATN POLY_C4 = 0x1.55548ep-3f;
  const __CLC_FLOATN POLY_C5 = 0x1.fffff8p-2f;

  __CLC_FLOATN fn = __clc_rint(x.hi * LOG2E);

  __CLC_EP_PAIR term_hi = __clc_ep_fast_add(__clc_mad(fn, -LN2_HI, x.hi), x.lo);
  __CLC_EP_PAIR term_med = __clc_ep_fast_sub(term_hi, fn * LN2_MED);
  __CLC_EP_PAIR t = __clc_ep_fast_sub(term_med, fn * LN2_LO);

  __CLC_FLOATN th = t.hi;

  __CLC_FLOATN poly1 = __clc_mad(th, POLY_C1, POLY_C2);
  __CLC_FLOATN poly2 = __clc_mad(th, poly1, POLY_C3);
  __CLC_FLOATN poly3 = __clc_mad(th, poly2, POLY_C4);
  __CLC_FLOATN p = __clc_mad(th, poly3, POLY_C5);

  __CLC_EP_PAIR sqr_t = __clc_ep_sqr(t);
  __CLC_EP_PAIR poly_term = __clc_ep_mul(sqr_t, p);
  __CLC_EP_PAIR inner_sum = __clc_ep_fast_add(t, poly_term);
  __CLC_EP_PAIR r = __clc_ep_fast_add(1.0f, inner_sum);

  return __clc_ep_ldexp(r, __CLC_CONVERT_INTN(fn));
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST __CLC_EP_PAIR __clc_ep_ln(__CLC_GENTYPE a) {
  __CLC_INTN a_exp;
  __CLC_GENTYPE m = __clc_frexp(a, &a_exp);
  __CLC_INTN b = m < (2.0f / 3.0f);
  m = __clc_ldexp(m, b);
  __CLC_INTN e = a_exp - b;

  __CLC_EP_PAIR x = __clc_ep_div(m - 1.0f, __clc_ep_fast_add(1.0f, m));
  __CLC_EP_PAIR s = __clc_ep_sqr(x);
  __CLC_GENTYPE t = s.hi;
  __CLC_GENTYPE p = __clc_mad(t, __clc_mad(t, 0x1.ed89c2p-3f, 0x1.23e988p-2f),
                              0x1.999bdep-2f);

  // ln(2)*e + 2*x + x^3(c3 + x^2*p)
  __CLC_EP_PAIR r = __clc_ep_add(
      __clc_ep_mul(__clc_ep_make_pair(__CLC_FP_LIT(0x1.62e430p-1),
                                      __CLC_FP_LIT(-0x1.05c610p-29)),
                   __CLC_CONVERT_GENTYPE(e)),
      __clc_ep_fast_add(
          __clc_ep_ldexp(x, 1),
          __clc_ep_mul(__clc_ep_mul(s, x),
                       __clc_ep_fast_add(
                           __clc_ep_make_pair(__CLC_FP_LIT(0x1.555554p-1),
                                              __CLC_FP_LIT(0x1.e72020p-29)),
                           __clc_ep_mul(s, p)))));
  return r;
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST __CLC_FLOATN __clc_ep_ln_hi(__CLC_EP_PAIR a,
                                                              __CLC_INTN ea) {
  __CLC_INTN a_hi_exp;
  __CLC_FLOATN a_hi_m = __clc_frexp(a.hi, &a_hi_exp);
  __CLC_INTN b = a_hi_m < (2.0f / 3.0f);
  __CLC_INTN e = a_hi_exp - b;
  __CLC_EP_PAIR m = __clc_ep_ldexp(a, -e);
  __CLC_EP_PAIR x =
      __clc_ep_div(__clc_ep_fast_add(-1.0f, m), __clc_ep_fast_add(1.0f, m));
  __CLC_FLOATN s = x.hi * x.hi;
  __CLC_FLOATN p = __clc_mad(s, __clc_mad(s, 0x1.36db58p-2f, 0x1.992b46p-2f),
                             0x1.5555b4p-1f);

  const __CLC_EP_PAIR ln2 = __clc_ep_make_pair(__CLC_FP_LIT(0x1.62e430p-1),
                                               __CLC_FP_LIT(-0x1.05c610p-29));

  __CLC_EP_PAIR r =
      __clc_ep_add(__clc_ep_mul(ln2, __CLC_CONVERT_FLOATN(e + ea)),
                   __clc_ep_fast_add(__clc_ep_ldexp(x, 1), s * x.hi * p));
  return r.hi;
}

#elif __CLC_FPSIZE == 64

_CLC_DEF _CLC_OVERLOAD _CLC_CONST __CLC_GENTYPE __clc_ep_exp(__CLC_EP_PAIR x) {
  __CLC_GENTYPE z = __clc_exp(x.hi);
  __CLC_GENTYPE zz = __clc_mad(z, x.lo, z);
  return __clc_isinf(z) ? z : zz;
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST __CLC_EP_PAIR
__clc_ep_exp_extended(__CLC_EP_PAIR x) {
  const __CLC_DOUBLEN LOG2E = 0x1.71547652b82fep+0;
  const __CLC_DOUBLEN LN2_HI = 0x1.62e42fefa3000p-1;
  const __CLC_DOUBLEN LN2_MED = 0x1.3de6af278e000p-42;
  const __CLC_DOUBLEN LN2_LO = 0x1.9cc01f97b57a0p-83;

  const __CLC_DOUBLEN POLY_C1 = 0x1.ade156a5dcb37p-26;
  const __CLC_DOUBLEN POLY_C2 = 0x1.28af3fca7ab0cp-22;
  const __CLC_DOUBLEN POLY_C3 = 0x1.71dee623fde64p-19;
  const __CLC_DOUBLEN POLY_C4 = 0x1.a01997c89e6b0p-16;
  const __CLC_DOUBLEN POLY_C5 = 0x1.a01a014761f6ep-13;
  const __CLC_DOUBLEN POLY_C6 = 0x1.6c16c1852b7b0p-10;
  const __CLC_DOUBLEN POLY_C7 = 0x1.1111111122322p-7;
  const __CLC_DOUBLEN POLY_C8 = 0x1.55555555502a1p-5;
  const __CLC_DOUBLEN POLY_C9 = 0x1.5555555555511p-3;
  const __CLC_DOUBLEN POLY_C10 = 0x1.000000000000bp-1;

  __CLC_DOUBLEN dn = __clc_rint(x.hi * LOG2E);

  __CLC_EP_PAIR term_hi = __clc_ep_fast_add(__clc_mad(dn, -LN2_HI, x.hi), x.lo);
  __CLC_EP_PAIR term_med = __clc_ep_fast_sub(term_hi, dn * LN2_MED);
  __CLC_EP_PAIR t = __clc_ep_fast_sub(term_med, dn * LN2_LO);

  __CLC_DOUBLEN th = t.hi;

  __CLC_DOUBLEN poly1 = __clc_mad(th, POLY_C1, POLY_C2);
  __CLC_DOUBLEN poly2 = __clc_mad(th, poly1, POLY_C3);
  __CLC_DOUBLEN poly3 = __clc_mad(th, poly2, POLY_C4);
  __CLC_DOUBLEN poly4 = __clc_mad(th, poly3, POLY_C5);
  __CLC_DOUBLEN poly5 = __clc_mad(th, poly4, POLY_C6);
  __CLC_DOUBLEN poly6 = __clc_mad(th, poly5, POLY_C7);
  __CLC_DOUBLEN poly7 = __clc_mad(th, poly6, POLY_C8);
  __CLC_DOUBLEN poly8 = __clc_mad(th, poly7, POLY_C9);
  __CLC_DOUBLEN p = __clc_mad(th, poly8, POLY_C10);

  __CLC_EP_PAIR sqr_t = __clc_ep_sqr(t);
  __CLC_EP_PAIR poly_term = __clc_ep_mul(sqr_t, p);
  __CLC_EP_PAIR inner_sum = __clc_ep_fast_add(t, poly_term);
  __CLC_EP_PAIR r = __clc_ep_fast_add(1.0, inner_sum);

  return __clc_ep_ldexp(r, __CLC_CONVERT_INTN(dn));
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST __CLC_EP_PAIR __clc_ep_ln(__CLC_DOUBLEN a) {
  __CLC_INTN a_exp;
  __CLC_DOUBLEN m = __clc_frexp(a, &a_exp);
  __CLC_INTN b = __CLC_CONVERT_INTN(m < __CLC_FP_LIT(2.0 / 3.0));
  m = __clc_ldexp(m, b);
  __CLC_INTN e = a_exp - b;

  __CLC_EP_PAIR x = __clc_ep_div(m - 1.0, __clc_ep_fast_add(1.0, m));
  __CLC_EP_PAIR s = __clc_ep_sqr(x);
  __CLC_DOUBLEN t = s.hi;
  __CLC_DOUBLEN p = __clc_mad(t, __clc_mad(t, __clc_mad(t, __clc_mad(t,
             __clc_mad(t, __clc_mad(t, __clc_mad(t, __clc_mad(t,
                 0x1.dee674222de17p-4, 0x1.a6564968915a9p-4), 0x1.e25e43abe935ap-4), 0x1.110ef47e6c9c2p-3),
                 0x1.3b13bcfa74449p-3), 0x1.745d171bf3c30p-3), 0x1.c71c71c7792cep-3), 0x1.24924924920dap-2),
                 0x1.999999999999cp-2);

  // ln(2)*e + 2*x + x^3(c3 + x^2*p)
  __CLC_EP_PAIR r = __clc_ep_add(
      __clc_ep_mul(__clc_ep_make_pair(__CLC_FP_LIT(0x1.62e42fefa39efp-1),
                                      __CLC_FP_LIT(0x1.abc9e3b39803fp-56)),
                   __CLC_CONVERT_GENTYPE(e)),
      __clc_ep_fast_add(
          __clc_ep_ldexp(x, 1),
          __clc_ep_mul(
              __clc_ep_mul(s, x),
              __clc_ep_fast_add(
                  __clc_ep_make_pair(__CLC_FP_LIT(0x1.5555555555555p-1),
                                     __CLC_FP_LIT(0x1.543b0d5df274dp-55)),
                  __clc_ep_mul(s, p)))));
  return r;
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST __CLC_DOUBLEN __clc_ep_ln_hi(__CLC_EP_PAIR a,
                                                               __CLC_INTN ea) {
  __CLC_INTN a_hi_exp;
  __CLC_DOUBLEN m_hi = __clc_frexp(a.hi, &a_hi_exp);

  __CLC_LONGN b = m_hi < (2.0 / 3.0);
  __CLC_INTN e = a_hi_exp - __CLC_CONVERT_INTN(b);
  __CLC_EP_PAIR m = __clc_ep_ldexp(a, -e);
  __CLC_EP_PAIR x =
      __clc_ep_div(__clc_ep_fast_add(-1.0, m), __clc_ep_fast_add(1.0, m));
  __CLC_DOUBLEN s = x.hi * x.hi;

  const __CLC_DOUBLEN ln_c1 = 0x1.3ab76bf559e2bp-3;
  const __CLC_DOUBLEN ln_c2 = 0x1.385386b47b09ap-3;
  const __CLC_DOUBLEN ln_c3 = 0x1.7474dd7f4df2ep-3;
  const __CLC_DOUBLEN ln_c4 = 0x1.c71c016291751p-3;
  const __CLC_DOUBLEN ln_c5 = 0x1.249249b27acf1p-2;
  const __CLC_DOUBLEN ln_c6 = 0x1.99999998ef7b6p-2;
  const __CLC_DOUBLEN ln_c7 = 0x1.5555555555780p-1;

  __CLC_DOUBLEN q1 = __clc_mad(s, ln_c1, ln_c2);
  __CLC_DOUBLEN q2 = __clc_mad(s, q1, ln_c3);
  __CLC_DOUBLEN q3 = __clc_mad(s, q2, ln_c4);
  __CLC_DOUBLEN q4 = __clc_mad(s, q3, ln_c5);
  __CLC_DOUBLEN q5 = __clc_mad(s, q4, ln_c6);
  __CLC_DOUBLEN p = __clc_mad(s, q5, ln_c7);

  const __CLC_EP_PAIR ln2 = __clc_ep_make_pair(
      __CLC_FP_LIT(0x1.62e42fefa39efp-1), __CLC_FP_LIT(0x1.abc9e3b39803fp-56));

  __CLC_EP_PAIR r =
      __clc_ep_add(__clc_ep_mul(ln2, __CLC_CONVERT_DOUBLEN(e + ea)),
                   __clc_ep_fast_add(__clc_ep_ldexp(x, 1), s * x.hi * p));
  return r.hi;
}

#endif

#undef __CLC_EP_USE_FMA

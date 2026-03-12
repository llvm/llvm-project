//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifdef __CLC_SCALAR

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

_CLC_OVERLOAD
static bool samesign(__CLC_GENTYPE x, __CLC_GENTYPE y) {
  return __clc_signbit(x) == __clc_signbit(y);
}

_CLC_DEF
_CLC_OVERLOAD _CLC_CONST __CLC_EP_PAIR __clc_ep_absv(__CLC_EP_PAIR a) {
  return __clc_signbit(a.hi) ? -a : a;
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST __CLC_EP_PAIR
__clc_ep_conditional_sign_match(__CLC_EP_PAIR a, __CLC_EP_PAIR b) {
  return samesign(a.hi, b.hi) ? a : -a;
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST __CLC_EP_PAIR
__clc_ep_make_pair(__CLC_GENTYPE a, __CLC_GENTYPE b) {
  return (__CLC_EP_PAIR)(b, a);
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
                                                               int e) {
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

#undef __CLC_EP_USE_FMA
#endif

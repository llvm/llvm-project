//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

/* Refer to the exp routine for the underlying algorithm */
#if __CLC_FPSIZE == 32

_CLC_OVERLOAD _CLC_CONST _CLC_DEF __CLC_FLOATN __clc_expm1(__CLC_FLOATN x) {
  const __CLC_FLOATN x_max = 0x1.62e42ep+6f; // 64 * ln(2)
  const __CLC_FLOATN ln2_tail =
      -0x1.05c610p-29f; // small correction term for ln(2)

  const __CLC_FLOATN c1 = 0x1.a26762p-13f; // ~1.250000e-4
  const __CLC_FLOATN c2 = 0x1.6d2e00p-10f; // ~ 9.999999e-4
  const __CLC_FLOATN c3 = 0x1.110ff2p-7f;  // ~8.333333e-3
  const __CLC_FLOATN c4 = 0x1.555502p-5f;  // ~4.166667e-2
  const __CLC_FLOATN c5 = 0x1.555556p-3f;  // ~1.666667e-1

  __CLC_FLOATN fn = __clc_rint(x * (__CLC_FLOATN)M_LOG2E);

  __CLC_FLOATN t =
      __clc_fma(-fn, ln2_tail, __clc_fma(-fn, (__CLC_FLOATN)M_LN2, x));

  __CLC_FLOATN q0 = __clc_mad(t, c1, c2);
  __CLC_FLOATN q1 = __clc_mad(t, q0, c3);
  __CLC_FLOATN q2 = __clc_mad(t, q1, c4);
  __CLC_FLOATN q3 = __clc_mad(t, q2, c5);
  __CLC_FLOATN p = __clc_fma(t, t * __clc_mad(t, q3, 0.5f), t);

  __CLC_INTN e = fn == 128.0f ? 127 : __CLC_CONVERT_INTN(fn);
  __CLC_FLOATN s = __clc_ldexp(__CLC_FP_LIT(1.0), e);

  __CLC_FLOATN z = __clc_fma(s, p, s - 1.0f);
  z = fn == 128.0f ? 2.0f * z : z;

  z = x > x_max ? __CLC_GENTYPE_INF : z;
  z = x < -17.0f ? -1.0f : z;
  return z;
}

#elif __CLC_FPSIZE == 64

_CLC_OVERLOAD _CLC_CONST _CLC_DEF __CLC_DOUBLEN __clc_expm1(__CLC_DOUBLEN x) {
  const __CLC_DOUBLEN max_expm1_arg = 0x1.62e42fefa39efp+9;
  const __CLC_DOUBLEN min_expm1_arg = -37.0;

  const __CLC_DOUBLEN lnof2_by_64_tail = 0x1.abc9e3b39803fp-56;

  __CLC_DOUBLEN dn = __clc_rint(x * M_LOG2E);

  // Reduced argument t = x - dn * (ln2/64), split into head and tail for
  // precision
  __CLC_DOUBLEN t = __clc_mad(-dn, lnof2_by_64_tail, __clc_mad(-dn, M_LN2, x));

  const __CLC_DOUBLEN c0 = 0x1.1f32ea9d67f34p-29;
  const __CLC_DOUBLEN c1 = 0x1.af4eb2a1b768bp-26;
  const __CLC_DOUBLEN c2 = 0x1.27e500e0ac05bp-22;
  const __CLC_DOUBLEN c3 = 0x1.71de01b889c29p-19;
  const __CLC_DOUBLEN c4 = 0x1.a01a0197bcfd8p-16;
  const __CLC_DOUBLEN c5 = 0x1.a01a01ac1a723p-13;
  const __CLC_DOUBLEN c6 = 0x1.6c16c16c18931p-10;
  const __CLC_DOUBLEN c7 = 0x1.1111111110056p-7;
  const __CLC_DOUBLEN c8 = 0x1.5555555555552p-5;
  const __CLC_DOUBLEN c9 = 0x1.5555555555557p-3;
  const __CLC_DOUBLEN c10 = 0x1.0000000000000p-1;

  __CLC_DOUBLEN q0 = __clc_mad(t, c0, c1);
  __CLC_DOUBLEN q1 = __clc_mad(t, q0, c2);
  __CLC_DOUBLEN q2 = __clc_mad(t, q1, c3);
  __CLC_DOUBLEN q3 = __clc_mad(t, q2, c4);
  __CLC_DOUBLEN q4 = __clc_mad(t, q3, c5);
  __CLC_DOUBLEN q5 = __clc_mad(t, q4, c6);
  __CLC_DOUBLEN q6 = __clc_mad(t, q5, c7);
  __CLC_DOUBLEN q7 = __clc_mad(t, q6, c8);
  __CLC_DOUBLEN q8 = __clc_mad(t, q7, c9);
  __CLC_DOUBLEN p = __clc_mad(t, t * __clc_mad(t, q8, c10), t);

  __CLC_INTN e =
      __CLC_CONVERT_INTN(dn == 1024.0) ? 1023 : __CLC_CONVERT_INTN(dn);

  // s = 2^e
  __CLC_DOUBLEN s = __clc_ldexp(__CLC_FP_LIT(1.0), e);

  // expm1(x) ~= s * p + (s - 1)
  __CLC_DOUBLEN z = __clc_mad(s, p, s - 1.0);

  z = dn == 1024.0 ? 2.0 * z : z;

  z = x > max_expm1_arg ? __CLC_GENTYPE_INF : z;
  z = x < min_expm1_arg ? -1.0 : z;

  return z;
}

#elif __CLC_FPSIZE == 16

_CLC_OVERLOAD _CLC_DEF __CLC_HALFN __clc_expm1(__CLC_HALFN x) {
  __CLC_FLOATN x_float = __CLC_CONVERT_FLOATN(x);
  __CLC_FLOATN float_expm1 =
      __clc_exp2_fast(x_float * (__CLC_FLOATN)M_LOG2E) - 1.0f;
  __CLC_HALFN p = __clc_fma(x, x * __clc_fma(x, 0x1.555556p-3h, 0.5h), x);
  return __clc_fabs(x) < 0x1.0p-6h ? p : __CLC_CONVERT_HALFN(float_expm1);
}

#endif

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if __CLC_FPSIZE == 32

_CLC_DEF _CLC_OVERLOAD __CLC_GENTYPE __clc_tan_reduced_eval(__CLC_GENTYPE x,
                                                            __CLC_INTN is_odd) {
  __CLC_GENTYPE s = x * x;

  __CLC_GENTYPE a = __clc_mad(s, -0x1.19dba6p-6f, 0x1.8a8b0ep-2f);
  __CLC_GENTYPE b = __clc_mad(s, __clc_mad(s, 0x1.2e2900p-6f, -0x1.07266ep-1f),
                              0x1.27e84ap+0f);
  __CLC_GENTYPE p = s * __clc_div_fast(a, b);

  __CLC_GENTYPE t = __clc_fma(p, x, x);
  __CLC_GENTYPE tt = __clc_fma(p, x, -(t - x));
  __CLC_GENTYPE tr = -__clc_recip_fast(t);
  __CLC_GENTYPE e = __clc_fma(tt, tr, __clc_fma(t, tr, 1.0f));
  tr = __clc_fma(e, tr, tr);

  return is_odd ? tr : t;
}

_CLC_DEF _CLC_OVERLOAD __CLC_GENTYPE __clc_tan(__CLC_GENTYPE x) {
  x = __clc_select(x, __CLC_GENTYPE_NAN, __clc_isinf(x));

  __CLC_GENTYPE absx = __clc_fabs(x);
  __CLC_UINTN x_signbit = __CLC_AS_UINTN(x) & SIGNBIT_SP32;

  __CLC_GENTYPE r0;
  __CLC_INTN regn = __clc_argReductionS(&r0, absx);

  __CLC_GENTYPE t = __clc_tan_reduced_eval(r0, regn & 1);

  return __CLC_AS_FLOATN(__CLC_AS_UINTN(t) ^ x_signbit);
}

#elif __CLC_FPSIZE == 64

_CLC_DEF _CLC_OVERLOAD __CLC_GENTYPE __clc_tan(__CLC_GENTYPE x) {
  x = __clc_select(x, __CLC_GENTYPE_NAN,
                   __CLC_CONVERT_S_GENTYPE(__clc_isinf(x)));

  __CLC_GENTYPE y = __clc_fabs(x);

  __CLC_BIT_INTN is_medium = y < 0x1.0p+30;

  __CLC_INTN regn_m, regn_l;
  __CLC_GENTYPE r_m, r_l, rr_m, rr_l;

  __clc_remainder_piby2_medium(y, &r_m, &rr_m, &regn_m);
  __clc_remainder_piby2_large(y, &r_l, &rr_l, &regn_l);

  __CLC_GENTYPE r = is_medium ? r_m : r_l;
  __CLC_GENTYPE rr = is_medium ? rr_m : rr_l;
  __CLC_INTN regn = __CLC_CONVERT_INTN(is_medium) ? regn_m : regn_l;

  __CLC_GENTYPE lead, tail;
  __clc_tan_piby4(r, rr, &lead, &tail);

  __CLC_LONGN t =
      __CLC_AS_LONGN(__CLC_CONVERT_BIT_INTN((regn & 1) != 0) ? tail : lead);
  t ^= __CLC_CONVERT_BIT_INTN(x < 0.0) << 63;

  return __CLC_AS_GENTYPE(t);
}

#elif __CLC_FPSIZE == 16

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __clc_tan(__CLC_GENTYPE x) {
  return __CLC_CONVERT_GENTYPE(__clc_tan(__CLC_CONVERT_FLOATN(x)));
}

#endif

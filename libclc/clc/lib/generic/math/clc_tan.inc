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

  __CLC_GENTYPE t = __clc_tan_reduced_eval(r0, (regn & 1) != 0);

  return __CLC_AS_FLOATN(__CLC_AS_UINTN(t) ^ x_signbit);
}

#elif __CLC_FPSIZE == 64

_CLC_DEF _CLC_OVERLOAD __CLC_DOUBLEN __clc_tan(__CLC_DOUBLEN x) {
  x = __clc_select(x, __CLC_GENTYPE_NAN, __CLC_CONVERT_LONGN(__clc_isinf(x)));

  __CLC_DOUBLEN absx = __clc_fabs(x);

  __CLC_DOUBLEN reduced_lo, reduced_hi;
  __CLC_INTN regn = __clc_argReductionS(&reduced_lo, &reduced_hi, absx);

  __CLC_DOUBLEN t =
      __clc_tan_reduced_eval(reduced_hi, reduced_lo, (regn & 1) != 0);

  t = __CLC_AS_DOUBLEN(__CLC_AS_LONGN(t) ^
                       (__CLC_AS_LONGN(x) & (__CLC_LONGN)SIGNBIT_DP64));
  return __CLC_AS_DOUBLEN(t);
}

#elif __CLC_FPSIZE == 16

_CLC_DEF _CLC_OVERLOAD __CLC_HALFN __clc_tan_reduced_eval(__CLC_HALFN x,
                                                          __CLC_INTN is_odd) {
  __CLC_HALFN s = x * x;

  __CLC_HALFN t =
      __clc_mad(s, __clc_mad(s, 0x1.794p-4h, 0x1.e3cp-4h), 0x1.57p-2h);
  t = __clc_mad(x, s * t, x);

  __CLC_HALFN tr = -1.0h / t;
  return __CLC_CONVERT_SHORTN(is_odd) ? tr : t;
}

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __clc_tan(__CLC_GENTYPE x) {
  x = __clc_select(x, __CLC_GENTYPE_NAN, __CLC_CONVERT_SHORTN(__clc_isinf(x)));
  __CLC_HALFN absx = __clc_fabs(x);

  __CLC_HALFN reduced;
  __CLC_INTN n = __clc_argReductionS(&reduced, absx);

  __CLC_HALFN t = __clc_tan_reduced_eval(reduced, (n & 1) != 0);

  __CLC_S_GENTYPE result_i =
      (__CLC_AS_SHORTN(t) ^
       (__CLC_AS_SHORTN(x) & (__CLC_S_GENTYPE)SIGNBIT_FP16));
  return __CLC_AS_HALFN(result_i);
}

#endif

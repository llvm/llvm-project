//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if __CLC_FPSIZE == 32

_CLC_OVERLOAD _CLC_DEF __CLC_FLOATN __clc_sin(__CLC_FLOATN x) {
  x = __clc_select(x, __CLC_GENTYPE_NAN, __clc_isinf(x));

  __CLC_FLOATN absx = __clc_fabs(x);

  __CLC_FLOATN r0;
  __CLC_INTN regn = __clc_argReductionS(&r0, absx);
  __CLC_SINCOS_RET_GENTYPE eval = __clc_sincos_reduced_eval(r0);

  __CLC_FLOATN s = (regn & 1) != 0 ? eval.cos : eval.sin;

  return __CLC_AS_FLOATN(__CLC_AS_UINTN(s) ^ (regn > 1 ? SIGNBIT_SP32 : 0) ^
                         (__CLC_AS_UINTN(x) ^ __CLC_AS_UINTN(absx)));
}

#elif __CLC_FPSIZE == 16

_CLC_OVERLOAD _CLC_DEF __CLC_HALFN __clc_sin(__CLC_HALFN x) {
  x = __clc_select(x, __CLC_GENTYPE_NAN,
                   __CLC_CONVERT_S_GENTYPE(__clc_isinf(x)));

  __CLC_HALFN absx = __clc_fabs(x);

  __CLC_HALFN reduced;
  __CLC_INTN n = __clc_argReductionS(&reduced, absx);

  __CLC_SINCOS_RET_GENTYPE eval = __clc_sincos_reduced_eval(reduced);

  __CLC_HALFN s = __CLC_CONVERT_S_GENTYPE((n & 1) == 0) ? eval.sin : eval.cos;
  __CLC_S_GENTYPE flip = __CLC_CONVERT_S_GENTYPE(n > 1)
                             ? (__CLC_S_GENTYPE)SIGNBIT_FP16
                             : (__CLC_S_GENTYPE)0;

  __CLC_S_GENTYPE result_i =
      __CLC_AS_SHORTN(s) ^
      (flip ^ (__CLC_AS_SHORTN(x) & (__CLC_S_GENTYPE)SIGNBIT_FP16));
  return __CLC_AS_HALFN(result_i);
}

#elif __CLC_FPSIZE == 64

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __clc_sin(__CLC_GENTYPE x) {
  x = __clc_select(x, __CLC_GENTYPE_NAN, __CLC_CONVERT_LONGN(__clc_isinf(x)));

  __CLC_DOUBLEN absx = __clc_fabs(x);

  __CLC_DOUBLEN reduced_lo, reduced_hi;
  __CLC_INTN regn = __clc_argReductionS(&reduced_lo, &reduced_hi, absx);

  __CLC_SINCOS_RET_GENTYPE eval =
      __clc_sincos_reduced_eval(reduced_hi, reduced_lo);

  __CLC_DOUBLEN s = __CLC_CONVERT_LONGN((regn & 1) == 0) ? eval.sin : eval.cos;

  s = __CLC_AS_DOUBLEN(__CLC_AS_ULONGN(s) ^
                       (__CLC_CONVERT_LONGN(regn > 1)
                            ? (__CLC_ULONGN)SIGNBIT_DP64
                            : (__CLC_ULONGN)0) ^
                       (__CLC_AS_ULONGN(x) ^ __CLC_AS_ULONGN(absx)));

  return __CLC_AS_GENTYPE(s);
}

#endif

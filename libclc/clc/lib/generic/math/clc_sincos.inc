//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if __CLC_FPSIZE == 32

_CLC_OVERLOAD _CLC_DEF __CLC_FLOATN
__clc_sincos(__CLC_FLOATN x, private __CLC_FLOATN *cos_out) {
  x = __clc_select(x, __CLC_GENTYPE_NAN, __clc_isinf(x));

  __CLC_FLOATN absx = __clc_fabs(x);

  __CLC_FLOATN r0;
  __CLC_INTN regn = __clc_argReductionS(&r0, absx);
  const __CLC_SINCOS_RET_GENTYPE eval = __clc_sincos_reduced_eval(r0);

  __CLC_FLOATN s = (regn & 1) != 0 ? eval.cos : eval.sin;
  __CLC_FLOATN sin_result =
      __CLC_AS_FLOATN(__CLC_AS_UINTN(s) ^ (regn > 1 ? SIGNBIT_SP32 : 0) ^
                      (__CLC_AS_UINTN(x) ^ __CLC_AS_UINTN(absx)));

  __CLC_FLOATN c = (regn & 1) != 0 ? -eval.sin : eval.cos;
  *cos_out = __CLC_AS_FLOATN(__CLC_AS_UINTN(c) ^ (regn > 1 ? SIGNBIT_SP32 : 0));

  return sin_result;
}

#elif __CLC_FPSIZE == 16

_CLC_OVERLOAD _CLC_DEF __CLC_HALFN __clc_sincos(__CLC_HALFN x,
                                                private __CLC_HALFN *cos_out) {
  __CLC_FLOATN cos_result;
  __CLC_FLOATN sin_result = __clc_sincos(__CLC_CONVERT_FLOATN(x), &cos_result);
  *cos_out = __CLC_CONVERT_HALFN(cos_result);
  return __CLC_CONVERT_HALFN(sin_result);
}

#elif __CLC_FPSIZE == 64

_CLC_OVERLOAD _CLC_DEF __CLC_DOUBLEN
__clc_sincos(__CLC_DOUBLEN x, private __CLC_DOUBLEN *cos_out) {
  x = __clc_select(x, __CLC_GENTYPE_NAN,
                   __CLC_CONVERT_S_GENTYPE(__clc_isinf(x)));

  __CLC_GENTYPE absx = __clc_fabs(x);

  __CLC_GENTYPE reduced_lo, reduced_hi;
  __CLC_INTN regn = __clc_argReductionS(&reduced_lo, &reduced_hi, absx);

  __CLC_SINCOS_RET_GENTYPE eval =
      __clc_sincos_reduced_eval(reduced_hi, reduced_lo);

  __CLC_LONGN flip = __CLC_CONVERT_LONGN(regn > 1) ? (__CLC_LONGN)SIGNBIT_DP64
                                                   : (__CLC_LONGN)0;
  __CLC_LONGN odd = __CLC_CONVERT_LONGN((regn & 1) != 0);

  __CLC_DOUBLEN sin_result = odd ? eval.cos : eval.sin;
  sin_result =
      __CLC_AS_DOUBLEN(__CLC_AS_LONGN(sin_result) ^ flip ^
                       (__CLC_AS_LONGN(x) & (__CLC_LONGN)SIGNBIT_DP64));

  __CLC_DOUBLEN c = odd ? -eval.sin : eval.cos;
  *cos_out = __CLC_AS_DOUBLEN(__CLC_AS_LONGN(c) ^ flip);

  return sin_result;
}

#endif

#define __CLC_SINCOS_DEF(addrspace)                                            \
  _CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __clc_sincos(                           \
      __CLC_GENTYPE x, addrspace __CLC_GENTYPE *cos_out) {                     \
    __CLC_GENTYPE cos_result;                                                  \
    __CLC_GENTYPE sin_result = __clc_sincos(x, &cos_result);                   \
    *cos_out = cos_result;                                                     \
    return sin_result;                                                         \
  }

__CLC_SINCOS_DEF(local)
__CLC_SINCOS_DEF(global)
#if _CLC_DISTINCT_GENERIC_AS_SUPPORTED
__CLC_SINCOS_DEF(generic)
#endif

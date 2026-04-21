//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

_CLC_DEF _CLC_OVERLOAD __CLC_GENTYPE
__clc_sincospi(__CLC_GENTYPE x, __private __CLC_GENTYPE *cos_out) {
  x = __clc_select(x, __CLC_GENTYPE_NAN,
                   __CLC_CONVERT_S_GENTYPE(__clc_isinf(x)));

  __CLC_GENTYPE absx = __clc_fabs(x);

  __CLC_GENTYPE reduced;
  __CLC_INTN n = __clc_piArgReductionS(&reduced, absx);

  __CLC_SINCOS_RET_GENTYPE eval = __clc_sincospi_reduced_eval(reduced);

  __CLC_S_GENTYPE flip = __CLC_CONVERT_S_GENTYPE(n > 1)
                             ? __CLC_GENTYPE_S_SIGNBIT
                             : (__CLC_S_GENTYPE)0;
  __CLC_S_GENTYPE odd = __CLC_CONVERT_S_GENTYPE((n & 1) != 0);
  __CLC_GENTYPE s = odd ? eval.cos : eval.sin;

  __CLC_S_GENTYPE sin_val = __CLC_AS_S_GENTYPE(s) ^ flip ^
                            (__CLC_AS_S_GENTYPE(absx) ^ __CLC_AS_S_GENTYPE(x));

  __CLC_GENTYPE c = odd ? -eval.sin : eval.cos;
  *cos_out =
      __CLC_AS_GENTYPE(__CLC_CONVERT_S_GENTYPE(__CLC_AS_S_GENTYPE(c) ^ flip));
  return __CLC_AS_GENTYPE(sin_val);
}

#define __CLC_SINCOSPI_DEF(addrspace)                                          \
  _CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __clc_sincospi(                         \
      __CLC_GENTYPE x, addrspace __CLC_GENTYPE *cos_out) {                     \
                                                                               \
    __CLC_GENTYPE cos_result;                                                  \
    __CLC_GENTYPE sin_result = __clc_sincospi(x, &cos_result);                 \
    *cos_out = cos_result;                                                     \
    return sin_result;                                                         \
  }

__CLC_SINCOSPI_DEF(local)
__CLC_SINCOSPI_DEF(global)
#if _CLC_DISTINCT_GENERIC_AS_SUPPORTED
__CLC_SINCOSPI_DEF(generic)
#endif

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

_CLC_OVERLOAD _CLC_CONST _CLC_DEF __CLC_GENTYPE __clc_logb(__CLC_GENTYPE x) {
  __CLC_GENTYPE ret = __CLC_CONVERT_GENTYPE(__clc_frexp_exp(x) - (__CLC_INTN)1);

  __CLC_GENTYPE ax = __clc_fabs(x);

  ret = __clc_select(ax, ret, __CLC_CONVERT_U_GENTYPE(__clc_isfinite(ax)));

  ret = __clc_select(ret, -__CLC_GENTYPE_INF,
                     __CLC_CONVERT_U_GENTYPE(x == __CLC_FP_LIT(0.0)));

  return ret;
}

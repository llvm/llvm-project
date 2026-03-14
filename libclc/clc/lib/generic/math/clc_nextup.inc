//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

_CLC_OVERLOAD _CLC_CONST _CLC_DEF __CLC_GENTYPE __clc_nextup(__CLC_GENTYPE x) {
  __CLC_S_GENTYPE y = __CLC_AS_S_GENTYPE(x + __CLC_FP_LIT(0.0));
  __CLC_S_GENTYPE ix =
      y + (y >= (__CLC_S_GENTYPE)0 ? (__CLC_S_GENTYPE)1 : (__CLC_S_GENTYPE)-1);
  __CLC_GENTYPE converted = __CLC_AS_GENTYPE(ix);
  return __clc_isnan(x) || x == __CLC_GENTYPE_INF ? x : converted;
}

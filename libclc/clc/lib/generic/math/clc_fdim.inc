//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __clc_fdim(__CLC_GENTYPE x,
                                                __CLC_GENTYPE y) {
  return __clc_select(
      __builtin_elementwise_max(x - y, (__CLC_GENTYPE)__CLC_FP_LIT(0.0)),
      __CLC_GENTYPE_NAN,
      __CLC_CONVERT_BIT_INTN(__clc_isnan(x) || __clc_isnan(y)));
}

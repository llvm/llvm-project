//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __clc_minmag(__CLC_GENTYPE x,
                                                  __CLC_GENTYPE y) {
  const __CLC_GENTYPE res = __clc_select(
      y, x, __CLC_CONVERT_BIT_INTN(__clc_isless(__clc_fabs(x), __clc_fabs(y))));
  return __clc_select(
      res, __clc_fmin(x, y),
      __CLC_CONVERT_BIT_INTN(__clc_isnan(x) || __clc_isnan(y) ||
                             __clc_isequal(__clc_fabs(x), __clc_fabs(y))));
}

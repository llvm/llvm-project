//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

_CLC_OVERLOAD _CLC_CONST _CLC_DEF __CLC_INTN __clc_ilogb(__CLC_GENTYPE x) {
  __CLC_INTN r = __clc_frexp_exp(x) - 1;

  r = __clc_select(r, (__CLC_INTN)FP_ILOGBNAN,
                   __CLC_CONVERT_UINTN(__clc_isnan(x)));
  r = __clc_select(r, (__CLC_INTN)INT_MAX, __CLC_CONVERT_UINTN(__clc_isinf(x)));
  r = __clc_select(r, (__CLC_INTN)FP_ILOGB0,
                   __CLC_CONVERT_UINTN(x == __CLC_FP_LIT(0.0)));

  return r;
}

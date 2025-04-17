//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

_CLC_DEF _CLC_OVERLOAD __CLC_GENTYPE __clc_sign(__CLC_GENTYPE x) {
  __CLC_BIT_INTN ret_zero = __clc_isnan(x) || x == __CLC_FP_LIT(0.0);
  __CLC_GENTYPE ret_val =
      __clc_select((__CLC_GENTYPE)__CLC_FP_LIT(1.0),
                   (__CLC_GENTYPE)__CLC_FP_LIT(0.0), ret_zero);
  return __clc_copysign(ret_val, x);
}

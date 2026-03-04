//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __clc_smoothstep(__CLC_GENTYPE edge0,
                                                      __CLC_GENTYPE edge1,
                                                      __CLC_GENTYPE x) {
  __CLC_GENTYPE t = __clc_clamp((x - edge0) / (edge1 - edge0),
                                __CLC_FP_LIT(0.0), __CLC_FP_LIT(1.0));
  return t * t * (__CLC_FP_LIT(3.0) - __CLC_FP_LIT(2.0) * t);
}

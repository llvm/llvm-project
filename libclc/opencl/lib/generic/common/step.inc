//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE step(__CLC_GENTYPE edge, __CLC_GENTYPE x) {
  return __clc_step(edge, x);
}

#if !defined(__CLC_SCALAR)

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE step(__CLC_SCALAR_GENTYPE edge,
                                          __CLC_GENTYPE x) {
  return __clc_step((__CLC_GENTYPE)edge, x);
}

#endif

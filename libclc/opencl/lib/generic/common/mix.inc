//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE mix(__CLC_GENTYPE x, __CLC_GENTYPE y,
                                         __CLC_GENTYPE a) {
  return __clc_mad(y - x, a, x);
}

#ifndef __CLC_SCALAR
_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE mix(__CLC_GENTYPE x, __CLC_GENTYPE y,
                                         __CLC_SCALAR_GENTYPE a) {
  return mix(x, y, (__CLC_GENTYPE)a);
}
#endif

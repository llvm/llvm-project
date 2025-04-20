//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __clc_clamp(__CLC_GENTYPE x,
                                                 __CLC_GENTYPE y,
                                                 __CLC_GENTYPE z) {
  return (x > z ? z : (x < y ? y : x));
}

#ifndef __CLC_SCALAR
_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __clc_clamp(__CLC_GENTYPE x,
                                                 __CLC_SCALAR_GENTYPE y,
                                                 __CLC_SCALAR_GENTYPE z) {
  return (x > (__CLC_GENTYPE)z ? (__CLC_GENTYPE)z
                               : (x < (__CLC_GENTYPE)y ? (__CLC_GENTYPE)y : x));
}
#endif

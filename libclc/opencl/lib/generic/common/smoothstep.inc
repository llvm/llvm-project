//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE smoothstep(__CLC_GENTYPE edge0,
                                                __CLC_GENTYPE edge1,
                                                __CLC_GENTYPE x) {
  return __clc_smoothstep(edge0, edge1, x);
}

#if !defined(__CLC_SCALAR)

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE smoothstep(__CLC_SCALAR_GENTYPE edge0,
                                                __CLC_SCALAR_GENTYPE edge1,
                                                __CLC_GENTYPE x) {
  return __clc_smoothstep((__CLC_GENTYPE)edge0, (__CLC_GENTYPE)edge1, x);
}

#endif

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

_CLC_OVERLOAD _CLC_DEF __CLC_U_GENTYPE __clc_abs(__CLC_GENTYPE x) {
  return __builtin_astype((__CLC_GENTYPE)(x > (__CLC_GENTYPE)(0) ? x : -x),
                          __CLC_U_GENTYPE);
}

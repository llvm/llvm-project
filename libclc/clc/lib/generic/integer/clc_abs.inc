//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifdef __CLC_GEN_S

_CLC_OVERLOAD _CLC_DEF __CLC_U_GENTYPE __clc_abs(__CLC_GENTYPE x) {
  return __builtin_astype(__builtin_elementwise_abs(x), __CLC_U_GENTYPE);
}

#else

_CLC_OVERLOAD _CLC_DEF __CLC_U_GENTYPE __clc_abs(__CLC_GENTYPE x) { return x; }

#endif

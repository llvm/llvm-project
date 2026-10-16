//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifdef __CLC_SCALAR

typedef struct __CLC_XCONCAT(__clc_lgamma_r_ret_, __CLC_GENTYPE) {
  __CLC_GENTYPE result;
  __CLC_INTN sign;
} __CLC_XCONCAT(__clc_lgamma_r_ret_, __CLC_GENTYPE);

#define __CLC_LGAMMA_R_RET_GENTYPE                                             \
  __CLC_XCONCAT(__clc_lgamma_r_ret_, __CLC_GENTYPE)
_CLC_DECL _CLC_OVERLOAD _CLC_CONST __CLC_LGAMMA_R_RET_GENTYPE
__clc_lgamma_r_stret(__CLC_GENTYPE x);

#endif

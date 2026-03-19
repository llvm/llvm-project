//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

typedef struct __CLC_XCONCAT(__clc_sincos_ret_, __CLC_GENTYPE) {
  __CLC_GENTYPE sin, cos;
} __CLC_XCONCAT(__clc_sincos_ret_, __CLC_GENTYPE);

#define __CLC_SINCOS_RET_GENTYPE __CLC_XCONCAT(__clc_sincos_ret_, __CLC_GENTYPE)

_CLC_DEF _CLC_OVERLOAD __CLC_SINCOS_RET_GENTYPE
__clc_sincos_reduced_eval(__CLC_HALFN x);

_CLC_DEF _CLC_OVERLOAD __CLC_INTN __clc_argReductionS(private __CLC_HALFN *r,
                                                      __CLC_HALFN x);

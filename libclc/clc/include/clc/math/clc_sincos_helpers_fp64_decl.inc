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
__clc_sincos_reduced_eval(__CLC_DOUBLEN x, __CLC_DOUBLEN y);

_CLC_DEF _CLC_OVERLOAD __CLC_DOUBLEN __clc_tan_reduced_eval(__CLC_DOUBLEN x,
                                                            __CLC_DOUBLEN y,
                                                            __CLC_INTN is_odd);

_CLC_DECL _CLC_OVERLOAD void __clc_sincos_piby4(__CLC_DOUBLEN x,
                                                __CLC_DOUBLEN xx,
                                                private __CLC_DOUBLEN *sinval,
                                                private __CLC_DOUBLEN *cosval);

_CLC_DECL _CLC_OVERLOAD void __clc_tan_piby4(__CLC_DOUBLEN x, __CLC_DOUBLEN xx,
                                             private __CLC_DOUBLEN *leadval,
                                             private __CLC_DOUBLEN *tailval);

_CLC_DECL _CLC_OVERLOAD __CLC_INTN __clc_remainder_piby2_small(
    __CLC_DOUBLEN x, private __CLC_DOUBLEN *r, private __CLC_DOUBLEN *rr);

_CLC_DECL _CLC_OVERLOAD __CLC_INTN __clc_remainder_piby2_large(
    __CLC_DOUBLEN x, private __CLC_DOUBLEN *r, private __CLC_DOUBLEN *rr);

_CLC_DEF _CLC_OVERLOAD __CLC_INTN __clc_argReductionS(private __CLC_DOUBLEN *r,
                                                      private __CLC_DOUBLEN *rr,
                                                      __CLC_DOUBLEN x);

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

_CLC_DECL _CLC_OVERLOAD void __clc_sincos_piby4(__CLC_DOUBLEN x,
                                                __CLC_DOUBLEN xx,
                                                private __CLC_DOUBLEN *sinval,
                                                private __CLC_DOUBLEN *cosval);

_CLC_DECL _CLC_OVERLOAD void __clc_tan_piby4(__CLC_DOUBLEN x, __CLC_DOUBLEN xx,
                                             private __CLC_DOUBLEN *leadval,
                                             private __CLC_DOUBLEN *tailval);

_CLC_DECL _CLC_OVERLOAD void
__clc_remainder_piby2_medium(__CLC_DOUBLEN x, private __CLC_DOUBLEN *r,
                             private __CLC_DOUBLEN *rr,
                             private __CLC_INTN *regn);

_CLC_DECL _CLC_OVERLOAD void
__clc_remainder_piby2_large(__CLC_DOUBLEN x, private __CLC_DOUBLEN *r,
                            private __CLC_DOUBLEN *rr,
                            private __CLC_INTN *regn);

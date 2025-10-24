//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

_CLC_DECL _CLC_OVERLOAD __CLC_FLOATN __clc_sinf_piby4(__CLC_FLOATN x,
                                                      __CLC_FLOATN y);
_CLC_DECL _CLC_OVERLOAD __CLC_FLOATN __clc_cosf_piby4(__CLC_FLOATN x,
                                                      __CLC_FLOATN y);

_CLC_DECL _CLC_OVERLOAD void __clc_sincos_piby4(__CLC_FLOATN x,
                                                private __CLC_FLOATN *sinval,
                                                private __CLC_FLOATN *cosval);

_CLC_DECL _CLC_OVERLOAD __CLC_FLOATN __clc_tanf_piby4(__CLC_FLOATN x,
                                                      __CLC_INTN regn);

_CLC_DECL _CLC_OVERLOAD __CLC_INTN __clc_argReductionS(private __CLC_FLOATN *r,
                                                       private __CLC_FLOATN *rr,
                                                       __CLC_FLOATN x);

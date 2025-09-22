//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

_CLC_OVERLOAD _CLC_CONST _CLC_DECL __CLC_GENTYPE smoothstep(__CLC_GENTYPE edge0,
                                                            __CLC_GENTYPE edge1,
                                                            __CLC_GENTYPE x);
_CLC_OVERLOAD _CLC_CONST _CLC_DECL __CLC_GENTYPE smoothstep(float edge0,
                                                            float edge1,
                                                            __CLC_GENTYPE x);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_CONST _CLC_DECL __CLC_GENTYPE smoothstep(double edge0,
                                                            double edge1,
                                                            __CLC_GENTYPE x);
#endif

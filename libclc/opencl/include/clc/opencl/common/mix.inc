//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

_CLC_OVERLOAD _CLC_CONST _CLC_DECL __CLC_GENTYPE mix(__CLC_GENTYPE a,
                                                     __CLC_GENTYPE b,
                                                     __CLC_GENTYPE c);

#ifndef __CLC_SCALAR
_CLC_OVERLOAD _CLC_CONST _CLC_DECL __CLC_GENTYPE mix(__CLC_GENTYPE a,
                                                     __CLC_GENTYPE b,
                                                     __CLC_SCALAR_GENTYPE c);
#endif

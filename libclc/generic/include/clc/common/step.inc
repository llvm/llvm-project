//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

_CLC_OVERLOAD _CLC_DECL __CLC_GENTYPE step(__CLC_GENTYPE edge, __CLC_GENTYPE x);
_CLC_OVERLOAD _CLC_DECL __CLC_GENTYPE step(float edge, __CLC_GENTYPE x);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DECL __CLC_GENTYPE step(double edge, __CLC_GENTYPE x);
#endif

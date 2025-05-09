//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define __CLC_DECLARE_SINCOS(ADDRSPACE, TYPE) \
  _CLC_OVERLOAD _CLC_DEF TYPE sincos (TYPE x, ADDRSPACE TYPE * cosval) { \
    *cosval = cos(x); \
    return sin(x); \
  }

__CLC_DECLARE_SINCOS(global, __CLC_GENTYPE)
__CLC_DECLARE_SINCOS(local, __CLC_GENTYPE)
__CLC_DECLARE_SINCOS(private, __CLC_GENTYPE)

#undef __CLC_DECLARE_SINCOS

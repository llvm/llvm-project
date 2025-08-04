//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define __CLC_DECLARE_SINCOS(ADDRSPACE, TYPE)                                  \
  _CLC_OVERLOAD _CLC_DEF TYPE __clc_sincos(TYPE x, ADDRSPACE TYPE *cosval) {   \
    *cosval = __clc_cos(x);                                                    \
    return __clc_sin(x);                                                       \
  }

__CLC_DECLARE_SINCOS(global, __CLC_GENTYPE)
__CLC_DECLARE_SINCOS(local, __CLC_GENTYPE)
__CLC_DECLARE_SINCOS(private, __CLC_GENTYPE)
#if _CLC_DISTINCT_GENERIC_AS_SUPPORTED
__CLC_DECLARE_SINCOS(generic, __CLC_GENTYPE)
#endif

#undef __CLC_DECLARE_SINCOS

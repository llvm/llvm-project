//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE
__clc_sincos(__CLC_GENTYPE x, private __CLC_GENTYPE *cosval) {
  *cosval = __clc_cos(x);
  return __clc_sin(x);
}

#define __CLC_SINCOS_DEF(addrspace)                                            \
  _CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __clc_sincos(                           \
      __CLC_GENTYPE x, addrspace __CLC_GENTYPE *cos_out) {                     \
    __CLC_GENTYPE cos_result;                                                  \
    __CLC_GENTYPE sin_result = __clc_sincos(x, &cos_result);                   \
    *cos_out = cos_result;                                                     \
    return sin_result;                                                         \
  }

__CLC_SINCOS_DEF(local)
__CLC_SINCOS_DEF(global)
#if _CLC_DISTINCT_GENERIC_AS_SUPPORTED
__CLC_SINCOS_DEF(generic)
#endif

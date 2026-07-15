//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifdef __CLC_SCALAR
#define __CLC_LGAMMA_R_DEF(addrspace)                                          \
  _CLC_DEF _CLC_OVERLOAD __CLC_GENTYPE __clc_lgamma_r(                         \
      __CLC_GENTYPE x, addrspace __CLC_INTN *signp) {                          \
    __CLC_LGAMMA_R_RET_GENTYPE result = __clc_lgamma_r_stret(x);               \
    *signp = result.sign;                                                      \
    return result.result;                                                      \
  }

__CLC_LGAMMA_R_DEF(private)
__CLC_LGAMMA_R_DEF(local)
__CLC_LGAMMA_R_DEF(global)
#if _CLC_DISTINCT_GENERIC_AS_SUPPORTED
__CLC_LGAMMA_R_DEF(generic)
#endif
#endif // __CLC_SCALAR

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __clc_modf(__CLC_GENTYPE x,
                                                private __CLC_GENTYPE *iptr) {
  *iptr = __clc_trunc(x);
  return __clc_copysign(__clc_isinf(x) ? __CLC_FP_LIT(0.0) : x - *iptr, x);
}

#define __CLC_MODF_DEF(addrspace)                                              \
  _CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __clc_modf(                             \
      __CLC_GENTYPE x, addrspace __CLC_GENTYPE *iptr) {                        \
    __CLC_GENTYPE private_iptr;                                                \
    __CLC_GENTYPE ret = __clc_modf(x, &private_iptr);                          \
    *iptr = private_iptr;                                                      \
    return ret;                                                                \
  }

__CLC_MODF_DEF(local);
__CLC_MODF_DEF(global);

#if _CLC_DISTINCT_GENERIC_AS_SUPPORTED
__CLC_MODF_DEF(generic);
#endif

#undef __CLC_MODF_DEF

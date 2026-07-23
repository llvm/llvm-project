//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifdef __CLC_SCALAR
#define __CLC_REMQUO_DEF(addrspace)                                            \
  _CLC_DEF _CLC_OVERLOAD __CLC_GENTYPE __clc_remquo(                           \
      __CLC_GENTYPE x, __CLC_GENTYPE y, addrspace __CLC_INTN *quo_out) {       \
    __CLC_REMQUO_RET_GENTYPE result = __clc_remquo_stret(x, y);                \
    *quo_out = result.quo;                                                     \
    return result.rem;                                                         \
  }

__CLC_REMQUO_DEF(private)
__CLC_REMQUO_DEF(local)
__CLC_REMQUO_DEF(global)
#if _CLC_DISTINCT_GENERIC_AS_SUPPORTED
__CLC_REMQUO_DEF(generic)
#endif
#endif // __CLC_SCALAR

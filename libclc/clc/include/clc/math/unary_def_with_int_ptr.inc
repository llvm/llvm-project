//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/utils.h>

#ifndef __CLC_IMPL_FUNCTION
#define __CLC_IMPL_FUNCTION(x) __CLC_CONCAT(__clc_, x)
#endif

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __CLC_FUNCTION(__CLC_GENTYPE x,
                                                    private __CLC_INTN *iptr) {
  return __CLC_IMPL_FUNCTION(__CLC_FUNCTION)(x, iptr);
}

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __CLC_FUNCTION(__CLC_GENTYPE x,
                                                    global __CLC_INTN *iptr) {
  return __CLC_IMPL_FUNCTION(__CLC_FUNCTION)(x, iptr);
}

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __CLC_FUNCTION(__CLC_GENTYPE x,
                                                    local __CLC_INTN *iptr) {
  return __CLC_IMPL_FUNCTION(__CLC_FUNCTION)(x, iptr);
}

#if _CLC_DISTINCT_GENERIC_AS_SUPPORTED
_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __CLC_FUNCTION(__CLC_GENTYPE x,
                                                    generic __CLC_INTN *iptr) {
  return __CLC_IMPL_FUNCTION(__CLC_FUNCTION)(x, iptr);
}
#endif

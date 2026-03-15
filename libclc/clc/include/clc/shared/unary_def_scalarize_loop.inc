//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/utils.h>

#if __CLC_VECSIZE_OR_1 >= 2

#ifndef __CLC_IMPL_FUNCTION
#define __CLC_IMPL_FUNCTION __CLC_FUNCTION
#endif

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __CLC_FUNCTION(__CLC_GENTYPE x) {
  union {
    __CLC_GENTYPE vec;
    __CLC_SCALAR_GENTYPE arr[__CLC_VECSIZE_OR_1];
  } u_x, u_result;
  u_x.vec = x;
  for (int i = 0; i < __CLC_VECSIZE_OR_1; ++i)
    u_result.arr[i] = __CLC_IMPL_FUNCTION(u_x.arr[i]);
  return u_result.vec;
}

#endif // __CLC_VECSIZE_OR_1 >= 2

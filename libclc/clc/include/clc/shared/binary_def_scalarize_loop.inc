//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clc/utils.h"

#ifndef __CLC_IMPL_FUNCTION
#define __CLC_IMPL_FUNCTION __CLC_FUNCTION
#endif

#ifndef __CLC_RET_SCALAR_TYPE
#define __CLC_RET_SCALAR_TYPE __CLC_SCALAR_GENTYPE
#endif

#ifndef __CLC_ARG1_SCALAR_TYPE
#define __CLC_ARG1_SCALAR_TYPE __CLC_SCALAR_GENTYPE
#endif

#ifndef __CLC_ARG2_SCALAR_TYPE
#define __CLC_ARG2_SCALAR_TYPE __CLC_SCALAR_GENTYPE
#endif

#define __CLC_RET_TYPE __CLC_XCONCAT(__CLC_RET_SCALAR_TYPE, __CLC_VECSIZE)
#define __CLC_ARG1_TYPE __CLC_XCONCAT(__CLC_ARG1_SCALAR_TYPE, __CLC_VECSIZE)
#define __CLC_ARG2_TYPE __CLC_XCONCAT(__CLC_ARG2_SCALAR_TYPE, __CLC_VECSIZE)

#if __CLC_VECSIZE_OR_1 >= 2

_CLC_OVERLOAD _CLC_DEF __CLC_RET_TYPE __CLC_FUNCTION(__CLC_ARG1_TYPE x,
                                                     __CLC_ARG2_TYPE y) {
  union {
    __CLC_ARG1_TYPE vec;
    __CLC_ARG1_SCALAR_TYPE arr[__CLC_VECSIZE_OR_1];
  } u_x;

  union {
    __CLC_ARG2_TYPE vec;
    __CLC_ARG2_SCALAR_TYPE arr[__CLC_VECSIZE_OR_1];
  } u_y;

  union {
    __CLC_RET_TYPE vec;
    __CLC_RET_SCALAR_TYPE arr[__CLC_VECSIZE_OR_1];
  } u_result;

  u_x.vec = x;
  u_y.vec = y;
  for (int i = 0; i < __CLC_VECSIZE_OR_1; ++i) {
    u_result.arr[i] = __CLC_IMPL_FUNCTION(u_x.arr[i], u_y.arr[i]);
  }

  return u_result.vec;
}

#endif // __CLC_VECSIZE_OR_1 >= 2

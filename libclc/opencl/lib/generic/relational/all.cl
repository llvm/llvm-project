//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/opencl/relational/all.h>
#include <clc/relational/clc_all.h>

#define __CLC_ALL_ID(TYPE) _CLC_OVERLOAD _CLC_DEF int all(TYPE v)

#define __CLC_ALL_VECTORIZE(TYPE)                                              \
  __CLC_ALL_ID(TYPE) { return __clc_all(v); }                                  \
  __CLC_ALL_ID(TYPE##2) { return __clc_all(v); }                               \
  __CLC_ALL_ID(TYPE##3) { return __clc_all(v); }                               \
  __CLC_ALL_ID(TYPE##4) { return __clc_all(v); }                               \
  __CLC_ALL_ID(TYPE##8) { return __clc_all(v); }                               \
  __CLC_ALL_ID(TYPE##16) { return __clc_all(v); }

__CLC_ALL_VECTORIZE(char)
__CLC_ALL_VECTORIZE(short)
__CLC_ALL_VECTORIZE(int)
__CLC_ALL_VECTORIZE(long)

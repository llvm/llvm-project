//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/opencl/relational/any.h>
#include <clc/relational/clc_any.h>

#define __CLC_ANY_ID(TYPE) _CLC_OVERLOAD _CLC_DEF int any(TYPE v)

#define __CLC_ANY_VECTORIZE(TYPE)                                              \
  __CLC_ANY_ID(TYPE) { return __clc_any(v); }                                  \
  __CLC_ANY_ID(TYPE##2) { return __clc_any(v); }                               \
  __CLC_ANY_ID(TYPE##3) { return __clc_any(v); }                               \
  __CLC_ANY_ID(TYPE##4) { return __clc_any(v); }                               \
  __CLC_ANY_ID(TYPE##8) { return __clc_any(v); }                               \
  __CLC_ANY_ID(TYPE##16) { return __clc_any(v); }

__CLC_ANY_VECTORIZE(char)
__CLC_ANY_VECTORIZE(short)
__CLC_ANY_VECTORIZE(int)
__CLC_ANY_VECTORIZE(long)

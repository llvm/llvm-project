//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/opencl/relational/any.h>
#include <clc/relational/clc_any.h>

#define ANY_ID(TYPE) _CLC_OVERLOAD _CLC_DEF int any(TYPE v)

#define ANY_VECTORIZE(TYPE)                                                    \
  ANY_ID(TYPE) { return __clc_any(v); }                                        \
  ANY_ID(TYPE##2) { return __clc_any(v); }                                     \
  ANY_ID(TYPE##3) { return __clc_any(v); }                                     \
  ANY_ID(TYPE##4) { return __clc_any(v); }                                     \
  ANY_ID(TYPE##8) { return __clc_any(v); }                                     \
  ANY_ID(TYPE##16) { return __clc_any(v); }

ANY_VECTORIZE(char)
ANY_VECTORIZE(short)
ANY_VECTORIZE(int)
ANY_VECTORIZE(long)

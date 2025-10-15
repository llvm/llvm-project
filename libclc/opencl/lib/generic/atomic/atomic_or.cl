//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/opencl/atomic/atomic_or.h>

#define __CLC_IMPL(TYPE, AS)                                                   \
  _CLC_OVERLOAD _CLC_DEF TYPE atomic_or(volatile AS TYPE *p, TYPE val) {       \
    return __sync_fetch_and_or(p, val);                                        \
  }

__CLC_IMPL(int, global)
__CLC_IMPL(unsigned int, global)
__CLC_IMPL(int, local)
__CLC_IMPL(unsigned int, local)
#undef __CLC_IMPL

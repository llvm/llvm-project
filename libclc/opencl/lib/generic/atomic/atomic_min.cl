//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/atomic/clc_atomic_fetch_min.h>
#include <clc/opencl/atomic/atomic_min.h>

#define __CLC_IMPL(TYPE, AS, OP)                                               \
  _CLC_OVERLOAD _CLC_DEF TYPE atomic_min(volatile AS TYPE *p, TYPE val) {      \
    return __clc_atomic_fetch_min(p, val, __ATOMIC_RELAXED,                    \
                                  __MEMORY_SCOPE_DEVICE);                      \
  }

__CLC_IMPL(int, global, min)
__CLC_IMPL(unsigned int, global, umin)
__CLC_IMPL(int, local, min)
__CLC_IMPL(unsigned int, local, umin)
#undef __CLC_IMPL

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/atomic/clc_atomic_compare_exchange.h>
#include <clc/opencl/atomic/atomic_cmpxchg.h>

#define __CLC_IMPL(TYPE, AS)                                                   \
  _CLC_OVERLOAD _CLC_DEF TYPE atomic_cmpxchg(volatile AS TYPE *p, TYPE cmp,    \
                                             TYPE val) {                       \
    return __clc_atomic_compare_exchange(p, cmp, val, __ATOMIC_RELAXED,        \
                                         __ATOMIC_RELAXED,                     \
                                         __MEMORY_SCOPE_DEVICE);               \
  }

__CLC_IMPL(int, global)
__CLC_IMPL(unsigned int, global)
__CLC_IMPL(int, local)
__CLC_IMPL(unsigned int, local)
#undef __CLC_IMPL

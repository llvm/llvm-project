//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/atomic/clc_atomic_exchange.h>
#include <clc/atomic/clc_atomic_flag_test_and_set.h>

#define __CLC_ATOMIC_FLAG_TRUE 1

#define __CLC_DEFINE_ATOMIC_FLAG_TEST_AND_SET(ADDRSPACE)                       \
  _CLC_OVERLOAD _CLC_DEF bool __clc_atomic_flag_test_and_set(                  \
      ADDRSPACE int *Ptr, int MemoryOrder, int MemoryScope) {                  \
    return (bool)__clc_atomic_exchange(Ptr, __CLC_ATOMIC_FLAG_TRUE,            \
                                       MemoryOrder, MemoryScope);              \
  }

__CLC_DEFINE_ATOMIC_FLAG_TEST_AND_SET(global)
__CLC_DEFINE_ATOMIC_FLAG_TEST_AND_SET(local)
#if _CLC_GENERIC_AS_SUPPORTED
__CLC_DEFINE_ATOMIC_FLAG_TEST_AND_SET()
#endif

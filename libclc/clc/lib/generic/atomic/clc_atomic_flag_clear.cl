//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/atomic/clc_atomic_flag_clear.h>
#include <clc/atomic/clc_atomic_store.h>

#define __CLC_ATOMIC_FLAG_FALSE 0

#define __CLC_DEFINE_ATOMIC_FLAG_CLEAR(ADDRSPACE)                              \
  _CLC_OVERLOAD _CLC_DEF void __clc_atomic_flag_clear(                         \
      ADDRSPACE int *Ptr, int MemoryOrder, int MemoryScope) {                  \
    __clc_atomic_store(Ptr, __CLC_ATOMIC_FLAG_FALSE, MemoryOrder,              \
                       MemoryScope);                                           \
  }

__CLC_DEFINE_ATOMIC_FLAG_CLEAR(global)
__CLC_DEFINE_ATOMIC_FLAG_CLEAR(local)
#if _CLC_GENERIC_AS_SUPPORTED
__CLC_DEFINE_ATOMIC_FLAG_CLEAR()
#endif

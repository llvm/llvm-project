//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __CLC_ATOMIC_CLC_ATOMIC_FLAG_CLEAR_H__
#define __CLC_ATOMIC_CLC_ATOMIC_FLAG_CLEAR_H__

#include <clc/internal/clc.h>

#define __CLC_DECLARE_ATOMIC_FLAG_CLEAR(ADDRSPACE)                             \
  _CLC_OVERLOAD _CLC_DECL void __clc_atomic_flag_clear(                        \
      ADDRSPACE int *Ptr, int MemoryOrder, int MemoryScope);

__CLC_DECLARE_ATOMIC_FLAG_CLEAR(global)
__CLC_DECLARE_ATOMIC_FLAG_CLEAR(local)
#if _CLC_GENERIC_AS_SUPPORTED
__CLC_DECLARE_ATOMIC_FLAG_CLEAR()
#endif

#endif // __CLC_ATOMIC_CLC_ATOMIC_FLAG_CLEAR_H__

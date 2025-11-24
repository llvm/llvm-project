//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __CLC_ATOMIC_CLC_ATOMIC_FLAG_TEST_AND_SET_H__
#define __CLC_ATOMIC_CLC_ATOMIC_FLAG_TEST_AND_SET_H__

#include <clc/internal/clc.h>

#define __CLC_DECLARE_ATOMIC_FLAG_TEST_AND_SET(ADDRSPACE)                      \
  _CLC_OVERLOAD _CLC_DECL bool __clc_atomic_flag_test_and_set(                 \
      ADDRSPACE int *Ptr, int MemoryOrder, int MemoryScope);

__CLC_DECLARE_ATOMIC_FLAG_TEST_AND_SET(global)
__CLC_DECLARE_ATOMIC_FLAG_TEST_AND_SET(local)
#if _CLC_GENERIC_AS_SUPPORTED
__CLC_DECLARE_ATOMIC_FLAG_TEST_AND_SET()
#endif

#endif // __CLC_ATOMIC_CLC_ATOMIC_FLAG_TEST_AND_SET_H__

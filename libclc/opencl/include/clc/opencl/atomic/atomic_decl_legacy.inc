//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define __CLC_DECLARE_ATOMIC(ADDRSPACE, TYPE)                                  \
  _CLC_OVERLOAD _CLC_DECL TYPE __CLC_FUNCTION(volatile ADDRSPACE TYPE *, TYPE);

#define __CLC_DECLARE_ATOMIC_ADDRSPACE(TYPE)                                   \
  __CLC_DECLARE_ATOMIC(global, TYPE)                                           \
  __CLC_DECLARE_ATOMIC(local, TYPE)

__CLC_DECLARE_ATOMIC_ADDRSPACE(int)
__CLC_DECLARE_ATOMIC_ADDRSPACE(uint)

#undef __CLC_DECLARE_ATOMIC_ADDRSPACE
#undef __CLC_DECLARE_ATOMIC

#undef __CLC_FUNCTION

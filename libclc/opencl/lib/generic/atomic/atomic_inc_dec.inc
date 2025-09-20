//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifdef __CLC_SCALAR

#if __CLC_GENSIZE == 32

#define __CLC_DEFINE_ATOMIC(ADDRSPACE)                                         \
  _CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __CLC_FUNCTION(                         \
      volatile ADDRSPACE __CLC_GENTYPE *Ptr) {                                 \
    return __CLC_IMPL_FUNCTION(Ptr, __ATOMIC_SEQ_CST, __MEMORY_SCOPE_DEVICE);  \
  }

__CLC_DEFINE_ATOMIC(global)
__CLC_DEFINE_ATOMIC(local)

#undef __CLC_DEFINE_ATOMIC

#endif // __CLC_GENSIZE == 32

#endif // __CLC_SCALAR

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// MemoryOrder is memory order supported by Clang __scoped_atomic* builtins.
// MemoryScope is memory scope supported by Clang __scoped_atomic* builtins.

#ifdef __CLC_SCALAR
#if defined(__CLC_FPSIZE) || (__CLC_GENSIZE >= 32)

#ifdef __CLC_NO_VALUE_ARG
#define __CLC_DECLARE_ATOMIC(ADDRSPACE)                                        \
  _CLC_OVERLOAD _CLC_DECL __CLC_GENTYPE __CLC_FUNCTION(                        \
      volatile ADDRSPACE __CLC_GENTYPE *Ptr, int MemoryOrder,                  \
      int MemoryScope);
#elif defined(__CLC_RETURN_VOID)
#define __CLC_DECLARE_ATOMIC(ADDRSPACE)                                        \
  _CLC_OVERLOAD _CLC_DECL void __CLC_FUNCTION(                                 \
      volatile ADDRSPACE __CLC_GENTYPE *Ptr, __CLC_GENTYPE Value,              \
      int MemoryOrder, int MemoryScope);
#elif defined(__CLC_COMPARE_EXCHANGE)
#define __CLC_DECLARE_ATOMIC(ADDRSPACE)                                        \
  _CLC_OVERLOAD _CLC_DECL __CLC_GENTYPE __CLC_FUNCTION(                        \
      volatile ADDRSPACE __CLC_GENTYPE *Ptr, __CLC_GENTYPE Comparator,         \
      __CLC_GENTYPE Value, int MemoryOrderEqual, int MemoryOrderUnequal,       \
      int MemoryScope);
#else
#define __CLC_DECLARE_ATOMIC(ADDRSPACE)                                        \
  _CLC_OVERLOAD _CLC_DECL __CLC_GENTYPE __CLC_FUNCTION(                        \
      volatile ADDRSPACE __CLC_GENTYPE *Ptr, __CLC_GENTYPE Value,              \
      int MemoryOrder, int MemoryScope);
#endif

__CLC_DECLARE_ATOMIC(global)
__CLC_DECLARE_ATOMIC(local)
#if _CLC_GENERIC_AS_SUPPORTED
__CLC_DECLARE_ATOMIC()
#endif

#undef __CLC_DECLARE_ATOMIC

#endif // defined(__CLC_FPSIZE) || (__CLC_GENSIZE >= 32)
#endif // __CLC_SCALAR

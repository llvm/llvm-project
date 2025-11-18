//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifdef __CLC_SCALAR

#if defined(__SPIR32__) || defined(CLC_NVPTX)
#if (defined(__CLC_FPSIZE) && __CLC_FPSIZE <= 32) ||                           \
    (defined(__CLC_GENSIZE) && (__CLC_GENSIZE == 32))
#define __CLC_HAS_ATOMIC
#endif
#else // defined(__SPIR32__) || defined(CLC_NVPTX)
#if defined(__CLC_FPSIZE) || (__CLC_GENSIZE >= 32)
#define __CLC_HAS_ATOMIC
#endif
#endif // defined(__SPIR32__) || defined(CLC_NVPTX)

#ifdef __CLC_HAS_ATOMIC

#ifdef __CLC_FPSIZE

#define __CLC_DEFINE_ATOMIC(ADDRSPACE)                                         \
  _CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __clc_atomic_compare_exchange(          \
      volatile ADDRSPACE __CLC_GENTYPE *Ptr, __CLC_GENTYPE Comparator,         \
      __CLC_GENTYPE Value, int MemoryOrderEqual, int MemoryOrderUnequal,       \
      int MemoryScope) {                                                       \
    __CLC_U_GENTYPE Comp = __CLC_AS_U_GENTYPE(Comparator);                     \
    __scoped_atomic_compare_exchange_n(                                        \
        (ADDRSPACE __CLC_U_GENTYPE *)Ptr, &Comp, __CLC_AS_U_GENTYPE(Value),    \
        false, MemoryOrderEqual, MemoryOrderUnequal, MemoryScope);             \
    return __CLC_AS_GENTYPE(Comp);                                             \
  }

#else

#define __CLC_DEFINE_ATOMIC(ADDRSPACE)                                         \
  _CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __clc_atomic_compare_exchange(          \
      volatile ADDRSPACE __CLC_GENTYPE *Ptr, __CLC_GENTYPE Comparator,         \
      __CLC_GENTYPE Value, int MemoryOrderEqual, int MemoryOrderUnequal,       \
      int MemoryScope) {                                                       \
    __scoped_atomic_compare_exchange_n(Ptr, &Comparator, Value, false,         \
                                       MemoryOrderEqual, MemoryOrderUnequal,   \
                                       MemoryScope);                           \
    return Comparator;                                                         \
  }

#endif // __CLC_FPSIZE

__CLC_DEFINE_ATOMIC(global)
__CLC_DEFINE_ATOMIC(local)
#if _CLC_GENERIC_AS_SUPPORTED
__CLC_DEFINE_ATOMIC()
#endif

#undef __CLC_DEFINE_ATOMIC

#endif // __CLC_HAS_ATOMIC
#undef __CLC_HAS_ATOMIC

#endif // __CLC_SCALAR

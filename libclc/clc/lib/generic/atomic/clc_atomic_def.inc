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

#ifndef __CLC_PTR_CASTTYPE
#define __CLC_PTR_CASTTYPE __CLC_GENTYPE
#endif

#ifndef __CLC_AS_RETTYPE
#define __CLC_AS_RETTYPE(x) x
#endif

#ifdef __CLC_NO_VALUE_ARG
#define __CLC_DEFINE_ATOMIC(ADDRSPACE)                                         \
  _CLC_OVERLOAD _CLC_DECL __CLC_GENTYPE __CLC_FUNCTION(                        \
      volatile ADDRSPACE __CLC_GENTYPE *Ptr, int MemoryOrder,                  \
      int MemoryScope) {                                                       \
    return __CLC_AS_RETTYPE(__CLC_IMPL_FUNCTION(                               \
        (ADDRSPACE __CLC_PTR_CASTTYPE *)Ptr, MemoryOrder, MemoryScope));       \
  }
#elif defined(__CLC_INC_DEC)
#define __CLC_DEFINE_ATOMIC(ADDRSPACE)                                         \
  _CLC_OVERLOAD _CLC_DECL __CLC_GENTYPE __CLC_FUNCTION(                        \
      volatile ADDRSPACE __CLC_GENTYPE *Ptr, int MemoryOrder,                  \
      int MemoryScope) {                                                       \
    return __CLC_AS_RETTYPE(                                                   \
        __CLC_IMPL_FUNCTION((ADDRSPACE __CLC_PTR_CASTTYPE *)Ptr,               \
                            (__CLC_GENTYPE)1, MemoryOrder, MemoryScope));      \
  }
#elif defined(__CLC_RETURN_VOID)
#define __CLC_DEFINE_ATOMIC(ADDRSPACE)                                         \
  _CLC_OVERLOAD _CLC_DECL void __CLC_FUNCTION(                                 \
      volatile ADDRSPACE __CLC_GENTYPE *Ptr, __CLC_GENTYPE Value,              \
      int MemoryOrder, int MemoryScope) {                                      \
    __CLC_IMPL_FUNCTION((ADDRSPACE __CLC_PTR_CASTTYPE *)Ptr, Value,            \
                        MemoryOrder, MemoryScope);                             \
  }
#else
#define __CLC_DEFINE_ATOMIC(ADDRSPACE)                                         \
  _CLC_OVERLOAD _CLC_DECL __CLC_GENTYPE __CLC_FUNCTION(                        \
      volatile ADDRSPACE __CLC_GENTYPE *Ptr, __CLC_GENTYPE Value,              \
      int MemoryOrder, int MemoryScope) {                                      \
    return __CLC_AS_RETTYPE(                                                   \
        __CLC_IMPL_FUNCTION((ADDRSPACE __CLC_PTR_CASTTYPE *)Ptr, Value,        \
                            MemoryOrder, MemoryScope));                        \
  }
#endif

__CLC_DEFINE_ATOMIC(global)
__CLC_DEFINE_ATOMIC(local)
#if _CLC_GENERIC_AS_SUPPORTED
__CLC_DEFINE_ATOMIC()
#endif

#undef __CLC_DEFINE_ATOMIC

#endif // __CLC_HAS_ATOMIC
#undef __CLC_HAS_ATOMIC

#endif // __CLC_SCALAR

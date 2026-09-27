//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifdef __CLC_SCALAR

#if defined(__opencl_c_fp64) && (defined(cl_khr_int64_base_atomics) &&         \
                                 defined(cl_khr_int64_extended_atomics))
#define __CLC_HAVE_64_ATOMIC
#endif
#if defined(__CLC_FPSIZE) &&                                                   \
    (__CLC_FPSIZE < 64 || defined(__CLC_HAVE_64_ATOMIC))
#define __CLC_HAVE_FP_ATOMIC
#endif
#if defined(__CLC_GENSIZE) &&                                                  \
    ((__CLC_GENSIZE == 32) ||                                                  \
     (__CLC_GENSIZE == 64 && defined(__CLC_HAVE_64_ATOMIC)))
#define __CLC_HAVE_INT_ATOMIC
#endif
#if defined(__CLC_HAVE_FP_ATOMIC) || defined(__CLC_HAVE_INT_ATOMIC)

#define __CLC_ATOMIC_GENTYPE __CLC_XCONCAT(atomic_, __CLC_GENTYPE)

#define __CLC_FUNCTION_EXPLICIT __CLC_XCONCAT(__CLC_FUNCTION, _explicit)

#ifdef __CLC_NO_VALUE_ARG
#define __CLC_DEFINE_ATOMIC(ADDRSPACE)                                         \
  _CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __CLC_FUNCTION_EXPLICIT(                \
      volatile ADDRSPACE __CLC_ATOMIC_GENTYPE *Ptr, memory_order Order,        \
      memory_scope Scope) {                                                    \
    return __CLC_IMPL_FUNCTION((ADDRSPACE __CLC_GENTYPE *)Ptr, Order,          \
                               __opencl_get_clang_memory_scope(Scope));        \
  }
#elif defined(__CLC_RETURN_VOID)
#define __CLC_DEFINE_ATOMIC(ADDRSPACE)                                         \
  _CLC_OVERLOAD _CLC_DEF void __CLC_FUNCTION_EXPLICIT(                         \
      volatile ADDRSPACE __CLC_ATOMIC_GENTYPE *Ptr, __CLC_GENTYPE Value,       \
      memory_order Order, memory_scope Scope) {                                \
    __CLC_IMPL_FUNCTION((ADDRSPACE __CLC_GENTYPE *)Ptr, Value, Order,          \
                        __opencl_get_clang_memory_scope(Scope));               \
  }
#elif defined(__CLC_COMPARE_EXCHANGE)
#define __CLC_DEFINE_ATOMIC(ADDRSPACE)                                         \
  _CLC_OVERLOAD _CLC_DEF bool __CLC_FUNCTION_EXPLICIT(                         \
      volatile ADDRSPACE __CLC_ATOMIC_GENTYPE *Ptr,                            \
      ADDRSPACE __CLC_GENTYPE *Expected, __CLC_GENTYPE Desired,                \
      memory_order Success, memory_order Failure, memory_scope Scope) {        \
    __CLC_GENTYPE Comparator = *Expected;                                      \
    __CLC_GENTYPE RetValue = __CLC_IMPL_FUNCTION(                              \
        (ADDRSPACE __CLC_GENTYPE *)Ptr, Comparator, Desired, Success, Failure, \
        __opencl_get_clang_memory_scope(Scope));                               \
    if (Comparator != RetValue) {                                              \
      *Expected = RetValue;                                                    \
      return false;                                                            \
    }                                                                          \
    return true;                                                               \
  }
#else
#define __CLC_DEFINE_ATOMIC(ADDRSPACE)                                         \
  _CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __CLC_FUNCTION_EXPLICIT(                \
      volatile ADDRSPACE __CLC_ATOMIC_GENTYPE *Ptr, __CLC_GENTYPE Value,       \
      memory_order Order, memory_scope Scope) {                                \
    return __CLC_IMPL_FUNCTION((ADDRSPACE __CLC_GENTYPE *)Ptr, Value, Order,   \
                               __opencl_get_clang_memory_scope(Scope));        \
  }
#endif

__CLC_DEFINE_ATOMIC(global)
__CLC_DEFINE_ATOMIC(local)
#if _CLC_GENERIC_AS_SUPPORTED
__CLC_DEFINE_ATOMIC()
#endif

#undef __CLC_DEFINE_ATOMIC

#if defined(__opencl_c_atomic_scope_device)

#ifdef __CLC_NO_VALUE_ARG
#define __CLC_DEFINE_ATOMIC(ADDRSPACE)                                         \
  _CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __CLC_FUNCTION_EXPLICIT(                \
      volatile ADDRSPACE __CLC_ATOMIC_GENTYPE *Ptr, memory_order Order) {      \
    return __CLC_FUNCTION_EXPLICIT(Ptr, Order, __OPENCL_MEMORY_SCOPE_DEVICE);  \
  }
#elif defined(__CLC_RETURN_VOID)
#define __CLC_DEFINE_ATOMIC(ADDRSPACE)                                         \
  _CLC_OVERLOAD _CLC_DEF void __CLC_FUNCTION_EXPLICIT(                         \
      volatile ADDRSPACE __CLC_ATOMIC_GENTYPE *Ptr, __CLC_GENTYPE Value,       \
      memory_order Order) {                                                    \
    __CLC_FUNCTION_EXPLICIT(Ptr, Value, Order, __OPENCL_MEMORY_SCOPE_DEVICE);  \
  }
#elif defined(__CLC_COMPARE_EXCHANGE)
#define __CLC_DEFINE_ATOMIC(ADDRSPACE)                                         \
  _CLC_OVERLOAD _CLC_DEF bool __CLC_FUNCTION_EXPLICIT(                         \
      volatile ADDRSPACE __CLC_ATOMIC_GENTYPE *Ptr,                            \
      ADDRSPACE __CLC_GENTYPE *Expected, __CLC_GENTYPE Desired,                \
      memory_order Success, memory_order Failure) {                            \
    return __CLC_FUNCTION_EXPLICIT(Ptr, Expected, Desired, Success, Failure,   \
                                   __OPENCL_MEMORY_SCOPE_DEVICE);              \
  }
#else
#define __CLC_DEFINE_ATOMIC(ADDRSPACE)                                         \
  _CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __CLC_FUNCTION_EXPLICIT(                \
      volatile ADDRSPACE __CLC_ATOMIC_GENTYPE *Ptr, __CLC_GENTYPE Value,       \
      memory_order Order) {                                                    \
    return __CLC_FUNCTION_EXPLICIT(Ptr, Value, Order,                          \
                                   __OPENCL_MEMORY_SCOPE_DEVICE);              \
  }
#endif

__CLC_DEFINE_ATOMIC(global)
__CLC_DEFINE_ATOMIC(local)
#if _CLC_GENERIC_AS_SUPPORTED
__CLC_DEFINE_ATOMIC()
#endif

#undef __CLC_DEFINE_ATOMIC

#endif // defined(__opencl_c_atomic_scope_device)

#if defined(__opencl_c_atomic_order_seq_cst) &&                                \
    defined(__opencl_c_atomic_scope_device)

#ifdef __CLC_NO_VALUE_ARG
#define __CLC_DEFINE_ATOMIC(ADDRSPACE)                                         \
  _CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __CLC_FUNCTION(                         \
      volatile ADDRSPACE __CLC_ATOMIC_GENTYPE *Ptr) {                          \
    return __CLC_FUNCTION_EXPLICIT(Ptr, __ATOMIC_SEQ_CST,                      \
                                   __OPENCL_MEMORY_SCOPE_DEVICE);              \
  }
#elif defined(__CLC_RETURN_VOID)
#define __CLC_DEFINE_ATOMIC(ADDRSPACE)                                         \
  _CLC_OVERLOAD _CLC_DEF void __CLC_FUNCTION(                                  \
      volatile ADDRSPACE __CLC_ATOMIC_GENTYPE *Ptr, __CLC_GENTYPE Value) {     \
    __CLC_FUNCTION_EXPLICIT(Ptr, Value, __ATOMIC_SEQ_CST,                      \
                            __OPENCL_MEMORY_SCOPE_DEVICE);                     \
  }
#elif defined(__CLC_COMPARE_EXCHANGE)
#define __CLC_DEFINE_ATOMIC(ADDRSPACE)                                         \
  _CLC_OVERLOAD _CLC_DEF bool __CLC_FUNCTION(                                  \
      volatile ADDRSPACE __CLC_ATOMIC_GENTYPE *Ptr,                            \
      ADDRSPACE __CLC_GENTYPE *Expected, __CLC_GENTYPE Desired) {              \
    return __CLC_FUNCTION_EXPLICIT(Ptr, Expected, Desired, __ATOMIC_SEQ_CST,   \
                                   __ATOMIC_SEQ_CST,                           \
                                   __OPENCL_MEMORY_SCOPE_DEVICE);              \
  }
#else
#define __CLC_DEFINE_ATOMIC(ADDRSPACE)                                         \
  _CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __CLC_FUNCTION(                         \
      volatile ADDRSPACE __CLC_ATOMIC_GENTYPE *Ptr, __CLC_GENTYPE Value) {     \
    return __CLC_FUNCTION_EXPLICIT(Ptr, Value, __ATOMIC_SEQ_CST,               \
                                   __OPENCL_MEMORY_SCOPE_DEVICE);              \
  }
#endif

__CLC_DEFINE_ATOMIC(global)
__CLC_DEFINE_ATOMIC(local)
#if _CLC_GENERIC_AS_SUPPORTED
__CLC_DEFINE_ATOMIC()
#endif

#undef __CLC_DEFINE_ATOMIC

#endif // defined(__opencl_c_atomic_order_seq_cst) &&
       // defined(__opencl_c_atomic_scope_device)

#endif // __CLC_HAVE_FP_ATOMIC || __CLC_HAVE_INT_ATOMIC

#undef __CLC_HAVE_INT_ATOMIC
#undef __CLC_HAVE_FP_ATOMIC
#undef __CLC_HAVE_64_ATOMIC

#endif // __CLC_SCALAR

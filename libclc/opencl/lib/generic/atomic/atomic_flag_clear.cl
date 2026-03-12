//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/atomic/clc_atomic_flag_clear.h>
#include <clc/opencl/utils.h>

#if defined(__opencl_c_atomic_order_seq_cst) &&                                \
    defined(__opencl_c_atomic_scope_device)

#define __CLC_DEFINE_ATOMIC_FLAG_CLEAR(ADDRSPACE)                              \
  _CLC_OVERLOAD _CLC_DEF void atomic_flag_clear(                               \
      volatile ADDRSPACE atomic_flag *object) {                                \
    __clc_atomic_flag_clear((ADDRSPACE int *)object, __ATOMIC_SEQ_CST,         \
                            __MEMORY_SCOPE_DEVICE);                            \
  }

__CLC_DEFINE_ATOMIC_FLAG_CLEAR(global)
__CLC_DEFINE_ATOMIC_FLAG_CLEAR(local)
#if defined(__opencl_c_generic_address_space)
__CLC_DEFINE_ATOMIC_FLAG_CLEAR()
#endif

#endif // defined(__opencl_c_atomic_order_seq_cst) &&
       // defined(__opencl_c_atomic_scope_device)

#if defined(__opencl_c_atomic_scope_device)

#define __CLC_DEFINE_ATOMIC_FLAG_CLEAR_ORDER(ADDRSPACE)                        \
  _CLC_OVERLOAD _CLC_DEF void atomic_flag_clear_explicit(                      \
      volatile ADDRSPACE atomic_flag *object, memory_order order) {            \
    __clc_atomic_flag_clear((ADDRSPACE int *)object, order,                    \
                            __MEMORY_SCOPE_DEVICE);                            \
  }

__CLC_DEFINE_ATOMIC_FLAG_CLEAR_ORDER(global)
__CLC_DEFINE_ATOMIC_FLAG_CLEAR_ORDER(local)
#if defined(__opencl_c_generic_address_space)
__CLC_DEFINE_ATOMIC_FLAG_CLEAR_ORDER()
#endif

#endif // defined(__opencl_c_atomic_scope_device)

#define __CLC_DEFINE_ATOMIC_FLAG_CLEAR_ORDER_SCOPE(ADDRSPACE)                  \
  _CLC_OVERLOAD _CLC_DEF void atomic_flag_clear_explicit(                      \
      volatile ADDRSPACE atomic_flag *object, memory_order order,              \
      memory_scope scope) {                                                    \
    __clc_atomic_flag_clear((ADDRSPACE int *)object, order,                    \
                            __opencl_get_clang_memory_scope(scope));           \
  }

__CLC_DEFINE_ATOMIC_FLAG_CLEAR_ORDER_SCOPE(global)
__CLC_DEFINE_ATOMIC_FLAG_CLEAR_ORDER_SCOPE(local)
#if defined(__opencl_c_generic_address_space)
__CLC_DEFINE_ATOMIC_FLAG_CLEAR_ORDER_SCOPE()
#endif

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/atomic/clc_atomic_flag_test_and_set.h>
#include <clc/opencl/utils.h>

#if defined(__opencl_c_atomic_order_seq_cst) &&                                \
    defined(__opencl_c_atomic_scope_device)

#define __CLC_DEFINE_ATOMIC_FLAG_TEST_AND_SET(ADDRSPACE)                       \
  _CLC_OVERLOAD _CLC_DEF bool atomic_flag_test_and_set(                        \
      volatile ADDRSPACE atomic_flag *object) {                                \
    return __clc_atomic_flag_test_and_set(                                     \
        (ADDRSPACE int *)object, __ATOMIC_SEQ_CST, __MEMORY_SCOPE_DEVICE);     \
  }

__CLC_DEFINE_ATOMIC_FLAG_TEST_AND_SET(global)
__CLC_DEFINE_ATOMIC_FLAG_TEST_AND_SET(local)
#if defined(__opencl_c_generic_address_space)
__CLC_DEFINE_ATOMIC_FLAG_TEST_AND_SET()
#endif

#undef __CLC_DEFINE_ATOMIC_FLAG_TEST_AND_SET

#endif // defined(__opencl_c_atomic_order_seq_cst) &&
       // defined(__opencl_c_atomic_scope_device)

#if defined(__opencl_c_atomic_scope_device)

#define __CLC_DEFINE_ATOMIC_FLAG_TEST_AND_SET(ADDRSPACE)                       \
  _CLC_OVERLOAD _CLC_DEF bool atomic_flag_test_and_set_explicit(               \
      volatile ADDRSPACE atomic_flag *object, memory_order order) {            \
    return __clc_atomic_flag_test_and_set((ADDRSPACE int *)object, order,      \
                                          __MEMORY_SCOPE_DEVICE);              \
  }

__CLC_DEFINE_ATOMIC_FLAG_TEST_AND_SET(global)
__CLC_DEFINE_ATOMIC_FLAG_TEST_AND_SET(local)
#if defined(__opencl_c_generic_address_space)
__CLC_DEFINE_ATOMIC_FLAG_TEST_AND_SET()
#endif

#undef __CLC_DEFINE_ATOMIC_FLAG_TEST_AND_SET

#endif // defined(__opencl_c_atomic_scope_device)

#define __CLC_DEFINE_ATOMIC_FLAG_TEST_AND_SET(ADDRSPACE)                       \
  _CLC_OVERLOAD _CLC_DEF bool atomic_flag_test_and_set_explicit(               \
      volatile ADDRSPACE atomic_flag *object, memory_order order,              \
      memory_scope scope) {                                                    \
    return __clc_atomic_flag_test_and_set(                                     \
        (ADDRSPACE int *)object, order,                                        \
        __opencl_get_clang_memory_scope(scope));                               \
  }

__CLC_DEFINE_ATOMIC_FLAG_TEST_AND_SET(global)
__CLC_DEFINE_ATOMIC_FLAG_TEST_AND_SET(local)
#if defined(__opencl_c_generic_address_space)
__CLC_DEFINE_ATOMIC_FLAG_TEST_AND_SET()
#endif

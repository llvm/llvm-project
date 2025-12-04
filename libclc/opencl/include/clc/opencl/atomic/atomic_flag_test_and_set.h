//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __CLC_OPENCL_ATOMIC_ATOMIC_FLAG_TEST_AND_SET_H__
#define __CLC_OPENCL_ATOMIC_ATOMIC_FLAG_TEST_AND_SET_H__

#include <clc/opencl/opencl-base.h>
#include <clc/opencl/types.h>

#if defined(__opencl_c_atomic_order_seq_cst) &&                                \
    defined(__opencl_c_atomic_scope_device)
_CLC_OVERLOAD _CLC_DECL bool
atomic_flag_test_and_set(volatile __global atomic_flag *);
_CLC_OVERLOAD _CLC_DECL bool
atomic_flag_test_and_set(volatile __local atomic_flag *);
#if defined(__opencl_c_generic_address_space)
_CLC_OVERLOAD _CLC_DECL bool atomic_flag_test_and_set(volatile atomic_flag *);
#endif // defined(__opencl_c_generic_address_space)
#endif

#if defined(__opencl_c_atomic_scope_device)
_CLC_OVERLOAD _CLC_DECL bool
atomic_flag_test_and_set_explicit(volatile __global atomic_flag *,
                                  memory_order);
_CLC_OVERLOAD _CLC_DECL bool
atomic_flag_test_and_set_explicit(volatile __local atomic_flag *, memory_order);
#if defined(__opencl_c_generic_address_space)
_CLC_OVERLOAD _CLC_DECL bool
atomic_flag_test_and_set_explicit(volatile atomic_flag *, memory_order);
#endif // defined(__opencl_c_generic_address_space)
#endif

_CLC_OVERLOAD _CLC_DECL bool
atomic_flag_test_and_set_explicit(volatile __global atomic_flag *, memory_order,
                                  memory_scope);
_CLC_OVERLOAD _CLC_DECL bool
atomic_flag_test_and_set_explicit(volatile __local atomic_flag *, memory_order,
                                  memory_scope);
#if defined(__opencl_c_generic_address_space)
_CLC_OVERLOAD _CLC_DECL bool
atomic_flag_test_and_set_explicit(volatile atomic_flag *, memory_order,
                                  memory_scope);
#endif // defined(__opencl_c_generic_address_space)

#endif // __CLC_OPENCL_ATOMIC_ATOMIC_FLAG_TEST_AND_SET_H__

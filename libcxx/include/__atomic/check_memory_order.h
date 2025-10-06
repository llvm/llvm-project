//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ATOMIC_CHECK_MEMORY_ORDER_H
#define _LIBCPP___ATOMIC_CHECK_MEMORY_ORDER_H

#include <__config>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

#define _LIBCPP_CHECK_STORE_MEMORY_ORDER(__m)                                                                          \
  _LIBCPP_LEGACY_DIAGNOSE_IF(                                                                                                 \
      "atomic-memory-ordering",                                                                                        \
      "warning",                                                                                                       \
      "memory order argument to atomic operation is invalid",                                                          \
      __m == memory_order_consume || __m == memory_order_acquire || __m == memory_order_acq_rel)

#define _LIBCPP_CHECK_LOAD_MEMORY_ORDER(__m)                                                                           \
  _LIBCPP_LEGACY_DIAGNOSE_IF(                                                                                                 \
      "atomic-memory-ordering",                                                                                        \
      "warning",                                                                                                       \
      "memory order argument to atomic operation is invalid",                                                          \
      __m == memory_order_release || __m == memory_order_acq_rel)

#define _LIBCPP_CHECK_EXCHANGE_MEMORY_ORDER(__m, __f)                                                                  \
  _LIBCPP_LEGACY_DIAGNOSE_IF(                                                                                                 \
      "atomic-memory-ordering",                                                                                        \
      "warning",                                                                                                       \
      "memory order argument to atomic operation is invalid",                                                          \
      __f == memory_order_release || __f == memory_order_acq_rel)

#define _LIBCPP_CHECK_WAIT_MEMORY_ORDER(__m)                                                                           \
  _LIBCPP_LEGACY_DIAGNOSE_IF(                                                                                                 \
      "atomic-memory-ordering",                                                                                        \
      "warning",                                                                                                       \
      "memory order argument to atomic operation is invalid",                                                          \
      __m == memory_order_release || __m == memory_order_acq_rel)

#endif // _LIBCPP___ATOMIC_CHECK_MEMORY_ORDER_H

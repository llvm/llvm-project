//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ATOMIC_TO_FAILURE_ORDER_H
#define _LIBCPP___ATOMIC_TO_FAILURE_ORDER_H

#include <__atomic/memory_order.h>
#include <__config>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

_LIBCPP_HIDE_FROM_ABI inline _LIBCPP_CONSTEXPR memory_order __to_failure_order(memory_order __order) {
  // Avoid switch statement to make this a constexpr.
  return __order == memory_order_release
           ? memory_order_relaxed
           : (__order == memory_order_acq_rel ? memory_order_acquire : __order);
}

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___ATOMIC_TO_FAILURE_ORDER_H

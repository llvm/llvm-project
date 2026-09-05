//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ATOMIC_TO_GCC_ORDER_H
#define _LIBCPP___ATOMIC_TO_GCC_ORDER_H

#include <__atomic/memory_order.h>
#include <__config>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

#if defined(__ATOMIC_RELAXED) && defined(__ATOMIC_CONSUME) && defined(__ATOMIC_ACQUIRE) &&                             \
    defined(__ATOMIC_RELEASE) && defined(__ATOMIC_ACQ_REL) && defined(__ATOMIC_SEQ_CST)

_LIBCPP_HIDE_FROM_ABI inline int __to_gcc_order(memory_order __order) {
  switch (__order) {
  case memory_order_relaxed:
    return __ATOMIC_RELAXED;
  case memory_order_acquire:
    return __ATOMIC_ACQUIRE;
  case memory_order_release:
    return __ATOMIC_RELEASE;
  case memory_order_acq_rel:
    return __ATOMIC_ACQ_REL;
  case memory_order_seq_cst:
    return __ATOMIC_SEQ_CST;
  case memory_order_consume:
    return __ATOMIC_CONSUME;
  }
}

_LIBCPP_HIDE_FROM_ABI inline int __to_gcc_failure_order(memory_order __order) {
  switch (__order) {
  case memory_order_relaxed:
    return __ATOMIC_RELAXED;
  case memory_order_acquire:
    return __ATOMIC_ACQUIRE;
  case memory_order_release:
    return __ATOMIC_RELAXED;
  case memory_order_seq_cst:
    return __ATOMIC_SEQ_CST;
  case memory_order_acq_rel:
    return __ATOMIC_ACQUIRE;
  case memory_order_consume:
    return __ATOMIC_CONSUME;
  }
}

#endif

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___ATOMIC_TO_GCC_ORDER_H

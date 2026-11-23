//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___CXX03___ATOMIC_MEMORY_ORDER_H
#define _LIBCPP___CXX03___ATOMIC_MEMORY_ORDER_H

#include <__cxx03/__config>
#include <__cxx03/__type_traits/is_same.h>
#include <__cxx03/__type_traits/underlying_type.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

// Figure out what the underlying type for `memory_order` would be if it were
// declared as an unscoped enum (accounting for -fshort-enums). Use this result
// to pin the underlying type in C++20.
enum __legacy_memory_order { __mo_relaxed, __mo_consume, __mo_acquire, __mo_release, __mo_acq_rel, __mo_seq_cst };

using __memory_order_underlying_t = underlying_type<__legacy_memory_order>::type;

enum memory_order {
  memory_order_relaxed = __mo_relaxed,
  memory_order_consume = __mo_consume,
  memory_order_acquire = __mo_acquire,
  memory_order_release = __mo_release,
  memory_order_acq_rel = __mo_acq_rel,
  memory_order_seq_cst = __mo_seq_cst,
};

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___CXX03___ATOMIC_MEMORY_ORDER_H

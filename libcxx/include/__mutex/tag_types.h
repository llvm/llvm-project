//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___MUTEX_TAG_TYPES_H
#define _LIBCPP___MUTEX_TAG_TYPES_H

#include <__config>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

#ifndef _LIBCPP_HAS_NO_THREADS

_LIBCPP_BEGIN_NAMESPACE_STD

struct _LIBCPP_TYPE_VIS defer_lock_t {
  explicit defer_lock_t() = default;
};

struct _LIBCPP_TYPE_VIS try_to_lock_t {
  explicit try_to_lock_t() = default;
};

struct _LIBCPP_TYPE_VIS adopt_lock_t {
  explicit adopt_lock_t() = default;
};

#  if !defined(_LIBCPP_CXX03_LANG)
/* inline */ constexpr defer_lock_t defer_lock   = defer_lock_t();
/* inline */ constexpr try_to_lock_t try_to_lock = try_to_lock_t();
/* inline */ constexpr adopt_lock_t adopt_lock   = adopt_lock_t();
#  endif

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_HAS_NO_THREADS

#endif // _LIBCPP___MUTEX_TAG_TYPES_H

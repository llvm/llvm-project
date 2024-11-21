// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___UTILITY_SCOPE_GUARD_H
#define _LIBCPP___UTILITY_SCOPE_GUARD_H

#include <__assert>
#include <__config>
#include <__utility/move.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _Func>
class __scope_guard {
  _Func __func_;
  bool __moved_from_;

public:
  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR __scope_guard(_Func __func) : __func_(std::move(__func)) {}
  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 ~__scope_guard() { __func_(); }

  __scope_guard(const __scope_guard&) = delete;

// C++17 has mandatory RVO, so we don't need the move constructor anymore to make __make_scope_guard work.
#if _LIBCPP_STD_VER <= 14
  __scope_guard(__scope_guard&& __other) : __func_(__other.__func_) {
    _LIBCPP_ASSERT_INTERNAL(!__other.__moved_from_, "Cannot move twice from __scope_guard");
    __other.__moved_from_ = true;
  }
#else
  __scope_guard(__scope_guard&&) = delete;
#endif

  __scope_guard& operator=(const __scope_guard&) = delete;
  __scope_guard& operator=(__scope_guard&&)      = delete;
};

template <class _Func>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX14 __scope_guard<_Func> __make_scope_guard(_Func __func) {
  return __scope_guard<_Func>(std::move(__func));
}

_LIBCPP_END_NAMESPACE_STD

_LIBCPP_POP_MACROS

#endif // _LIBCPP___UTILITY_SCOPE_GUARD_H

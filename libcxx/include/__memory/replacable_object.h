//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___MEMORY_REPLACABLE_OBJECT_H
#define _LIBCPP___MEMORY_REPLACABLE_OBJECT_H

#include <__config>
#include <__memory/addressof.h>
#include <__memory/construct_at.h>
#include <__type_traits/is_assignable.h>
#include <__type_traits/is_nothrow_assignable.h>
#include <__type_traits/is_nothrow_constructible.h>
#include <__utility/exception_guard.h>
#include <__utility/forward.h>
#include <__utility/move.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _Tp>
class __replacable_object {
  union {
    _Tp __value_;
  };
  bool __has_object_ = true;

public:
  template <class... _Args>
  _LIBCPP_CONSTEXPR_SINCE_CXX14 __replacable_object(_Args&&... __args) : __value_(std::forward<_Args>(__args)...) {}

  __replacable_object(const __replacable_object&)            = delete;
  __replacable_object& operator=(const __replacable_object&) = delete;

  _LIBCPP_CONSTEXPR_SINCE_CXX20 ~__replacable_object() {
    if (__has_object_)
      std::__destroy_at(std::addressof(__value_));
  }

  _LIBCPP_CONSTEXPR_SINCE_CXX14 void __replace(_Tp&& __v) {
#ifndef _LIBCPP_CXX03_LANG
    // Try to ensure that no exception is thrown
    if constexpr (is_nothrow_move_assignable<_Tp>::value) {
      __value_ = std::move(__v);
    } else if constexpr (is_nothrow_move_constructible<_Tp>::value) {
      std::__destroy_at(std::addressof(__value_));
      std::__construct_at(std::addressof(__value_), std::move(__v));
    } else
#endif
    {
      auto __guard = std::__make_exception_guard([&] { __has_object_ = false; });
      std::__destroy_at(std::addressof(__value_));
      std::__construct_at(std::addressof(__value_), std::move(__v));
      __guard.__complete();
    }
  }

  _LIBCPP_CONSTEXPR_SINCE_CXX14 _Tp& __get() { return __value_; }
  _LIBCPP_CONSTEXPR const _Tp& __get() const { return __value_; }
};

_LIBCPP_END_NAMESPACE_STD

_LIBCPP_POP_MACROS

#endif // _LIBCPP___MEMORY_REPLACABLE_OBJECT_H
